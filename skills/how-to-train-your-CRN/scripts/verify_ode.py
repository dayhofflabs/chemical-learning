"""Post-training ODE verification: does the dynamics actually reach the root?

A root finder returns *a* root. Newton is perfectly happy to land on a saddle
or an unstable branch that the real dynamics would never settle into, so the
fitted response is not trustworthy until the mass-action (or Hill, or
whatever) ODE has been integrated to steady state at every grid point and
agreed. This pass is the arbiter behind the success gate:

    R^2 >= 0.98 on the ODE-verified response  AND  RMS(ODE, root finder) < 1e-2

Strategy (ported from batch_ode_sweep_v2)
-----------------------------------------
* Each point is integrated single-shot to ``t_max`` with the stiffest solver
  available; only on failure is it escalated down a ladder of looser
  tolerances with staged time checkpoints.
* ``dt0`` is chosen adaptively from the RHS magnitude, ``|x|/|F(x)|``. Fixed
  ``dt0`` is why stiff points used to stall for minutes and then report a
  bogus failure.
* The quasistatic pass warm-starts each point from the previous converged ODE
  state, and a failed step is retried by **bisecting the control interval**
  (linear interpolation between consecutive control vectors, up to
  ``2**max_bisect`` substeps) so a stiff jump is walked rather than jumped.
* No wall-clock cap by default. A timeout here does not mean divergence, and
  capping it manufactures phantom failures.

Requires implicit solvers to be importable *and runnable*: some
diffrax/equinox combinations throw for every implicit solver at construction
time. ``probe_solvers`` detects that and the ladder falls back to an explicit
solver, which is **not** equivalent for stiff systems: treat those results as
indicative and fix the environment.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

DEFAULT_CHAIN = [
    {'name': 'kvaerno5', 'atol': 1e-4, 'rtol': 1e-3, 'max_steps': 5_000_000},
    {'name': 'kvaerno3', 'atol': 1e-3, 'rtol': 1e-3, 'max_steps': 5_000_000},
    {'name': 'implicit_euler', 'atol': 1e-2, 'rtol': 1e-2,
     'max_steps': 10_000_000},
]

#: Used only when no implicit solver runs in this environment. An explicit
#: solver's step is capped by stability, not accuracy, so a single shot to
#: t_max=1e9 would need ~1e9 steps: these entries are staged-only and get a
#: default wall-clock cap.
EXPLICIT_FALLBACK = [
    {'name': 'tsit5', 'atol': 1e-6, 'rtol': 1e-6, 'max_steps': 1_000_000,
     'explicit_fallback': True},
]

#: Wall-clock cap applied only when the chain degraded to explicit solvers.
FALLBACK_TIMEOUT_S = 120.0

TIME_STAGES = [1e2, 1e3, 1e4, 1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 1e9]

DEFAULTS = {
    't_max': 1e9,
    'residual_tol': 1e-4,
    'event_tol': 5e-5,
    'max_bisect': 5,
    'mode': 'both',
    'wall_clock_timeout': 0.0,
}


def _solver_object(name):
    import diffrax as dfx
    return {
        'kvaerno5': dfx.Kvaerno5,
        'kvaerno3': dfx.Kvaerno3,
        'implicit_euler': dfx.ImplicitEuler,
        'tsit5': dfx.Tsit5,
        'dopri8': dfx.Dopri8,
    }[name]()


def probe_solvers(names):
    """Return the subset of ``names`` that actually integrates a stiff scalar."""
    import diffrax as dfx

    ok = []
    for name in names:
        try:
            solver = _solver_object(name)
            dfx.diffeqsolve(
                dfx.ODETerm(lambda t, y, a: -1e3 * y + 1.0), solver,
                t0=0.0, t1=1.0, dt0=1e-4, y0=jnp.array([1.0]),
                stepsize_controller=dfx.PIDController(atol=1e-6, rtol=1e-6),
                saveat=dfx.SaveAt(t1=True), max_steps=100_000, throw=False)
            ok.append(name)
        except Exception:
            continue
    return ok


def resolve_chain(chain=None, log=print):
    """Drop unavailable solvers; fall back to explicit if none survive."""
    chain = list(chain or DEFAULT_CHAIN)
    available = set(probe_solvers([c['name'] for c in chain]))
    usable = [c for c in chain if c['name'] in available]
    if usable:
        dropped = [c['name'] for c in chain if c['name'] not in available]
        if dropped:
            log(f"  WARNING: solvers unavailable in this environment: {dropped}")
        return usable

    fallback = [c for c in EXPLICIT_FALLBACK
                if c['name'] in set(probe_solvers(
                    [c['name'] for c in EXPLICIT_FALLBACK]))]
    if not fallback:
        raise RuntimeError(
            "no diffrax solver runs in this environment; ODE verification "
            "cannot be trusted. Fix the diffrax/equinox pin.")
    log("  WARNING: no implicit solver runs here (diffrax/equinox pin). "
        f"Falling back to {[c['name'] for c in fallback]} — NOT equivalent "
        "for stiff systems; results are indicative only.")
    return fallback


def verify(F, params, controls, x0s, target, readout_idx, reference_preds,
           cfg=None, chain=None, log=print):
    """Integrate to steady state at every control point.

    Args:
        F: residual ``(x, params, control) -> (n_state,)``, i.e. ``x_dot``.
        x0s: (K, n_state) root-finder steady states — the warm pass starts here.
        reference_preds: (K,) root-finder readout, for the RMS comparison.

    Returns a dict of per-point arrays plus summary scalars, or None if
    verification could not run.
    """
    import diffrax as dfx

    cfg = {**DEFAULTS, **(cfg or {})}
    t_max = float(cfg['t_max'])
    residual_tol = float(cfg['residual_tol'])
    event_tol = float(cfg['event_tol'])
    max_bisect = int(cfg['max_bisect'])
    mode = str(cfg['mode'])
    timeout = float(cfg['wall_clock_timeout'])
    timeout = float('inf') if timeout <= 0 else timeout

    chain = resolve_chain(chain, log=log)
    log(f"  solver chain: {[c['name'] for c in chain]}")
    if any(c.get('explicit_fallback') for c in chain) and timeout == float('inf'):
        timeout = FALLBACK_TIMEOUT_S
        log(f"  degraded chain: applying a {timeout:.0f}s wall-clock cap so a "
            "stability-limited explicit solver cannot run away. Set "
            "ode_verification.wall_clock_timeout to override.")

    controls = jnp.asarray(controls)
    x0s = jnp.asarray(x0s)
    n_points, n_state = x0s.shape

    def _make_integrator(cfg_i):
        solver = _solver_object(cfg_i['name'])
        ctrl = dfx.PIDController(atol=cfg_i['atol'], rtol=cfg_i['rtol'])

        @jax.jit
        def integrate(u0, control, dt0, t1, p):
            def rhs(_t, u, _args):
                return F(jnp.maximum(u, 1e-15), p, control)

            sol = dfx.diffeqsolve(
                dfx.ODETerm(rhs), solver, t0=0.0, t1=t1, dt0=dt0, y0=u0,
                saveat=dfx.SaveAt(t1=True), stepsize_controller=ctrl,
                max_steps=int(cfg_i['max_steps']), throw=False)
            x_final = jnp.maximum(sol.ys[-1], 1e-15)
            return x_final, jnp.max(jnp.abs(F(x_final, p, control)))

        return integrate

    integrators = {c['name']: _make_integrator(c) for c in chain}

    def adaptive_dt0(x, control):
        Fv = F(jnp.maximum(x, 1e-15), params, control)
        return jnp.clip(jnp.max(jnp.abs(x)) / (jnp.max(jnp.abs(Fv)) + 1e-30),
                        1e-12, 1.0)

    def _run(name, u0, control, t1):
        dt0 = adaptive_dt0(u0, control)
        x, res = integrators[name](u0, control, dt0, jnp.asarray(float(t1)),
                                   params)
        x = jax.block_until_ready(x)
        res = float(res)
        return x, res

    def _single_shot(name, u0, control):
        x, res = _run(name, u0, control, t_max)
        return x, res, res <= event_tol

    def _staged(name, u0, control):
        x, res = u0, float('inf')
        for t1 in TIME_STAGES:
            if t1 > t_max:
                break
            x, res = _run(name, x, control, t1)
            if res <= event_tol:
                return x, res, True
        return x, res, False

    def integrate_escalating(u0, control):
        x, res = u0, float('inf')
        for i, cfg_i in enumerate(chain):
            # The first solver gets one cheap shot straight to t_max; later ones
            # walk the time checkpoints, continuing from where the last stopped.
            staged = i > 0 or bool(cfg_i.get('explicit_fallback'))
            runner = _staged if staged else _single_shot
            x, res, done = runner(cfg_i['name'], x, control)
            if done or res <= residual_tol:
                return x, res, cfg_i['name']
        return x, res, 'failed'

    def integrate_bisecting(x_prev, control_prev, control_target):
        """Walk control_prev -> control_target, doubling substeps until it holds."""
        x, res = x_prev, float('inf')
        for level in range(max_bisect + 1):
            alphas = np.linspace(0.0, 1.0, 2 ** level + 1)[1:]
            x, ok = x_prev, True
            for alpha in alphas:
                control = (1.0 - alpha) * control_prev + alpha * control_target
                x, res, _ = integrate_escalating(x, control)
                if res > residual_tol:
                    ok = False
                    break
            if ok:
                return x, res, level
        return x, res, max_bisect

    out = {}
    t_start = time.time()

    for pass_name, enabled in (('warm', mode in ('warm', 'both')),
                               ('quasistatic', mode in ('quasistatic', 'both'))):
        states = np.zeros((n_points, n_state))
        residuals = np.zeros(n_points)
        okmask = np.zeros(n_points, dtype=bool)
        solvers_used = [''] * n_points
        bisections = np.zeros(n_points, dtype=int)

        if not enabled:
            out[pass_name] = {'states': states, 'residuals': residuals,
                              'ok': okmask, 'solvers': solvers_used,
                              'bisections': bisections, 'ran': False}
            continue

        x_prev = None
        n_attempted = 0
        for k in range(n_points):
            if time.time() - t_start > timeout:
                log(f"  {pass_name}: wall-clock cap hit at point {k}/{n_points}"
                    " — statistics below cover the attempted points only")
                break
            n_attempted = k + 1

            if pass_name == 'warm' or x_prev is None:
                x, res, used = integrate_escalating(x0s[k], controls[k])
                level = 0
            else:
                x, res, level = integrate_bisecting(
                    x_prev, controls[k - 1], controls[k])
                used = 'bisect' if level > 0 else 'quasistatic'
                if res > residual_tol:
                    # One bad point must not poison the rest of the sweep.
                    x, res, used = integrate_escalating(x0s[k], controls[k])

            states[k] = np.asarray(x)
            residuals[k] = res
            okmask[k] = res <= residual_tol
            solvers_used[k] = used
            bisections[k] = level
            x_prev = x if okmask[k] else jnp.asarray(x0s[k])

        out[pass_name] = {'states': states, 'residuals': residuals,
                          'ok': okmask, 'solvers': solvers_used,
                          'bisections': bisections, 'ran': True,
                          'n_attempted': n_attempted}

    target = np.asarray(target)
    reference_preds = np.asarray(reference_preds)
    summary = {'n_points': n_points,
               'chain': [c['name'] for c in chain],
               'wall_time_s': time.time() - t_start}

    for pass_name, res in out.items():
        # A pass that did not run has no statistics. Reporting zeros here reads
        # as a catastrophic failure of a pass nobody asked for.
        summary[f'{pass_name}_ran'] = bool(res['ran'])
        if not res['ran']:
            continue
        preds = (res['states'][:, int(readout_idx)]
                 if readout_idx is not None else np.zeros(n_points))
        res['readout'] = preds

        # A wall-clock cap truncates the sweep. Points that were never
        # attempted are still zeros; folding them into R^2 turns "we ran out of
        # time" into "the fit is catastrophically wrong".
        n_att = int(res.get('n_attempted', n_points))
        summary[f'{pass_name}_n_attempted'] = n_att
        summary[f'{pass_name}_truncated'] = n_att < n_points
        sl = slice(0, n_att)
        p_att, t_att, r_att = preds[sl], target[sl], reference_preds[sl]

        ss_res = float(np.sum((p_att - t_att) ** 2))
        ss_tot = float(np.sum((t_att - t_att.mean()) ** 2)) if n_att else 0.0
        summary[f'{pass_name}_n_ok'] = int(res['ok'].sum())
        summary[f'{pass_name}_max_residual'] = float(
            res['residuals'][sl].max() if n_att else 0.0)
        summary[f'{pass_name}_rms_vs_reference'] = float(
            np.sqrt(np.mean((p_att - r_att) ** 2))) if n_att else float('inf')
        summary[f'{pass_name}_r_squared'] = (
            1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))
        summary[f'{pass_name}_n_bisected'] = int((res['bisections'] > 0).sum())

    return {'passes': out, 'summary': summary}


def success_gate(summary, pass_name='quasistatic', r2_min=0.98, rms_max=1e-2):
    """The paper's gate: ODE-verified R^2 and ODE-vs-root-finder agreement.

    Deliberately separate from the trainer's ``exit_reason``. A run can exit
    with ``grad_norm``/``loss_plateau`` (trainer says "converged") and still
    fail this.
    """
    r2 = summary.get(f'{pass_name}_r_squared', float('nan'))
    rms = summary.get(f'{pass_name}_rms_vs_reference', float('inf'))
    n_ok = summary.get(f'{pass_name}_n_ok', 0)
    return {
        'success': bool(r2 >= r2_min and rms < rms_max
                        and n_ok == summary['n_points']),
        'r_squared': r2,
        'rms_vs_reference': rms,
        'n_ok': n_ok,
        'n_points': summary['n_points'],
        # Distinguishes "the fit is wrong" from "verification ran out of time".
        'truncated': bool(summary.get(f'{pass_name}_truncated', False)),
    }


def comparison_plot(path, grid_axis, target, reference_preds, result):
    """target vs root finder vs ODE passes. Cheap, and the fastest way for a
    human to see that a 'converged' run landed on the wrong branch."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(grid_axis, target, 'k-', lw=2, label='target')
    ax.plot(grid_axis, reference_preds, 'o--', ms=4, label='root finder')
    for name, res in result['passes'].items():
        if res.get('ran') and 'readout' in res:
            ax.plot(grid_axis, res['readout'], 's:', ms=3, label=f'ODE {name}')
    ax.set_xlabel('control grid index / value')
    ax.set_ylabel('readout')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
