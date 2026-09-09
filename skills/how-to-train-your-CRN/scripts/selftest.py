#!/usr/bin/env python3
"""Check a Problem against the residual contract before training it.

    python selftest.py --plugin ../examples/mass_action.py [--cfg cfg.json]

Runs seven checks, in the order things actually go wrong:

1. params is a flat dict of finite real arrays
2. F is finite at the state floor and at the initial guess
3. |F| scale vs trust_tol / newton_atol  (both are ABSOLUTE thresholds)
4. controls vary smoothly along the grid ordering
5. the grid solves, and dF/dx is well conditioned at the roots
6. the roots are dynamically stable (max Re eig(J) < 0) — an unstable root is
   a root the ODE verification will refuse to reproduce
7. the adjoint gradient matches central finite differences of the re-solved
   loss  <-- the one that catches a wrong residual

Exit code is nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model as M  # noqa: E402
from gradients import make_grad_fn  # noqa: E402
from solvers import make_adjoint_solver, solve_grid  # noqa: E402
from train_crn import load_plugin  # noqa: E402

jax.config.update("jax_enable_x64", True)

TIGHT = {'newton_maxiter': 300, 'newton_atol': 1e-12, 'newton_rtol': 1e-10}


class Report:
    def __init__(self):
        self.failed = []

    def check(self, ok, name, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))
        if not ok:
            self.failed.append(name)
        return ok

    def note(self, msg):
        print(f"         {msg}")


def main():
    ap = argparse.ArgumentParser(
        description="Check a Problem against the residual contract. "
                    "Exit code is nonzero on any failure. Run this on a "
                    "new plugin BEFORE training against it — a wrong "
                    "residual otherwise presents as 'training plateaus "
                    "for no reason'.",
        epilog="example: python selftest.py --plugin examples/minimal.py")
    ap.add_argument('--plugin', required=True,
                    help='path to a plugin file exposing build(cfg) -> Problem')
    ap.add_argument('--cfg',
                    help='JSON file or inline JSON string passed to build()')
    ap.add_argument('--seed', type=int, default=0,
                    help='PRNG key for init_params (default: 0)')
    ap.add_argument('--fd_probes', type=int, default=3,
                    help='finite-difference probes per parameter leaf '
                         '(default: 3)')
    ap.add_argument('--fd_eps', type=float, default=1e-6,
                    help='FD step size, scaled by |param| (default: 1e-6)')
    args = ap.parse_args()

    cfg = {}
    if args.cfg:
        cfg = (json.load(open(args.cfg)) if os.path.exists(args.cfg)
               else json.loads(args.cfg))

    plugin = load_plugin(args.plugin, os.getcwd())
    problem = plugin.build(cfg)
    mdl = problem.model
    F = mdl.residual
    controls = jnp.asarray(problem.controls)
    target = jnp.asarray(problem.target)
    K, N = controls.shape[0], mdl.n_state
    rep = Report()

    print(f"\nmodel '{mdl.name}': {N} states, {K} grid points, "
          f"floor={mdl.state_floor:g}")

    # 1. parameters
    params = dict(mdl.init_params(jax.random.PRNGKey(args.seed)))
    flat_ok = all(isinstance(v, (jnp.ndarray, np.ndarray)) and v.ndim >= 1
                  for v in params.values())
    finite_ok = all(bool(jnp.all(jnp.isfinite(v))) for v in params.values())
    sizes = M.param_sizes(params)
    rep.check(flat_ok and finite_ok, "params: flat dict of finite arrays",
              ", ".join(f"{k}[{sizes[k]}]" for k in sorted(params)))
    unlogged = [k for k in params if not k.startswith('log')
                and k not in ('dagger', 'mu', 'drive')]
    if unlogged:
        rep.note(f"leaves not obviously log-parameterized: {unlogged} — "
                 "confirm they are genuinely unconstrained reals")

    # 2. F finite at the floor and at the initial guess
    x0s = M.initial_guess_grid(mdl, controls)
    floor_state = jnp.full(N, max(mdl.state_floor, 1e-300))
    F_floor = F(floor_state, params, controls[0])
    F_init = jax.vmap(F, in_axes=(0, None, 0))(x0s, params, controls)
    rep.check(bool(jnp.all(jnp.isfinite(F_floor))),
              "F finite at the state floor")
    rep.check(bool(jnp.all(jnp.isfinite(F_init))),
              "F finite at the initial guess")

    # 3. residual scale
    scale = float(jnp.max(jnp.abs(F_init)))
    rep.check(np.isfinite(scale), "residual scale measurable",
              f"max|F| at init = {scale:.3e}")
    if scale > 0:
        rep.note(f"suggested trust_tol ~ {max(scale * 1e-8, 1e-12):.1e}, "
                 f"newton_atol ~ {max(scale * 1e-10, 1e-14):.1e} "
                 "(defaults 1e-6 / 1e-10 assume max|F| = O(1))")

    # 4. control continuity
    if K > 2:
        diffs = np.linalg.norm(np.diff(np.asarray(controls), axis=0), axis=-1)
        span = float(np.linalg.norm(controls[-1] - controls[0])) or 1.0
        worst = float(diffs.max() / span)
        rep.check(worst < 0.5, "controls vary smoothly along the grid",
                  f"largest consecutive jump = {worst:.1%} of grid span")

    # 5. solve + conditioning
    X, res, iters = solve_grid('vanilla_newton', F, params, controls, x0s,
                              state_floor=mdl.state_floor, **TIGHT)
    n_conv = int(np.sum(res <= 1e-8 * max(scale, 1.0)))
    rep.check(n_conv > 0, "grid solves from the initial guess",
              f"{n_conv}/{K} points reached max|F| <= 1e-8*scale; "
              f"worst residual {float(res.max()):.2e}")
    if n_conv == 0:
        rep.note("nothing converged — supply a better Model.initial_guess, or "
                 "try solver_mode=quasistatic_ptc which walks the grid")

    worst = int(np.argmax(res))
    J = jax.jacobian(lambda x: F(x, params, controls[worst]))(X[worst])
    cond = float(jnp.linalg.cond(J))
    rep.check(np.isfinite(cond) and cond < 1e12,
              "dF/dx nonsingular at the roots",
              f"cond(J) = {cond:.2e} at the worst point")

    # 6. dynamical stability of the roots
    eigs = np.linalg.eigvals(np.asarray(
        jax.vmap(lambda x, c: jax.jacobian(lambda z: F(z, params, c))(x),
                 in_axes=(0, 0))(X, controls)))
    max_re = float(np.max(np.real(eigs)))
    rep.check(max_re < 0, "roots are dynamically stable",
              f"max Re eig(J) = {max_re:.3e}")
    if max_re >= 0:
        rep.note("the root finder found an unstable root or a saddle; the ODE "
                 "verification will not reproduce it. Use a ptc_* solver mode.")

    # 7. adjoint vs finite differences
    adjoint_solve = make_adjoint_solver('dense', F)
    grad_fn = make_grad_fn(F, problem.loss_fn, adjoint_solve)
    _, grads, _ = grad_fn(X, params, controls, target,
                          jnp.ones(K, dtype=bool))

    def loss_at(p):
        Xp, resp, _ = solve_grid('vanilla_newton', F, p, controls, x0s,
                                 state_floor=mdl.state_floor, **TIGHT)
        return float(problem.loss_fn(Xp, target)), float(resp.max())

    rng = np.random.RandomState(args.seed)
    errors = []
    for name in sorted(params):
        size = sizes[name]
        picks = rng.choice(size, size=min(args.fd_probes, size), replace=False)
        for idx in picks:
            base = np.asarray(params[name]).ravel()
            eps = args.fd_eps * max(1.0, abs(float(base[idx])))
            perturbed = []
            for sign in (+1, -1):
                bumped = base.copy()
                bumped[idx] += sign * eps
                p = dict(params)
                p[name] = jnp.asarray(bumped.reshape(params[name].shape))
                perturbed.append(loss_at(p))
            (l_plus, r_plus), (l_minus, r_minus) = perturbed
            if max(r_plus, r_minus) > 1e-6 * max(scale, 1.0):
                rep.note(f"skipped {name}[{idx}]: perturbed solve did not "
                         f"converge (residual {max(r_plus, r_minus):.1e})")
                continue
            fd = (l_plus - l_minus) / (2 * eps)
            adj = float(np.asarray(grads[name]).ravel()[idx])
            denom = max(abs(fd), abs(adj), 1e-12)
            errors.append((abs(fd - adj) / denom, name, int(idx), fd, adj))

    if errors:
        errors.sort(reverse=True)
        worst_err, wname, widx, wfd, wadj = errors[0]
        rep.check(worst_err < 1e-4,
                  "adjoint gradient matches finite differences",
                  f"worst relative error {worst_err:.2e} at {wname}[{widx}] "
                  f"(fd={wfd:.6e}, adjoint={wadj:.6e}) over {len(errors)} probes")
        if worst_err >= 1e-4:
            rep.note("a mismatch here usually means F is not differentiable "
                     "where you think, or the loss reads a state the residual "
                     "does not actually determine")
    else:
        rep.check(False, "adjoint gradient matches finite differences",
                  "no probe converged; fix checks 5-6 first")

    print()
    if rep.failed:
        print(f"FAILED: {', '.join(rep.failed)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == '__main__':
    sys.exit(main())
