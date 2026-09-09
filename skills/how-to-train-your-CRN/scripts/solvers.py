"""Batched steady-state solvers for a generic residual F(x, params, control).

Model-agnostic: nothing here knows what the state or the parameters mean. All
grid points are solved inside a single JIT call — ``lax.while_loop`` per point,
no Python-level iteration and no host-device sync.

Modes
-----
``vanilla_newton``      vmapped Newton with Armijo backtracking, warm-started
                        from the previous epoch's solution.
``vanilla_krylov``      same forward solve; matrix-free GMRES adjoint.
``quasistatic_newton``  sequential Newton, each point warm-started from the
                        previous *grid point*.
``ptc``                 pseudo-transient continuation: implicit Euler on
                        ``x_dot = F`` with SER step adaptation. ``dt -> inf``
                        recovers Newton, so it costs Newton but inherits the
                        ODE's preference for *stable* roots.
``ptc_krylov``          ptc + GMRES adjoint.
``ptc_fresh``           vmapped Ptc restarted from ``fresh_x0`` every epoch.
``quasistatic_ptc``     sequential Ptc, warm-started from the previous point.

Choosing between them: see reference/solver-modes.md. The short version is
that vmapped Ptc warm-started at a fixed point degrades to Newton (the
pseudo-trajectory has zero length, so SER's stabilising phase never fires) —
``ptc_fresh`` and ``quasistatic_ptc`` exist to defeat that.

Caches are keyed on ``id(F)``: rebuild the residual and you silently pay
recompilation. Build it once and pass the same object around.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_cache: dict = {}

DEFAULTS = {
    'newton_maxiter': 200,
    'newton_atol': 1e-10,
    'newton_rtol': 1e-8,
    'armijo_c': 1e-4,
    'armijo_rho': 0.5,
    'armijo_maxbt': 10,
    'ptc_maxiter': 500,
    'ptc_dt0': 1e-2,
    'ptc_dt_max': 1e8,
    'ptc_dt_growth': 2.0,
    'state_floor': 1e-15,
}


def _opt(kwargs, key):
    return kwargs.get(key, DEFAULTS[key])


# ============================================================
# Single-point solvers (fully on-device)
# ============================================================

def _make_newton(F, maxiter, atol, rtol, armijo_c, armijo_rho, armijo_maxbt,
                 floor):
    def solve(params, control, x0):
        def resid(x):
            return F(x, params, control)

        F0 = resid(x0)
        res0 = jnp.max(jnp.abs(F0))
        tol = atol + rtol * res0

        def cond_fn(state):
            _, res, i = state
            return (res > tol) & (i < maxiter)

        def body_fn(state):
            x, _, i = state
            Fval = resid(x)
            J = jax.jacobian(resid)(x)
            dx = jnp.linalg.solve(J, -Fval)
            phi0 = 0.5 * jnp.dot(Fval, Fval)
            slope = jnp.dot(Fval, J @ dx)

            def backtrack(alpha, _):
                x_try = jnp.maximum(x + alpha * dx, floor)
                F_try = resid(x_try)
                good = 0.5 * jnp.dot(F_try, F_try) <= phi0 + armijo_c * alpha * slope
                return jnp.where(good, alpha, alpha * armijo_rho), None

            alpha, _ = jax.lax.scan(backtrack, 1.0, None, length=armijo_maxbt)
            x_new = jnp.maximum(x + alpha * dx, floor)
            x_new = jnp.where(jnp.all(jnp.isfinite(x_new)), x_new, x)
            return (x_new, jnp.max(jnp.abs(resid(x_new))), i + 1)

        x, res, n = jax.lax.while_loop(cond_fn, body_fn, (x0, res0, 0))
        return x, n, res

    return solve


def _make_ptc(F, maxiter, atol, rtol, dt0, dt_max, dt_growth, floor):
    """Implicit Euler pseudo-timestepping: (I/dt - J) delta = F(x_n).

    SER adapts dt from the residual ratio: as |F| shrinks dt ramps toward
    dt_max (recovering Newton); as it grows dt falls back to dt0.
    """
    def solve(params, control, x0):
        def resid(x):
            return F(x, params, control)

        eye = jnp.eye(x0.shape[0])
        F0 = resid(x0)
        res0 = jnp.max(jnp.abs(F0))
        nrm0 = jnp.linalg.norm(F0) + 1e-30
        tol = atol + rtol * res0

        def cond_fn(state):
            _, res, i, _ = state
            return (res > tol) & (i < maxiter)

        def body_fn(state):
            x, _, i, dt = state
            Fval = resid(x)
            J = jax.jacobian(resid)(x)
            delta = jnp.linalg.solve(eye / dt - J, Fval)
            x_new = jnp.maximum(x + delta, floor)

            ok = jnp.all(jnp.isfinite(x_new))
            x_new = jnp.where(ok, x_new, x)
            F_new = resid(x_new)
            nrm_new = jnp.linalg.norm(F_new) + 1e-30

            dt_ser = dt0 * nrm0 / nrm_new
            dt_new = jnp.minimum(jnp.minimum(dt_ser, dt * dt_growth), dt_max)
            dt_new = jnp.where(ok, jnp.maximum(dt_new, dt0), dt0)
            return (x_new, jnp.max(jnp.abs(F_new)), i + 1, dt_new)

        x, res, n, _ = jax.lax.while_loop(
            cond_fn, body_fn, (x0, res0, 0, jnp.asarray(float(dt0))))
        return x, n, res

    return solve


def _get_single(F, kind, kwargs):
    floor = _opt(kwargs, 'state_floor')
    if kind == 'newton':
        sig = ('newton', _opt(kwargs, 'newton_maxiter'),
               _opt(kwargs, 'newton_atol'), _opt(kwargs, 'newton_rtol'),
               _opt(kwargs, 'armijo_c'), _opt(kwargs, 'armijo_rho'),
               _opt(kwargs, 'armijo_maxbt'), floor)
        builder = lambda: _make_newton(F, *sig[1:])
    else:
        sig = ('ptc', _opt(kwargs, 'ptc_maxiter'),
               _opt(kwargs, 'newton_atol'), _opt(kwargs, 'newton_rtol'),
               _opt(kwargs, 'ptc_dt0'), _opt(kwargs, 'ptc_dt_max'),
               _opt(kwargs, 'ptc_dt_growth'), floor)
        builder = lambda: _make_ptc(F, *sig[1:])
    key = (id(F),) + sig
    if key not in _cache:
        _cache[key] = builder()
    return _cache[key], key


# ============================================================
# Grid sweeps
# ============================================================

def _get_vmapped(solve_single, key):
    vkey = key + ('vmap',)
    if vkey not in _cache:
        @jax.jit
        def solve_vmap(params, controls, x0s):
            return jax.vmap(solve_single, in_axes=(None, 0, 0))(
                params, controls, x0s)
        _cache[vkey] = solve_vmap
    return _cache[vkey]


def _get_sequential(solve_single, key):
    skey = key + ('scan',)
    if skey not in _cache:
        @jax.jit
        def solve_scan(params, controls, x0_first):
            def step(x_prev, control):
                x, n, res = solve_single(params, control, x_prev)
                return x, (x, res, n)

            _, (X, res, n) = jax.lax.scan(step, x0_first, controls)
            return X, n, res
        _cache[skey] = solve_scan
    return _cache[skey]


def _vmapped_mode(kind, F, params, controls, x0s, kwargs, x0_override=None):
    solve_single, key = _get_single(F, kind, kwargs)
    solve = _get_vmapped(solve_single, key)
    X, n_iters, residuals = solve(
        params, controls, x0s if x0_override is None else x0_override)
    return X, np.asarray(residuals), np.asarray(n_iters)


def _sequential_mode(kind, F, params, controls, x0s, kwargs):
    solve_single, key = _get_single(F, kind, kwargs)
    solve = _get_sequential(solve_single, key)
    X, n_iters, residuals = solve(params, controls, x0s[0])
    return X, np.asarray(residuals), np.asarray(n_iters)


def _newton(F, params, controls, x0s, **kw):
    return _vmapped_mode('newton', F, params, controls, x0s, kw)


def _newton_quasistatic(F, params, controls, x0s, **kw):
    return _sequential_mode('newton', F, params, controls, x0s, kw)


def _ptc(F, params, controls, x0s, **kw):
    return _vmapped_mode('ptc', F, params, controls, x0s, kw)


def _ptc_fresh(F, params, controls, x0s, **kw):
    fresh = kw.get('fresh_x0')
    if fresh is None:
        raise ValueError("ptc_fresh needs fresh_x0=(K, n_state)")
    floor = _opt(kw, 'state_floor')
    return _vmapped_mode('ptc', F, params, controls, x0s, kw,
                         x0_override=jnp.maximum(fresh, floor))


def _ptc_quasistatic(F, params, controls, x0s, **kw):
    return _sequential_mode('ptc', F, params, controls, x0s, kw)


SOLVER_MODES = {
    'vanilla_newton': _newton,
    'vanilla_krylov': _newton,
    'quasistatic_newton': _newton_quasistatic,
    'ptc': _ptc,
    'ptc_krylov': _ptc,
    'ptc_fresh': _ptc_fresh,
    'quasistatic_ptc': _ptc_quasistatic,
}

#: solver mode -> adjoint linear solve used by the training loop.
SOLVER_ADJOINT_MODES = {
    'vanilla_newton': 'dense',
    'vanilla_krylov': 'gmres',
    'quasistatic_newton': 'dense',
    'ptc': 'dense',
    'ptc_krylov': 'gmres',
    'ptc_fresh': 'dense',
    'quasistatic_ptc': 'dense',
}


def solve_grid(mode, F, params, controls, x0s, **kwargs):
    """Solve F(x, params, control) = 0 at every control in the grid.

    Args:
        mode: key of ``SOLVER_MODES``.
        F: residual ``(x, params, control) -> (n_state,)``.
        controls: (K, ...) control grid.
        x0s: (K, n_state) initial guesses (previous epoch's solutions).
        fresh_x0: (K, n_state), required by ``ptc_fresh``.

    Returns:
        X: (K, n_state) steady states.
        residuals: (K,) numpy ``max|F(x*)|``.
        iters: (K,) numpy iteration counts.
    """
    if mode not in SOLVER_MODES:
        raise ValueError(
            f"unknown solver mode '{mode}'; choose from {sorted(SOLVER_MODES)}")
    return SOLVER_MODES[mode](F, params, controls, x0s, **kwargs)


# ============================================================
# Adjoint linear solve:  J^T lam = dL/dx
# ============================================================

def _gmres(A, b, tol, atol, restart, maxiter):
    # jax renamed gmres' `tol` to `rtol`; both spellings still ship in the
    # wild, so resolve it at trace time.
    fn = jax.scipy.sparse.linalg.gmres
    try:
        return fn(A, b, rtol=tol, atol=atol, restart=restart, maxiter=maxiter,
                  solve_method='batched')[0]
    except TypeError:
        return fn(A, b, tol=tol, atol=atol, restart=restart, maxiter=maxiter,
                  solve_method='batched')[0]


def make_adjoint_solver(mode, F, **kwargs):
    """Return ``solve(x_star, params, control, dLdx) -> lam``.

    ``dense`` materialises J and factorises it: O(N^3), fine to a few hundred
    states. ``gmres`` never forms J — the J^T-vector product is the VJP of F
    w.r.t. x — and wins once N is large enough that the factorisation or the
    N x N storage hurts.
    """
    if mode == 'dense':
        def solve(x_star, params, control, dLdx):
            J = jax.jacobian(lambda x: F(x, params, control))(x_star)
            return jnp.linalg.solve(J.T, dLdx)
        return solve

    if mode == 'gmres':
        tol = float(kwargs.get('krylov_tol', 1e-8))
        atol = float(kwargs.get('krylov_atol', 0.0))
        restart = int(kwargs.get('krylov_restart', 30))
        maxiter = int(kwargs.get('krylov_maxiter', 200))

        def solve(x_star, params, control, dLdx):
            _, vjp = jax.vjp(lambda x: F(x, params, control), x_star)
            return _gmres(lambda v: vjp(v)[0], dLdx, tol, atol, restart, maxiter)
        return solve

    raise ValueError(f"unknown adjoint mode '{mode}'; choose dense or gmres")
