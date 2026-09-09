"""Adjoint gradients, solver-stability penalties and Adam, over a param dict.

The gradient of a loss defined on steady states, without differentiating
through the solver:

    F(x*, theta, r) = 0
    J^T lam_k = dL/dx*_k          (J = dF/dx at x*_k)
    dL/dtheta = -sum_k VJP_theta(F_k)(lam_k)

Cost is one linear solve per grid point, memory is independent of how many
iterations the solver took, and the whole thing is agnostic to what F is.

The loss seam is ``dL/dX``. It is obtained by differentiating ``loss_fn`` with
respect to the *whole* (K, n_state) steady-state grid, so a loss that does not
decompose per grid point — curve shape, ratios between points, a norm of the
response, derivative matching — costs exactly the same and needs no new code.

Trust masking caveat: zeroing an unconverged point's gradient contribution is
exact only for a loss that decomposes over grid points. For a global loss,
dropping a point biases the objective; prefer ``trust_policy='abort'`` there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_grad_fn(F, loss_fn, adjoint_solve, *,
                 lambda_residual=0.0, penalty_kind='l2', hinge_threshold=1e-6,
                 lambda_spectral=0.0, spectral_kind='sym', spectral_margin=0.0,
                 lambda_condition=0.0, condition_log_threshold=9.0,
                 lambda_params=None):
    """Build ``grad_fn(X, params, controls, target, trust_mask)``.

    Returns ``(total_loss, grads, aux)`` where ``aux`` holds the penalty
    values. The penalties all hold ``x*`` fixed via ``stop_gradient``, so
    their gradient flows only through F's explicit parameter dependence —
    they are cheap, and they steer the parameters toward regions where the
    solver behaves rather than toward a better fit:

      * ``lambda_residual``  -- ``mean ||F||^2`` (or a hinge above
        ``hinge_threshold``): pushes toward solvable configurations.
      * ``lambda_spectral``  -- pushes the Jacobian spectrum into the left
        half-plane. ``'sym'`` uses the numerical abscissa
        ``max eig(0.5(J+J^T))``, an autodiff-safe upper bound on
        ``max Re eig(J)``; ``'eig'`` uses ``max Re eig(J)`` directly, which is
        tighter but NaN-prone at eigenvalue crossings.
      * ``lambda_condition`` -- hinge on ``log10 cond(J)``.
      * ``lambda_params``    -- ``{name: weight}`` plain L2 on a parameter
        leaf; this is the physics knob (e.g. constraining drives), not a
        solver aid.
    """
    lambda_params = dict(lambda_params or {})

    def _residual_penalty(params, X, controls):
        X_sg = jax.lax.stop_gradient(X)
        Fv = jax.vmap(F, in_axes=(0, None, 0))(X_sg, params, controls)
        if penalty_kind == 'hinge':
            return jnp.mean(
                jax.nn.relu(jnp.linalg.norm(Fv, axis=-1) - hinge_threshold) ** 2)
        return jnp.mean(jnp.sum(Fv * Fv, axis=-1))

    def _jac_scores(x, params, control):
        J = jax.jacobian(lambda z: F(z, params, control))(x)
        if spectral_kind == 'eig':
            eigs = jnp.linalg.eigvals(J)
            score = jnp.max(jnp.real(eigs))
        else:
            eigs = jnp.linalg.eigvalsh(0.5 * (J + J.T))
            score = jnp.max(eigs)
        mag = jnp.abs(eigs)
        log_cond = (jnp.log10(jnp.maximum(jnp.max(mag), 1e-30))
                    - jnp.log10(jnp.maximum(jnp.min(mag), 1e-30)))
        return score, log_cond

    def _jac_penalty(params, X, controls):
        X_sg = jax.lax.stop_gradient(X)
        scores, log_conds = jax.vmap(
            _jac_scores, in_axes=(0, None, 0))(X_sg, params, controls)
        l_spec = jnp.mean(jax.nn.relu(scores + spectral_margin) ** 2)
        l_cond = jnp.mean(jax.nn.relu(log_conds - condition_log_threshold) ** 2)
        total = lambda_spectral * l_spec + lambda_condition * l_cond
        return total, (l_spec, l_cond)

    @jax.jit
    def grad_fn(X, params, controls, target, trust_mask):
        fit_loss, dLdX = jax.value_and_grad(loss_fn)(X, target)

        lam = jax.vmap(adjoint_solve, in_axes=(0, None, 0, 0))(
            X, params, controls, dLdX)

        def _point_grad(x, control, lam_k):
            _, vjp = jax.vjp(lambda p: F(x, p, control), params)
            return vjp(lam_k)[0]

        per_point = jax.vmap(_point_grad, in_axes=(0, 0, 0))(X, controls, lam)
        mask = trust_mask.astype(X.dtype)
        grads = {k: -jnp.tensordot(mask, v, axes=(0, 0))
                 for k, v in per_point.items()}

        total = fit_loss
        aux = {'l_fit': fit_loss,
               'l_residual': jnp.array(0.0),
               'l_spectral': jnp.array(0.0),
               'l_condition': jnp.array(0.0),
               'l_params': jnp.array(0.0)}

        if lambda_residual > 0:
            l_res, g_res = jax.value_and_grad(_residual_penalty)(
                params, X, controls)
            total = total + lambda_residual * l_res
            grads = {k: grads[k] + lambda_residual * g_res[k] for k in grads}
            aux['l_residual'] = l_res

        if lambda_spectral > 0 or lambda_condition > 0:
            (l_jac, (l_spec, l_cond)), g_jac = jax.value_and_grad(
                _jac_penalty, has_aux=True)(params, X, controls)
            total = total + l_jac
            grads = {k: grads[k] + g_jac[k] for k in grads}
            aux['l_spectral'] = l_spec
            aux['l_condition'] = l_cond

        if lambda_params:
            l_par = jnp.array(0.0)
            for name, weight in lambda_params.items():
                l_par = l_par + weight * jnp.sum(params[name] ** 2)
                grads[name] = grads[name] + 2.0 * weight * params[name]
            total = total + l_par
            aux['l_params'] = l_par

        return total, grads, aux

    return grad_fn


# ============================================================
# Adam over a flat param dict
# ============================================================

def adam_update(params, grads, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    new_m = {k: b1 * m[k] + (1 - b1) * grads[k] for k in params}
    new_v = {k: b2 * v[k] + (1 - b2) * grads[k] ** 2 for k in params}
    out = {}
    for k in params:
        mhat = new_m[k] / (1 - b1 ** t)
        vhat = new_v[k] / (1 - b2 ** t)
        out[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
    return out, new_m, new_v


# ============================================================
# Response Jacobian / Fisher information
# ============================================================

def response_jacobian(F, adjoint_solve, X, params, controls, readout_idx,
                      param_order):
    """(K, P) sensitivity of the readout to every parameter.

    Reuses the adjoint: seeding ``dL/dx = e_readout`` makes the adjoint
    gradient literally ``d pred_k / d theta``, so this costs one linear solve
    per grid point, same as one training step.
    """
    n_state = X.shape[1]
    seed = jnp.zeros(n_state).at[int(readout_idx)].set(1.0)

    def one_point(x, control):
        lam = adjoint_solve(x, params, control, seed)
        _, vjp = jax.vjp(lambda p: F(x, p, control), params)
        g = vjp(lam)[0]
        return jnp.concatenate([-jnp.ravel(g[k]) for k in param_order])

    return jax.vmap(one_point, in_axes=(0, 0))(X, controls)


def fisher_summary(resp_jac, param_order, param_sizes, top_k=5):
    """Eigen-summary of ``J^T J``: effective parameter count and, for the
    leading eigenvectors, how the sensitivity splits across parameter leaves.
    """
    import numpy as np

    fisher = np.asarray(resp_jac).T @ np.asarray(resp_jac)
    eigvals, eigvecs = np.linalg.eigh(fisher)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    if eigvals[0] > 0:
        eff = int(np.sum(eigvals > 0.01 * eigvals[0]))
    else:
        eff = 0

    bounds, start = {}, 0
    for name in param_order:
        bounds[name] = (start, start + param_sizes[name])
        start += param_sizes[name]

    modes = []
    for k in range(min(top_k, len(eigvals))):
        w = eigvecs[:, k] ** 2
        modes.append({'eigenvalue': float(eigvals[k]),
                      **{f'frac_{n}': float(np.sum(w[a:b]))
                         for n, (a, b) in bounds.items()}})

    return {'eigenvalues': eigvals.tolist(),
            'effective_parameter_count': eff,
            'top_modes': modes}
