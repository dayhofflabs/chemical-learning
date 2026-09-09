# Differentiating a steady state

## The problem with backpropagating a solver

The steady state `x*` is defined implicitly: `F(x*, θ, c) = 0`. The obvious
approach — unroll the Newton iterations and backprop — costs memory
proportional to the iteration count, and the iteration count is *data
dependent*, so it changes as training progresses. Under `vmap` over a grid of
K control points, that memory is multiplied by K.

The implicit function theorem removes the solver from the gradient entirely.
Differentiating `F(x*(θ), θ) = 0`:

```
(∂F/∂x)·(dx*/dθ) + ∂F/∂θ = 0
dx*/dθ = −J⁻¹ (∂F/∂θ)        J = ∂F/∂x at (x*, θ)
```

Then for a scalar loss `L(x*)`:

```
dL/dθ = (dL/dx*)ᵀ (dx*/dθ) = −(dL/dx*)ᵀ J⁻¹ (∂F/∂θ)
```

Never form `J⁻¹`. Solve for the adjoint `λ` once, then push it through a VJP:

```
Jᵀ λ = dL/dx*                       # (n_state,) linear solve
dL/dθ = −VJP_θ(F)(λ)                # one reverse-mode pass over F
```

Memory is now independent of how hard the solve was. The gradient depends only
on the *converged point*, which is why the trust gate matters so much: an
unconverged `x*` gives a mathematically valid gradient of the wrong function.

## Over a grid

Each control point contributes independently:

```
dL/dθ = − Σ_k VJP_θ(F(x*_k, ·, c_k))(λ_k)
```

`gradients.make_grad_fn` implements exactly this:

```python
fit_loss, dLdX = jax.value_and_grad(loss_fn)(X, target)          # (K, n_state)
lam = jax.vmap(adjoint_solve, in_axes=(0, None, 0, 0))(X, params, controls, dLdX)
per_point = jax.vmap(_point_grad)(X, controls, lam)              # dict of (K, ...)
grads = {k: -jnp.tensordot(mask, v, axes=(0, 0)) for k, v in per_point.items()}
```

The mask contraction *is* the sum over k, so trust masking is free.

`jax.value_and_grad(loss_fn)` is the whole reason this generalizes: the trainer
never assumes the loss is a sum over grid points. See the loss-seam note in
`residual-contract.md`.

## Dense vs Krylov

`make_adjoint_solver(mode, F)`:

* `dense` — build `J` with `jax.jacobian`, `jnp.linalg.solve(J.T, dLdx)`.
  O(n³), exact, no tuning. Correct choice up to a few hundred species.
* `gmres` — matrix-free, `Jᵀv` products via `jax.linear_transpose`. Needed
  when `n_state` is large enough that the dense Jacobian doesn't fit, or when
  `J` is very sparse. Convergence is not guaranteed; `krylov_tol` too loose
  gives a systematically biased gradient that looks like a stubborn loss floor.

Which one a solver mode uses is fixed by `SOLVER_ADJOINT_MODES`. Krylov modes
are the `*_krylov` variants.

## Verifying it

If `F` is wrong, or wrong *where you think it is differentiable*, the adjoint
gradient is silently wrong and training just plateaus. `scripts/selftest.py`
check 7 catches this by central-differencing the fully re-solved loss:

```
L(θ ± ε e_i) via a fresh tight solve  →  fd = (L₊ − L₋)/2ε
```

compared against the adjoint entry. Measured on the shipped examples:
worst relative error `2.0e-09` (hill_cascade) and `1.5e-07` (mass_action) over
10 probes each. Anything above `1e-4` is a bug in `F`, not floating point.

The failure mode this catches most often: the loss reads a state that `F` does
not actually constrain (a species with no reactions and no dilution), so `J` is
singular in that direction and the "gradient" is whatever the linear solve
happened to return.

## Free lunch: response Jacobians and Fisher information

The adjoint machinery gives you sensitivities with no extra code. Seed
`dL/dx = e_readout` instead of the real loss gradient, and the "gradient" that
comes back is literally `d pred_k/dθ` — one row of the response Jacobian per
grid point. `gradients.response_jacobian` does this;
`gradients.fisher_summary` contracts it into `Fᵢⱼ = Σ_k ∂pred_k/∂θᵢ ∂pred_k/∂θⱼ`
and reports the eigenvalue spectrum plus each parameter class's share of the
trace.

That last number is the useful one: it tells you which parameter class the fit
actually depends on, which is how you decide what to freeze next.
