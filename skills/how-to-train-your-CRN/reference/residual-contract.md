# The residual contract

The trainer knows nothing about chemistry. It knows one function:

```python
F(x, params, control) -> (n_state,)
```

`x` is the state (concentrations), `params` a **flat dict of unconstrained real
arrays**, `control` one row of the control grid. The steady state is `F = 0`.
Everything else in this skill is machinery for finding those roots and
differentiating through them.

## The seven invariants

| # | Invariant | Why it exists | Checked by |
|---|-----------|---------------|------------|
| 1 | `F` is a pure JAX function, traceable and twice differentiable in `x` and `params` | it goes inside `jit`, `vmap`, `jacobian` and `vjp` | trace failure |
| 2 | `F` returns exactly `n_state` finite values, including at `x = state_floor` | the solvers clamp to the floor every iteration; one `inf` there poisons the whole vmapped grid | selftest 2 |
| 3 | `F` is autonomous — no time, no `x`-history, no Python-level state | there is no trajectory, only a root | review |
| 4 | `params` values are **unconstrained reals** | Adam is unprojected; it will happily propose a negative rate constant | selftest 1 (warns) |
| 5 | `F` depends on `control` continuously | quasistatic modes warm-start from the previous grid point | selftest 4 |
| 6 | `dF/dx` is nonsingular at the roots you care about | the adjoint solve is `Jᵀλ = dL/dx`; a singular `J` means no gradient | selftest 5 |
| 7 | The roots are *stable* (`max Re eig(J) < 0`) | an unstable root is a real root of `F` that no physical system ever sits at, and the ODE verification will reject it | selftest 6 |

Invariant 4 in practice: every positive quantity is stored as its log.
`log_Km`, `log_Vmax`, `log_h`, `log_gamma`. Exponentiate inside `F`. This is
not a style preference — it is what makes plain Adam a valid optimizer here,
and it removes invariant-2 problems for free (`exp(anything)` is positive, so
denominators like `Km^h + s^h` cannot vanish).

Invariant 3 is where a "let's just add a decay term that depends on the
previous state" instinct goes to die. If your system genuinely has no steady
state — a limit cycle, say — this whole approach is the wrong tool. Train
against the ODE trajectory instead.

## Writing `F`: products and powers

Mass-action-style products `∏ x_i^ν_ia` should go through logs rather than
`prod`:

```python
logx = jnp.log(x)                     # x >= state_floor > 0, always finite
v_fwd = jnp.exp(base + NU_PLUS.T @ logx)
```

`jnp.prod(x ** nu, axis=0)` is mathematically the same, numerically worse
(underflows to exactly 0 for long polymers, and `0 ** 0` gradients are a
minefield). The exp/log form also makes thermodynamic parameterizations fall
out algebraically — see `examples/mass_action.py`.

Powers with a *trained* exponent need the same treatment:
`s ** h` becomes `jnp.exp(h * jnp.log(s))`, because `s ** h` with `s` at the
floor and `h` an array gives you a NaN gradient rather than a large number.

## The `Model` and `Problem` objects

```python
Model(n_state, residual, init_params,
      initial_guess=None,     # control -> x0.  Default: 0.1 everywhere.
      diagnostics=None,       # (x, params, control) -> {str: float}, logged
      state_labels=None,
      state_floor=1e-15,      # x is clamped to this every solver iteration
      name='model')

Problem(model, controls, target, loss_fn,
        readout_idx=None,     # for reporting/R², not for the gradient
        meta={})
```

A plugin is any Python file exposing `build(cfg) -> Problem`. `cfg` is the
`"cfg"` block of the experiment JSON. Everything system-specific — topology,
target construction, which species is swept — lives in the plugin, not in
`scripts/`.

`initial_guess` matters more than it looks, and what makes a good one depends
entirely on your parameterization — the control is not generally a state, so
there is no shape-agnostic recipe. A bad initial guess shows up as selftest
check 5 failing on the first grid point only; the quasistatic solver modes walk
the rest of the grid from there.

## The loss seam

```python
loss_fn(X, target) -> scalar        # X is the whole (K, n_state) grid
```

The trainer computes `dL/dX = jax.grad(loss_fn)(X, target)` and pushes that
through the adjoint. Consequences:

* The loss may read **any** state, or all of them, or couple grid points to
  each other. Total variation across the sweep, a slope constraint, a
  Wasserstein distance to a target distribution — all cost the same as a
  pointwise SSE, because the adjoint is solved per grid point regardless.
* `target` can be any shape `loss_fn` understands. The trainer never indexes
  it.
* **Trust masking is only exact for per-point-decomposable losses.** The mask
  zeros the contribution of untrusted grid points *after* `dL/dX` is computed.
  If your loss couples points, an untrusted point still leaked into `dL/dX` at
  the trusted points. With `trust_policy='abort'` this is moot; with `'mask'`
  and a coupled loss, treat the masked gradient as approximate.

Provided: `readout_sse_loss(idx)`, `readout_relative_loss(idx, eps)`.
