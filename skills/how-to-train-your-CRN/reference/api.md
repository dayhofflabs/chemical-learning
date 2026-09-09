# Public API

Everything an LLM re-implementing this skill needs to import lives in
`scripts/`. Nothing else is public. There are no classes with methods —
just factories that return functions plus two dataclasses.

## From `model`

| symbol | purpose |
|--------|---------|
| `Model(n_state, residual, init_params, initial_guess=None, diagnostics=None, state_labels=None, state_floor=1e-15, name='model')` | Frozen dataclass. Holds the residual and metadata. |
| `Problem(model, controls, target, loss_fn, readout_idx=None, meta={})` | Frozen dataclass. Model + control grid + loss. What `build(cfg)` returns. |
| `readout_sse_loss(idx)` | `sum((X[:, idx] - target)**2)`. Decomposes per grid point. |
| `readout_relative_loss(idx, eps=1e-8)` | Same, normalized by `|target|`. |
| `zeros_like`, `leaf_norms`, `total_norm`, `clip_leaves`, `apply_masks` | Param-dict utilities the trainer uses. |
| `build_freeze_masks(params, freeze, freeze_partial, log)` | 1.0/0.0 gradient masks. Frozen entries stay at init, Adam moments stay zero. |
| `apply_init_override(params, overrides, log)` | Set whole leaves to a constant before training (e.g. `{"drive": 0.0}`). |
| `initial_guess_grid(model, controls)` | `(K, n_state)` stack of per-point guesses. |
| `params_to_npz`, `params_from_npz`, `param_order`, `param_sizes` | Checkpoint I/O. |

## From `solvers`

| symbol | purpose |
|--------|---------|
| `SOLVER_MODES` | dict of mode → forward-solver function. Keys enumerate the seven modes. |
| `SOLVER_ADJOINT_MODES` | dict of mode → `'dense'`/`'gmres'`. |
| `solve_grid(mode, F, params, controls, x0s, **kw)` | Solve `F=0` at every control. Returns `(X, residuals, iters)`. Compiles are cached on `id(F)` — build `F` once. |
| `make_adjoint_solver(mode, F, **kw)` | Returns `solve(x_star, params, control, dLdx) -> lam` implementing `J^T lam = dLdx`. |

## From `gradients`

| symbol | purpose |
|--------|---------|
| `make_grad_fn(F, loss_fn, adjoint_solve, *, lambda_residual=0., lambda_spectral=0., lambda_condition=0., lambda_params={}, ...)` | Returns JIT'd `grad_fn(X, params, controls, target, trust_mask) -> (loss, grads, aux)`. Implements the adjoint gradient plus optional solver-stability penalties. |
| `adam_update(params, grads, m, v, t, lr, b1, b2, eps)` | One Adam step over the flat param dict. |
| `response_jacobian(F, adjoint_solve, X, params, controls, readout_idx, param_order)` | `(K, P)` sensitivities `d pred_k / d theta_i`. Reuses the adjoint. |
| `fisher_summary(resp_jac, param_order, param_sizes, top_k=5)` | Eigen-summary of `J^T J`: effective parameter count, per-leaf trace fractions. |

## From `verify_ode`

| symbol | purpose |
|--------|---------|
| `verify(F, params, controls, x0s, target, readout_idx, reference_preds, cfg=None, chain=None, log=print)` | Independent ODE integration with the escalation ladder. Returns `{'passes': {...}, 'summary': {...}}`. |
| `success_gate(summary, pass_name='quasistatic', r2_min=0.98, rms_max=1e-2)` | The final gate: R² and RMS against the root finder plus all-points-converged. |
| `comparison_plot(path, grid_axis, target, reference_preds, result)` | Target vs root finder vs ODE passes. |
| `resolve_chain(chain, log)` | Probe which diffrax solvers actually run; fall back to explicit + warn if none do. |
| `DEFAULT_CHAIN`, `EXPLICIT_FALLBACK`, `TIME_STAGES`, `FALLBACK_TIMEOUT_S`, `DEFAULTS` | Tunable constants. |

## From `checkpoints`

| symbol | purpose |
|--------|---------|
| `make_logger(path)` | `(log, handle)` — `log(msg)` writes to stdout and file. |
| `save_checkpoint(path, epoch, params, m, v, x0s, history, best)` | npz with params, Adam moments, x0s and history. |
| `load_checkpoint(path, names)` | Dict of the same. |
| `latest_checkpoint(ckpt_dir)` | Path of the highest-numbered `step_*.npz`, or `None`. |

## The plugin contract

A plugin is any Python file exposing:

```python
def build(cfg: dict) -> Problem: ...
```

`cfg` is the `"cfg"` block of the experiment JSON. Everything system-specific
— topology, target construction, which species is swept — lives in the
plugin. The trainer never inspects it.

Read `examples/minimal.py` first (the contract with no chemistry), then
`examples/mass_action.py` for a thermodynamically consistent parameterization
and `examples/michaelis_menten.py` for a saturating, non-mass-action one.
