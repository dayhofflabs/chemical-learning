# experiment.json, checkpoints and outputs

## experiment.json

```json
{
  "id": "hill_cascade",
  "outdir": "runs/hill_cascade",
  "plugin": "../examples/michaelis_menten.py",
  "cfg":      { "...": "passed verbatim to the plugin's build(cfg)" },
  "training": { "...": "overrides of TRAINING_DEFAULTS" },
  "ode_verification": { "mode": "quasistatic" }
}
```

`plugin` and `outdir` resolve relative to the **experiment file's** directory.
Resolution order for every training key is CLI flag → `"training"` block →
`TRAINING_DEFAULTS`.

## Training keys that change answers

| key | default | what it does |
|-----|---------|--------------|
| `solver_mode` | `quasistatic_ptc` | see `solver-modes.md` |
| `lr` | `0.008` | Adam step |
| `max_epoch` | `10000` | |
| `trust_tol` | `1e-6` | **absolute** `max\|F\|` gate per grid point |
| `trust_policy` | `mask` | `mask` zeros untrusted gradients, `abort` stops |
| `solver_unstable_frac` | `0.5` | abort if more than this fraction is untrusted |
| `grad_clip` | `10.0` | global norm clip |
| `freeze` | `[]` | leaf names held at their init value |
| `freeze_partial` | `{}` | `{"mu": {"fraction": 0.5, "seed": 0}}` |
| `init_override` | `{}` | `{"drive": 0.0}` — set a whole leaf to a constant |
| `lambda_residual` | `0.0` | penalize `‖F(x*)‖` (via `stop_gradient(x*)`) |
| `lambda_spectral` | `0.0` | penalize positive `max eig(½(J+Jᵀ))` |
| `spectral_kind` | `sym` | `sym` = autodiff-safe upper bound; `real` = tighter, NaN-prone |
| `lambda_condition` | `0.0` | hinge on `log10 cond(J)` above `condition_log_threshold` |
| `lambda_params` | `{}` | per-leaf L2, `{"drive": 1e-3}` |
| `gradient_norm_threshold` | `2e-4` | exit criterion |
| `loss_var_threshold` | `1e-6` | plateau exit, over `loss_var_window` epochs |

### Freezing

Freeze masks multiply the **gradient**, not the parameter. A frozen entry
therefore stays at exactly its initialization value and its Adam moments stay
at exactly zero — no drift, no momentum leaking through. Two consequences:

* `freeze` is how you ask "how much of the fit does this parameter class
  actually buy me?" Combine with the Fisher trace fractions in the log.
* Freezing a class **at exactly zero** takes both keys:
  `"freeze": ["drive"], "init_override": {"drive": 0.0}`. For the mass-action
  parameterization that is precisely the detailed-balance-constrained model —
  `D = 0` on every reaction. `init_override` alone leaves it trainable;
  `freeze` alone pins it at whatever the random init produced.

### Solver-stability penalties

All of them apply `stop_gradient` to `x*` before evaluating, so they push on
the parameters without trying to differentiate back through the solve a second
time. Use them when the trust gate keeps tripping: they make the *next* epoch's
solve easier rather than papering over this one.

`spectral_kind='sym'` uses the numerical abscissa `max eig(½(J+Jᵀ))`, which is
an upper bound on `max Re eig(J)` and is a symmetric-eigenvalue problem, so its
gradient is stable. `'real'` computes the true spectral abscissa; it is tighter
but its gradient NaNs at eigenvalue crossings and near-defective Jacobians.
Start with `sym`.

## Checkpoints

`checkpoints/step_NNNNNN.npz`, written every `checkpoint_every` epochs. Keys:

* `p_<leaf>` — parameters
* `m_<leaf>`, `v_<leaf>` — Adam first and second moments
* `h_<key>` + `history_keys` — logged history arrays
* `epoch`, `best_loss`

Resume is automatic from the latest checkpoint; `--no_resume` starts clean.
Adam moments are part of the checkpoint on purpose: resuming without them
produces a visible loss bump that looks like a bug in the gradient. Verified:
stopping at epoch 299 with loss `5.218039e-03` and resuming gives
`5.207001e-03` at epoch 300 — continuous, no bump.

Note that `history` is recorded every `log_every` epochs, so it is
*subsampled*, while the plateau-detection window counts real epochs. The two
must not be conflated on resume: seeding the per-epoch loss window from the
subsampled history would make a "50 epoch" plateau window span
`50 × log_every` epochs and miscount the total. Hence `epochs_this_run` and
`total_epochs` are reported separately.

## Outputs

| file | contents |
|------|----------|
| `training.log` | the full run log, including the ODE-verification warnings |
| `summary.parquet` | one row per run: config, exit reason, losses, R², gate, Fisher, timings |
| `result.json` | the same row as JSON, plus the full input `spec`, for quick reads |
| `ode_states.npz` | per-pass ODE states, residuals, ok-masks, solver used per point |
| `ode_comparison.png` | root-finder vs ODE response against the target |

`summary.parquet` is the spine: one row per run means a sweep is a
`pd.concat` of directories, and every number the gate depends on is in it.
