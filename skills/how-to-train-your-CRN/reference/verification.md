# ODE verification and the success gate

## Why the root finder is not enough

A root of `F` is not necessarily a state the system reaches. It can be a
saddle, an unstable node, or — most commonly — one branch of a multistable
system that a real relaxation would never select from the given initial
condition. Trust gating only tells you `‖F(x*)‖` is small; it says nothing
about whether `x*` is dynamically reachable.

So the final answer is always produced by integrating `ẋ = F(x)` to steady
state and reading the response off *that*, independently of the root finder.

## Two passes

* **warm** — start each grid point at the root finder's `x*_k` and integrate.
  Asks: "is this root stable?" If the trajectory leaves and settles elsewhere,
  the root was a saddle.
* **quasistatic** — start grid point k from the *converged trajectory* of point
  k−1, walking the control grid. Asks: "does a slowly swept experiment trace
  this curve?" This is the pass that reproduces hysteresis.

`mode: 'warm' | 'quasistatic' | 'both'`. With multistability, run `both`: where
they disagree is exactly where the branch structure matters.

## The escalation ladder

Stiffness is the whole difficulty. A CRN steady state can involve timescales
separated by ten orders of magnitude, so:

1. **Solver chain** — `kvaerno5` (tight) → `kvaerno3` (looser) →
   `implicit_euler` (looser still). Each is tried in turn; the first whose
   residual falls below `residual_tol` wins. Loosening tolerance is what gets
   a stiff point unstuck, so this is a ladder of *decreasing* accuracy on
   purpose.
2. **Staged time checkpoints** — `[1e2, 1e3, ..., 1e9]`. Rather than one shot
   to `t_max`, integrate to each checkpoint and stop as soon as
   `max|F| ≤ event_tol`. Points that settle fast cost almost nothing.
3. **Adaptive `dt0`** — `clip(max|x| / max|F(x)|, 1e-12, 1)`. A fixed `dt0` is
   the classic cause of a stiff point stalling for minutes and then reporting
   a spurious failure.
4. **Control bisection** — in the quasistatic pass, if stepping from `c_{k−1}`
   to `c_k` fails, insert substeps by linear interpolation
   `(1−α)c_{k−1} + α c_k`, doubling the substep count up to `2^max_bisect`.
   This handles a control step that crosses a bifurcation.

If a quasistatic point still fails after bisection, it falls back to a warm
integration from `x*_k`, so one bad point cannot poison the rest of the sweep.

## No wall-clock cap by default

A timeout does not mean divergence. In an earlier version of this pipeline a
180 s cap manufactured phantom failures: 177 of 2700 runs were recorded as
"quasistatic-fail" when in truth they had never finished the pass at all, and
the adaptive-`dt0` version then solved one of them in 1.6 s. So
`wall_clock_timeout` defaults to 0 (off), and a slow verification is a slow
verification, not a failure.

The one exception is the degraded-chain case below, where the cap is a
correctness guard rather than a patience limit.

When a cap *does* fire, it truncates the sweep, and points that were never
attempted must not be scored. Statistics are computed over the attempted prefix
only, and the summary carries `<pass>_n_attempted` and `<pass>_truncated` so a
truncated pass is distinguishable from a bad fit. `success_gate` still fails a
truncated pass — it requires all `n_points` to have converged — but it reports
`truncated: True`, which is the difference between "retrain" and "fix the
solver".

## Check that implicit solvers actually run

`resolve_chain` does not trust a version check — it *probes*, by actually
integrating a stiff scalar problem with each solver, and drops the ones that
throw. Some diffrax/equinox pairings break every implicit solver while the
explicit ones stay fine, and they break at construction time rather than
returning a bad answer, so probing catches it for the price of one tiny solve.
If nothing implicit survives, the chain falls back to explicit `tsit5` and
warns loudly that the result is indicative only.

That fallback is a genuine degradation, and it changes the cost model: an
explicit solver's step size is limited by stability, not accuracy, so a single
shot to `t_max = 1e9` needs ~1e9 steps and simply never returns. Under a
degraded chain the pipeline therefore (a) uses staged time checkpoints from the
first solver instead of a single shot, and (b) applies a 120 s wall-clock cap.
Both are visible in the log. If you see the degradation warning, fix the pin
before trusting any stiff result.

## The success gate

**Trainer convergence is not success.** `exit_reason ∈ {grad_norm,
loss_plateau}` means Adam stopped moving; it says nothing about the fit. The
gate that matters, applied to the ODE-verified response:

```
R² ≥ 0.98   between the ODE response and the target
AND  RMS(ODE readout − root-finder readout) < 1e-2
```

`success_gate(summary, pass_name, r2_min=0.98, rms_max=1e-2)` returns this plus
the reason it failed. Both conditions are load-bearing: the first says the model
does the job, the second says the object you trained is the object you verified.
A high R² with a large RMS means you fitted a root the dynamics don't visit.

Reported per pass, so `warm` can pass while `quasistatic` fails — that is the
signature of a bistable system whose branch is not reachable by a slow sweep.
