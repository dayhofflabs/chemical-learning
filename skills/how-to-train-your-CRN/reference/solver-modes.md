# Solver modes

Seven modes, from three orthogonal choices: **Newton vs pseudo-transient**,
**vmapped vs sequential over the grid**, **dense vs Krylov adjoint**.

| mode | root finder | grid traversal | adjoint | use it when |
|------|-------------|----------------|---------|-------------|
| `vanilla_newton` | Newton + Armijo | vmap | dense | fast, well-conditioned, single stable root |
| `vanilla_krylov` | Newton + Armijo | vmap | gmres | same, but `n_state` too big for a dense J |
| `quasistatic_newton` | Newton + Armijo | sequential | dense | grid points are close, initial guess is poor |
| `ptc` | pseudo-transient | vmap | dense | Newton lands on saddles — **but see the trap** |
| `ptc_krylov` | pseudo-transient | vmap | gmres | as `ptc`, large systems |
| `ptc_fresh` | pseudo-transient, fresh IC each epoch | vmap | dense | defeats the trap, costs a full transient every epoch |
| `quasistatic_ptc` | pseudo-transient | sequential | dense | **default. Start here.** |

## Newton with Armijo

Plain Newton on a CRN residual diverges: the update can drive concentrations
negative, and `log(x)` then produces NaN. Two guards, both required:

1. **Positivity clamp** — `x ← max(x + αδ, state_floor)` after every step.
2. **Armijo backtracking** — halve `α` from 1 until `‖F(x+αδ)‖ ≤ (1−cα)‖F(x)‖`,
   up to `armijo_maxbt` times. Implemented as a `lax.scan` so it stays inside
   `jit`, with a NaN guard that treats a NaN residual as "no improvement".

Convergence is `max|F| ≤ atol + rtol·‖F₀‖` inside a `lax.while_loop`.

## Pseudo-transient continuation

A Newton step solves `Jδ = −F`. An implicit Euler step of `ẋ = F(x)` with step
`dt` solves

```
(I/dt − J) δ = F
```

so `dt → ∞` recovers Newton exactly, and small `dt` gives a damped, dynamics-
following step. The step size adapts by SER (switched evolution relaxation):

```
dt ← clip(dt0 · ‖F₀‖ / ‖F‖, dt, dt_max)
```

Early on `‖F‖` is large, `dt` is small, and the iteration crawls along the
actual trajectory. As `‖F‖` collapses, `dt` blows up and the last few steps are
Newton steps with quadratic convergence.

The payoff: PTC inherits the ODE's **root selection**. It converges to stable
fixed points and walks away from saddles and unstable roots, at the cost of a
linear solve per step rather than a full stiff integration. This is the single
most important knob in the whole pipeline, because Newton has no preference for
stable roots and will cheerfully hand you a saddle that no physical system
occupies (selftest check 6).

## The warm-start-at-saddle trap

**Read this before choosing `ptc`.**

Training warm-starts each epoch's solve from the previous epoch's `x*`. If the
previous `x*` was a fixed point, then at iteration 0 of the new solve `‖F‖ ≈
‖F₀‖`, so SER immediately sets `dt ≈ dt0 · 1 = dt0`... and then the very next
iteration sees `‖F‖` already tiny, so `dt → dt_max`. The pseudo-trajectory has
zero length. PTC has silently degraded to Newton, and it stays on whatever
branch — including a saddle — it started on.

The symptom is diagnostic: `ptc` and `vanilla_newton` produce *identical*
iterate counts and losses after the first few epochs. If you see that, you are
not getting stable-root selection.

Two fixes, both shipped:

* `ptc_fresh` — discard the warm start, re-run the full transient from
  `Model.initial_guess` every epoch. Correct, and expensive.
* `quasistatic_ptc` — warm-start from the previous **grid point**, not the
  previous epoch. Grid point k starts at `x*_{k−1}`, which is a genuinely
  different state, so the transient has real length. Grid point 0 starts from
  `initial_guess`. This is why it is the default: it gets stable-root selection
  at roughly the cost of one warm Newton solve per point.

Measured on the paper's networks, `quasistatic_ptc` was ~560× faster than
integrating the ODE per grid point, and was the only mode that converged
reliably across the sweep.

## Sequential modes and multistability

`quasistatic_*` walk the grid with `lax.scan`, carrying `x`. Two consequences
that are not bugs:

* **No parallelism over K.** For a 32-point grid this is usually still faster
  than vmapped-from-cold, because each solve starts from an excellent guess.
* **The result depends on sweep direction.** With multiple stable roots — Hill
  kinetics with feedback is bistable almost by default — a quasistatic sweep
  traces a *branch*, and reversing the control grid traces a different one.
  That is hysteresis, and it is physically real. It also means your "response
  curve" is not a function of the control alone.

If you have multistability and you care which branch you are on: run the ODE
verification in `quasistatic` mode, which reproduces the same branch-following,
and compare against `warm` mode, which starts each point from the root finder's
answer. Divergence between the two is the tell.

## Caches and `id(F)`

Compiled solvers are cached keyed on `id(F)`. If you rebuild the residual
closure (call `build(cfg)` again), you get a new object identity and a full
silent recompile. Build the `Problem` once per process.

## Trust gating

After every solve, per-point `max|F(x*_k)|` is compared against `trust_tol`:

* `trust_policy='mask'` — untrusted points' gradient contributions are zeroed.
  Training continues on the points that solved.
* `trust_policy='abort'` — stop the run.

Either way, if the untrusted fraction exceeds `solver_unstable_frac` (default
0.5), the run aborts.

**`trust_tol` is absolute, not relative.** A residual whose natural scale is
`max|F| ≈ 60` at initialization (the mass-action example) needs a different
`trust_tol` than one at `≈1`. `selftest.py` measures the scale and suggests a
value; use it. Setting `trust_tol` too tight silently masks everything and the
run aborts on `solver_unstable_frac`; too loose and you train on garbage roots.
