---
name: how-to-train-your-CRN
description: Train a chemical reaction network — or any parameterized steady-state system — to match a target response, by implicit (adjoint) differentiation of its steady state. Use when the user wants to fit rate constants, thermodynamic parameters, or kinetic parameters so that a network's steady state reproduces a target curve across a swept control; when they mention CRN training, NESS fitting, steady-state response fitting, mass action or Hill/Michaelis-Menten networks; or when they need to differentiate through a root-finding solve rather than an ODE trajectory.
---

# How to train your CRN

Fit parameters `θ` so that the **steady state** of a network reproduces a target
response across a swept control.

```
for each control point c_k:   solve  F(x*_k, θ, c_k) = 0
minimize                      L(X*, target)              X* = [x*_1 ... x*_K]
gradient                      dL/dθ  via the adjoint of F, never through the solver
```

The pipeline is agnostic about the chemistry. You supply `F`; it supplies root
finding, adjoint gradients, trust gating, freezing, checkpointing, ODE
verification and a success gate.

## Reading order

1. **This file.** The contract, why the adjoint, why the trust gate, the
   failure playbook. Stop here for a working mental model.
2. **`examples/minimal.py`** — the contract in ~40 lines, no chemistry.
3. **`reference/residual-contract.md`** — the seven invariants and their
   rationale; the loss seam.
4. **`reference/api.md`** — the exact functions you'll import.
5. Deep dives, when they bite: `solver-modes.md`, `adjoint.md`,
   `verification.md`, `schema.md` (`experiment.json` keys).

## The one thing to get right

**You write the residual. Everything else is already written.**

```python
F(x, params, control) -> (n_state,)     # x_dot; the steady state is F = 0
```

Mass action, Michaelis-Menten, Hill, thermodynamically-constrained mass action,
a Moran-style network, anything — the trainer never inspects it. It only needs
`F` to be a pure JAX function of a state, a **flat dict of unconstrained real
parameter arrays**, and one control vector.

Three worked residuals ship with this skill, in the order to read them:

* `examples/minimal.py` — a linear chain, one parameter class, ~40 lines.
  The contract with no chemistry to distract.
* `examples/mass_action.py` — thermodynamically consistent mass action over
  general stoichiometry (`ν⁺`, `ν⁻`), with `mu`/`dagger`/`drive` parameters.
* `examples/michaelis_menten.py` — a Hill cascade. Saturating, irreversible,
  no thermodynamics. Deliberately *not* mass action, to show the contract does
  not assume it.

## Workflow

1. **Write a plugin** exposing `build(cfg) -> Problem`. Copy `minimal.py`.
2. **Run the self-test.** Not optional:
   ```bash
   python scripts/selftest.py --plugin examples/minimal.py
   python scripts/selftest.py --plugin path/to/your_plugin.py --cfg cfg.json
   ```
   It checks the contract mechanically and finite-difference-verifies the
   adjoint gradient. A wrong `F` otherwise presents as "training plateaus
   for no reason".
3. **Write an experiment JSON**, pointing at the plugin. See
   `experiments/*.json` and `reference/schema.md`.
4. **Train.**
   ```bash
   python scripts/train_crn.py --experiment experiments/hill_cascade.json
   python scripts/train_crn.py --experiment experiments/mass_action.json \
       --solver_mode quasistatic_ptc --lr 0.01
   ```
   CLI flags override the JSON's `"training"` block. `--help` on both
   scripts lists every flag.
5. **Read the success gate, not the loss.** See below.

## Why the adjoint, and not backprop through the solver

`F(x*, θ) = 0` defines `x*` implicitly. Differentiate it:

```
J (dx*/dθ) + ∂F/∂θ = 0,      J = ∂F/∂x
dL/dθ = −(dL/dx*)ᵀ J⁻¹ (∂F/∂θ)
```

so solve one linear system for the adjoint and push it through one VJP:

```
Jᵀ λ_k = dL/dx*_k            per control point
dL/dθ  = − Σ_k VJP_θ(F)(λ_k)
```

Memory is **independent of the solver's iteration count** — which matters
because that count is data-dependent and grows as training gets hard. The
gradient depends only on the converged point, which is exactly why the trust
gate below exists.

Details, dense-vs-Krylov, and the free response-Jacobian/Fisher machinery:
`reference/adjoint.md`.

## The loss is a seam, not a formula

```python
loss_fn(X, target) -> scalar      # X is the whole (K, n_state) grid
```

The trainer computes `dL/dX = jax.grad(loss_fn)(X, target)` and feeds it to the
adjoint. So a loss may read any state, all states, or **couple grid points to
each other** — a total-variation penalty across the sweep, a monotonicity
constraint, a distance between distributions — at no extra cost, because the
adjoint is solved per point regardless.

The one caveat: trust masking is only *exact* for per-point-decomposable
losses. See `reference/residual-contract.md`.

## Solver modes: start with `quasistatic_ptc`

Seven modes; the default is right most of the time. The decision that actually
matters is Newton vs pseudo-transient continuation (PTC):

* **Newton** finds *a* root. It has no preference for stable ones, and will
  hand you a saddle that no physical system ever occupies.
* **PTC** solves `(I/dt − J)δ = F`, an implicit Euler step, with `dt` adapted
  from `‖F₀‖/‖F‖`. Small `dt` follows the trajectory; `dt → ∞` *is* Newton. So
  it inherits the ODE's stable-root selection at roughly Newton's cost.

**The trap**: warm-starting PTC at last epoch's fixed point gives a
zero-length pseudo-trajectory, so the step adaptation never fires and PTC
silently degrades to Newton. The tell is `ptc` and `vanilla_newton` producing
identical iterate counts. `quasistatic_ptc` defeats it by warm-starting from
the previous **grid point** instead — a genuinely different state. That is why
it is the default.

Full table, multistability and hysteresis: `reference/solver-modes.md`.

## Trust gating: the most common source of silent garbage

After each solve, per-point `max|F(x*_k)|` is compared to `trust_tol`.
Untrusted points get their gradient contribution zeroed (`trust_policy='mask'`)
or abort the run (`'abort'`); exceeding `solver_unstable_frac` aborts either
way.

**`trust_tol` is absolute.** The mass-action example starts at `max|F| ≈ 60`,
the Hill example at `≈ 1.3`. The same `trust_tol` cannot be right for both. Too
tight and everything is masked and the run aborts; too loose and you train
confidently on unconverged roots. `selftest.py` measures your residual scale
and prints a suggested value — use it.

## Freezing is how you do science here

Freeze masks multiply the **gradient**, so a frozen entry stays at exactly its
init value and its Adam moments stay exactly zero.

* `"freeze": ["mu"]` — ablate a whole parameter class and see what the fit
  loses. Cross-reference with the Fisher trace fractions in the log to find out
  which class the fit actually depends on.
* `"freeze": ["drive"], "init_override": {"drive": 0.0}` — pin a class at
  *exactly zero*. For the mass-action parameterization this is precisely the
  detailed-balance-constrained model. Both keys are required: `init_override`
  alone leaves it trainable, `freeze` alone pins it at its random init.

## Success is not convergence

The trainer's `exit_reason ∈ {grad_norm, loss_plateau}` means Adam stopped
moving. It says nothing about whether the model works. The gate that counts is
applied to the **ODE-verified** response:

```
R² ≥ 0.98 against the target        AND    RMS(ODE readout − root-finder readout) < 1e-2
```

Both matter. The first says the network does the job; the second says the
object you trained is the object you verified. High R² with large RMS means you
fitted a root the dynamics never visit.

Verification integrates `ẋ = F(x)` to steady state independently of the root
finder, in a `warm` pass (start at `x*`: is this root stable?) and a
`quasistatic` pass (walk the control grid: does a slow sweep trace this
curve?). Where the two disagree, you have branch structure. Escalation ladder
and rationale: `reference/verification.md`.

### Check that implicit solvers actually run

The chain is resolved by *probing* — actually integrating a stiff scalar with
each solver — rather than by a version check, because some diffrax/equinox
pairings break every implicit solver at construction time. Whatever throws is
dropped. If nothing implicit survives, the ladder falls back to explicit
`tsit5` with a loud warning and a wall-clock cap. An explicit solver on a stiff
CRN is not equivalent, so if you see that warning, fix the pin before trusting
a stiff result.

## Failure playbook

| symptom | cause | fix |
|---------|-------|-----|
| selftest check 7 fails (adjoint ≠ FD) | `F` is wrong, or the loss reads a state `F` doesn't constrain (singular `J`) | fix `F`; check every state has a reaction or a dilution term |
| NaN loss at epoch 0 | `log`/`**` at the state floor, or a raw (non-log) positive parameter went negative | log-parameterize; use `exp(h·log s)` not `s**h` |
| `trusted=0/K`, immediate abort | `trust_tol` far below the residual scale | run selftest, use its suggestion |
| loss plateaus high, gradients tiny | genuinely stuck, or Krylov adjoint tolerance too loose (biased gradient) | switch to `dense`; check Fisher effective-parameter count |
| `ptc` behaves exactly like `vanilla_newton` | the warm-start-at-saddle trap | `quasistatic_ptc` or `ptc_fresh` |
| selftest check 6 fails (`max Re eig ≥ 0`) | root finder found a saddle | a `ptc_*` mode |
| ODE verification disagrees with root finder | multistability; the root isn't reachable | run `mode: both`, compare passes |
| verification runs forever | stiff, and the implicit chain degraded to explicit | check the log for the degraded-chain warning; get a diffrax/equinox pair whose implicit solvers run |
| loss jumps on resume | Adam moments not restored | they are in the checkpoint; don't hand-roll a reload |
| silent recompile every epoch | solver cache is keyed on `id(F)`; you rebuilt the closure | build the `Problem` once per process |

## Reachability is a topology question, not an optimizer question

Check that the target is *reachable* before you tune anything. A feedforward
cascade with dilution has a **monotone** readout in the swept feed, so it
cannot fit a non-monotone target at all: against a Gaussian bump the Hill
example plateaus at R² ≈ 0.32 forever, and no learning rate rescues it. The
tell is in the log already — `Fisher: 1 effective parameters of 13`. When R² is
stuck and the effective-parameter count is tiny, suspect the topology before
you touch the optimizer.

Non-monotone targets need a topology that can fold a monotone input: an
incoherent feedforward branch, molecular sequestration, substrate inhibition.

## Non-negotiables

* `jax.config.update("jax_enable_x64", True)`. Double precision is not
  optional: the adjoint solve and the convergence tests both live near
  `1e-10`.
* Log-parameterize every positive quantity. Adam is unprojected and will
  propose negative rate constants.
* JIT warmup before timing anything, or your first epoch's cost is a compile.
* If your system has no steady state (a limit cycle), this is the wrong tool.

## Files

```
scripts/
  model.py        Model / Problem / Params contract, losses, freeze masks, npz io
  solvers.py      Newton+Armijo, PTC, grid traversal, adjoint solvers
  gradients.py    adjoint gradient, penalties, Adam, response Jacobian, Fisher
  verify_ode.py   independent ODE verification + success gate
  train_crn.py    CLI, training loop, reporting
  checkpoints.py  checkpoint IO and the run logger
  selftest.py     contract validator incl. finite-difference adjoint check
examples/
  minimal.py           linear chain, ~40 lines — read this first
  mass_action.py       thermodynamic mass action, general stoichiometry
  michaelis_menten.py  Hill cascade (non-mass-action)
  targets.py           target construction — deliberately not in scripts/
experiments/           demo experiment.json for each example
reference/
  residual-contract.md  the 7 invariants, writing F, the loss seam
  api.md                the import surface — every public function
  adjoint.md            derivation, dense vs Krylov, verifying, Fisher
  solver-modes.md       all 7 modes, the saddle trap, multistability
  verification.md       escalation ladder, the gate, the diffrax caveat
  schema.md             experiment.json, training keys, checkpoints, outputs
```
