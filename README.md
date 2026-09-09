# Chemical Learning: Inverse Design of Steady-State Chemical Computation

```ascii
        __                   _            __
  _____/ /_  ___  ____ ___  (_)________ _/ /
 / ___/ __ \/ _ \/ __ `__ \/ / ___/ __ `/ /
/ /__/ / / /  __/ / / / / / / /__/ /_/ / /
\___/_/ /_/\___/_/ /_/ /_/_/\___/\__,_/_/
    __                      _
   / /__  ____ __________  (_)___  ____ _
  / / _ \/ __ `/ ___/ __ \/ / __ \/ __ `/
 / /  __/ /_/ / /  / / / / / / / / /_/ /
/_/\___/\__,_/_/  /_/ /_/_/_/ /_/\__, /
                                /____/
```

Official implementation of the methods in **"Local energetic coupling enhances the expressivity of chemical computation"** — training chemical
reaction networks by implicit differentiation so that a steady-state concentration reproduces a target function of an environmental input.

Two ways to use this repository:

- **[`crn_training.ipynb`](crn_training.ipynb)** — the complete methods in one
  self-contained notebook: topology, energetic parameterization, steady-state
  solver, implicit-differentiation gradients, a live dashboard, and the paper's
  ODE verification and success gate. Runs on a CPU in about an hour; no data
  download.
- **[`skills/how-to-train-your-CRN`](skills/how-to-train-your-CRN/SKILL.md)** —
  the same method as an **agent skill**, so a coding agent can set up and run
  this kind of training on networks and targets of your choice. See [Use the agent skill](#use-the-agent-skill).

## Overview

A network of reversible ligation/cleavage reactions sits in a continuously
stirred tank reactor (CSTR), fed from a reservoir of fixed composition and
drained at rate $\gamma$:

$$\dot c_i = \sum_\alpha S_{i\alpha} J_\alpha + \gamma\,(c^{\text{ext}}_i - c_i)$$

The **input** is the reservoir concentration of one monomer, swept over a grid;
the **output** is the steady-state concentration of one chosen species.
We inverse-design the physically interpretable energetic parameters from which reaction kinetics and thus CRN dynamics derive from:

- $\mu^\circ_i$ — standard chemical potentials (one per species),
- $G^\ddagger_\alpha$ — transition-state energies (one per reaction),
- $D_\alpha$ — thermodynamic drives (one per reaction),

— via $k^+_\alpha = \exp(G^\ddagger_\alpha - \mu^\circ_{\text{L}} - \mu^\circ_{\text{R}} - D_\alpha/2)$
and $k^-_\alpha = \exp(G^\ddagger_\alpha - \mu^\circ_{\text{P}} + D_\alpha/2)$, so that
**every network the optimizer produces is thermodynamically consistent by
construction**. $D_\alpha$ enters the two directions with opposite sign — it
tilts a reaction and is the only class that can break detailed balance;
$G^\ddagger_\alpha$ enters both with the same sign and only scales the rate.

Training works by:

1. **Steady states by pseudo-transient continuation (PTC)** — implicit Euler on
   $\dot c = F(c)$ with an adaptive pseudo-timestep. As $\Delta t \to \infty$
   this *is* Newton, but starting small makes it follow the physical dynamics,
   so it is attracted to *dynamically stable* steady states rather than to
   whatever root happens to be nearest.
2. **Gradients by the implicit function theorem** — differentiating
   $F(c^*(\theta);\theta) = 0$ gives $\partial\mathcal{L}/\partial\theta =
   -\lambda^\top \partial F/\partial\theta$ with $\lambda = \mathbb{J}^{-\mathsf T}
   \partial\mathcal{L}/\partial c^*$. One linear solve per grid point, and cost
   independent of how many solver iterations it took to get there.
   `lax.while_loop` is not reverse-mode differentiable, so backpropagating
   through PTC is not an option anyway.
3. **Trust masking and regularizers** — points that failed to converge are
   excluded from the gradient; penalties keep the root a real, stable,
   well-conditioned one.
4. **Independent ODE verification** — the root finder is thrown away and
   $\dot c = F(c)$ is integrated with a stiff solver (Kvaerno5) in a
   quasistatic sweep, because the claim is about what a *chemical system* does,
   not about what a root finder found.

**Training success needs:** every grid point converged ($\max_i|F_i| \le 10^{-4}$), 
the ODE agreeing with the root finder (RMSE $< 10^{-2}$), and a good fit ($R^2 \ge 0.98$). 
A failed fit is a normal, informative outcome — the studies in the paper are precisely about which
(topology, target, trainable-set) triples *can* be fit.

## Quick Start

### Installation

The notebook needs only JAX, Diffrax, NumPy, Matplotlib, and Jupyter:

```bash
pip install "jax[cpu]" diffrax numpy matplotlib jupyterlab
```

The stored run used **jax 0.11.1, diffrax 0.7.2, numpy 2.5.2** on Python 3.12.
No GPU is required. To use GPU install `jax[cuda]`.


### Run it

```bash
jupyter lab crn_training.ipynb
```

Then run all cells. The default configuration trains the `abc3` network — a
3-letter alphabet up to length 3, so **39 species, 60 reactions, 160
parameters** — to match a random trigonometric target, with only the
thermodynamic drives $D_\alpha$ trainable (60 of 160 parameters).

Everything you are meant to change lives in three clearly marked cells:

| Knob | Section | What it sets |
|------|---------|--------------|
| **KNOB A** | §2 | The network: monomer alphabet, maximum polymer length, which species is the input and which is the readout |
| **KNOB B** | §3 | The target: `gaussian`, `sigmoid`, `legendre`, `trig`, or `tent`, and its parameters |
| **KNOB C** | §4 | Which energetic classes train, which are frozen at zero, and the init seed and scale |

Every numerical setting from the paper lives in one `CONFIG` dictionary near the
top. Nothing further down the notebook hard-codes any of those numbers.

### What to expect

A dashboard of eleven panels refreshes in place every 25 epochs; each panel
exists to catch a specific failure mode (the table in §6 says which). In the
stored run the solver held all 32 grid points trusted throughout at
`max|F| ≈ 1e-8`, costing roughly 0.25 s/epoch, so the 16,000-epoch cap is about
an hour — though most runs stop earlier on the stationary-point or plateau test.
That stored run ends at $R^2 = 0.952$: a **failure** under the gate above, which
is the expected outcome for a single sample from a bin whose success rate is
60%.

## Reproducing the Paper's Studies

**Scaling study** — how expressivity grows with network size. Sweep KNOB A over
topologies and KNOB B over target complexity, with an ensemble of seeds at each
fixed complexity. Expressivity scales logarithmically with network size,
predicted primarily by the number of reactions.

**Freezing study** — which energetic class carries the capacity. Fix `abc3` and
$H = 2$, then sweep KNOB C over the seven non-empty subsets of
$\{G^\ddagger, \mu^\circ, D\}$ with $\gamma$ frozen at $\gamma = 1$ throughout.
Published success rates over $7 \times 128 = 896$ runs:

| trainable class | success rate |
|---|---|
| $D_\alpha$ (drives) alone | 60% |
| $\mu^\circ_i$ alone | 8% |
| $G^\ddagger_\alpha$ alone | 0% |
| all three | 73% |

Nonequilibrium drive is the most effective single resource for steady-state
expressivity. 

## Use the Agent Skill

`skills/how-to-train-your-CRN` is an
[Agent Skill](https://agentskills.io): a directory containing a `SKILL.md`
with YAML frontmatter, plus runnable `scripts/`, worked `examples/`, and
`reference/` documents. It generalizes the notebook — you write a residual
`F(x, params, control)` for *any* parameterized steady-state system, and the
skill supplies root finding, adjoint gradients, trust gating, freezing,
checkpointing, ODE verification and the success gate.

To install it, just copy the directory to where your coding agent looks for skills. For Claude Code:

```bash
git clone https://github.com/dayhofflabs/chemical-learning.git

# personal — available in every project
cp -r chemical-learning/skills/how-to-train-your-CRN ~/.claude/skills/

# or per-project — commit it and your whole team has it
mkdir -p .claude/skills && \
  cp -r chemical-learning/skills/how-to-train-your-CRN .claude/skills/
```

Other agents read skills from their own directory — `.agents/skills/` for Codex,
Cursor, Copilot, Gemini CLI, Amp and others, `~/.cursor/skills/` for Cursor
globally, and so on; check your agent's documentation. Symlink instead of
copying if you want `git pull` to update the skill in place.

If you would rather not look the path up, the community
[`skills`](https://github.com/vercel-labs/skills) CLI does it for you, from git
or from a local clone, with no account and no publishing step:

```bash
npx skills add dayhofflabs/chemical-learning         # detects your agents
npx skills add ./chemical-learning -a claude-code    # from a local clone
```

Then ask your agent to train a network. Start by reading
[`SKILL.md`](skills/how-to-train-your-CRN/SKILL.md), which is written to be read
top to bottom; its `scripts/` commands assume the skill's own directory as the
working directory.

## Repository Structure

```
chemical-learning/
├── crn_training.ipynb              # the complete method, end to end
└── skills/                         # agent skills distilled from the research code
    ├── LICENSE
    ├── README.md
    └── how-to-train-your-CRN/
        ├── SKILL.md                # entry point: contract, rationale, failure playbook
        ├── scripts/                # the trainer
        │   ├── model.py            #   Model/Problem contract, losses, freeze masks
        │   ├── solvers.py          #   Newton+Armijo, PTC, grid traversal, adjoints
        │   ├── gradients.py        #   adjoint gradient, penalties, Adam, Fisher
        │   ├── verify_ode.py       #   independent ODE verification + success gate
        │   ├── train_crn.py        #   CLI, training loop, reporting
        │   ├── checkpoints.py      #   checkpoint IO and the run logger
        │   └── selftest.py         #   contract validator + finite-difference check
        ├── examples/               # three worked residuals, easiest first
        │   ├── minimal.py          #   a linear chain in ~40 lines, no chemistry
        │   ├── mass_action.py      #   thermodynamic mass action, general stoichiometry
        │   ├── michaelis_menten.py #   a Hill cascade — deliberately not mass action
        │   └── targets.py          #   target construction
        ├── experiments/            # a demo experiment.json per example
        └── reference/              # residual contract, API, adjoint, solver modes,
                                    #   verification, experiment.json schema
```

## Citation

If you use this code in your research, please cite:

(To be updated with correct publication bibtex)
```bibtex
@article{tuccio2026chemical,
  title = {Free-energy driving governs the expressivity of steady-state chemical computation},
  author = {Tuccio, Marco and Rocks, Jason W. and Goldford, Joshua E.},
  year = {2026},
}
```

## License

This code is licensed under **PolyForm Noncommercial License 1.0.0**.

- ✅ **Noncommercial use**: Free to use and modify for noncommercial purposes
- ✅ **Research and education**: Permitted for academic, research, and educational purposes
- ❌ **Commercial use**: Prohibited without separate commercial licensing
- 📧 **Commercial inquiries**: [info@dayhofflabs.com](mailto:info@dayhofflabs.com)

See [skills/LICENSE](skills/LICENSE) for full terms or visit [https://polyformproject.org/licenses/noncommercial/1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)

## Contributing

This repository is maintained by Dayhoff Labs. For questions or issues, please
open a GitHub issue.
