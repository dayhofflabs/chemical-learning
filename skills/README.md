# robots

Agent skills distilled out of jaxcheml research code — self-contained, so an
LLM (or a person) can run them without importing `jaxcheml`.

| skill | what it does |
|-------|--------------|
| [how-to-train-your-CRN](how-to-train-your-CRN/SKILL.md) | Train any parameterized steady-state network against any target by adjoint differentiation of `F(x, θ, c) = 0`. You write the residual; it handles root finding, gradients, trust gating, freezing, ODE verification and the success gate. |

Each skill directory holds `SKILL.md` (the entry point), runnable `scripts/`,
worked `examples/`, and `reference/` documents for the details `SKILL.md` only
summarizes.
