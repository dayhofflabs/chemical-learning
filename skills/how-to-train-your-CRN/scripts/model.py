"""Problem contract for the steady-state CRN trainer.

Everything downstream — solvers, adjoint gradients, penalties, ODE
verification — reads only the interface defined here. Writing a new
kinetics means writing a ``build(cfg) -> Problem`` factory and nothing else.

The residual ``F(x, params, control) -> (n_state,)`` **is** ``x_dot``, and
``params`` is a **flat dict of unconstrained real arrays** (Adam is
unprojected: log-parameterize every positive quantity). See
``reference/residual-contract.md`` for the seven invariants and the reasons
behind them; ``selftest.py`` checks them before training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

#: Parameters are always a flat dict of named real-valued arrays.
Params = Mapping[str, jnp.ndarray]


@dataclass(frozen=True)
class Model:
    """A residual plus the metadata the trainer needs around it."""

    n_state: int
    residual: Callable[[jnp.ndarray, Params, jnp.ndarray], jnp.ndarray]
    init_params: Callable[[jax.Array], Params]
    #: (control,) -> (n_state,) initial guess. Defaults to 0.1 everywhere,
    #: which is rarely a good guess — supply one.
    initial_guess: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    #: (x, params, control) -> dict of scalars. Model-specific physics
    #: (fluxes, dissipation, cycle affinities); logged verbatim.
    diagnostics: Callable[..., Mapping[str, float]] | None = None
    state_labels: Sequence[str] | None = None
    #: Positivity clamp applied to every solver iterate. Set to -inf for a
    #: state that is not a concentration.
    state_floor: float = 1e-15
    name: str = "model"


@dataclass(frozen=True)
class Problem:
    """A model plus the control grid, the target and the loss."""

    model: Model
    #: (K, ...) — one control per grid point, ordered so consecutive points
    #: are close (see contract item 6).
    controls: jnp.ndarray
    #: Whatever ``loss_fn`` consumes; conventionally (K,).
    target: jnp.ndarray
    #: (X (K, n_state), target) -> scalar. Differentiated w.r.t. X to seed
    #: the adjoint, so any differentiable function of the whole steady-state
    #: grid works, not just per-point readouts.
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    #: Reporting only (R^2, comparison plots). None for losses with no
    #: single scalar readout.
    readout_idx: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Default losses
# ============================================================

def readout_sse_loss(readout_idx: int):
    """Sum of squared errors on one state variable across the grid.

    The loss used by the arXiv pipeline. Decomposes per grid point, so
    trust masking is exact for it.
    """
    idx = int(readout_idx)

    def loss_fn(X, target):
        return jnp.sum((X[:, idx] - target) ** 2)

    return loss_fn


def readout_relative_loss(readout_idx: int, eps: float = 1e-8):
    """SSE normalised per point — use when the target spans decades."""
    idx = int(readout_idx)

    def loss_fn(X, target):
        return jnp.sum(((X[:, idx] - target) / (jnp.abs(target) + eps)) ** 2)

    return loss_fn


# ============================================================
# Param-dict utilities (the trainer treats params as one pytree)
# ============================================================

def zeros_like(params: Params) -> dict:
    return {k: jnp.zeros_like(v) for k, v in params.items()}


def leaf_norms(tree: Params) -> dict[str, float]:
    return {k: float(jnp.linalg.norm(v)) for k, v in tree.items()}


def total_norm(tree: Params) -> float:
    return float(sum(jnp.linalg.norm(v) for v in tree.values()))


def clip_leaves(tree: Params, max_norm: float) -> dict:
    """Clip each leaf independently to ``max_norm`` (the v10 convention)."""
    out = {}
    for k, v in tree.items():
        n = jnp.linalg.norm(v)
        out[k] = jnp.where(n > max_norm, v * max_norm / (n + 1e-30), v)
    return out


def apply_masks(tree: Params, masks: Params) -> dict:
    return {k: v * masks[k] for k, v in tree.items()}


def build_freeze_masks(params: Params, freeze: Sequence[str] | None,
                       freeze_partial: Mapping[str, Mapping[str, Any]] | None,
                       log=print) -> dict:
    """1.0 = trainable, 0.0 = frozen; masks multiply the *gradient*.

    Frozen entries therefore keep exactly their initial value and their Adam
    moments stay zero. ``freeze_partial[name] = {'fraction': f, 'seed': s}``
    freezes a random subset of one leaf's entries.
    """
    freeze = set(freeze or [])
    freeze_partial = dict(freeze_partial or {})
    unknown = (freeze | set(freeze_partial)) - set(params)
    if unknown:
        raise ValueError(
            f"freeze targets {sorted(unknown)} are not parameter names; "
            f"available: {sorted(params)}")

    masks = {}
    for name, value in params.items():
        size = int(np.prod(value.shape))
        if name in freeze_partial:
            cfg = freeze_partial[name]
            frac = float(cfg['fraction'])
            n_freeze = int(round(frac * size))
            mask = np.ones(size)
            if n_freeze >= size:
                mask[:] = 0.0
            elif n_freeze > 0:
                rng = np.random.RandomState(int(cfg['seed']))
                mask[rng.choice(size, size=n_freeze, replace=False)] = 0.0
            masks[name] = jnp.asarray(mask.reshape(value.shape))
            log(f"  partial freeze: {name} {frac:.0%} "
                f"({int(size - mask.sum())}/{size} entries, seed={cfg['seed']})")
        elif name in freeze:
            masks[name] = jnp.zeros_like(value)
            log(f"  full freeze: {name} ({size} entries)")
        else:
            masks[name] = jnp.ones_like(value)
    return masks


def apply_init_override(params: Params, overrides: Mapping[str, Any],
                        log=print) -> dict:
    """Set whole leaves to a constant before training.

    ``{"drive": 0.0}`` is how a class is frozen *at exactly zero* rather than
    at whatever ``init_params`` produced.
    """
    out = dict(params)
    for name, value in (overrides or {}).items():
        if name not in out:
            raise ValueError(
                f"init_override target '{name}' is not a parameter name; "
                f"available: {sorted(out)}")
        out[name] = jnp.full_like(out[name], float(value))
        log(f"  init override: {name} = {float(value)}")
    return out


def initial_guess_grid(model: Model, controls: jnp.ndarray) -> jnp.ndarray:
    """(K, n_state) stack of per-point initial guesses."""
    if model.initial_guess is None:
        return jnp.full((controls.shape[0], model.n_state), 0.1)
    return jax.vmap(model.initial_guess)(controls)


def params_to_npz(params: Params, prefix: str) -> dict:
    return {f"{prefix}{k}": np.asarray(v) for k, v in params.items()}


def params_from_npz(data, prefix: str, names: Sequence[str]) -> dict:
    return {k: jnp.asarray(data[f"{prefix}{k}"]) for k in names}


def param_order(params: Params) -> list[str]:
    """Canonical leaf ordering (sorted), so flattened layouts round-trip."""
    return sorted(params)


def param_sizes(params: Params) -> dict[str, int]:
    return {k: int(np.prod(v.shape)) for k, v in params.items()}
