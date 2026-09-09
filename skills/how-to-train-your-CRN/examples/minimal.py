"""The bare minimum: a linear chain with chemostat feed.

No chemistry. Read this first to see the residual contract in isolation:
one parameter class, no thermodynamics, no saturating rates, ~40 lines.

    dx_0/dt = gamma * (control - x_0) - k_0 * x_0
    dx_i/dt = k_{i-1} * x_{i-1} - k_i * x_i         (i > 0)

The control is the external feed of x_0. The readout is the last species.
At steady state the readout is a linear function of the control, so a
linear target is reachable — the point of a minimal example is that the
optimizer should have no excuse to fail.

    python scripts/selftest.py --plugin examples/minimal.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'scripts'))
from model import Model, Problem, readout_sse_loss  # noqa: E402


def build(cfg):
    N = int(cfg.get('n_stages', 3))
    n_points = int(cfg.get('n_points', 16))

    def residual(x, params, control):
        k = jnp.exp(params['log_k'])                # (N,) positive rates
        gamma = jnp.exp(params['log_gamma'][0])
        F0 = gamma * (control - x[0]) - k[0] * x[0]
        F_rest = k[:-1] * x[:-1] - k[1:] * x[1:]
        return jnp.concatenate([jnp.array([F0]), F_rest])

    def init_params(key):
        return {'log_k': jnp.zeros(N), 'log_gamma': jnp.zeros(1)}

    def initial_guess(control):
        return jnp.ones(N) * jnp.maximum(control, 1e-3)

    controls = jnp.linspace(0.5, 2.0, n_points)     # scalar control per point
    target = 0.4 * np.asarray(controls) + 0.1       # linear, reachable

    return Problem(
        model=Model(n_state=N, residual=residual, init_params=init_params,
                    initial_guess=initial_guess,
                    state_labels=[f"X{i}" for i in range(N)],
                    name='minimal_chain'),
        controls=controls,
        target=jnp.asarray(target),
        loss_fn=readout_sse_loss(N - 1),
        readout_idx=N - 1,
        meta={'grid_axis': np.asarray(controls).tolist()},
    )
