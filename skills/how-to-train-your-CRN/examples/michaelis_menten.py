"""Hill / Michaelis-Menten cascade — a residual that is not mass action.

An enzymatic pathway ``X0 -> X1 -> ... -> Xn`` in a chemostat, with saturating
kinetics on every step:

    v_a = Vmax_a * s_a^h_a / (Km_a^h_a + s_a^h_a)          s_a = substrate of a
    F   = S v + gamma (c_ext - x)

Nothing about this is mass action: rates saturate, there is no reverse flux, no
thermodynamic parameterization and no detailed balance. The trainer does not
care. What matters is that the contract holds, and this example exists to show
the three places where a non-mass-action residual needs thought:

1. **Denominator positivity.** ``Km^h + s^h`` must stay strictly positive at
   ``x = state_floor``. It does here because ``Km > 0`` by construction
   (``exp(log_Km)``), which is the general lesson: log-parameterize and the
   awkward cases disappear.
2. **The exponent is a parameter too.** Hill coefficients are positive reals,
   so they are trained as ``log_h``. Adam would happily send a raw ``h``
   negative and invert the response.
3. **Multistability.** Turn on ``feedback`` and the cascade becomes bistable.
   A quasistatic sweep then traces a *branch* with hysteresis, not a function:
   the response depends on sweep direction, and only the ODE verification pass
   tells you which branch you landed on. See reference/solver-modes.md.
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
from targets import make_target  # noqa: E402


def build(cfg):
    n_stages = int(cfg.get('n_stages', 4))          # reactions
    N = n_stages + 1                                 # species X0..Xn
    R = n_stages
    feedback = bool(cfg.get('feedback', False))

    # S[i, a]: reaction a consumes X_a, produces X_{a+1}
    S_np = np.zeros((N, R))
    for a in range(R):
        S_np[a, a] = -1.0
        S_np[a + 1, a] = 1.0
    S = jnp.asarray(S_np)
    substrate_idx = jnp.arange(R)                    # X_a is substrate of a

    def rates(x, params):
        s = x[substrate_idx]
        h = jnp.exp(params['log_h'])
        km = jnp.exp(params['log_Km'])
        vmax = jnp.exp(params['log_Vmax'])
        s_h = jnp.exp(h * jnp.log(s))                # s**h, safe for s > 0
        v = vmax * s_h / (km ** h + s_h)
        if feedback:
            # product of the cascade activates its own first step -> bistability
            p = x[-1]
            ka = jnp.exp(params['log_Ka'][0])
            gain = 1.0 + jnp.exp(params['log_gain'][0]) * p ** 2 / (ka ** 2 + p ** 2)
            v = v.at[0].multiply(gain)
        return v

    def residual(x, params, control):
        flow = jnp.exp(params['log_gamma'][0]) * (control - x)
        return S @ rates(x, params) + flow

    scale = float(cfg.get('init_scale', 0.2))

    def init_params(key):
        keys = jax.random.split(key, 6)
        params = {
            'log_Vmax': jax.random.normal(keys[0], (R,)) * scale,
            'log_Km': jax.random.normal(keys[1], (R,)) * scale,
            'log_h': jnp.zeros(R) + jax.random.normal(keys[2], (R,)) * scale,
            'log_gamma': jnp.zeros(1),
        }
        if feedback:
            params['log_Ka'] = jnp.zeros(1)
            params['log_gain'] = jnp.zeros(1) + jnp.log(4.0)
        return params

    def initial_guess(control):
        return jnp.maximum(control, 1e-3)

    def diagnostics(x, params, control):
        v = rates(x, params)
        return {'total_flux': float(jnp.sum(v)),
                'min_saturation': float(jnp.min(
                    x[substrate_idx] / (jnp.exp(params['log_Km'])
                                        + x[substrate_idx])))}

    ctrl = cfg.get('control', {})
    lo, hi = ctrl.get('range', [0.5, 5.0])
    n_points = int(ctrl.get('n_points', 24))
    grid_axis = np.linspace(float(lo), float(hi), n_points)

    # Only X0 is fed; everything else is pure dilution (c_ext = 0).
    controls = np.zeros((n_points, N))
    controls[:, 0] = grid_axis

    readout_idx = int(ctrl.get('readout_idx', N - 1))
    target = make_target(
        cfg.get('target', {'kind': 'gaussian', 'amp': 0.5, 'floor': 0.05}),
        grid_axis)

    return Problem(
        model=Model(n_state=N, residual=residual, init_params=init_params,
                    initial_guess=initial_guess, diagnostics=diagnostics,
                    state_labels=[f"X{i}" for i in range(N)],
                    name='hill_cascade'),
        controls=jnp.asarray(controls),
        target=jnp.asarray(target),
        loss_fn=readout_sse_loss(readout_idx),
        readout_idx=readout_idx,
        meta={'grid_axis': grid_axis.tolist(), 'feedback': feedback},
    )
