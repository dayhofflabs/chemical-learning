"""Thermodynamically consistent mass action in a chemostat.

The parameterization from the arXiv paper, written against *general*
stoichiometry rather than the (left, right, product) index triples of the
original solver — so unimolecular, bimolecular and higher-order reactions all
work:

    k+_a = exp(dagger_a - sum_i nu+_ia mu_i - D_a/2)
    k-_a = exp(dagger_a - sum_i nu-_ia mu_i + D_a/2)
    v+_a = k+_a prod_i x_i^nu+_ia          v-_a = k-_a prod_i x_i^nu-_ia
    F    = S (v+ - v-) + gamma (c_ext - x),   S = nu- - nu+

Three parameter classes, each with a different physical role:

* ``mu``     (N,)  standard chemical potentials — per species, so they enter
                   every reaction that touches the species.
* ``dagger`` (R,)  transition-state energies — symmetric, cancel in k+/k-, so
                   they set timescales, not equilibria.
* ``drive``  (R,)  antisymmetric drive. The **only** class that can break
                   detailed balance on a single reaction: D = 0 everywhere
                   means detailed balance holds around every cycle.
* ``log_gamma`` (1,) chemostat exchange rate.

Everything is log-parameterized, which is what makes unprojected Adam safe.

The network builder below is example scaffolding, not part of the contract —
swap in your own stoichiometry.
"""

from __future__ import annotations

import itertools
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'scripts'))
from model import Model, Problem, readout_sse_loss  # noqa: E402
from targets import make_target  # noqa: E402


# ============================================================
# Example network: polymer ligation over an alphabet
# ============================================================

def ligation_network(alphabet: str, max_length: int):
    """All strings up to ``max_length``, with every split ``u + v <-> s``."""
    species = []
    for length in range(1, max_length + 1):
        species += [''.join(p) for p in itertools.product(alphabet,
                                                          repeat=length)]
    index = {s: i for i, s in enumerate(species)}

    reactions = []
    for s in species:
        for cut in range(1, len(s)):
            reactions.append((s[:cut], s[cut:], s))

    n, r = len(species), len(reactions)
    nu_plus = np.zeros((n, r))
    nu_minus = np.zeros((n, r))
    for a, (left, right, prod) in enumerate(reactions):
        nu_plus[index[left], a] += 1.0
        nu_plus[index[right], a] += 1.0
        nu_minus[index[prod], a] += 1.0

    labels = [f"{u}+{v}<->{s}" for u, v, s in reactions]
    return species, labels, nu_plus, nu_minus


# ============================================================
# build()
# ============================================================

def build(cfg):
    chem = cfg.get('chemistry', {'alphabet': 'ab', 'max_length': 3})
    species, labels, nu_plus, nu_minus = ligation_network(
        chem['alphabet'], int(chem['max_length']))

    index = {s: i for i, s in enumerate(species)}
    N, R = nu_plus.shape
    S = jnp.asarray(nu_minus - nu_plus)
    NP = jnp.asarray(nu_plus)
    NM = jnp.asarray(nu_minus)

    def residual(x, params, control):
        # exp/log form of prod x^nu: x is clamped to state_floor > 0 by the
        # solvers, so log(x) is always finite here.
        logx = jnp.log(x)
        base = params['dagger']
        v_fwd = jnp.exp(base - NP.T @ params['mu'] - params['drive'] / 2
                        + NP.T @ logx)
        v_rev = jnp.exp(base - NM.T @ params['mu'] + params['drive'] / 2
                        + NM.T @ logx)
        flow = jnp.exp(params['log_gamma'][0]) * (control - x)
        return S @ (v_fwd - v_rev) + flow

    init_scale = float(cfg.get('init_scale', 0.0))

    def init_params(key):
        if init_scale <= 0:
            return {'dagger': jnp.zeros(R), 'mu': jnp.zeros(N),
                    'drive': jnp.zeros(R), 'log_gamma': jnp.zeros(1)}
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return {'dagger': jax.random.normal(k1, (R,)) * init_scale,
                'mu': jax.random.normal(k2, (N,)) * init_scale,
                'drive': jax.random.normal(k3, (R,)) * init_scale,
                'log_gamma': jax.random.normal(k4, (1,)) * init_scale}

    def initial_guess(control):
        return jnp.maximum(control, 1e-6)

    def diagnostics(x, params, control):
        logx = jnp.log(jnp.maximum(x, 1e-15))
        base = params['dagger']
        v_fwd = jnp.exp(base - NP.T @ params['mu'] - params['drive'] / 2
                        + NP.T @ logx)
        v_rev = jnp.exp(base - NM.T @ params['mu'] + params['drive'] / 2
                        + NM.T @ logx)
        # sigma_a = (v+ - v-) ln(v+/v-) >= 0 : entropy production per reaction
        sigma = (v_fwd - v_rev) * (jnp.log(v_fwd + 1e-300)
                                   - jnp.log(v_rev + 1e-300))
        affinity = S.T @ params['mu'] + params['drive']
        return {'entropy_production': float(jnp.sum(sigma)),
                'max_affinity': float(jnp.max(jnp.abs(affinity))),
                'total_flux': float(jnp.sum(jnp.abs(v_fwd - v_rev)))}

    # --- Control grid: sweep one species' external concentration ---
    ctrl = cfg.get('control', {})
    variable = ctrl.get('variable_species', species[0])
    lo, hi = ctrl.get('range', [1.0, 5.0])
    n_points = int(ctrl.get('n_points', 32))
    fixed_value = float(ctrl.get('fixed_monomer_value', 1.0))
    monomer_len = min(len(s) for s in species)

    grid_axis = np.linspace(float(lo), float(hi), n_points)
    controls = np.zeros((n_points, N))
    for s, i in index.items():
        if len(s) == monomer_len and s != variable:
            controls[:, i] = fixed_value
    controls[:, index[variable]] = grid_axis
    for name, value in (ctrl.get('fixed_species') or {}).items():
        controls[:, index[name]] = float(value)

    readout = ctrl.get('readout_species', species[-1])
    if readout not in index:
        raise ValueError(f"readout '{readout}' not in species {species[:8]}...")

    target = make_target(cfg.get('target', {'kind': 'sine'}), grid_axis)

    return Problem(
        model=Model(n_state=N, residual=residual, init_params=init_params,
                    initial_guess=initial_guess, diagnostics=diagnostics,
                    state_labels=species, name='mass_action'),
        controls=jnp.asarray(controls),
        target=jnp.asarray(target),
        loss_fn=readout_sse_loss(index[readout]),
        readout_idx=index[readout],
        meta={'species': species, 'reaction_labels': labels,
              'grid_axis': grid_axis.tolist(),
              'variable_species': variable, 'readout_species': readout},
    )
