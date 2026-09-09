"""Target construction for the examples.

Deliberately outside ``scripts/``: how you generate targets is your business,
not the trainer's. The trainer only ever sees the array ``Problem.target``.
"""

from __future__ import annotations

import numpy as np


def make_target(cfg, grid):
    """Build a (K,) target on ``grid``.

    cfg = {"kind": "file", "path": "target.npy"}
        | {"kind": "sine",  "freq": 1.2, "amp": 0.2, "offset": 0.25}
        | {"kind": "gaussian", "center": None, "width": 0.7, "amp": 0.4,
           "floor": 0.02}
        | {"kind": "values", "values": [...]}
    """
    kind = cfg.get('kind', 'sine')
    grid = np.asarray(grid, dtype=float)
    u = ((grid - grid.min()) / (grid.max() - grid.min())
         if grid.max() > grid.min() else np.zeros_like(grid))

    if kind == 'file':
        values = np.load(cfg['path'])
        if values.shape != grid.shape:
            raise ValueError(
                f"target file has shape {values.shape}, grid is {grid.shape}")
        return values
    if kind == 'values':
        return np.asarray(cfg['values'], dtype=float)
    if kind == 'sine':
        return (cfg.get('offset', 0.25)
                + cfg.get('amp', 0.2)
                * np.sin(2 * np.pi * cfg.get('freq', 1.2) * u))
    if kind == 'gaussian':
        center = cfg.get('center')
        center = 0.5 if center is None else float(center)
        return (cfg.get('floor', 0.02)
                + cfg.get('amp', 0.4)
                * np.exp(-((u - center) ** 2) / (2 * cfg.get('width', 0.7) ** 2)))
    raise ValueError(f"unknown target kind '{kind}'")
