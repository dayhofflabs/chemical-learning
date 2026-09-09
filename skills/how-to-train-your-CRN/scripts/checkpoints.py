"""Checkpoint IO and the run logger.

Adam moments are part of the checkpoint on purpose: resuming without them
produces a visible loss bump that reads as a gradient bug.
"""

from __future__ import annotations

import glob
import os

import jax.numpy as jnp
import numpy as np

import model as M


def make_logger(path):
    handle = open(path, 'a')

    def log(msg=""):
        print(msg, flush=True)
        handle.write(str(msg) + "\n")
        handle.flush()

    return log, handle


def save_checkpoint(path, epoch, params, m, v, x0s, history, best):
    np.savez(
        path, epoch=epoch,
        **M.params_to_npz(params, 'p_'),
        **M.params_to_npz(m, 'm_'),
        **M.params_to_npz(v, 'v_'),
        x0s=np.asarray(x0s),
        best_loss=best['loss'], best_epoch=best['epoch'],
        best_preds=(best['preds'] if best['preds'] is not None
                    else np.array([])),
        best_states=(best['states'] if best['states'] is not None
                     else np.array([])),
        history_keys=np.array(sorted(history)),
        **{f"h_{k}": np.asarray(v_) for k, v_ in history.items()})


def load_checkpoint(path, names):
    data = np.load(path, allow_pickle=True)
    history = {k: list(data[f"h_{k}"]) for k in data['history_keys']}
    best_preds = np.array(data['best_preds'])
    best_states = np.array(data['best_states'])
    return {
        'epoch': int(data['epoch']),
        'params': M.params_from_npz(data, 'p_', names),
        'm': M.params_from_npz(data, 'm_', names),
        'v': M.params_from_npz(data, 'v_', names),
        'x0s': jnp.asarray(data['x0s']),
        'history': history,
        'best': {'loss': float(data['best_loss']),
                 'epoch': int(data['best_epoch']),
                 'preds': best_preds if best_preds.size else None,
                 'states': best_states if best_states.size else None},
    }


def latest_checkpoint(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, 'step_*.npz'))
    if not files:
        return None
    return max(files, key=lambda f: int(os.path.basename(f)[5:-4]))
