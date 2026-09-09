#!/usr/bin/env python3
"""Train a CRN against a target by implicit differentiation of its steady state.

Model-agnostic: the residual, the control grid, the target and the loss all
come from a plugin exposing ``build(cfg) -> Problem`` (see model.py). This file
owns the training loop, freezing, trust gating, checkpointing and reporting.

Usage:
    python train_crn.py --experiment path/to/experiment.json
    python train_crn.py --experiment ... --solver_mode quasistatic_ptc --lr 0.01

experiment.json:
    {
      "id": "demo",
      "outdir": "runs/demo",
      "plugin": "../examples/mass_action.py",
      "cfg": { ... passed to build() ... },
      "training": { "max_epoch": 4000, "lr": 0.008, "freeze": ["mu"], ... },
      "ode_verification": { "mode": "quasistatic" }
    }
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model as M  # noqa: E402
import verify_ode  # noqa: E402
from checkpoints import (  # noqa: E402
    latest_checkpoint, load_checkpoint, make_logger, save_checkpoint,
)
from gradients import (  # noqa: E402
    adam_update, fisher_summary, make_grad_fn, response_jacobian,
)
from solvers import (  # noqa: E402
    SOLVER_ADJOINT_MODES, SOLVER_MODES, make_adjoint_solver, solve_grid,
)

jax.config.update("jax_enable_x64", True)

TRAINING_DEFAULTS = {
    'lr': 0.008,
    'max_epoch': 10000,
    'init_seed': 0,
    'solver_mode': 'quasistatic_ptc',
    'newton_maxiter': 200,
    'newton_atol': 1e-10,
    'newton_rtol': 1e-8,
    'armijo_c': 1e-4,
    'armijo_rho': 0.5,
    'armijo_maxbt': 10,
    'ptc_maxiter': 500,
    'ptc_dt0': 1e-2,
    'ptc_dt_max': 1e8,
    'ptc_dt_growth': 2.0,
    'trust_tol': 1e-6,
    'trust_policy': 'mask',          # 'mask' | 'abort'
    'solver_unstable_frac': 0.5,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_eps': 1e-8,
    'grad_clip': 10.0,
    'lambda_residual': 0.0,
    'residual_penalty_kind': 'l2',
    'lambda_spectral': 0.0,
    'spectral_kind': 'sym',
    'spectral_margin': 0.0,
    'lambda_condition': 0.0,
    'condition_log_threshold': 9.0,
    'lambda_params': {},
    'krylov_tol': 1e-8,
    'krylov_atol': 0.0,
    'krylov_restart': 30,
    'krylov_maxiter': 200,
    'gradient_norm_threshold': 2e-4,
    'loss_var_threshold': 1e-6,
    'loss_var_window': 50,
    'log_every': 10,
    'checkpoint_every': 500,
    'print_every': 50,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a CRN plugin against a target via implicit "
                    "differentiation of the steady state. CLI flags "
                    "override the experiment's 'training' block, which "
                    "overrides TRAINING_DEFAULTS.",
        epilog="example: python train_crn.py --experiment "
               "experiments/hill_cascade.json --solver_mode quasistatic_ptc")
    p.add_argument('--experiment', required=True,
                   help='path to experiment.json (see reference/schema.md)')
    p.add_argument('--no_resume', action='store_true',
                   help='ignore any existing checkpoint; start clean')
    p.add_argument('--no_ode', action='store_true',
                   help='skip ODE verification and the success gate')
    p.add_argument('--lr', type=float, help='Adam learning rate')
    p.add_argument('--max_epoch', type=int,
                   help='hard cap on training epochs')
    p.add_argument('--solver_mode', choices=sorted(SOLVER_MODES),
                   help='root-finder mode (see reference/solver-modes.md)')
    p.add_argument('--trust_tol', type=float,
                   help='ABSOLUTE max|F| gate per grid point')
    return p.parse_args()


def load_plugin(path, base_dir):
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(base_dir, path))
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location("crn_plugin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["crn_plugin"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, 'build'):
        raise AttributeError(f"plugin {path} must define build(cfg) -> Problem")
    return mod


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    exp_path = os.path.abspath(args.experiment)
    with open(exp_path) as f:
        spec = json.load(f)
    base_dir = os.path.dirname(exp_path)

    outdir = spec.get('outdir', os.path.join(base_dir, 'runs', spec['id']))
    if not os.path.isabs(outdir):
        outdir = os.path.normpath(os.path.join(base_dir, outdir))
    ckpt_dir = os.path.join(outdir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    log, log_handle = make_logger(os.path.join(outdir, 'training.log'))

    cfg = {**TRAINING_DEFAULTS, **spec.get('training', {})}
    for key in ('lr', 'max_epoch', 'solver_mode', 'trust_tol'):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)

    # --- Problem ---
    plugin = load_plugin(spec['plugin'], base_dir)
    problem = plugin.build(spec.get('cfg', {}))
    mdl = problem.model
    F = mdl.residual
    controls = jnp.asarray(problem.controls)
    target = jnp.asarray(problem.target)
    n_points = controls.shape[0]
    N = mdl.n_state

    log("=" * 64)
    log(f"Training {spec['id']}  [{mdl.name}]")
    log(f"  states={N}  grid={n_points}  outdir={outdir}")

    # --- Parameters, overrides, freeze masks ---
    params = dict(mdl.init_params(jax.random.PRNGKey(int(cfg['init_seed']))))
    params = M.apply_init_override(params, cfg.get('init_override', {}), log=log)
    masks = M.build_freeze_masks(
        params, cfg.get('freeze'), cfg.get('freeze_partial'), log=log)
    names = sorted(params)
    sizes = M.param_sizes(params)
    n_params = sum(sizes.values())
    log(f"  params: " + ", ".join(f"{k}[{sizes[k]}]" for k in names)
        + f"  (P={n_params})")

    # --- Solver + gradient ---
    solver_mode = cfg['solver_mode']
    adjoint_mode = SOLVER_ADJOINT_MODES[solver_mode]
    adjoint_solve = make_adjoint_solver(
        adjoint_mode, F,
        krylov_tol=cfg['krylov_tol'], krylov_atol=cfg['krylov_atol'],
        krylov_restart=cfg['krylov_restart'],
        krylov_maxiter=cfg['krylov_maxiter'])
    grad_fn = make_grad_fn(
        F, problem.loss_fn, adjoint_solve,
        lambda_residual=float(cfg['lambda_residual']),
        penalty_kind=cfg['residual_penalty_kind'],
        hinge_threshold=float(cfg['trust_tol']),
        lambda_spectral=float(cfg['lambda_spectral']),
        spectral_kind=cfg['spectral_kind'],
        spectral_margin=float(cfg['spectral_margin']),
        lambda_condition=float(cfg['lambda_condition']),
        condition_log_threshold=float(cfg['condition_log_threshold']),
        lambda_params=cfg['lambda_params'])

    solver_kwargs = {k: cfg[k] for k in (
        'newton_maxiter', 'newton_atol', 'newton_rtol', 'armijo_c',
        'armijo_rho', 'armijo_maxbt', 'ptc_maxiter', 'ptc_dt0', 'ptc_dt_max',
        'ptc_dt_growth')}
    solver_kwargs['state_floor'] = mdl.state_floor
    fresh_x0 = M.initial_guess_grid(mdl, controls)
    if solver_mode == 'ptc_fresh':
        solver_kwargs['fresh_x0'] = fresh_x0

    log(f"  solver={solver_mode}  adjoint={adjoint_mode}  "
        f"trust_tol={cfg['trust_tol']:g} ({cfg['trust_policy']})")
    log(f"  lr={cfg['lr']}  max_epoch={cfg['max_epoch']}  "
        f"clip={cfg['grad_clip']}")

    # --- State ---
    m = M.zeros_like(params)
    v = M.zeros_like(params)
    x0s = fresh_x0
    history = {k: [] for k in (
        'epoch', 'loss', 'l_fit', 'l_residual', 'l_spectral', 'l_condition',
        'l_params', 'n_trusted', 'max_solver_residual', 'total_iters',
        'max_iters', 'grad_norm')}
    for name in names:
        history[f'gnorm_{name}'] = []
    best = {'loss': float('inf'), 'epoch': 0, 'preds': None, 'states': None}
    all_losses = []
    start_epoch = 0
    exit_reason = 'max_epoch'

    ckpt = None if args.no_resume else latest_checkpoint(ckpt_dir)
    if ckpt:
        state = load_checkpoint(ckpt, names)
        params, m, v, x0s = state['params'], state['m'], state['v'], state['x0s']
        history = {**history, **state['history']}
        best = state['best']
        # NOT seeded from history['loss']: that is subsampled at log_every, so
        # seeding it would make the plateau window span log_every times more
        # epochs than it claims, and would miscount total epochs.
        all_losses = []
        start_epoch = state['epoch'] + 1
        log(f"  resumed from {os.path.basename(ckpt)} at epoch {start_epoch}")

    # --- JIT warmup (so epoch timings are honest) ---
    t_warm = time.time()
    X, res, _ = solve_grid(solver_mode, F, params, controls, x0s,
                           **solver_kwargs)
    X.block_until_ready()
    out = grad_fn(X, params, controls, target,
                  jnp.ones(n_points, dtype=bool))
    jax.block_until_ready(out[0])
    log(f"  JIT warmup: {time.time() - t_warm:.1f}s "
        f"(init max|F|={float(np.max(res)):.2e})")
    if float(np.max(res)) > 1e3 * cfg['trust_tol']:
        log("  NOTE: initial max|F| is far above trust_tol. If it stays there, "
            "check that trust_tol matches your residual's natural scale.")

    # --- Training loop ---
    t0 = time.time()
    preds = None
    epoch = start_epoch - 1
    for epoch in range(start_epoch, int(cfg['max_epoch'])):
        X, point_res, point_iters = solve_grid(
            solver_mode, F, params, controls, x0s, **solver_kwargs)
        x0s = X

        trust_mask = jnp.asarray(point_res) <= float(cfg['trust_tol'])
        n_trusted = int(jnp.sum(trust_mask))
        n_untrusted = n_points - n_trusted

        if n_untrusted > float(cfg['solver_unstable_frac']) * n_points:
            exit_reason = 'solver_unstable'
            log(f"  epoch {epoch:5d} | {n_untrusted}/{n_points} untrusted "
                f"— solver unstable, stopping")
            break
        if n_untrusted and cfg['trust_policy'] == 'abort':
            exit_reason = 'solver_unstable'
            log(f"  epoch {epoch:5d} | {n_untrusted} untrusted and "
                f"trust_policy=abort — stopping")
            break

        loss, grads, aux = grad_fn(X, params, controls, target, trust_mask)
        loss = float(loss)
        all_losses.append(loss)

        if problem.readout_idx is not None:
            preds = np.asarray(X[:, problem.readout_idx])
        if loss < best['loss']:
            best = {'loss': loss, 'epoch': epoch,
                    'preds': None if preds is None else preds.copy(),
                    'states': np.asarray(X)}

        gnorms = M.leaf_norms(grads)
        total_gnorm = sum(gnorms.values())
        if total_gnorm < float(cfg['gradient_norm_threshold']):
            exit_reason = 'grad_norm'
            log(f"  epoch {epoch:5d} | |grad|={total_gnorm:.2e} — stationary, "
                f"stopping")
            break

        window = int(cfg['loss_var_window'])
        if len(all_losses) >= window:
            old = all_losses[-window]
            if old > 0 and abs(old - loss) / old < float(
                    cfg['loss_var_threshold']):
                exit_reason = 'loss_plateau'
                log(f"  epoch {epoch:5d} | plateau over {window} epochs — "
                    f"stopping")
                break

        grads = M.clip_leaves(grads, float(cfg['grad_clip']))
        grads = M.apply_masks(grads, masks)
        params, m, v = adam_update(
            params, grads, m, v, epoch + 1, float(cfg['lr']),
            b1=float(cfg['adam_beta1']), b2=float(cfg['adam_beta2']),
            eps=float(cfg['adam_eps']))

        if epoch % int(cfg['log_every']) == 0:
            history['epoch'].append(epoch)
            history['loss'].append(loss)
            for key in ('l_fit', 'l_residual', 'l_spectral', 'l_condition',
                        'l_params'):
                history[key].append(float(aux[key]))
            history['n_trusted'].append(n_trusted)
            history['max_solver_residual'].append(float(np.max(point_res)))
            history['total_iters'].append(int(np.sum(point_iters)))
            history['max_iters'].append(int(np.max(point_iters)))
            history['grad_norm'].append(total_gnorm)
            for name in names:
                history[f'gnorm_{name}'].append(gnorms[name])

        if epoch % int(cfg['checkpoint_every']) == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f'step_{epoch:06d}.npz'),
                epoch, params, m, v, x0s, history, best)

        if epoch % int(cfg['print_every']) == 0:
            log(f"  epoch {epoch:5d} | {time.time() - t0:6.1f}s | "
                f"loss={loss:.6e} | trusted={n_trusted}/{n_points} | "
                f"iters={int(np.sum(point_iters))} | |grad|={total_gnorm:.3e}")

    total_time = time.time() - t0
    if not all_losses:
        log("no epoch completed; aborting before analysis")
        log_handle.close()
        return

    save_checkpoint(os.path.join(ckpt_dir, f'step_{epoch:06d}.npz'),
                    epoch, params, m, v, x0s, history, best)

    # --- Final solve + reporting ---
    X, point_res, _ = solve_grid(solver_mode, F, params, controls, x0s,
                                 **solver_kwargs)
    preds = (np.asarray(X[:, problem.readout_idx])
             if problem.readout_idx is not None else None)
    target_np = np.asarray(target)
    r_squared = float('nan')
    if preds is not None and target_np.shape == preds.shape:
        ss_tot = float(np.sum((target_np - target_np.mean()) ** 2))
        if ss_tot > 0:
            r_squared = 1.0 - float(np.sum((preds - target_np) ** 2)) / ss_tot

    converged = exit_reason in ('grad_norm', 'loss_plateau')
    log("=" * 64)
    log(f"exit_reason={exit_reason}  (trainer 'converged'={converged}; this is "
        f"NOT the success gate)")
    log(f"best loss {best['loss']:.6e} @ epoch {best['epoch']}  |  "
        f"final {all_losses[-1]:.6e}  |  epochs {len(all_losses)} "
        f"(total {start_epoch + len(all_losses)})  |  "
        f"{total_time:.1f}s")
    log(f"root-finder R^2 = {r_squared:.5f}  max|F| = "
        f"{float(np.max(point_res)):.2e}")

    row = {
        'id': spec['id'],
        'model': mdl.name,
        'n_state': N,
        'n_grid': n_points,
        'n_params': n_params,
        'param_names': names,
        'param_sizes': [sizes[k] for k in names],
        'solver_mode': solver_mode,
        'lr': float(cfg['lr']),
        'max_epoch': int(cfg['max_epoch']),
        'init_seed': int(cfg['init_seed']),
        'frozen': sorted(set(cfg.get('freeze') or [])),
        'target': target_np.ravel().tolist(),
        'readout_idx': problem.readout_idx,
        'exit_reason': exit_reason,
        'trainer_converged': converged,
        'best_loss': best['loss'],
        'best_epoch': best['epoch'],
        'final_loss': all_losses[-1],
        'epochs_this_run': len(all_losses),
        'total_epochs': start_epoch + len(all_losses),
        'wall_time_s': total_time,
        'r_squared_rootfinder': r_squared,
        'max_solver_residual_final': float(np.max(point_res)),
        'predictions_final': None if preds is None else preds.tolist(),
        **{f'{k}_final': np.asarray(params[k]).ravel().tolist() for k in names},
        **{f'hist_{k}': list(map(float, val)) for k, val in history.items()},
    }

    # --- Fisher information (reuses the adjoint) ---
    if problem.readout_idx is not None:
        try:
            resp = response_jacobian(F, adjoint_solve, X, params, controls,
                                     problem.readout_idx, names)
            fisher = fisher_summary(np.asarray(resp), names, sizes)
            row['eff_parameters'] = fisher['effective_parameter_count']
            row['fisher_eigenvalues'] = fisher['eigenvalues']
            log(f"Fisher: {fisher['effective_parameter_count']} effective "
                f"parameters of {n_params}")
        except Exception as e:
            log(f"WARNING: Fisher failed: {e}")

    # --- Model-specific diagnostics ---
    if mdl.diagnostics is not None:
        try:
            diags = [mdl.diagnostics(X[k], params, controls[k])
                     for k in range(n_points)]
            for key in diags[0]:
                row[f'diag_{key}'] = float(
                    np.mean([float(d[key]) for d in diags]))
        except Exception as e:
            log(f"WARNING: diagnostics failed: {e}")

    # --- ODE verification + success gate ---
    if not args.no_ode:
        log("\n--- ODE verification ---")
        try:
            result = verify_ode.verify(
                F, params, controls, X, target_np, problem.readout_idx,
                preds if preds is not None else np.zeros(n_points),
                cfg=spec.get('ode_verification'), log=log)
            summary = result['summary']
            for pass_name in ('warm', 'quasistatic'):
                if f'{pass_name}_n_ok' in summary:
                    log(f"  {pass_name}: {summary[f'{pass_name}_n_ok']}"
                        f"/{n_points} ok, R^2="
                        f"{summary[f'{pass_name}_r_squared']:.5f}, "
                        f"rms_vs_rootfinder="
                        f"{summary[f'{pass_name}_rms_vs_reference']:.2e}")
            gate_pass = ('quasistatic'
                         if result['passes']['quasistatic'].get('ran')
                         else 'warm')
            gate = verify_ode.success_gate(summary, pass_name=gate_pass)
            log(f"  SUCCESS GATE ({gate_pass}): {gate['success']}  "
                f"(R^2>=0.98 and rms<1e-2 and all points converged)")
            row.update({f'ode_{k}': v for k, v in summary.items()
                        if not isinstance(v, list)})
            row['ode_chain'] = summary['chain']
            row['success'] = gate['success']
            np.savez(os.path.join(outdir, 'ode_states.npz'),
                     **{f"{p}_states": r['states']
                        for p, r in result['passes'].items()},
                     **{f"{p}_residuals": r['residuals']
                        for p, r in result['passes'].items()})
            if problem.readout_idx is not None:
                axis = (np.asarray(problem.meta.get('grid_axis'))
                        if problem.meta.get('grid_axis') is not None
                        else np.arange(n_points))
                verify_ode.comparison_plot(
                    os.path.join(outdir, 'ode_comparison.png'),
                    axis, target_np, preds, result)
        except Exception as e:
            log(f"WARNING: ODE verification failed: {e}")
            traceback.print_exc()

    try:
        import pandas as pd
        pd.DataFrame([row]).to_parquet(
            os.path.join(outdir, 'summary.parquet'), index=False)
        log(f"wrote {os.path.join(outdir, 'summary.parquet')}")
    except Exception as e:
        log(f"WARNING: parquet failed: {e}")

    with open(os.path.join(outdir, 'result.json'), 'w') as f:
        json.dump({**row, 'spec': spec,
                   'completed': datetime.now().isoformat()},
                  f, indent=2, default=str)
    log_handle.close()


if __name__ == '__main__':
    main()
