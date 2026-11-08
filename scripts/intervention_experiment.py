"""
Legacy script. The paper uses only its training recipe (train_model and its schedule, through
scripts/train_matched.py). Its own evaluation compares a spectral-measure prediction with the
frozen-parameter ERF under three A initializations; it is not used in the paper, and its outputs
are not released.

Three conditions:
  1. "slow" — A entries clustered near zero: linspace(0.1, 2, d_state)
     → many slow poles, heavy ERF tails expected
  2. "default" — A entries spread: linspace(1, 16, d_state)
     → moderate spread, baseline behavior
  3. "fast" — A entries large: linspace(5, 50, d_state)
     → few slow poles, lighter ERF tails expected

For each condition:
  1. Train single-layer Mamba (same hyperparameters, 12 epochs)
  2. Extract conductances and compute spectral measure
  3. Compute frozen-parameter ERF and spectral prediction
  4. Compare slopes and shapes across conditions

Usage:
    python3 scripts/intervention_experiment.py --condition slow
    python3 scripts/intervention_experiment.py --condition default
    python3 scripts/intervention_experiment.py --condition fast
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.single_layer_mamba import SingleLayerMamba
from src.model.scan_ref import selective_scan_ref
from src.extraction.conductances import compute_conductances
from src.utils.logspace import cumulative_log_retention
from src.data.wikitext import get_dataloader


A_CONFIGS = {
    'slow': {'low': 0.1, 'high': 2.0, 'desc': 'Clustered near zero (heavy tails expected)'},
    'default': {'low': 1.0, 'high': 16.0, 'desc': 'Standard Mamba spread'},
    'fast': {'low': 5.0, 'high': 50.0, 'desc': 'Large A (light tails expected)'},
}


def create_model_with_A_init(d_model, d_inner, d_state, a_low, a_high):
    """Create a SingleLayerMamba with custom A initialization."""
    model = SingleLayerMamba(
        d_model=d_model, d_inner=d_inner, d_state=d_state, vocab_size=50257,
    )
    # Override A_log initialization
    with torch.no_grad():
        A_log = torch.log(
            torch.linspace(a_low, a_high, d_state)
            .unsqueeze(0).expand(d_inner, -1).clone()
        )
        model.A_log.copy_(A_log)
    return model


def make_optimizer(model, lr, no_wd=()):
    """AdamW with weight decay 0.1, except the parameters named in no_wd, which get none."""
    if not no_wd:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    no_wd = sorted(set(no_wd))
    named = dict(model.named_parameters())
    missing = set(no_wd) - set(named)
    if missing:
        raise ValueError(f'no such parameters: {sorted(missing)}')
    groups = [{'params': [p for n, p in named.items() if n not in no_wd]},
              {'params': [named[n] for n in no_wd], 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=0.1, betas=(0.9, 0.95))


def train_model(model, args, device):
    """Train the model. Returns training history."""
    model = model.to(device)
    train_loader = get_dataloader(
        split='train', seq_len=args.seq_len,
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = get_dataloader(
        split='validation', seq_len=args.seq_len,
        batch_size=args.batch_size, shuffle=False,
    )

    optimizer = make_optimizer(model, args.lr, getattr(args, 'no_wd', ()))
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_ppl': []}

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_tokens = 0.0, 0
        t0 = time.time()

        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            logits, _ = model(input_ids=input_ids)
            loss = loss_fn(logits.view(-1, 50257), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * target_ids.numel()
            train_tokens += target_ids.numel()

        train_loss = train_loss_sum / train_tokens

        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                logits, _ = model(input_ids=input_ids)
                loss = loss_fn(logits.view(-1, 50257), target_ids.view(-1))
                val_loss_sum += loss.item() * target_ids.numel()
                val_tokens += target_ids.numel()

        val_loss = val_loss_sum / val_tokens
        val_ppl = min(math.exp(val_loss), 1e6)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | ppl={val_ppl:.1f} | {elapsed:.0f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_ppl': val_ppl,
                'epoch': epoch + 1, 'args': vars(args),
            }, os.path.join(args.save_dir, f'model_{args.condition}.pt'))

    return history


def evaluate_spectral_prediction(model, device, max_lag=1024, n_batches=5, n_targets=15):
    """Compute frozen-param ERF and spectral prediction for a trained model."""
    model.eval()
    model.extract_mode = True

    val_loader = get_dataloader(
        split='validation', batch_size=2, seq_len=2048,
        shuffle=False, num_workers=0,
    )

    targets = sorted(np.random.choice(range(1024, 2048), size=n_targets, replace=False).tolist())

    erf_frozen = torch.zeros(max_lag, device=device)
    h_dyn = torch.zeros(max_lag, device=device)
    count = 0

    for batch_idx, (input_ids, _) in enumerate(val_loader):
        if batch_idx >= n_batches:
            break
        input_ids = input_ids.to(device)

        # Frozen-param ERF
        with torch.no_grad():
            emb = model.embedding(input_ids)
            xz = model.in_proj(emb)
            x_ssm, z = xz.split(model.d_inner, dim=-1)
            xc = x_ssm.transpose(1, 2)
            xc = model.conv1d(xc)[:, :, :x_ssm.shape[1]].transpose(1, 2)
            xc = F.silu(xc)
            dt_f = F.softplus(model.dt_proj(xc))
            Bt_f = model.B_proj(xc)
            Ct_f = model.C_proj(xc)

        u_live = xc.detach().requires_grad_(True)
        y_ssm, _ = selective_scan_ref(model.A, Bt_f, Ct_f, dt_f, u_live, model.D)

        for T in targets:
            if u_live.grad is not None:
                u_live.grad.zero_()
            loss = (y_ssm[:, T, :] ** 2).sum()
            loss.backward(retain_graph=True)
            gsq = (u_live.grad ** 2).sum(dim=-1)
            for tau in range(min(max_lag, T)):
                erf_frozen[tau] += gsq[:, T - tau].mean()

        # h_dyn: the prediction with time-varying conductance products
        with torch.no_grad():
            _, inter = model(input_ids=input_ids)
            g = compute_conductances(inter['A'], inter['delta_t'])
            cumret = cumulative_log_retention(g, dim=1)

            for T in targets:
                c = inter['C_t'][:, T, :]
                yT = inter['y'][:, T, :]
                yTsq = yT ** 2
                log_end = cumret[:, T - 1, :, :]

                for tau in range(min(max_lag, T)):
                    start = T - tau
                    if tau == 0:
                        ret = torch.ones_like(log_end)
                    elif start == 0:
                        ret = torch.exp(cumret[:, T, :, :])
                    else:
                        ret = torch.exp(cumret[:, T, :, :] - cumret[:, start, :, :])
                    b = inter['B_bar'][:, start, :, :]
                    pch = (c[:, None, :] * b * ret).sum(dim=-1)
                    h_dyn[tau] += (yTsq * pch ** 2).sum(dim=-1).mean()

        count += 1

    n_total = count * len(targets)
    erf_frozen /= n_total
    h_dyn /= n_total

    # Normalize
    erf_frozen = erf_frozen / erf_frozen[1].clamp(min=1e-30)
    h_dyn = h_dyn / h_dyn[1].clamp(min=1e-30)

    # Conductance stats — get a fresh batch on the right device
    val_loader2 = get_dataloader(
        split='validation', batch_size=2, seq_len=2048,
        shuffle=False, num_workers=0,
    )
    input_ids_fresh, _ = next(iter(val_loader2))
    input_ids_fresh = input_ids_fresh.to(device)
    with torch.no_grad():
        _, inter = model(input_ids=input_ids_fresh)
        g = compute_conductances(inter['A'], inter['delta_t'])
        g_flat = g.cpu().numpy().flatten()

    return {
        'erf_frozen': erf_frozen.cpu().numpy(),
        'h_dyn': h_dyn.cpu().numpy(),
        'g_stats': {
            'mean': float(np.mean(g_flat)),
            'frac_below_0.01': float((g_flat < 0.01).mean()),
            'frac_below_0.05': float((g_flat < 0.05).mean()),
            'frac_above_0.99': float((g_flat > 0.99).mean()),
        },
        'A_range': [float(model.A.min()), float(model.A.max())],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', required=True, choices=['slow', 'default', 'fast'])
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_inner', type=int, default=512)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--skip_train', action='store_true',
                        help='Skip training, load existing checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = A_CONFIGS[args.condition]
    print(f"{'='*60}")
    print(f"Intervention experiment: {args.condition}")
    print(f"  A range: linspace({cfg['low']}, {cfg['high']}, {args.d_state})")
    print(f"  {cfg['desc']}")
    print(f"{'='*60}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results/intervention', exist_ok=True)

    ckpt_path = os.path.join(args.save_dir, f'model_{args.condition}.pt')

    if args.skip_train and os.path.exists(ckpt_path):
        print(f"Loading existing checkpoint: {ckpt_path}")
        model = create_model_with_A_init(
            args.d_model, args.d_inner, args.d_state, cfg['low'], cfg['high'],
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        val_ppl = ckpt['val_ppl']
    else:
        print("\nTraining...")
        model = create_model_with_A_init(
            args.d_model, args.d_inner, args.d_state, cfg['low'], cfg['high'],
        )
        print(f"A init: [{model.A.min().item():.4f}, {model.A.max().item():.4f}]")
        history = train_model(model, args, device)
        val_ppl = min(history['val_ppl'])

        with open(f'results/intervention/training_{args.condition}.json', 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\nVal PPL: {val_ppl:.1f}")
    print(f"\nEvaluating spectral prediction...")

    torch.manual_seed(42)
    np.random.seed(42)
    results = evaluate_spectral_prediction(model, device)

    # Compute slopes
    h, ef = results['h_dyn'], results['erf_frozen']
    slopes = {}
    for name, arr in [('h_dyn', h), ('erf_frozen', ef)]:
        for t1, t2 in [(10, 100), (100, 500)]:
            if arr[t1] > 1e-15 and arr[t2] > 1e-15:
                slopes[f'{name}_{t1}_{t2}'] = float(
                    np.log10(arr[t2] / arr[t1]) / np.log10(t2 / t1)
                )

    output = {
        'condition': args.condition,
        'A_config': cfg,
        'val_ppl': float(val_ppl),
        'A_range': results['A_range'],
        'g_stats': results['g_stats'],
        'slopes': slopes,
        'h_dyn': results['h_dyn'].tolist(),
        'erf_frozen': results['erf_frozen'].tolist(),
        'key_lags': {
            str(tau): {
                'h_dyn': float(h[tau]),
                'erf_frozen': float(ef[tau]),
                'ratio': float(h[tau] / ef[tau]) if ef[tau] > 1e-15 else None,
            }
            for tau in [1, 10, 50, 100, 500]
        },
    }

    out_path = f'results/intervention/{args.condition}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"\nConductance stats:")
    for k, v in results['g_stats'].items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSlopes:")
    for k, v in slopes.items():
        print(f"  {k}: {v:.3f}")
    print(f"\nKey lags (h_dyn / erf_frozen):")
    for tau in [10, 50, 100, 500]:
        r = h[tau] / ef[tau] if ef[tau] > 1e-15 else float('nan')
        print(f"  τ={tau}: {r:.2f}")


if __name__ == '__main__':
    main()
