"""
Legacy script. App. I comes from scripts/scan_probe_large_sample.py, which reuses
load_pretrained_mamba from here; this script's own sweep is not used in the paper, and its
outputs are not released.

Full per-layer sweep of a pretrained Mamba (default: Mamba-370M, 48 layers). It caches layer
inputs from a single forward pass, then compares the spectral prediction with the
frozen-parameter ERF for each layer in sequence.

Smoke test: --smoke_test evaluates 3 layers (0, 23, 47) at low fidelity (~3-5 min).
Full run: --layer_indices all evaluates all 48 layers (~2-4 hours).

Usage:
    python3 scripts/pretrained_full_sweep.py --smoke_test
    python3 scripts/pretrained_full_sweep.py --layer_indices all
    python3 scripts/pretrained_full_sweep.py --layer_indices 0,3,6,9,12,18,24,30,36,42,47
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.scan_ref import selective_scan_ref
from src.extraction.conductances import compute_conductances
from src.utils.logspace import cumulative_log_retention


def load_pretrained_mamba(model_name, device):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=torch.float32)
    model.eval()
    return model


def cache_layer_inputs(model, input_ids, layer_indices):
    """
    Run the full model forward once, cache the input to each requested layer.

    Returns dict {layer_idx: x_input_tensor}.
    """
    cache = {}
    target_set = set(layer_indices)
    with torch.no_grad():
        x = model.backbone.embedding(input_ids)
        for i, layer in enumerate(model.backbone.layers):
            if i in target_set:
                cache[i] = x.detach().clone()
            residual = x
            x_normed = layer.norm(x)
            x_out = layer.mixer(x_normed)
            x = residual + x_out
    return cache


def evaluate_layer(model, layer_idx, x_input, device, max_lag, n_targets, seed):
    """
    Evaluate spectral prediction vs. frozen-parameter ERF for a single layer,
    using the cached input x_input (= input to that layer's residual block).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    layer = model.backbone.layers[layer_idx]
    mixer = layer.mixer

    L = x_input.shape[1]
    d_inner = mixer.d_inner
    d_state = mixer.d_state

    # Recompute the SSM intermediates with the reference scan
    with torch.no_grad():
        x_normed = layer.norm(x_input)
        xz = mixer.in_proj(x_normed)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = x_ssm.transpose(1, 2)
        x_conv = mixer.conv1d(x_conv)[:, :, :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_dbl = mixer.x_proj(x_conv)
        dt_rank = mixer.dt_proj.in_features
        dt_raw, B_t, C_t = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        delta_t = F.softplus(mixer.dt_proj(dt_raw))
        A = -torch.exp(mixer.A_log.float())

    # Frozen-parameter ERF: backprop through recurrent chain only
    targets = sorted(np.random.choice(range(L // 2, L),
                                      size=min(n_targets, L // 2),
                                      replace=False).tolist())

    u_live = x_conv.detach().requires_grad_(True)
    y_ssm, inter = selective_scan_ref(
        A, B_t, C_t, delta_t, u_live,
        mixer.D.float() if hasattr(mixer, 'D') else None,
    )

    erf_frozen = torch.zeros(max_lag, device=device)
    for T in targets:
        if u_live.grad is not None:
            u_live.grad.zero_()
        loss = (y_ssm[:, T, :] ** 2).sum()
        loss.backward(retain_graph=True)
        gsq = (u_live.grad ** 2).sum(dim=-1)
        for tau in range(min(max_lag, T)):
            erf_frozen[tau] += gsq[:, T - tau].mean()

    # Spectral prediction
    g = compute_conductances(A, delta_t)
    cumret = cumulative_log_retention(g, dim=1)
    B_bar = inter['B_bar']

    h_dyn = torch.zeros(max_lag, device=device)
    with torch.no_grad():
        for T in targets:
            c = C_t[:, T, :]
            yT = inter['y'][:, T, :]
            yTsq = yT ** 2
            log_cumT = cumret[:, T, :, :]
            for tau in range(min(max_lag, T)):
                start = T - tau
                if tau == 0:
                    ret = torch.ones_like(log_cumT)
                elif start == 0:
                    ret = torch.exp(log_cumT)
                else:
                    ret = torch.exp(log_cumT - cumret[:, start, :, :])
                b = B_bar[:, start, :, :]
                pch = (c[:, None, :] * b * ret).sum(dim=-1)
                h_dyn[tau] += (yTsq * pch ** 2).sum(dim=-1).mean()

    n_total = len(targets)
    erf_frozen /= n_total
    h_dyn /= n_total
    erf_frozen = erf_frozen / erf_frozen[1].clamp(min=1e-30)
    h_dyn = h_dyn / h_dyn[1].clamp(min=1e-30)

    # Sign ablation, per layer: each modal residue |C_T * B_bar_s| replaces C_T * B_bar_s
    h_unsigned = torch.zeros(max_lag, device=device)
    with torch.no_grad():
        for T in targets:
            c = C_t[:, T, :]
            yT = inter['y'][:, T, :]
            yTsq = yT ** 2
            log_cumT = cumret[:, T, :, :]
            for tau in range(min(max_lag, T)):
                start = T - tau
                if tau == 0:
                    ret = torch.ones_like(log_cumT)
                elif start == 0:
                    ret = torch.exp(log_cumT)
                else:
                    ret = torch.exp(log_cumT - cumret[:, start, :, :])
                b = B_bar[:, start, :, :]
                # Unsigned: take |c*b| instead of c*b before summing
                pch_unsigned = (c[:, None, :].abs() * b.abs() * ret).sum(dim=-1)
                h_unsigned[tau] += (yTsq * pch_unsigned ** 2).sum(dim=-1).mean()
    h_unsigned /= n_total
    h_unsigned = h_unsigned / h_unsigned[1].clamp(min=1e-30)

    # Summary at key lags
    results = {
        'layer': layer_idx, 'd_inner': d_inner, 'd_state': d_state,
        'h_dyn': h_dyn.cpu().tolist(),
        'erf_frozen': erf_frozen.cpu().tolist(),
        'h_unsigned': h_unsigned.cpu().tolist(),
    }
    for tau in [10, 50, 100, 500]:
        if tau < max_lag:
            h, e, u = h_dyn[tau].item(), erf_frozen[tau].item(), h_unsigned[tau].item()
            results[f'ratio_tau{tau}'] = h / e if e > 1e-15 else None
            results[f'unsigned_overestimate_tau{tau}'] = u / e if e > 1e-15 else None
    for t1, t2 in [(10, 100), (100, min(500, max_lag - 1))]:
        for name, arr in [('h_dyn', h_dyn), ('erf_frozen', erf_frozen)]:
            v1, v2 = arr[t1].item(), arr[t2].item()
            if v1 > 1e-15 and v2 > 1e-15:
                results[f'{name}_slope_{t1}_{t2}'] = float(
                    np.log10(v2 / v1) / np.log10(t2 / t1))

    g_flat = g.cpu().numpy().flatten()
    results['g_stats'] = {
        'mean': float(np.mean(g_flat)),
        'frac_below_0.01': float((g_flat < 0.01).mean()),
        'frac_above_0.99': float((g_flat > 0.99).mean()),
    }
    return results


def parse_layer_indices(arg, n_layers):
    if arg == 'all':
        return list(range(n_layers))
    return sorted(set(int(x) for x in arg.split(',') if x.strip()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='state-spaces/mamba-370m')
    parser.add_argument('--max_lag', type=int, default=512)
    parser.add_argument('--n_targets', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--layer_indices', default='all',
                        help='Comma-separated indices, or "all"')
    parser.add_argument('--smoke_test', action='store_true',
                        help='Quick run: 3 layers (0, mid, last), low fidelity')
    parser.add_argument('--results_path', default=None,
                        help='default results/pretrained/full_sweep[_smoke]_<model>.json')
    args = parser.parse_args()

    if args.smoke_test:
        args.max_lag = 128
        args.n_targets = 3
        args.seq_len = 512

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('results/pretrained', exist_ok=True)

    print(f"{'='*60}\nPretrained full sweep: {args.model}\n{'='*60}")
    print("Loading model...")
    model = load_pretrained_mamba(args.model, device)
    n_layers = len(model.backbone.layers)
    print(f"Model has {n_layers} layers")

    if args.smoke_test:
        layer_indices = sorted(set([0, n_layers // 2, n_layers - 1]))
        print(f"[SMOKE TEST] layers={layer_indices}, "
              f"max_lag={args.max_lag}, n_targets={args.n_targets}")
    else:
        layer_indices = parse_layer_indices(args.layer_indices, n_layers)
        print(f"Evaluating {len(layer_indices)} layers: "
              f"{layer_indices[:5]}{'...' if len(layer_indices) > 5 else ''}")

    # Tokenize input once
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    text = ("The history of artificial intelligence began in antiquity, "
            "with myths and stories of artificial beings endowed with intelligence. ") * 30
    input_ids = tokenizer.encode(text, return_tensors='pt')[:, :args.seq_len].to(device)
    print(f"Input shape: {input_ids.shape}")

    # Cache layer inputs in one forward pass
    print(f"\nCaching layer inputs (single forward pass)...")
    t_cache = time.time()
    cache = cache_layer_inputs(model, input_ids, layer_indices)
    print(f"  cached {len(cache)} layer inputs in {time.time() - t_cache:.0f}s "
          f"(memory: {sum(t.numel() * 4 for t in cache.values()) / 1e9:.2f} GB)")

    # Evaluate each layer
    all_results = {'model': args.model, 'n_layers': n_layers,
                   'config': vars(args), 'layers': {}}
    t0 = time.time()

    for k, layer_idx in enumerate(layer_indices):
        print(f"\nLayer {layer_idx} ({k+1}/{len(layer_indices)})...")
        t1 = time.time()
        try:
            r = evaluate_layer(
                model, layer_idx, cache[layer_idx], device,
                max_lag=args.max_lag, n_targets=args.n_targets, seed=args.seed,
            )
            r10 = r.get('ratio_tau10')
            r100 = r.get('ratio_tau100')
            hs = r.get('h_dyn_slope_10_100')
            es = r.get('erf_frozen_slope_10_100')
            ue100 = r.get('unsigned_overestimate_tau100')
            r10_s = f"{r10:.3f}" if isinstance(r10, float) else 'N/A'
            r100_s = f"{r100:.3f}" if isinstance(r100, float) else 'N/A'
            hs_s = f"{hs:.3f}" if isinstance(hs, float) else 'N/A'
            es_s = f"{es:.3f}" if isinstance(es, float) else 'N/A'
            ue_s = f"{ue100:.1e}" if isinstance(ue100, float) else 'N/A'
            print(f"  d_inner={r['d_inner']}, d_state={r['d_state']}")
            print(f"  ratio@10={r10_s}, ratio@100={r100_s}, "
                  f"slopes h={hs_s}, frozen={es_s}, unsigned/froz@100={ue_s}")
            print(f"  time {time.time() - t1:.0f}s "
                  f"(total {time.time() - t0:.0f}s)")

            # Strip full curves from summary (kept separately under 'curves')
            summary = {k: v for k, v in r.items()
                      if k not in ('h_dyn', 'erf_frozen', 'h_unsigned')}
            all_results['layers'][str(layer_idx)] = {
                **summary,
                'curves': {
                    'h_dyn': r['h_dyn'],
                    'erf_frozen': r['erf_frozen'],
                    'h_unsigned': r['h_unsigned'],
                },
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results['layers'][str(layer_idx)] = {'error': str(e)}

    out_path = args.results_path or (
                f'results/pretrained/full_sweep_smoke_{args.model.replace("/", "_")}.json'
                if args.smoke_test else
                f'results/pretrained/full_sweep_{args.model.replace("/", "_")}.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f)
    print(f"\nSaved to {out_path}")

    # Summary table
    print(f"\n{'='*82}")
    print(f"{'L':>3} | {'r@10':>7} | {'r@100':>7} | {'slope_h':>8} | "
          f"{'slope_f':>8} | {'g<0.01':>7} | {'unsigned/f@100':>14}")
    print('-' * 82)
    for idx_str in sorted(all_results['layers'].keys(), key=int):
        r = all_results['layers'][idx_str]
        if 'error' in r:
            print(f"{idx_str:>3} | ERROR: {r['error'][:60]}")
            continue
        r10 = r.get('ratio_tau10'); r100 = r.get('ratio_tau100')
        hs = r.get('h_dyn_slope_10_100'); es = r.get('erf_frozen_slope_10_100')
        gf = r.get('g_stats', {}).get('frac_below_0.01')
        ue = r.get('unsigned_overestimate_tau100')
        def fmt(v, w, dp=3, e=False):
            if v is None or not isinstance(v, float):
                return f"{'N/A':>{w}}"
            return f"{v:>{w}.{dp}{'e' if e else 'f'}}"
        print(f"{idx_str:>3} | {fmt(r10,7)} | {fmt(r100,7)} | "
              f"{fmt(hs,8)} | {fmt(es,8)} | {fmt(gf,7)} | {fmt(ue,14,1,e=True)}")


if __name__ == '__main__':
    main()
