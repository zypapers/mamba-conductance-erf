"""
Learned A, actual Delta and conductance statistics of a single-layer cohort.

For each final checkpoint, on the first N validation sequences (default 8), pooled over
sequences, positions and channels (and states, for the conductances):
  - learned |A| = exp(A_log): p5, median, p95;
  - the actual Delta_t = softplus(W_dt xi_t + b), not just softplus(b): p5, median, p95, and the
    median of softplus(b) for comparison;
  - |A| Delta and the conductances g = 1 - exp(A Delta): the coast fraction (g < 0.01), the
    median of g (and the corresponding median of |A| Delta), and the fraction with g > 0.99;
  - a 60-bin histogram of log10 g over [-6, 0] (values below 1e-6 counted in the first bin),
    for the figures.
The same code runs on the matched cohort and on a weight-decay control cohort (--tag), so the
two are compared on one definition.

Usage:
    python3 scripts/conductance_stats.py [--tag <tag>] [--n_seq 8]
Output: results/camera_ready/conductance_stats_matched[_<tag>].json. Runtime: about 1 min.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.camera_ready_single_layer import CONDS, DEV, load
from src.data.wikitext import get_dataloader


def pct(x, qs=(5, 50, 95)):
    return {f'p{q}': float(np.percentile(x, q)) for q in qs}


@torch.no_grad()
def stats(m, ids):
    _, inter = m(input_ids=ids)
    dt = inter['delta_t'].float()
    absA = torch.exp(m.A_log.float())
    adt = dt[..., None] * absA
    g = -torch.expm1(-adt)
    med_g = float(g.flatten().median())
    bias = m.dt_proj.bias.float()
    hist = torch.histc(torch.log10(g.clamp_min(1e-6)), bins=60, min=-6.0, max=0.0)
    return {
        'abs_A': pct(absA.cpu().numpy().ravel()),
        'delta': pct(dt.cpu().numpy().ravel()),
        'softplus_bias_median': float(torch.nn.functional.softplus(bias).median()),
        'coast_fraction_g_below_0.01': float((g < 0.01).double().mean()),
        'median_g': med_g,
        'median_abs_A_delta': float(-np.log1p(-med_g)),
        'fraction_g_above_0.99': float((g > 0.99).double().mean()),
        'log10_g_histogram': {'bins': 60, 'range': [-6.0, 0.0], 'clamped_below': 1e-6,
                              'fraction': (hist / hist.sum()).tolist()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='', help='weight-decay control cohort; default: the matched cohort')
    p.add_argument('--n_seq', type=int, default=8)
    p.add_argument('--seeds', default='1,2,3')
    args = p.parse_args()
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    loader = get_dataloader(split='validation', batch_size=args.n_seq, seq_len=2048, shuffle=False,
                            num_workers=0)
    ids = next(iter(loader))[0].to(DEV)
    out = {'cohort': cohort, 'n_sequences': args.n_seq, 'split': 'validation (first sequences)',
           'models': {}}
    for s in [int(x) for x in args.seeds.split(',')]:
        for c in CONDS:
            path = f'checkpoints/{cohort}/seed{s}/final_{c}.pt'
            r = stats(load(path), ids)
            out['models'][f'{c}_seed{s}'] = {'checkpoint': path, **r}
            print(f"{c:7s} seed {s}: |A| median {r['abs_A']['p50']:.2f} [p5 {r['abs_A']['p5']:.2f}, "
                  f"p95 {r['abs_A']['p95']:.2f}] | Delta median {r['delta']['p50']:.3f} [p5 {r['delta']['p5']:.4f}, "
                  f"p95 {r['delta']['p95']:.2f}] (softplus bias {r['softplus_bias_median']:.3f}) | "
                  f"coast {100 * r['coast_fraction_g_below_0.01']:.1f}% | median g {r['median_g']:.3f} "
                  f"(|A|Delta {r['median_abs_A_delta']:.2f}) | g>0.99 {100 * r['fraction_g_above_0.99']:.1f}%",
                  flush=True)
    path = f'results/camera_ready/conductance_stats_{cohort}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
