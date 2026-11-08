"""
How far weight decay could have moved the timescale parameters.

The WikiText recipe passes every parameter to one AdamW group with weight decay 0.1. This
script compares, for the nine matched-cohort final checkpoints, the learned A_log, dt_proj.bias
and D with their exact initial values (re-created from the seed, as scripts/train_matched.py
does) and with the decay-only prediction (initial value times the schedule's decay multiplier).
For reference it also summarizes A, dt_proj.bias and D in the released Mamba-1 checkpoints
(130M, 370M), whose module marks A_log and D as exempt from weight decay.

Per model: percentiles of initial and learned values, softplus of the bias, the rank
correlation of learned with initial values, and the learned spread left unexplained by the
decay-only map, std(learned - m * initial) / std(learned).

With --tag <tag>, the same for the weight-decay control cohort checkpoints/matched_<tag>/.
Parameters a checkpoint records as exempt (args.no_wd) get multiplier 1, so the decay-only
prediction for them is the initial value.

Output: results/camera_ready/weight_decay_drift.json (matched cohort and released checkpoints)
or results/camera_ready/weight_decay_drift_matched_<tag>.json. Runtime: under a minute, CPU.

Usage:
    python3 scripts/check_weight_decay_drift.py [--tag <tag>]
"""

import argparse
import glob
import json
import math
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.intervention_experiment as ie

STEPS, WARMUP, LR, WD = 86316, 500, 1e-3, 0.1


def lr_lambda(k):
    if k < WARMUP:
        return k / WARMUP
    return 0.5 * (1.0 + math.cos(math.pi * (k - WARMUP) / (STEPS - WARMUP)))


def softplus(x):
    return np.log1p(np.exp(x))


def pct(x):
    return {f'p{q}': float(np.percentile(x, q)) for q in (5, 50, 95)}


def compare(x0, x1, m):
    return {'initial': pct(x0), 'learned': pct(x1), 'decay_only_median': float(np.median(m * x0)),
            'spearman_initial_learned': float(spearmanr(x0, x1)[0]),
            'unexplained_by_decay_only': float(np.std(x1 - m * x0) / np.std(x1))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='', help='the weight-decay control cohort matched_<tag>')
    args = p.parse_args()
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    m = math.exp(sum(math.log(1 - LR * WD * lr_lambda(k)) for k in range(STEPS)))
    out = {'schedule': {'steps': STEPS, 'warmup': WARMUP, 'peak_lr': LR, 'weight_decay': WD,
                        'summed_lr': sum(LR * lr_lambda(k) for k in range(STEPS)),
                        'decay_only_multiplier': m},
           'cohort': cohort, 'matched': {}, 'released': {}}
    for seed in (1, 2, 3):
        for cond in ('slow', 'default', 'fast'):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            cfg = ie.A_CONFIGS[cond]
            m0 = ie.create_model_with_A_init(256, 512, 16, cfg['low'], cfg['high'])
            ck = torch.load(f'checkpoints/{cohort}/seed{seed}/final_{cond}.pt', map_location='cpu',
                            weights_only=False)
            sd, no_wd = ck['model_state_dict'], ck.get('args', {}).get('no_wd', [])
            mult = {n: 1.0 if n in no_wd else m for n in ('A_log', 'dt_proj.bias', 'D')}
            a0, a1 = m0.A_log.detach().numpy().ravel(), sd['A_log'].numpy().ravel()
            b0, b1 = m0.dt_proj.bias.detach().numpy(), sd['dt_proj.bias'].numpy()
            d0, d1 = m0.D.detach().numpy(), sd['D'].numpy()
            out['matched'][f'{cond}_seed{seed}'] = {
                'no_wd': no_wd,
                'A_log': {**compare(a0, a1, mult['A_log']), 'learned_mean': float(a1.mean()),
                          'decay_only_mean': float(mult['A_log'] * a0.mean()),
                          'learned_abs_A': pct(np.exp(a1))},
                'dt_bias': {**compare(b0, b1, mult['dt_proj.bias']), 'softplus_initial_median': float(np.median(softplus(b0))),
                            'softplus_learned_median': float(np.median(softplus(b1)))},
                'D': {'initial': pct(d0), 'learned': pct(d1),
                      'decay_only_median': float(np.median(mult['D'] * d0))},
            }
    for name in ('mamba-130m', 'mamba-370m') if not args.tag else ():
        files = sorted(glob.glob(os.path.expanduser(
            f'~/.cache/huggingface/hub/models--state-spaces--{name}/snapshots/*/pytorch_model.bin')))
        if not files:
            continue
        sd = torch.load(files[0], map_location='cpu', weights_only=True)
        layers = sorted({int(k.split('.')[2]) for k in sd if k.endswith('mixer.A_log')})
        get = lambda l, p: sd[f'backbone.layers.{l}.mixer.{p}'].float().numpy()
        A = np.concatenate([np.exp(get(l, 'A_log')).ravel() for l in layers])
        b = np.concatenate([get(l, 'dt_proj.bias') for l in layers])
        D = np.concatenate([get(l, 'D') for l in layers])
        out['released'][name] = {
            'layers': len(layers), 'abs_A': pct(A),
            'per_layer_median_abs_A': [float(np.median(np.exp(get(l, 'A_log')))) for l in layers],
            'dt_bias': pct(b), 'softplus_dt_bias_median': float(np.median(softplus(b))),
            'per_layer_softplus_dt_bias_median': [float(np.median(softplus(get(l, 'dt_proj.bias'))))
                                                  for l in layers],
            'D': pct(D),
            'initialization': 'A = 1..16 per state (median 8.5); softplus(dt bias) log-uniform on '
                              '[0.001, 0.1]; D = 1',
        }
    os.makedirs('results/camera_ready', exist_ok=True)
    path = f'results/camera_ready/weight_decay_drift{"_" + cohort if args.tag else ""}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"decay-only multiplier {m:.5f} (summed lr {out['schedule']['summed_lr']:.3f})")
    for k, v in out['matched'].items():
        print(f"{k:13s} dt bias median {v['dt_bias']['initial']['p50']:+.2f} -> {v['dt_bias']['learned']['p50']:+.3f} "
              f"(decay-only {v['dt_bias']['decay_only_median']:+.3f}), rho {v['dt_bias']['spearman_initial_learned']:.2f}, "
              f"unexplained {v['dt_bias']['unexplained_by_decay_only']:.2f} | A_log rho {v['A_log']['spearman_initial_learned']:+.2f}, "
              f"unexplained {v['A_log']['unexplained_by_decay_only']:.2f}, mean {v['A_log']['learned_mean']:+.3f} | "
              f"D median {v['D']['learned']['p50']:+.3f}")
    for k, v in out['released'].items():
        print(f"{k}: |A| median {v['abs_A']['p50']:.2f}, per-layer {min(v['per_layer_median_abs_A']):.2f}-"
              f"{max(v['per_layer_median_abs_A']):.2f}; dt bias median {v['dt_bias']['p50']:+.2f} "
              f"(softplus {v['softplus_dt_bias_median']:.3f}); D median {v['D']['p50']:.2f}")


if __name__ == '__main__':
    main()
