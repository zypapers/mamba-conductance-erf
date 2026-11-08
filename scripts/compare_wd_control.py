"""
Weight-decay control against the matched cohort.

The cohorts pair by seed and condition (same initial weights and data order). For E, X and S,
with the curve and normalization definitions of scripts/analyze_camera_ready_matched.py
(pooled mean over sequences, divided by its value at tau = 1):
  - recipe x initialization: per seed and initialization pair, the
    control's within-seed log10 ratio of normalized curves minus the matched cohort's, at the
    reported lags and bands, and the same difference of slope differences; mean and 95%
    t-interval across seeds (ratios exponentiated);
  - the recipe effect per initialization (descriptive): per seed, the control's normalized
    curve over the matched one's, and the slope difference; mean and t-interval;
  - side by side: final validation perplexity, the conductance and Delta statistics
    (scripts/conductance_stats.py) and the parameter drift (scripts/check_weight_decay_drift.py).

Usage:
    python3 scripts/compare_wd_control.py [--control matched_nowd_timescale]
Output: results/camera_ready/wd_control_comparison_<control>.json. Run with --control matched
as a check: every difference must then be exactly 0.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_camera_ready_matched import CONDS, LAGS, PAIRS, SPLITS, load, slopes, tint
from scripts.analyze_camera_ready_single_layer import BANDS, gm

SEEDS = [1, 2, 3]
WINDOWS = ['lsq_10_100', 'lsq_100_1000']


def norm_curves(cohort, split):
    f = {}
    for c in CONDS:
        for s in SEEDS:
            d, _ = load(c, s, split, False, cohort)
            f[(c, s)] = {k: d[k].mean(0) / d[k].mean(0)[1] for k in ['E', 'X', 'S']}
    return f


def summarize(lr, lb, dsl):
    """lr: seeds x lags log10 ratios (lag t at index t - 1); lb: seeds x bands; dsl: window -> seeds."""
    bm, blo, bhi = tint(lb)
    return {'ratio_at_lags_mean_lo_hi': {str(t): [float(10 ** v) for v in tint(lr[:, t - 1])] for t in LAGS},
            'ratio_bands_mean_t95': [(10 ** bm).tolist(), (10 ** blo).tolist(), (10 ** bhi).tolist()],
            'slope_difference_mean_lo_hi': {w: [float(v) for v in tint(dsl[w])] for w in WINDOWS}}


def side_by_side(control):
    def read(path):
        return json.load(open(path)) if os.path.exists(path) else None
    out = {}
    for name, cohort in [('matched', 'matched'), ('control', control)]:
        drift = read('results/camera_ready/weight_decay_drift.json' if cohort == 'matched'
                     else f'results/camera_ready/weight_decay_drift_{cohort}.json')
        cond = read(f'results/camera_ready/conductance_stats_{cohort}.json')
        ppl = {}
        for c in CONDS:
            for s in SEEDS:
                h = read(f'results/{cohort}/training_{c}_seed{s}.json')
                ppl[f'{c}_seed{s}'] = h['val_ppl'][-1] if h else None
        out[name] = {'final_val_ppl': ppl, 'conductance_stats': cond and cond['models'],
                     'parameter_drift': drift and drift['matched']}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', default='matched_nowd_timescale')
    args = p.parse_args()
    out = {'control': args.control,
           'seeds': SEEDS, 'bands': BANDS, 'lags': LAGS, 'recipe_x_init': {}, 'recipe_effect': {},
           'side_by_side': side_by_side(args.control)}
    for split in SPLITS:
        fm, fc = norm_curves('matched', split), norm_curves(args.control, split)
        out['recipe_x_init'][split], out['recipe_effect'][split] = {}, {}
        for a, b in PAIRS:
            res = {}
            for k in ['E', 'X', 'S']:
                lr = np.array([np.log10(fc[(a, s)][k][1:] / fc[(b, s)][k][1:])
                               - np.log10(fm[(a, s)][k][1:] / fm[(b, s)][k][1:]) for s in SEEDS])
                lb = np.array([np.log10(np.array(gm(fc[(a, s)][k])) / np.array(gm(fc[(b, s)][k])))
                               - np.log10(np.array(gm(fm[(a, s)][k])) / np.array(gm(fm[(b, s)][k])))
                               for s in SEEDS])
                dsl = {w: [(slopes(fc[(a, s)][k])[w] - slopes(fc[(b, s)][k])[w])
                           - (slopes(fm[(a, s)][k])[w] - slopes(fm[(b, s)][k])[w]) for s in SEEDS]
                       for w in WINDOWS}
                res[k] = summarize(lr, lb, dsl)
            out['recipe_x_init'][split][f'{a}_vs_{b}'] = res
        for c in CONDS:
            res = {}
            for k in ['E', 'X', 'S']:
                lr = np.array([np.log10(fc[(c, s)][k][1:] / fm[(c, s)][k][1:]) for s in SEEDS])
                lb = np.array([np.log10(np.array(gm(fc[(c, s)][k])) / np.array(gm(fm[(c, s)][k])))
                               for s in SEEDS])
                dsl = {w: [slopes(fc[(c, s)][k])[w] - slopes(fm[(c, s)][k])[w] for s in SEEDS]
                       for w in WINDOWS}
                res[k] = summarize(lr, lb, dsl)
            out['recipe_effect'][split][c] = res
    path = f'results/camera_ready/wd_control_comparison_{args.control}.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    e = out['recipe_x_init']['validation']['fast_vs_default']['E']
    print('validation, fast vs default, E: recipe x init ratio at tau = 100',
          e['ratio_at_lags_mean_lo_hi']['100'], '| slope difference 10-100',
          e['slope_difference_mean_lo_hi']['lsq_10_100'])
    print('wrote', path)


if __name__ == '__main__':
    main()
