"""
Summary for the matched cohort (the nine main language models).
Output: results/camera_ready/matched_summary.json.

Per checkpoint (condition x seed) and split:
  - the statistics of scripts/analyze_camera_ready_single_layer.py, with sequence-bootstrap
    intervals that are conditional on that checkpoint;
  - the exact selectivity decomposition per band: each part's energy and every pairwise term
    relative to the frozen part's energy, and total / frozen.
Across the training seeds of each condition:
  - for E, X and S: the geometric mean of the normalized curves f_r = E_r / E_r(1), with a
    back-transformed pointwise 95% t-interval, as full curves (for figures) and as bands;
  - per-seed least-squares slopes (tau = 10-100, 100-1000), with mean and t-interval.
Initialization effects, paired within seed (slow vs default, fast vs default, slow vs fast):
  - log10 ratios of normalized curves, with mean and t-interval across seeds, exponentiated,
    at bands and selected lags;
  - slope differences, with mean and t-interval.
Block form against the full model: per checkpoint, raw and normalized E/X over tau = 2-500,
and the largest |log10| over tau = 501-1023; plus the worst case over all checkpoints.

With --tag <tag>, the same summary for the weight-decay control cohort, read from
results/camera_ready/matched_<tag>/ and written to results/camera_ready/matched_<tag>_summary.json.

Usage:
    python3 scripts/analyze_camera_ready_matched.py [--partial] [--seeds 1,2,3] [--n_boot 1000] [--tag T]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyze_camera_ready_single_layer import BANDS, gm, lsq, stats as ckpt_stats, with_ci

CONDS = ['slow', 'default', 'fast']
LAGS = [10, 50, 100, 200, 500, 1000]
PAIRS = [('slow', 'default'), ('fast', 'default'), ('slow', 'fast')]
PARTS = ['frozen', 'dt_ret', 'dt_write', 'B', 'C']
SPLITS = ['validation', 'test']


def load(cond, seed, split, partial, cohort='matched'):
    z = np.load(f'results/camera_ready/{cohort}/single_layer_{cond}_seed{seed}_{split}'
                f'{"_partial" if partial else ""}.npz')
    skip = ('targets', 'subset', 'decomp_targets', 'meta')
    return {k: z[k] for k in z.files if k not in skip}, json.loads(str(z['meta']))


def tint(x):
    """Mean and pointwise 95% t-interval across seeds (axis 0)."""
    x = np.asarray(x, dtype=float)
    m = x.mean(0)
    if len(x) < 2:
        return m, m, m
    h = st.t.ppf(0.975, len(x) - 1) * x.std(0, ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def decomposition(mean):
    out = {}
    for a, b in BANDS:
        sl = slice(a, b + 1)
        f = mean['dec_E_frozen'][sl].sum()
        row = {f'E_{c}': float(mean[f'dec_E_{c}'][sl].sum() / f) for c in PARTS}
        row.update({k[4:]: float(mean[k][sl].sum() / f) for k in mean if k.startswith('dec_X_')})
        row['total_over_frozen'] = float(mean['dec_total'][sl].sum() / f)
        out[f'{a}-{b}'] = row
    return out


def slopes(f):
    return {'lsq_10_100': lsq(f, 10, 100), 'lsq_100_1000': lsq(f, 100, 1000)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    p.add_argument('--seeds', default='1,2,3')
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--tag', default='', help='the weight-decay control cohort matched_<tag>')
    args = p.parse_args()
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    seeds = [int(s) for s in args.seeds.split(',')]
    rng = np.random.default_rng(0)
    out = {'seeds': seeds,
           'bands': BANDS, 'lags': LAGS, 'n_boot': args.n_boot, 'per_checkpoint': {},
           'across_seeds': {}, 'paired': {}, 'block_agreement': {}}
    for split in SPLITS:
        f, raw, metas = {}, {}, {}
        out['per_checkpoint'][split] = {}
        for c in CONDS:
            for s in seeds:
                d, meta = load(c, s, split, args.partial, cohort)
                main_keys = {k: v for k, v in d.items() if not k.startswith('dec_')}
                n = len(d['E'])
                point = ckpt_stats({k: v.mean(0) for k, v in main_keys.items()})
                reps = [ckpt_stats({k: v[i].mean(0) for k, v in main_keys.items()})
                        for i in (rng.integers(0, n, n) for _ in range(args.n_boot))]
                out['per_checkpoint'][split][f'{c}_seed{s}'] = {
                    'meta': meta, 'n_sequences': n,
                    'sequence_bootstrap_conditional_on_checkpoint': with_ci(point, reps),
                    'decomposition': decomposition({k: v.mean(0) for k, v in d.items()
                                                    if k.startswith('dec_')})}
                raw[(c, s)] = {k: d[k].mean(0) for k in ['E', 'X', 'S']}
                f[(c, s)] = {k: v / v[1] for k, v in raw[(c, s)].items()}
        out['across_seeds'][split] = {}
        for c in CONDS:
            res = {}
            for k in ['E', 'X', 'S']:
                z = np.log10(np.array([f[(c, s)][k][1:] for s in seeds]))
                m, lo, hi = tint(z)
                zb = np.log10(np.array([gm(f[(c, s)][k]) for s in seeds]))
                bm, blo, bhi = tint(zb)
                sl = {w: tint([slopes(f[(c, s)][k])[w] for s in seeds]) for w in ['lsq_10_100', 'lsq_100_1000']}
                res[k] = {'geometric_mean_curve_lag1_to_1023': (10 ** m).tolist(),
                          'pointwise_t95_lo': (10 ** lo).tolist(), 'pointwise_t95_hi': (10 ** hi).tolist(),
                          'bands_geometric_mean': (10 ** bm).tolist(),
                          'bands_t95': [(10 ** blo).tolist(), (10 ** bhi).tolist()],
                          'per_seed_slopes': {str(s): slopes(f[(c, s)][k]) for s in seeds},
                          'slopes_mean_t95': {w: [float(v) for v in sl[w]] for w in sl}}
            out['across_seeds'][split][c] = res
        out['paired'][split] = {}
        for a, b in PAIRS:
            res = {}
            for k in ['E', 'X', 'S']:
                lr = np.array([np.log10(f[(a, s)][k][1:] / f[(b, s)][k][1:]) for s in seeds])
                lb = np.array([np.log10(np.array(gm(f[(a, s)][k])) / np.array(gm(f[(b, s)][k])))
                               for s in seeds])
                bm, blo, bhi = tint(lb)
                at = {str(t): [float(10 ** v) for v in tint(lr[:, t - 1])] for t in LAGS}
                dsl = {w: [float(v) for v in tint([slopes(f[(a, s)][k])[w] - slopes(f[(b, s)][k])[w]
                                                   for s in seeds])] for w in ['lsq_10_100', 'lsq_100_1000']}
                res[k] = {'ratio_bands_mean_t95': [(10 ** bm).tolist(), (10 ** blo).tolist(), (10 ** bhi).tolist()],
                          'ratio_at_lags_mean_lo_hi': at, 'slope_difference_mean_lo_hi': dsl}
            out['paired'][split][f'{a}_vs_{b}'] = res
        agree = {}
        for (c, s) in raw:
            rr = raw[(c, s)]['E'] / raw[(c, s)]['X']
            nr = f[(c, s)]['E'] / f[(c, s)]['X']
            agree[f'{c}_seed{s}'] = {'raw_range_2_500': [float(rr[2:501].min()), float(rr[2:501].max())],
                                     'norm_range_2_500': [float(nr[2:501].min()), float(nr[2:501].max())],
                                     'raw_max_abs_log10_501_1023': float(np.abs(np.log10(rr[501:])).max()),
                                     'norm_max_abs_log10_501_1023': float(np.abs(np.log10(nr[501:])).max())}
        agree['worst_over_checkpoints'] = {
            key: [min(v[key][0] for v in agree.values()), max(v[key][1] for v in agree.values())]
            for key in ['raw_range_2_500', 'norm_range_2_500']}
        out['block_agreement'][split] = agree
        print(split, 'done', flush=True)
    path = f'results/camera_ready/{cohort}_summary{"_partial" if args.partial else ""}.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
