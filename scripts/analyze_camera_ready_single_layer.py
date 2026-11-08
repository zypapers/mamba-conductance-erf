"""
Summary statistics for the legacy cohort (scripts/camera_ready_single_layer.py --cohort legacy)
-> results/camera_ready/single_layer_summary.json (not released). scripts/analyze_camera_ready_matched.py
reuses these statistics for the nine main models.

Curves are pooled means over sequences and targets, normalized at tau = 1
unless marked raw. Bands are geometric means over lag ranges. 95% CIs:
bootstrap over sequences (1000 resamples, seed 0); the A-init comparisons
resample the same sequences jointly for the three models (paired bootstrap).

Sections per model and split:
  - slopes (endpoint and least squares, tau = 10-100; least squares 100-1000)
    and normalized band values of S, Su, R, P, X, Xu, E;
  - ratios E/X, E/S, E/R, E/P, X/P, E/Xu, Su/S, Xu/X: bands, range over
    tau = 2-500, max |log10| over 2-500 and 501-1023, median |log10|;
  - raw ratios E/X, Su/S, Xu/X;
  - App. H: raw ratios of E_sub, Ddt, DB, DC to Dall, contributions at tau = 100;
  - App. D: normalized Dall / E_sub;
  - interference: per band, component energy ||g_s||^2 / ||g_d||^2, cross term
    2<g_d, g_s> / ||g_d||^2, net E_sub / Dall and the pooled cosine.

Usage:
    python3 scripts/analyze_camera_ready_single_layer.py [--partial]
"""

import argparse
import json

import numpy as np

BANDS = [(1, 1), (5, 15), (40, 60), (80, 120), (180, 220), (400, 600), (800, 1023)]
D_LAGS = [1, 5, 10, 50, 100, 200, 500, 1000]
H_LAGS = [1, 10, 100, 500, 1000]
N_BOOT = 1000
MODELS = ['slow', 'default', 'fast']
SPLITS = ['validation', 'test']
CURVES = ['S', 'Su', 'R', 'P', 'X', 'Xu', 'E']
RATIOS = [('E', 'X'), ('E', 'S'), ('E', 'R'), ('E', 'P'), ('X', 'P'), ('E', 'Xu'), ('Su', 'S'),
          ('Xu', 'X')]


def gm(r):
    return [float(np.exp(np.mean(np.log(r[a:b + 1])))) for a, b in BANDS]


def lsq(c, a, b):
    t = np.arange(a, b + 1)
    return float(np.polyfit(np.log10(t), np.log10(c[a:b + 1]), 1)[0])


def slopes(c):
    return {'endpoint_10_100': float(np.log10(c[100] / c[10])), 'lsq_10_100': lsq(c, 10, 100),
            'lsq_100_1000': lsq(c, 100, 1000)}


def stats(mean):
    """mean: name -> pooled raw curve over lags 0-1023."""
    n = {k: v / v[1] for k, v in mean.items()}
    out = {'slopes': {k: slopes(n[k]) for k in CURVES},
           'norm_bands': {k: gm(n[k]) for k in CURVES}, 'ratios': {}, 'raw_ratios': {}}
    for a, b in RATIOS:
        r = n[a] / n[b]
        lr = np.abs(np.log10(r))
        out['ratios'][f'{a}/{b}'] = {
            'bands': gm(r), 'range_2_500': [float(r[2:501].min()), float(r[2:501].max())],
            'max_abs_log10_2_500': float(lr[2:501].max()),
            'max_abs_log10_501_1023': float(lr[501:].max()),
            'median_abs_log10_2_1023': float(np.median(lr[2:]))}
    for a, b in [('E', 'X'), ('Su', 'S'), ('Xu', 'X')]:
        r = mean[a] / mean[b]
        out['raw_ratios'][f'{a}/{b}'] = {'bands': gm(r), 'at': {str(t): float(r[t]) for t in H_LAGS}}
    base = mean['Dall']
    out['appH_raw_ratio_to_Dall'] = {k: {str(t): float(mean[k][t] / base[t]) for t in H_LAGS}
                                     for k in ['E_sub', 'Ddt', 'DB', 'DC']}
    out['appH_contribution_tau100'] = {k: float((mean['E_sub'][100] - mean[k][100]) / base[100])
                                       for k in ['Ddt', 'DB', 'DC']}
    nd = (base / base[1]) / (mean['E_sub'] / mean['E_sub'][1])
    out['appD_Dall_over_E_norm'] = {'at': {str(t): float(nd[t]) for t in D_LAGS},
                                    'range_1_500': [float(nd[1:501].min()), float(nd[1:501].max())]}
    inter = {}
    for a, b in BANDS:
        d, s, x, f = (float(mean[k][a:b + 1].sum()) for k in ['Dall', 'Gss', 'Gx', 'E_sub'])
        inter[f'{a}-{b}'] = {'component_energy': s / d, 'cross': x / d, 'net_full_over_detached': f / d,
                             'pooled_cosine': x / 2 / np.sqrt(d * s)}
    out['interference'] = inter
    out['interference_identity_max_abs_dev'] = float(
        np.abs((base + mean['Gss'] + mean['Gx'])[1:] / mean['E_sub'][1:] - 1).max())
    return out


def with_ci(point, reps):
    """Attach 95% bootstrap intervals to every numeric leaf of point."""
    if isinstance(point, dict):
        return {k: with_ci(v, [r[k] for r in reps]) for k, v in point.items()}
    arr = np.array(reps, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=0)
    return {'point': point, 'ci95': [np.asarray(lo).tolist(), np.asarray(hi).tolist()]}


def load(cond, split, partial):
    z = np.load(f'results/camera_ready/single_layer_{cond}_{split}{"_partial" if partial else ""}.npz')
    return {k: z[k] for k in z.files if k not in ('targets', 'subset')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    rng = np.random.default_rng(0)
    out = {'bands': BANDS, 'n_boot': N_BOOT, 'note': 'see module docstring'}
    for split in SPLITS:
        data = {c: load(c, split, args.partial) for c in MODELS}
        n = len(data['slow']['E'])
        idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
        for c in MODELS:
            point = stats({k: v.mean(0) for k, v in data[c].items()})
            reps = [stats({k: v[i].mean(0) for k, v in data[c].items()}) for i in idx]
            out.setdefault(c, {})[split] = {'n_sequences': n, 'values': with_ci(point, reps)}

        def a_init(sel):
            s = {c: slopes(data[c]['E'][sel].mean(0) / data[c]['E'][sel].mean(0)[1]) for c in MODELS}
            return {f'{c}_minus_slow_{w}': s[c][w] - s['slow'][w]
                    for c in ['default', 'fast'] for w in ['lsq_10_100', 'lsq_100_1000']}
        out.setdefault('a_init_full_model_slope_differences', {})[split] = with_ci(
            a_init(slice(None)), [a_init(i) for i in idx])
        print(split, 'done', flush=True)
    path = f'results/camera_ready/single_layer_summary{"_partial" if args.partial else ""}.json'
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote', path)


if __name__ == '__main__':
    main()
