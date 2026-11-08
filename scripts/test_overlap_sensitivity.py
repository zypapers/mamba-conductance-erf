"""
Sensitivity of the test-split results to WikiText-103's test/train overlap (App. E).
The test split repeats one training article, which fills
most of its last two 2048-token sequences (136 and 137, 0-based). Every test-split quantity of the matched-cohort
analysis and of the weight-decay control analysis is recomputed with and without those two sequences, from the
per-sequence curves of both cohorts, with the definitions of scripts/analyze_camera_ready_matched.py and
scripts/compare_wd_control.py (pooled mean over sequences, normalized at tau = 1; paired within-seed log10 ratios
and slope differences, mean and 95% t-interval across seeds). The recomputation with all sequences must reproduce
the published summaries (asserted).

Reported:
  - every interval that excludes its null value (1 for a ratio, 0 for a slope difference) with all test sequences
    but not without the two, or the reverse (key verdict_changes): paired ratios at the
    reported lags and bands and slope differences for E, X and S in both cohorts; the control minus the main cohort;
    the recipe effect per initialization; the slow - fast curve differences (E or X minus S);
  - the test-split values printed in App. E, with and without the two sequences, and the largest changes;
  - the bounds: the largest raw E/X over tau = 1-500 per cohort (2.3x and 1.45x).

Usage:
    python3 scripts/test_overlap_sensitivity.py
Output: results/camera_ready/test_overlap_sensitivity.json
"""

import json
import os

import numpy as np
from scipy import stats as st

COHORTS = {'main': 'matched', 'control': 'matched_nowd_timescale'}
CONDS, SEEDS = ['slow', 'default', 'fast'], [1, 2, 3]
PAIRS = [('slow', 'default'), ('fast', 'default'), ('slow', 'fast')]
LAGS = [10, 50, 100, 200, 500, 1000]
WINDOWS = {'lsq_10_100': (10, 100), 'lsq_100_1000': (100, 1000)}
BANDS = [(1, 1), (5, 15), (40, 60), (80, 120), (180, 220), (400, 600), (800, 1023)]
MID_BANDS = ['5-15', '40-60', '80-120', '180-220', '400-600']
PARTS = ['dt_ret', 'dt_write', 'B', 'C']
EXCLUDED = [136, 137]


def t95(x):
    x = np.asarray(x, dtype=float)
    m, h = x.mean(0), st.t.ppf(0.975, len(x) - 1) * x.std(0, ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def slope(f, a, b):
    lx, ly = np.log10(np.arange(a, b + 1)), np.log10(f[a:b + 1])
    lx = lx - lx.mean()
    return float((lx * (ly - ly.mean())).sum() / (lx ** 2).sum())


def gm(f):
    return np.array([np.exp(np.mean(np.log(f[a:b + 1]))) for a, b in BANDS])


def curves(exclude):
    """Pooled raw curves and band decompositions per (cohort, condition, seed) on the test split."""
    raw, dec, n = {}, {}, {}
    for name, cohort in COHORTS.items():
        for c in CONDS:
            for s in SEEDS:
                z = np.load(f'results/camera_ready/{cohort}/single_layer_{c}_seed{s}_test.npz')
                keep = np.ones(z['E'].shape[0], bool)
                keep[exclude] = False
                n[name] = int(keep.sum())
                raw[(name, c, s)] = {k: z[k][keep].mean(0) for k in ['E', 'X', 'S', 'R', 'P']}
                d = {k: z[k][keep].mean(0) for k in z.files if k.startswith('dec_')}
                dec[(name, c, s)] = {}
                for a, b in BANDS:
                    sl = slice(a, b + 1)
                    fr = d['dec_E_frozen'][sl].sum()
                    dec[(name, c, s)][f'{a}-{b}'] = {**{p: float(d[f'dec_E_{p}'][sl].sum() / fr) for p in PARTS},
                                                     'total_over_frozen': float(d['dec_total'][sl].sum() / fr)}
    return raw, dec, n


def intervals(raw):
    """Every interval of the two analyses on the test split: name -> (mean, lo, hi, null)."""
    f = {key: {k: v / v[1] for k, v in r.items()} for key, r in raw.items()}
    out = {}

    def put(name, lr, lb, dsl):
        for t in LAGS:
            out[f'{name} ratio at {t}'] = [float(10 ** v) for v in t95(lr[:, t])] + [1.0]
        m, lo, hi = t95(lb)
        for i, (a, b) in enumerate(BANDS[1:], start=1):
            out[f'{name} ratio band {a}-{b}'] = [float(10 ** m[i]), float(10 ** lo[i]), float(10 ** hi[i]), 1.0]
        for w, ab in WINDOWS.items():
            out[f'{name} slope difference {w}'] = [float(v) for v in t95(dsl(*ab))] + [0.0]

    for a, b in PAIRS:
        for k in ['E', 'X', 'S']:
            for co in COHORTS:
                put(f'{co} {a}/{b} {k}',
                    np.array([np.log10(f[(co, a, s)][k] / f[(co, b, s)][k]) for s in SEEDS]),
                    np.array([np.log10(gm(f[(co, a, s)][k]) / gm(f[(co, b, s)][k])) for s in SEEDS]),
                    lambda lo, hi, co=co: [slope(f[(co, a, s)][k], lo, hi) - slope(f[(co, b, s)][k], lo, hi)
                                           for s in SEEDS])
            put(f'control-main {a}/{b} {k}',
                np.array([np.log10(f[('control', a, s)][k] / f[('control', b, s)][k])
                          - np.log10(f[('main', a, s)][k] / f[('main', b, s)][k]) for s in SEEDS]),
                np.array([np.log10(gm(f[('control', a, s)][k]) / gm(f[('control', b, s)][k]))
                          - np.log10(gm(f[('main', a, s)][k]) / gm(f[('main', b, s)][k])) for s in SEEDS]),
                lambda lo, hi: [(slope(f[('control', a, s)][k], lo, hi) - slope(f[('control', b, s)][k], lo, hi))
                                - (slope(f[('main', a, s)][k], lo, hi) - slope(f[('main', b, s)][k], lo, hi))
                                for s in SEEDS])
    for c in CONDS:
        for k in ['E', 'X', 'S']:
            put(f'recipe effect {c} {k}',
                np.array([np.log10(f[('control', c, s)][k] / f[('main', c, s)][k]) for s in SEEDS]),
                np.array([np.log10(gm(f[('control', c, s)][k]) / gm(f[('main', c, s)][k])) for s in SEEDS]),
                lambda lo, hi: [slope(f[('control', c, s)][k], lo, hi) - slope(f[('main', c, s)][k], lo, hi)
                                for s in SEEDS])
    for co in COHORTS:
        for k in ['E', 'X']:
            for w, ab in WINDOWS.items():
                d = [(slope(f[(co, 'slow', s)][k], *ab) - slope(f[(co, 'fast', s)][k], *ab))
                     - (slope(f[(co, 'slow', s)]['S'], *ab) - slope(f[(co, 'fast', s)]['S'], *ab)) for s in SEEDS]
                out[f'{co} slow-fast curve difference {k}-S {w}'] = [float(v) for v in t95(d)] + [0.0]
    return out


def check_published(iv):
    """Largest discrepancy between the all-sequence intervals and the published summaries."""
    worst = 0.0
    for co, path in [('main', 'matched_summary.json'), ('control', 'matched_nowd_timescale_summary.json')]:
        pub = json.load(open(f'results/camera_ready/{path}'))['paired']['test']
        for a, b in PAIRS:
            for k in ['E', 'X', 'S']:
                p = pub[f'{a}_vs_{b}'][k]
                for t in LAGS:
                    worst = max(worst, max(abs(x / y - 1) for x, y in zip(iv[f'{co} {a}/{b} {k} ratio at {t}'],
                                                                          p['ratio_at_lags_mean_lo_hi'][str(t)])))
                for w in WINDOWS:
                    worst = max(worst, max(abs(x - y) for x, y in zip(iv[f'{co} {a}/{b} {k} slope difference {w}'],
                                                                      p['slope_difference_mean_lo_hi'][w])))
                for i, (lo, hi) in enumerate(BANDS[1:], start=1):
                    worst = max(worst, max(abs(x / p['ratio_bands_mean_t95'][j][i] - 1)
                                           for j, x in enumerate(iv[f'{co} {a}/{b} {k} ratio band {lo}-{hi}'][:3])))
    cd = json.load(open('results/camera_ready/c3_numbers.json'))['control']['slow_minus_fast_curve_differences']['test']
    for k in ['E', 'X']:
        for w in WINDOWS:
            worst = max(worst, max(abs(x - y) for x, y in zip(iv[f'control slow-fast curve difference {k}-S {w}'],
                                                              cd[f'{k}_minus_S'][w]['mean_lo_hi'])))
    pub = json.load(open('results/camera_ready/wd_control_comparison_matched_nowd_timescale.json'))
    for name, grp in [('control-main', 'recipe_x_init'), ('recipe effect', 'recipe_effect')]:
        for key, p in pub[grp]['test'].items():
            for k in ['E', 'X', 'S']:
                base = f'{name} {"/".join(key.split("_vs_"))} {k}' if grp == 'recipe_x_init' else f'{name} {key} {k}'
                for t in LAGS:
                    worst = max(worst, max(abs(x / y - 1) for x, y in zip(iv[f'{base} ratio at {t}'],
                                                                          p[k]['ratio_at_lags_mean_lo_hi'][str(t)])))
                for w in WINDOWS:
                    worst = max(worst, max(abs(x - y) for x, y in zip(iv[f'{base} slope difference {w}'],
                                                                      p[k]['slope_difference_mean_lo_hi'][w])))
    return worst


def excludes_null(v):
    return v[1] > v[3] or v[2] < v[3]


def printed(iv, raw, dec):
    """The test-split values App. E prints."""
    rr = {key: r['E'] / r['X'] for key, r in raw.items()}
    ctl = [key for key in rr if key[0] == 'control']
    table = {f'{co} {a}/{b} {w}': iv[f'{co} {a}/{b} E slope difference {w}'][:3]
             for co, a, b in [('main', 'slow', 'fast'), ('control', 'slow', 'fast'), ('control', 'slow', 'default'),
                              ('control', 'fast', 'default'), ('control-main', 'slow', 'fast')] for w in WINDOWS}
    ratios = {'control slow/fast E at 1000': iv['control slow/fast E ratio at 1000'][:3],
              'control-main fast/default S at 100': iv['control-main fast/default S ratio at 100'][:3],
              'main slow/default E at 500': iv['main slow/default E ratio at 500'][:3],
              'control raw E/X over tau 1-500': [float(min(rr[k][1:501].min() for k in ctl)),
                                                 float(max(rr[k][1:501].max() for k in ctl))],
              'control min raw E/X over tau 501-1023': [float(min(rr[k][501:].min() for k in ctl))],
              'control dt_write energy over frozen, max over bands 5-600':
                  [max(dec[k][b]['dt_write'] for k in ctl for b in MID_BANDS)]}
    bounds = {co: float(max(rr[k][1:501].max() for k in rr if k[0] == co)) for co in COHORTS}
    return table, ratios, bounds


def main():
    raw_all, dec_all, n_all = curves([])
    raw_d, dec_d, n_d = curves(EXCLUDED)
    iv_all, iv_d = intervals(raw_all), intervals(raw_d)
    worst = check_published(iv_all)
    assert worst < 1e-12, f'recomputation does not reproduce the published summaries ({worst:.2e})'
    flips = {k: {'all': iv_all[k][:3], 'without': iv_d[k][:3]} for k in iv_all
             if excludes_null(iv_all[k]) != excludes_null(iv_d[k])}
    t_all, r_all, b_all = printed(iv_all, raw_all, dec_all)
    t_d, r_d, b_d = printed(iv_d, raw_d, dec_d)
    out = {'excluded_test_sequences': EXCLUDED, 'n_test_sequences': {'all': n_all, 'without': n_d},
           'max_discrepancy_all_vs_published': worst, 'n_intervals_checked': len(iv_all),
           'verdict_changes': flips,
           'appendix_E_printed_values': {
               'table_slope_differences': {k: {'all': t_all[k], 'without': t_d[k]} for k in t_all},
               'ratios_and_ranges': {k: {'all': r_all[k], 'without': r_d[k]} for k in r_all},
               'max_abs_change_table_slope_differences': max(abs(x - y) for k in t_all for x, y in zip(t_all[k], t_d[k])),
               'max_relative_change_ratios_and_ranges': max(abs(y / x - 1) for k in r_all for x, y in zip(r_all[k], r_d[k]))},
           'bounds_max_raw_E_over_X_tau_1_500': {co: {'all': b_all[co], 'without': b_d[co]} for co in COHORTS}}
    os.makedirs('results/camera_ready', exist_ok=True)
    with open('results/camera_ready/test_overlap_sensitivity.json', 'w') as fh:
        json.dump(out, fh, indent=1)
    print('checked', len(iv_all), 'intervals; reproduces the published summaries to', f'{worst:.1e}')
    for k, v in flips.items():
        print('null exclusion changes without the two sequences:', k, [round(x, 4) for x in v['all']], '->', [round(x, 4) for x in v['without']])
    print('largest changes in App. E values:', round(out['appendix_E_printed_values']['max_abs_change_table_slope_differences'], 4),
          '(slope difference),', round(out['appendix_E_printed_values']['max_relative_change_ratios_and_ranges'], 4), '(relative)')
    print('bounds:', out['bounds_max_raw_E_over_X_tau_1_500'])
    print('wrote results/camera_ready/test_overlap_sensitivity.json')


if __name__ == '__main__':
    main()
