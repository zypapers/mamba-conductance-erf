"""
Numbers for the weight-decay control (Sec. 4.5 and App. E),
collected from the outputs of scripts/launch_wd_control.sh. Nothing is recomputed from scratch except the
three-regime block-agreement summary, which follows scripts/section4_numbers.py on the control's
per-sequence curves.

For the control cohort (and the matched cohort alongside, where relevant):
  1. slow minus fast in the full-model ERF's slope over tau = 10-100 (keys primary_*): mean and 95% t-interval
     across seeds, on validation and test, classified in this order: 'detected in the predicted direction on both
     splits' if the interval's lower bound is above 0 on both splits; 'validation only, in the predicted direction'
     if it is above 0 on validation; the same two cases with the upper bound below 0 ('against the predicted
     direction'); otherwise 'not detected with three seeds'. The same contrast for the other pairs, the 100-1000
     window, and X and S (matched_<tag>_summary.json, key paired);
  2. recipe x initialization: the control's within-seed contrast minus the matched cohort's
     (wd_control_comparison_matched_<tag>.json, key recipe_x_init);
  3. parameter retention (weight_decay_drift_matched_<tag>.json);
  4. actual Delta and conductances (conductance_stats_matched_<tag>.json);
  5. scan-probe contrasts (paired S), and the direct test of whether E or X responds to the initialization
     differently from S: the slow - fast slope contrast's paired difference between curves (key
     slow_minus_fast_curve_differences);
  6. block form against the full model: per checkpoint and worst over checkpoints, raw and normalized E/X
     over tau = 2-500, the range beyond 500 (summary key block_agreement), the three regimes of
     Table 1 recomputed from the per-sequence curves, and the decomposition per band; and the headline
     agreement numbers on the control cohort, computed as scripts/abstract_numbers.py and scripts/section4_numbers.py
     compute it for the matched cohort (key claim2);
  7. final validation perplexity and best epoch per model, both cohorts, and the control minus its
     matched pair.
Per-seed E slopes of every model, both windows (key per_seed_E_slopes), from the summary.

Usage:
    python3 scripts/c3_numbers.py [--tag nowd_timescale] [--out PATH]
    python3 scripts/c3_numbers.py --tag '' --out /tmp/x.json   # dry run on the matched cohort itself
Output: results/camera_ready/c3_numbers.json by default.
"""

import argparse
import json
import os

import numpy as np
from scipy import stats as st

CONDS = ['slow', 'default', 'fast']
SEEDS = [1, 2, 3]
SPLITS = ['validation', 'test']
PAIRS = ['slow_vs_fast', 'slow_vs_default', 'fast_vs_default']
MODELS = [f'{c}_seed{s}' for s in SEEDS for c in CONDS]
LAGS = [1, 10, 50, 100, 200, 500, 1000]                     # the lag table of scripts/section4_numbers.py


def span(v):
    v = [float(x) for x in v]
    return [min(v), max(v)]


def read(path):
    return json.load(open(path)) if os.path.exists(path) else None


def contrasts(summ):
    out = {}
    for split in SPLITS:
        out[split] = {p: {k: {'slope_difference_mean_lo_hi': summ['paired'][split][p][k]['slope_difference_mean_lo_hi'],
                              'ratio_at_lags_mean_lo_hi': summ['paired'][split][p][k]['ratio_at_lags_mean_lo_hi']}
                          for k in ['E', 'X', 'S']} for p in PAIRS}
    prim = {split: out[split]['slow_vs_fast']['E']['slope_difference_mean_lo_hi']['lsq_10_100'] for split in SPLITS}
    lo_pos = {split: prim[split][1] > 0 for split in SPLITS}
    hi_neg = {split: prim[split][2] < 0 for split in SPLITS}
    if all(lo_pos.values()):
        verdict = 'detected in the predicted direction on both splits'
    elif lo_pos['validation']:
        verdict = 'validation only, in the predicted direction'
    elif all(hi_neg.values()):
        verdict = 'detected against the predicted direction on both splits'
    elif hi_neg['validation']:
        verdict = 'validation only, against the predicted direction'
    else:
        verdict = 'not detected with three seeds'
    return {'primary_slow_minus_fast_E_slope_10_100_mean_lo_hi': prim, 'primary_verdict': verdict,
            'all_pairs': out}


def regimes(cohort):
    out = {}
    for split in SPLITS:
        raw, norm, per_seq = {}, {}, {}
        for m in MODELS:
            z = np.load(f'results/camera_ready/{cohort}/single_layer_{m}_{split}.npz')
            e, x = z['E'].mean(0), z['X'].mean(0)
            raw[m], norm[m] = e / x, (e / e[1]) / (x / x[1])
            rs = z['E'][:, 1:501] / z['X'][:, 1:501]              # each sequence's own ratio, tau = 1-500
            per_seq[m] = {'n_sequences': int(rs.shape[0]), 'n_above_2.3': int((rs.max(1) > 2.3).sum()),
                          'max': float(rs.max())}
        out[split] = {'max_raw_E_over_X_tau_1_3': float(max(r[1:4].max() for r in raw.values())),
                      'max_raw_E_over_X_tau_4_500': float(max(r[4:501].max() for r in raw.values())),
                      'max_raw_E_over_X_tau_501_1023': float(max(r[501:].max() for r in raw.values())),
                      'min_raw_E_over_X_tau_1_1023': float(min(r[1:].min() for r in raw.values())),
                      'raw_E_over_X_tau_1_500': span([min(r[1:501].min() for r in raw.values()),
                                                      max(r[1:501].max() for r in raw.values())]),
                      'per_model_raw_max_tau_4_500': {m: float(raw[m][4:501].max()) for m in MODELS},
                      'per_model_raw_max_tau_1_500': {m: float(raw[m][1:501].max()) for m in MODELS},
                      'per_sequence_raw_E_over_X_tau_1_500': per_seq}
    return out


def claim2(cohort, summ):
    """The headline agreement numbers on a cohort, pooled over both splits as in scripts/abstract_numbers.py, with the lag
    table of scripts/section4_numbers.py (minimum, median and maximum of raw E/X over the nine) per split."""
    raw_lo, raw_hi, worst, table = np.inf, 0.0, {k: 0.0 for k in 'SRPX'}, {}
    for split in SPLITS:
        rows = []
        for m in MODELS:
            z = np.load(f'results/camera_ready/{cohort}/single_layer_{m}_{split}.npz')
            e = z['E'].mean(0)
            r = e / z['X'].mean(0)
            raw_lo, raw_hi = min(raw_lo, r[1:501].min()), max(raw_hi, r[1:501].max())
            for k in worst:
                q = z[k].mean(0)
                n = (e / e[1]) / (q / q[1])
                worst[k] = max(worst[k], float(10 ** np.abs(np.log10(n[1:501])).max()))
            rows.append(r)
        table[split] = {str(t): {'min': float(min(r[t] for r in rows)), 'median': float(np.median([r[t] for r in rows])),
                                 'max': float(max(r[t] for r in rows))} for t in LAGS}
    dsl, beyond = {w: 0.0 for w in ['lsq_10_100', 'lsq_100_1000']}, 0.0
    for split in SPLITS:
        for m in MODELS:
            sl = summ['per_checkpoint'][split][m]['sequence_bootstrap_conditional_on_checkpoint']['slopes']
            for w in dsl:
                dsl[w] = max(dsl[w], abs(sl['X'][w]['point'] - sl['E'][w]['point']))
            beyond = max(beyond, 10 ** summ['block_agreement'][split][m]['raw_max_abs_log10_501_1023'])
    return {'raw_E_over_X_tau_1_500': [float(raw_lo), float(raw_hi)], 'max_abs_slope_difference_X_minus_E': dsl,
            'max_raw_abs_log10_factor_beyond_500': float(beyond), 'worst_normalized_miss_vs_E_tau_1_500': worst,
            'lag_table_raw_E_over_X': table}


def per_seed_slopes(summ):
    return {split: {w: {f'{c}_seed{s}': summ['across_seeds'][split][c]['E']['per_seed_slopes'][str(s)][w]
                        for s in SEEDS for c in CONDS} for w in ['lsq_10_100', 'lsq_100_1000']} for split in SPLITS}


def curve_differences(summ):
    """Per seed, the slow - fast slope difference of E (or X) minus that of S, with the mean and 95% t-interval
    across seeds (as scripts/analyze_camera_ready_matched.py forms them). A contrast detected in one curve and not
    in another is not by itself a difference between the curves; this is the direct test."""
    a, out = summ['across_seeds'], {}
    for split in SPLITS:
        def sf(k, w):
            return np.array([a[split]['slow'][k]['per_seed_slopes'][str(s)][w]
                             - a[split]['fast'][k]['per_seed_slopes'][str(s)][w] for s in SEEDS])
        out[split] = {}
        for k in ['E', 'X']:
            for w in ['lsq_10_100', 'lsq_100_1000']:
                d = sf(k, w) - sf('S', w)
                h = st.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                out[split].setdefault(f'{k}_minus_S', {})[w] = {
                    'per_seed': [float(x) for x in d], 'mean_lo_hi': [float(d.mean()), float(d.mean() - h), float(d.mean() + h)]}
    return out


def decomposition(summ):
    per = summ['per_checkpoint']['validation']
    bands = list(per[MODELS[0]]['decomposition'])
    parts = ['E_dt_ret', 'E_dt_write', 'E_B', 'E_C']
    return {b: {**{p: span(per[m]['decomposition'][b][p] for m in MODELS) for p in parts},
                'total_over_frozen': span(per[m]['decomposition'][b]['total_over_frozen'] for m in MODELS),
                'largest_omitted': sorted({max(parts, key=lambda p: per[m]['decomposition'][b][p]) for m in MODELS})}
            for b in bands}


def cohort_block(cohort, summ):
    ba = summ['block_agreement']
    return {'block_agreement': {split: {m: ba[split][m] for m in MODELS} | {'worst': ba[split]['worst_over_checkpoints']}
                                for split in SPLITS},
            'regimes': regimes(cohort), 'claim2': claim2(cohort, summ), 'decomposition_validation': decomposition(summ),
            'per_seed_E_slopes': per_seed_slopes(summ), 'slow_minus_fast_curve_differences': curve_differences(summ)}


def training(cohort):
    rows = {}
    for m in MODELS:
        c, s = m.split('_seed')
        h = read(f'results/{cohort}/training_{c}_seed{s}.json')
        rows[m] = None if h is None else {'val_ppl_final': h['val_ppl'][-1], 'best_epoch': h['best_epoch'],
                                          'val_ppl_best': min(h['val_ppl'])}
    good = [r['val_ppl_final'] for r in rows.values() if r]
    return {'per_model': rows, 'val_ppl_final_range': span(good) if good else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='nowd_timescale')
    ap.add_argument('--out', default='results/camera_ready/c3_numbers.json')
    args = ap.parse_args()
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    summ = read(f'results/camera_ready/{cohort}_summary.json')
    msumm = read('results/camera_ready/matched_summary.json')
    comp = read(f'results/camera_ready/wd_control_comparison_{cohort}.json')
    drift = read(f'results/camera_ready/weight_decay_drift_{cohort}.json' if args.tag
                 else 'results/camera_ready/weight_decay_drift.json')
    cond = read(f'results/camera_ready/conductance_stats_{cohort}.json')
    out = {'cohort': cohort,
           'control': {'contrasts': contrasts(summ), **cohort_block(cohort, summ), 'training': training(cohort)},
           'matched_reference': {'contrasts': contrasts(msumm)['primary_slow_minus_fast_E_slope_10_100_mean_lo_hi'],
                                 'training': training('matched')}}
    if args.tag:
        ctl, ref = out['control']['training']['per_model'], out['matched_reference']['training']['per_model']
        out['control']['training']['val_ppl_final_minus_matched'] = {
            m: ctl[m]['val_ppl_final'] - ref[m]['val_ppl_final'] for m in MODELS if ctl[m] and ref[m]}
    if comp is not None:
        rx = comp['recipe_x_init']
        out['recipe_x_init'] = {split: {p: {k: rx[split][p][k]['slope_difference_mean_lo_hi'] for k in ['E', 'X', 'S']}
                                        for p in PAIRS} for split in SPLITS}
        out['recipe_x_init_primary_E_slope_10_100'] = {split: rx[split]['slow_vs_fast']['E']['slope_difference_mean_lo_hi']['lsq_10_100']
                                                       for split in SPLITS}
        out['recipe_effect'] = {split: {c: comp['recipe_effect'][split][c]['E']['slope_difference_mean_lo_hi']
                                        for c in CONDS} for split in SPLITS}
    if drift is not None:
        models = drift.get(cohort) or drift.get('matched')
        out['retention'] = {m: {'A_log': {k: models[m]['A_log'][k] for k in
                                          ['initial', 'learned', 'spearman_initial_learned', 'decay_only_median']},
                                'learned_abs_A': models[m]['A_log'].get('learned_abs_A'),
                                'dt_bias_softplus': [models[m]['dt_bias']['softplus_initial_median'],
                                                     models[m]['dt_bias']['softplus_learned_median']],
                                'dt_bias_spearman': models[m]['dt_bias']['spearman_initial_learned'],
                                'D_learned': models[m]['D']['learned'], 'no_wd': models[m].get('no_wd')}
                            for m in MODELS}
    if cond is not None:
        out['conductances'] = {m: {k: cond['models'][m][k] for k in
                                   ['abs_A', 'delta', 'median_abs_A_delta', 'coast_fraction_g_below_0.01',
                                    'median_g', 'fraction_g_above_0.99']} for m in MODELS}
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    c = out['control']
    print('slow - fast, E slope 10-100:', json.dumps(c['contrasts']['primary_slow_minus_fast_E_slope_10_100_mean_lo_hi']),
          '->', c['contrasts']['primary_verdict'])
    print('matched reference:', json.dumps(out['matched_reference']['contrasts']))
    if 'recipe_x_init_primary_E_slope_10_100' in out:
        print('recipe x init (slow - fast, E slope 10-100):', json.dumps(out['recipe_x_init_primary_E_slope_10_100']))
    for split in SPLITS:
        r = c['regimes'][split]
        print(split, 'raw E/X: 1-3 max %.4f | 4-500 max %.4f | 501-1023 max %.4f | min %.4f' % (
            r['max_raw_E_over_X_tau_1_3'], r['max_raw_E_over_X_tau_4_500'], r['max_raw_E_over_X_tau_501_1023'],
            r['min_raw_E_over_X_tau_1_1023']))
    k2 = c['claim2']
    print('block-formula agreement: raw E/X 1-500 %s | max |X-E| slope %s | beyond 500 %.4f | worst normalized miss %s' % (
        json.dumps(k2['raw_E_over_X_tau_1_500']), json.dumps(k2['max_abs_slope_difference_X_minus_E']),
        k2['max_raw_abs_log10_factor_beyond_500'], json.dumps(k2['worst_normalized_miss_vs_E_tau_1_500'])))
    print('val ppl (control):', c['training']['val_ppl_final_range'],
          '| matched:', out['matched_reference']['training']['val_ppl_final_range'])


if __name__ == '__main__':
    main()
