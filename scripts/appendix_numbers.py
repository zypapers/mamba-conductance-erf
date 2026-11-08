"""
Numbers for the paper's appendices C to J and L.4, recomputed from saved results.

- App. C and App. D's constant-input check: ranges over the nine models of
  results/camera_ready/griffiths_constant_input.json (scripts/griffiths_constant_input.py).
- App. D: the identity checks per model (results/camera_ready/matched/single_layer_checks.json);
  positivity and dynamic range of the full-model ERF E over tau = 1-1023 (per-sequence curves in
  results/camera_ready/matched/*.npz); the detached-selectivity ERF over E, both normalized at
  tau = 1 (matched_summary.json appD_Dall_over_E_norm).
- App. E: per model, the final validation perplexity (results/matched/training_*.json), the learned
  |A| percentiles, coast fraction, median g and fraction above 0.99
  (results/camera_ready/conductance_stats_matched.json), softplus of the dt bias
  (results/camera_ready/weight_decay_drift.json), and the least-squares slopes over tau = 10-100
  of E, X and S with sequence-bootstrap intervals; across seeds, the slope means with t-intervals
  (matched_summary.json).
- Apps. F and G: task accuracies of the 18 task runs by seed and by distance bucket, and the
  single selective-copy run at length 4096 (see appendix_fg).
- App. H: the five-term decomposition by band (each part's energy over the frozen part, the net of
  all pairwise terms, and total over frozen) and the one-at-a-time detachments over Dall (matched_summary.json); the task
  decomposition at the relevant pair under the margin and squared-logit probes
  (results/task_sensitivity_v2_pooled.json, results/task_probe_control.json).
- Apps. I and J: the scan-probe analyses of Mamba-370M and of the two-layer model (see appendix_ij).
- L.4: the sign ablation, Su/S and Xu/X by band, raw and normalized at tau = 1
  (matched_summary.json).

Validation split unless a key says test. Ranges are [min, max] over the nine models.

Usage:
    python3 scripts/appendix_numbers.py
Output: results/camera_ready/appendix_numbers.json
"""

import json

import numpy as np

CONDS = ['slow', 'default', 'fast']
SEEDS = [1, 2, 3]
MODELS = [f'{c}_seed{s}' for s in SEEDS for c in CONDS]
PARTS = ['E_dt_ret', 'E_dt_write', 'E_B', 'E_C']


def span(values):
    v = [float(x) for x in values]
    return [min(v), max(v)]


def appendix_d(summ):
    checks = json.load(open('results/camera_ready/matched/single_layer_checks.json'))['checks']
    keys = {'S_vs_scan_frozen_erf': 'S_vs_autograd_scan_frozen_erf_max_rel_dev',
            'X_vs_detach_all': 'X_vs_autograd_detach_all_max_rel_dev',
            'decomposition_sum_vs_full': 'decomposition_sum_vs_autograd_full_max_rel_dev',
            'E_reference_vs_training_path': 'E_reference_scan_vs_training_path_max_rel_dev'}
    identity = {m: {k: checks[f'single_layer_{m}'][v] for k, v in keys.items()} for m in MODELS}
    flow = {}
    for m in MODELS:
        e = np.load(f'results/camera_ready/matched/single_layer_{m}_validation.npz')['E'][:, 1:]
        mean = e.mean(0)
        flow[m] = {'min_per_sequence_value': float(e.min()),
                   'decades_tau1_to_1023': float(np.log10(mean[0] / mean[-1])),
                   'decades_max_over_min': float(np.log10(mean.max() / mean.min()))}
    per = summ['per_checkpoint']
    lags = list(per['validation'][MODELS[0]]['sequence_bootstrap_conditional_on_checkpoint']
                ['appD_Dall_over_E_norm']['at'])
    dall = {split: {t: span(per[split][m]['sequence_bootstrap_conditional_on_checkpoint']
                            ['appD_Dall_over_E_norm']['at'][t]['point'] for m in MODELS)
                    for t in lags} for split in ['validation', 'test']}
    return {'identity_per_model': identity,
            'identity_max': {k: max(identity[m][k] for m in MODELS) for k in keys},
            'gradient_flow': flow,
            'gradient_flow_ranges': {k: span(flow[m][k] for m in MODELS) for k in flow[MODELS[0]]},
            'Dall_over_E_normalized_at': dall}


def appendix_e(summ):
    cond = json.load(open('results/camera_ready/conductance_stats_matched.json'))['models']
    drift = json.load(open('results/camera_ready/weight_decay_drift.json'))
    per = summ['per_checkpoint']['validation']
    rows = {}
    for m in MODELS:
        c, s = m.split('_seed')
        tr = json.load(open(f'results/matched/training_{c}_seed{s}.json'))
        sl = per[m]['sequence_bootstrap_conditional_on_checkpoint']['slopes']
        st = cond[m]
        rows[m] = {'val_ppl_final': tr['val_ppl'][-1], 'best_epoch': tr['best_epoch'],
                   'abs_A_p5_p50_p95': [st['abs_A'][q] for q in ['p5', 'p50', 'p95']],
                   'softplus_dt_bias_median': drift['matched'][m]['dt_bias']['softplus_learned_median'],
                   'coast_fraction': st['coast_fraction_g_below_0.01'],
                   'median_g': st['median_g'],
                   'fraction_g_above_0.99': st['fraction_g_above_0.99'],
                   'slope_lsq_10_100': {k: sl[k]['lsq_10_100'] for k in ['E', 'X', 'S']}}
    across = {c: {k: summ['across_seeds']['validation'][c][k]['slopes_mean_t95']['lsq_10_100']
                  for k in ['E', 'X', 'S']} for c in CONDS}
    return {'per_model': rows, 'slopes_across_seeds_mean_t95_lsq_10_100': across,
            'schedule': drift['schedule'], 'conductance_sequences': 'first 8 validation sequences'}


def appendix_h(summ):
    per = summ['per_checkpoint']['validation']
    bands = list(per[MODELS[0]]['decomposition'])
    dec = {b: {**{p: span(per[m]['decomposition'][b][p] for m in MODELS) for p in PARTS},
               'total_over_frozen': span(per[m]['decomposition'][b]['total_over_frozen'] for m in MODELS)}
           for b in bands}
    largest = {b: {m: max(PARTS, key=lambda p: per[m]['decomposition'][b][p]) for m in MODELS}
               for b in bands}
    pairwise = {b: span(per[m]['decomposition'][b]['total_over_frozen'] - 1
                        - sum(per[m]['decomposition'][b][p] for p in PARTS) for m in MODELS) for b in bands}
    mid = [b for b in bands if b not in ('1-1', '800-1023')]
    others = [m for m in MODELS if m != 'default_seed3']
    eight = span(per[m]['decomposition'][b]['total_over_frozen'] for m in others for b in mid)
    ds3 = {b: {p: per['default_seed3']['decomposition'][b][p] for p in PARTS + ['total_over_frozen']}
           for b in bands}
    boot = {m: per[m]['sequence_bootstrap_conditional_on_checkpoint'] for m in MODELS}
    lags = list(boot[MODELS[0]]['appH_raw_ratio_to_Dall']['E_sub'])
    detach = {k: {t: span(boot[m]['appH_raw_ratio_to_Dall'][k][t]['point'] for m in MODELS) for t in lags}
              for k in ['E_sub', 'Ddt', 'DB', 'DC']}
    contrib = {k: span(boot[m]['appH_contribution_tau100'][k]['point'] for m in MODELS)
               for k in ['Ddt', 'DB', 'DC']}
    v2 = json.load(open('results/task_sensitivity_v2_pooled.json'))['runs']
    pc = json.load(open('results/task_probe_control.json'))['runs']
    tasks = {}
    for r in v2:
        rel = v2[r]['all']['relevant']
        sq = pc[r]['probes']['sq_logit']['classes']['all']['relevant']
        tasks[r] = {'margin': {'full_over_frozen': rel['over_frozen']['full'],
                               **{p: rel['over_frozen'][p] for p in PARTS},
                               'pairwise_net': rel['over_frozen']['full'] - 1
                               - sum(rel['over_frozen'][p] for p in PARTS),
                               'largest_part': rel['largest_part']},
                    'sq_logit': {'full_over_frozen': sq['over_frozen']['full'],
                                 'largest_part': sq['largest_part']}}
    return {'lm_decomposition_by_band': dec, 'lm_largest_part_by_band': largest,
            'lm_total_over_frozen_tau5_600_eight_models': eight, 'lm_default_seed3_by_band': ds3,
            'lm_pairwise_net_by_band': pairwise,
            'lm_one_at_a_time_over_Dall': detach, 'lm_lost_by_detaching_alone_tau100_over_Dall': contrib,
            'tasks_relevant_pair': tasks}


def appendix_fg():
    """Apps. F and G: the 18 task runs (results/{mqar,selcopy}/rerun/, seeds 42-44, paired
    training data, one fixed evaluation set) and the single selective-copy run at length 4096
    (results/selcopy/selcopy_intervention_seq4096.json; one seed, unpaired data)."""
    import statistics as st
    out = {}
    for task in ['mqar', 'selcopy']:
        runs = {s: json.load(open(f'results/{task}/rerun/{task}_seed{s}.json')) for s in [42, 43, 44]}
        dist = [p[0] for p in runs[42]['conditions']['slow']['all_pairs']]
        t = {'distance_range': [min(dist), max(dist)], 'conditions': {}}
        for c in CONDS:
            per = {s: runs[s]['conditions'][c] for s in runs}
            ov = [per[s]['overall_acc'] for s in per]
            buckets = {}
            for b, v in per[42]['per_bucket'].items():
                if v['total'] == 0:
                    continue
                accs = [per[s]['per_bucket'][b]['acc'] for s in per]
                buckets[b] = {'n_per_run': v['total'], 'per_seed': accs, 'mean': st.mean(accs), 'sd': st.stdev(accs)}
            t['conditions'][c] = {'per_seed': dict(zip(['42', '43', '44'], ov)), 'mean': st.mean(ov),
                                  'sd': st.stdev(ov), 'by_distance': buckets,
                                  'correct_per_seed': [sum(v['correct'] for v in per[s]['per_bucket'].values())
                                                       for s in per]}
        from scipy import stats as sps   # paired within seed, 95% t-interval across the three seeds
        t['paired_within_seed_t95'] = {}
        for other in ['default', 'fast']:
            dif = np.array([t['conditions']['slow']['per_seed'][s] - t['conditions'][other]['per_seed'][s]
                            for s in ['42', '43', '44']])
            h = sps.t.ppf(0.975, len(dif) - 1) * dif.std(ddof=1) / np.sqrt(len(dif))
            t['paired_within_seed_t95'][f'slow_minus_{other}'] = {
                'per_seed': dif.tolist(), 'mean_lo_hi': [float(dif.mean()), float(dif.mean() - h), float(dif.mean() + h)]}
        out[task] = t
    d = json.load(open('results/selcopy/selcopy_intervention_seq4096.json'))
    dist = [p[0] for p in d['conditions']['slow']['all_pairs']]
    out['selcopy_4096_single_run'] = {
        'distance_range': [min(dist), max(dist)],
        'conditions': {c: {'overall': d['conditions'][c]['overall_acc'],
                           'by_distance': {b: {'acc': v['acc'], 'n': v['total']}
                                           for b, v in d['conditions'][c]['per_bucket'].items() if v['total']}}
                       for c in CONDS}}
    return out


def appendix_ij():
    """Apps. I and J: scan-probe analyses of Mamba-370M (results/pretrained/scan_probe_large_mamba-370m.json)
    and of the two-layer model (results/two_layer/scan_probe_large.json), both with the trained recurrence.
    Identity residuals are the worse of the two checked targets (T = 1023, 2047)."""
    from scipy.stats import spearmanr
    out = {}
    for key, path in [('mamba370m', 'results/pretrained/scan_probe_large_mamba-370m.json'),
                      ('two_layer', 'results/two_layer/scan_probe_large.json')]:
        d = json.load(open(path))
        layers = d['layers']
        per = {}
        for name, v in layers.items():
            ic = v['identity_check'].values()
            per[name] = {'raw_identity': max(x['raw_max_rel_dev'] for x in ic),
                         'normalized_identity': max(x['normalized_ratio_max_dev_tau_1_1023'] for x in ic),
                         'slope_10_100': v['slope_ls']['10-100'], 'slope_100_1000': v['slope_ls']['100-1000'],
                         'coast_fraction': v['g_stats']['frac_below_0.01'],
                         'frac_above_0.99': v['g_stats']['frac_above_0.99'],
                         'sign_normalized_Su_over_S': v['sign_ablation_normalized_Su_over_S'],
                         'n_sequences': v['n_sequences']}
        raw = [p['raw_identity'] for p in per.values()]
        nrm = [p['normalized_identity'] for p in per.values()]
        out[key] = {'targets': d['targets'], 'seq_len': d['seq_len'], 'tokenizer': d['tokenizer'],
                    'identity_raw': {'max': max(raw), 'median': float(np.median(raw))},
                    'identity_normalized': {'max': max(nrm), 'median': float(np.median(nrm))},
                    'per_layer': per}
    p = out['mamba370m']['per_layer']
    names = list(p)
    idx = [int(n.lstrip('L').replace('layer', '')) if n.lstrip('L').replace('layer', '').isdigit() else i
           for i, n in enumerate(names)]
    coast = np.array([p[n]['coast_fraction'] for n in names])
    slope = np.array([p[n]['slope_10_100']['value'] for n in names])
    groups = {}
    for g, mask in [('coast_above_70pct', coast > 0.70), ('coast_below_5pct', coast < 0.05)]:
        sel = [n for n, m in zip(names, mask) if m]
        sl = [p[n]['slope_10_100']['value'] for n in sel]
        groups[g] = {'layers': sel, 'count': len(sel), 'slope_range': span(sl), 'slope_mean': float(np.mean(sl)),
                     'count_shallower_than_minus2': int(sum(x > -2 for x in sl))}
    sign = {}
    for t in ['10', '100', '500', '1000']:
        vals = [p[n]['sign_normalized_Su_over_S'][t] for n in names]
        sign[t] = {'range': span(vals), 'count_below_1': int(sum(x < 1 for x in vals)), 'of': len(vals)}
    out['mamba370m'].update({
        'n_chunks': json.load(open('results/pretrained/scan_probe_large_mamba-370m.json'))['n_chunks'],
        'coast_fraction_range': span(coast),
        'coast_min_layer': names[int(coast.argmin())], 'coast_max_layer': names[int(coast.argmax())],
        'spearman_coast_vs_slope_10_100': float(spearmanr(coast, slope)[0]),
        'spearman_coast_vs_layer_index': float(spearmanr(coast, idx)[0]),
        'groups': groups, 'sign_ablation_normalized_Su_over_S': sign})
    return out


def appendix_c_constant():
    """App. C and App. D's constant-input check: ranges over the nine models of
    results/camera_ready/griffiths_constant_input.json (scripts/griffiths_constant_input.py)."""
    d = json.load(open('results/camera_ready/griffiths_constant_input.json'))['models']
    b = [v['block'] for v in d.values()]
    sh = [v['shuffled'] for v in d.values()]
    po = [v['block_order_permuted'] for v in d.values()]
    const = [v[c] for v in d.values() for c in ['const500', 'const30000']]
    keys = ['max_ratio_1_1021', 'argmax_tau_1_1021', 'min_ratio', 'mean_ratio_50_200', 'mean_ratio_300_1000',
            'delta_window_mean', 'delta_window_sd']
    big = [k for k, v in d.items() if v['block']['max_ratio_1_1021'] > 5]
    return {'block': {k: span(x[k] for x in b) for k in
                      keys + ['ratio_larger_over_smaller', 'mean_delta_token500', 'mean_delta_token30000']},
            'shuffled': {k: span(x[k] for x in sh) for k in keys},
            'block_order_permuted_worst_of_seeds': {k: span(x[k] for x in po) for k in
                                                     ['max_ratio_1_1021', 'mean_ratio_50_200',
                                                      'mean_ratio_300_1000']},
            'block_peak_above_5': {'models': big,
                                   'block_peak': span(d[k]['block']['max_ratio_1_1021'] for k in big),
                                   'block_mean_50_200': span(d[k]['block']['mean_ratio_50_200'] for k in big),
                                   'shuffled_peak': span(d[k]['shuffled']['max_ratio_1_1021'] for k in big),
                                   'order_permuted_peak': span(d[k]['block_order_permuted']['max_ratio_1_1021']
                                                               for k in big)},
            'block_token30000_larger_delta_in_all': all(x['mean_delta_token30000'] > x['mean_delta_token500'] for x in b),
            'constant': {'max_abs_log10_ratio_1_1021': max(x['max_abs_log10_ratio_1_1021'] for x in const),
                         'max_ratio_1022_1023': max(x['max_ratio'] for x in const)},
            # spans over all (model, permutation seed) pairs, not each model's largest
            'permutation_arms_all_pairs': {
                arm: {**{k: span(x[k] for v in d.values() for x in v[key]['per_seed'])
                         for k in ['max_ratio_1_1021', 'mean_ratio_50_200', 'mean_ratio_300_1000']},
                      'max_factor_1_1021': span(10 ** x['max_abs_log10_ratio_1_1021']
                                                for v in d.values() for x in v[key]['per_seed'])}
                for arm, key in PERM_ARMS.items()},
            'permutation_peaks_in_block_peak_above_5_models': {
                arm: span(x['max_ratio_1_1021'] for k in big for x in d[k][key]['per_seed'])
                for arm, key in PERM_ARMS.items()},
            'block_peak_by_model': {k: v['block']['max_ratio_1_1021'] for k, v in d.items()},
            'permutation_peak_by_model': {arm: {k: span(x['max_ratio_1_1021'] for x in v[key]['per_seed'])
                                                for k, v in d.items()} for arm, key in PERM_ARMS.items()},
            'blocked_peak_above_every_conductances_permuted_peak_in_all_models':
                all(v['block']['max_ratio_1_1021'] > max(x['max_ratio_1_1021']
                                                         for x in v['block_conductances_permuted']['per_seed'])
                    for v in d.values()),
            'every_writes_permuted_peak_at_least_blocked_in_all_models':
                all(min(x['max_ratio_1_1021'] for x in v['block_writes_permuted']['per_seed'])
                    >= v['block']['max_ratio_1_1021'] for v in d.values()),
            # adjacent blocks exchanged: persistent and coupled, out of line with the readout
            'block_swapped': {
                'peak_by_model': {k: v['block_swapped']['max_ratio_1_1021'] for k, v in d.items()},
                'peak_in_block_peak_above_5_models': span(d[k]['block_swapped']['max_ratio_1_1021'] for k in big),
                'mean_50_200_in_block_peak_above_5_models': span(d[k]['block_swapped']['mean_ratio_50_200']
                                                                 for k in big),
                'peak_above_every_jointly_permuted_peak_in_block_peak_above_5_models':
                    all(d[k]['block_swapped']['max_ratio_1_1021'] > max(x['max_ratio_1_1021']
                                                                        for x in d[k]['block_order_permuted']['per_seed'])
                        for k in big),
                'peak_below_blocked_peak_in_block_peak_above_5_models':
                    all(d[k]['block_swapped']['max_ratio_1_1021'] < d[k]['block']['max_ratio_1_1021'] for k in big),
                'n_models_with_peak_below_blocked_peak':
                    sum(v['block_swapped']['max_ratio_1_1021'] < v['block']['max_ratio_1_1021'] for v in d.values())}}


PERM_ARMS = {'joint': 'block_order_permuted', 'conductances': 'block_conductances_permuted',
             'writes': 'block_writes_permuted'}


def appendix_l4(summ):
    per = summ['per_checkpoint']['validation']
    out = {}
    for key, kind in [('raw_ratios', 'raw'), ('ratios', 'normalized')]:
        for r in ['Su/S', 'Xu/X']:
            bands = [per[m]['sequence_bootstrap_conditional_on_checkpoint'][key][r]['bands']['point']
                     for m in MODELS]
            out[f'{kind}_{r}_bands'] = [span(b[i] for b in bands) for i in range(len(bands[0]))]
    out['bands'] = summ['bands']
    return out


def main():
    summ = json.load(open('results/camera_ready/matched_summary.json'))
    out = {'sources': 'see module docstring', 'models': MODELS,
           'C_and_D_constant_input': appendix_c_constant(),
           'D': appendix_d(summ), 'E': appendix_e(summ), 'FG': appendix_fg(), 'H': appendix_h(summ),
           'IJ': appendix_ij(), 'L4': appendix_l4(summ)}
    with open('results/camera_ready/appendix_numbers.json', 'w') as f:
        json.dump(out, f, indent=1)
    d = out['D']
    print('identity max', d['identity_max'])
    print('gradient flow', d['gradient_flow_ranges'])
    print('Dall/E normalized', d['Dall_over_E_normalized_at']['validation'])
    h = out['H']
    print('LM decomposition', json.dumps(h['lm_decomposition_by_band']))
    print('detachments', json.dumps(h['lm_one_at_a_time_over_Dall']))
    print('lost by detaching alone at 100', h['lm_lost_by_detaching_alone_tau100_over_Dall'])
    print('L4', json.dumps(out['L4']))


if __name__ == '__main__':
    main()
