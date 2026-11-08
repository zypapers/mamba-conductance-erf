"""
Numbers and curves for the paper's Section 4.3 (block formula against the full model), from the
matched cohort's per-sequence curves (results/camera_ready/matched/*.npz).

For each of the nine final checkpoints and each split, with curves pooled over sequences and targets:
  - raw E/X (full-model ERF over the block formula) at every lag 1-1023;
  - normalized E/S (full model over the Eq. 15 scan probe, both normalized at tau = 1);
  - the lag table of the three-regime characterization (Table 1; tau = 1, 10, 50, 100, 200, 500,
    1000): minimum, median and maximum over the nine checkpoints of raw E/X, normalized E/X and
    normalized E/S;
  - the largest raw E/X within the convolution window (tau = 1-3), over tau = 4-500 and over
    tau = 501-1023, and the smallest raw E/X over tau = 1-1023.
The validation curves of raw E/X are saved for the figure (Figure 1, panel d).

Also collected for the Section 4 text (key "text"): final validation perplexity of the nine models;
the language-model decomposition (largest Delta-write energy and total/frozen per band, per
checkpoint); the identity checks of Mamba-370M and the two-layer model; the 370M depth trend
(Spearman of the fraction of g below 0.01 with layer index); the task-model decomposition
(relevant pair: full/frozen, largest part, Delta-write energy range, runs where Delta-write energy
alone exceeds the full energy); and the initialization contrasts (paired slope differences of E,
and the scan-probe fast/default ratio at tau = 100).

Usage:
    python3 scripts/section4_numbers.py
Output: results/camera_ready/section4_numbers.json.
"""

import json

import numpy as np
from scipy.stats import spearmanr

CONDS, SEEDS, SPLITS = ['slow', 'default', 'fast'], [1, 2, 3], ['validation', 'test']
LAGS = [1, 10, 50, 100, 200, 500, 1000]


def stats(vals):
    v = np.asarray(vals, dtype=float)
    return {'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}


def text_numbers():
    t = {}
    ppl = [json.load(open(f'results/matched/training_{c}_seed{s}.json'))['val_ppl'][-1]
           for c in CONDS for s in SEEDS]
    t['final_val_ppl'] = [float(min(ppl)), float(max(ppl))]
    summ = json.load(open('results/camera_ready/matched_summary.json'))
    dec = {}
    for split in SPLITS:
        for k, v in summ['per_checkpoint'][split].items():
            d = v['decomposition']
            dec.setdefault(k, {})[split] = {
                'max_E_dt_write': float(max(b['E_dt_write'] for b in d.values())),
                'total_over_frozen_range': [float(min(b['total_over_frozen'] for b in d.values())),
                                            float(max(b['total_over_frozen'] for b in d.values()))],
                'band_400_600': {'E_dt_write': d['400-600']['E_dt_write'],
                                 'total_over_frozen': d['400-600']['total_over_frozen']}}
    t['lm_decomposition'] = dec
    p370 = json.load(open('results/pretrained/scan_probe_large_mamba-370m.json'))['layers']
    raw = [c['raw_max_rel_dev'] for v in p370.values() for c in v['identity_check'].values()]
    nrm = [c['normalized_ratio_max_dev_tau_1_1023'] for v in p370.values() for c in v['identity_check'].values()]
    frac = [v['g_stats']['frac_below_0.01'] for v in p370.values()]
    depth = [int(k[1:]) for k in p370]
    t['mamba370m'] = {'identity_raw_max': float(max(raw)), 'identity_normalized_max_tau_1_1023': float(max(nrm)),
                      'spearman_frac_below_0.01_vs_layer_index': float(spearmanr(frac, depth)[0])}
    tl = json.load(open('results/two_layer/scan_probe_large.json'))['layers']
    t['two_layer_identity'] = {
        'raw_max': float(max(c['raw_max_rel_dev'] for v in tl.values() for c in v['identity_check'].values())),
        'normalized_max_tau_1_1023': float(max(c['normalized_ratio_max_dev_tau_1_1023']
                                               for v in tl.values() for c in v['identity_check'].values()))}
    v2 = json.load(open('results/task_sensitivity_v2_pooled.json'))['runs']
    rel = {k: v['all']['relevant']['over_frozen'] for k, v in v2.items()}
    t['tasks_relevant_pair_margin'] = {
        'full_over_frozen': [float(min(r['full'] for r in rel.values())), float(max(r['full'] for r in rel.values()))]
        if all('full' in r for r in rel.values()) else None,
        'E_dt_write_over_frozen': [float(min(r['E_dt_write'] for r in rel.values())),
                                   float(max(r['E_dt_write'] for r in rel.values()))],
        'dt_write_largest': f"{sum(v['all']['relevant']['largest_part'] == 'dt_write' for v in v2.values())}/{len(v2)}",
        'runs_dt_write_energy_exceeds_full': {k: float(r['E_dt_write'] / r['full']) for k, r in rel.items()
                                              if 'full' in r and r['E_dt_write'] > r['full']}}
    pair = summ['paired']
    t['init_contrasts_validation'] = {
        'E_slope_difference_10_100': {k: pair['validation'][k]['E']['slope_difference_mean_lo_hi']['lsq_10_100']
                                      for k in pair['validation']},
        'S_fast_vs_default_ratio_tau_100': pair['validation']['fast_vs_default']['S']['ratio_at_lags_mean_lo_hi']['100'],
        'S_fast_vs_default_ratio_tau_100_test': pair['test']['fast_vs_default']['S']['ratio_at_lags_mean_lo_hi']['100']}
    return t


def main():
    out = {'source': 'results/camera_ready/matched/single_layer_<cond>_seed<S>_<split>.npz',
           'lags': LAGS, 'splits': {}}
    for split in SPLITS:
        raw_ex, norm_ex, norm_es, curves, per_seq = {}, {}, {}, {}, {}
        for c in CONDS:
            for s in SEEDS:
                z = np.load(f'results/camera_ready/matched/single_layer_{c}_seed{s}_{split}.npz')
                e, x, sp = z['E'].mean(0), z['X'].mean(0), z['S'].mean(0)
                key = f'{c}_seed{s}'
                raw_ex[key] = e / x
                norm_ex[key] = (e / e[1]) / (x / x[1])
                norm_es[key] = (e / e[1]) / (sp / sp[1])
                curves[key] = (e / x)[1:].tolist()
                rs = z['E'][:, 1:501] / z['X'][:, 1:501]          # each sequence's own ratio, tau = 1-500
                per_seq[key] = {'n_sequences': int(rs.shape[0]), 'n_above_2.3': int((rs.max(1) > 2.3).sum()),
                                'max': float(rs.max())}
        table = {}
        for t in LAGS:
            table[str(t)] = {'raw_E_over_X': stats([r[t] for r in raw_ex.values()]),
                             'normalized_E_over_X': stats([r[t] for r in norm_ex.values()]),
                             'normalized_E_over_S': stats([r[t] for r in norm_es.values()])}
        regimes = {
            'max_raw_E_over_X_tau_1_3': float(max(r[1:4].max() for r in raw_ex.values())),
            'max_raw_E_over_X_tau_4_500': float(max(r[4:501].max() for r in raw_ex.values())),
            'max_raw_E_over_X_tau_501_1023': float(max(r[501:].max() for r in raw_ex.values())),
            'min_raw_E_over_X_tau_1_1023': float(min(r[1:].min() for r in raw_ex.values())),
            'max_normalized_E_over_S_tau_1_500': float(max(np.abs(np.log10(r[1:501])).max()
                                                           for r in norm_es.values())),
        }
        regimes['max_normalized_E_over_S_tau_1_500'] = float(10 ** regimes['max_normalized_E_over_S_tau_1_500'])
        out['splits'][split] = {'lag_table': table, 'regimes': regimes,
                                'per_model_max_raw_E_over_X_tau_1_500': {k: float(r[1:501].max()) for k, r in raw_ex.items()},
                                'per_sequence_raw_E_over_X_tau_1_500': per_seq}
        if split == 'validation':
            out['raw_E_over_X_curves_validation_lag1_to_1023'] = curves
    out['text'] = text_numbers()
    with open('results/camera_ready/section4_numbers.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('text numbers:', json.dumps(out['text'], indent=1)[:3500])
    for split in SPLITS:
        print(split, json.dumps(out['splits'][split]['regimes']))
        for t in LAGS:
            r = out['splits'][split]['lag_table'][str(t)]
            print(f'  tau {t:4d}: raw E/X {r["raw_E_over_X"]["min"]:.2f}-{r["raw_E_over_X"]["max"]:.2f} '
                  f'(median {r["raw_E_over_X"]["median"]:.2f}) | norm E/S median {r["normalized_E_over_S"]["median"]:.3g} '
                  f'[{r["normalized_E_over_S"]["min"]:.3g}, {r["normalized_E_over_S"]["max"]:.3g}]')


if __name__ == '__main__':
    main()
