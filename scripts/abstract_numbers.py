"""
Every number in the paper's abstract and the introduction's contributions, recomputed
from saved results.

  - Language models (matched cohort, nine final checkpoints, validation and test): raw E/X over
    tau = 1-500 from the per-sequence curves; the largest |X - E| slope difference over
    tau = 10-100 and 100-1000 from matched_summary.json; the largest raw E/X beyond tau = 500;
    and the worst normalized miss against E over tau = 1-500 of each closed form in turn
    (S: Eq. 15 scan probe; R: S with the readout; P: block form without cross-channel terms;
    X: block formula).
  - Retrieval tasks: relevant-pair full/frozen energy ratio and largest part, margin probe
    (task_sensitivity_v2_pooled.json) and squared-logit probe (task_probe_control.json).
  - Task accuracy: overall accuracy per seed and condition over the 18 task runs (seeds 42-44).
  - Mamba-370M: per-layer fraction of conductances below 0.01, and its Spearman correlation
    with the least-squares scan-probe slope over tau = 10-100.

Usage:
    python3 scripts/abstract_numbers.py
Output: results/camera_ready/abstract_numbers.json.
"""

import json

import numpy as np
from scipy.stats import spearmanr

CONDS, SEEDS, SPLITS = ['slow', 'default', 'fast'], [1, 2, 3], ['validation', 'test']


def language_models():
    raw_lo, raw_hi, worst = np.inf, 0.0, {k: 0.0 for k in 'SRPX'}
    for split in SPLITS:
        for c in CONDS:
            for s in SEEDS:
                z = np.load(f'results/camera_ready/matched/single_layer_{c}_seed{s}_{split}.npz')
                e = z['E'].mean(0)
                r = e[1:501] / z['X'].mean(0)[1:501]
                raw_lo, raw_hi = min(raw_lo, r.min()), max(raw_hi, r.max())
                for k in worst:
                    q = z[k].mean(0)
                    n = (e / e[1]) / (q / q[1])
                    worst[k] = max(worst[k], float(10 ** np.abs(np.log10(n[1:501])).max()))
    summ = json.load(open('results/camera_ready/matched_summary.json'))
    dsl = {w: 0.0 for w in ['lsq_10_100', 'lsq_100_1000']}
    beyond = 0.0
    for split in SPLITS:
        for v in summ['per_checkpoint'][split].values():
            sl = v['sequence_bootstrap_conditional_on_checkpoint']['slopes']
            for w in dsl:
                dsl[w] = max(dsl[w], abs(sl['X'][w]['point'] - sl['E'][w]['point']))
        for k, v in summ['block_agreement'][split].items():
            if k != 'worst_over_checkpoints':
                beyond = max(beyond, 10 ** v['raw_max_abs_log10_501_1023'])
    return {'raw_E_over_X_tau_1_500': [float(raw_lo), float(raw_hi)],
            'max_abs_slope_difference_X_minus_E': dsl,
            'max_raw_E_over_X_beyond_500': float(beyond),
            'worst_normalized_miss_vs_E_tau_1_500': worst}


def retrieval():
    out = {}
    v2 = json.load(open('results/task_sensitivity_v2_pooled.json'))['runs']
    full = [v['all']['relevant']['over_frozen']['full'] if 'full' in v['all']['relevant']['over_frozen']
            else None for v in v2.values()]
    largest = [v['all']['relevant']['largest_part'] for v in v2.values()]
    pc = json.load(open('results/task_probe_control.json'))['runs']
    for probe in ['margin', 'sq_logit']:
        f = [r['probes'][probe]['classes']['all']['relevant']['over_frozen']['full'] for r in pc.values()]
        lg = [r['probes'][probe]['classes']['all']['relevant']['largest_part'] for r in pc.values()]
        out[probe] = {'runs': len(f), 'full_over_frozen': [float(min(f)), float(max(f))],
                      'dt_write_largest': f'{lg.count("dt_write")}/{len(lg)}'}
    out['margin_v2_records_dt_write_largest'] = f'{largest.count("dt_write")}/{len(largest)}'
    return out


def accuracy():
    out = {}
    for task in ['mqar', 'selcopy']:
        per = {c: [json.load(open(f'results/{task}/rerun/{task}_seed{s}.json'))['conditions'][c]['overall_acc']
                   for s in [42, 43, 44]] for c in CONDS}
        out[task] = {c: {'per_seed': v, 'mean': float(np.mean(v))} for c, v in per.items()}
    return out


def pretrained():
    p = json.load(open('results/pretrained/scan_probe_large_mamba-370m.json'))['layers']
    frac = np.array([v['g_stats']['frac_below_0.01'] for v in p.values()])
    slope = np.array([v['slope_ls']['10-100']['value'] for v in p.values()])
    return {'layers': len(p), 'frac_g_below_0.01': [float(frac.min()), float(frac.max())],
            'spearman_frac_vs_slope_10_100': float(spearmanr(frac, slope)[0])}


def main():
    out = {'language_models': language_models(),
           'retrieval_relevant_pair': retrieval(), 'task_accuracy': accuracy(),
           'mamba_370m': pretrained()}
    with open('results/camera_ready/abstract_numbers.json', 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
