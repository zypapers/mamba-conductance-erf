"""
Pooled summaries of the margin decomposition, from the per-query
records of scripts/task_sensitivity_v2.py. No GPU.

For each of the 18 task checkpoints and each decomposition class (relevant, other_pairs,
rest), it pools the energies over a group of queries (all, correct, incorrect) and divides
by the pooled frozen energy. It also reports which individual part has the largest energy.
The parts are not orthogonal, so a ratio to frozen energy is not an additive share.

Also reported, at the relevant pair:
  - per-query full/frozen: median and interquartile range;
  - the fraction of queries in which each part has the largest energy;
  - pooled full/frozen within source-distance buckets.

Input: results/task_sensitivity_v2.json and results/task_sensitivity_v2_records.npz.
Output: results/task_sensitivity_v2_pooled.json.

Usage:
    python3 scripts/task_sensitivity_v2_pooled.py
"""

import json

import numpy as np

BUCKETS = [(1, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096)]
PARTS = ['frozen', 'dt_ret', 'dt_write', 'B', 'C']


def pooled(dec, ix):
    """dec: [n_queries, n_keys] for one class -> ratios to pooled frozen energy, largest part."""
    v = dec.sum(0)
    v = v / v[ix['E_frozen']]
    E = {p: float(v[ix[f'E_{p}']]) for p in PARTS}
    return {'n': int(dec.shape[0]), 'over_frozen': {k: float(v[i]) for k, i in ix.items()},
            'largest_part': max(E, key=E.get)}


def main():
    meta = json.load(open('results/task_sensitivity_v2.json'))
    d = np.load('results/task_sensitivity_v2_records.npz')
    ix = {k: i for i, k in enumerate(meta['dkeys'])}
    out = {'note': 'see module docstring', 'runs': {}}
    for run in meta['runs']:
        dec, cor, dist = d[f'{run}__dec'], d[f'{run}__correct'], d[f'{run}__distance']
        groups = {'all': np.ones(len(cor), bool), 'correct': cor == 1, 'incorrect': cor == 0}
        r = {g: {c: pooled(dec[sel][:, ci], ix) for ci, c in enumerate(meta['dclasses'])}
             for g, sel in groups.items() if sel.any()}
        rel = dec[:, 0]
        q = rel[:, ix['full']] / rel[:, ix['E_frozen']]
        lead = np.argmax(rel[:, [ix[f'E_{p}'] for p in PARTS]], 1)
        r['relevant_per_query'] = {
            'full_over_frozen_median': float(np.median(q)),
            'full_over_frozen_iqr': [float(x) for x in np.percentile(q, [25, 75])],
            'largest_part_fraction': {p: float(np.mean(lead == i)) for i, p in enumerate(PARTS)}}
        r['relevant_by_distance'] = {
            f'{lo}-{hi}': {'n': int(((dist >= lo) & (dist < hi)).sum()),
                           'full_over_frozen': (float(rel[(dist >= lo) & (dist < hi)][:, ix['full']].sum()
                                                      / rel[(dist >= lo) & (dist < hi)][:, ix['E_frozen']].sum())
                                                if ((dist >= lo) & (dist < hi)).any() else None)}
            for lo, hi in BUCKETS}
        out['runs'][run] = r
        a = r['all']['relevant']
        print('%-22s full/frozen %6.2f  dt_write %6.2f  largest %s' % (
            run, a['over_frozen']['full'], a['over_frozen']['E_dt_write'], a['largest_part']))
    with open('results/task_sensitivity_v2_pooled.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote results/task_sensitivity_v2_pooled.json')


if __name__ == '__main__':
    main()
