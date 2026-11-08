"""
Probe control for the task margin decomposition.

The 18 task checkpoints, the diagnostic examples and the query positions of
scripts/task_sensitivity_v2.py stay fixed. Only the probe changes:
  margin    m = logit_T[correct] - mean of logit_T over the other valid answers (as in scripts/task_sensitivity_v2.py);
  sq_logit  ||logits_T||^2, the probe used for the language models.

For each probe and query, the five-part decomposition of
src/curves/selectivity_decomposition.py is computed at every past position. Two kinds of
energy sums are saved:
  - class sums (relevant, other_pairs, rest), as in scripts/task_sensitivity_v2.py;
  - lag-resolved sums over queries, lag = T - p for p < T, for all, correct and incorrect
    queries, with counts per lag.
The paper's claims concern ERF curves, so the lag-resolved sums are the main output.

Compare full/frozen ratios, component energies and cross terms within a probe. Absolute
energies are not comparable between probes. Near agreement of the full and frozen totals
can conceal cancellation, so the component energies and cross terms are kept.

Checks, enforced (above 1e-2 or nonfinite stops the run):
  - on the first batch of every run, for each probe: the frozen part against autograd with
    Delta, B, C detached, and the sum of the parts against the full autograd gradient;
  - for the first run of each task, the first batch's margin class sums against the
    records of scripts/task_sensitivity_v2.py (same computation).

Summaries per run, probe and group:
  - pooled class ratios to frozen energy;
  - per lag band, ratios of band sums of the per-lag mean energies to frozen. For full, also
    the geometric mean over the band of the per-lag full/frozen ratio, and that ratio divided
    by its value at lag 1 (the shape comparison after normalizing at tau = 1);
  - slopes of log mean energy on log lag over tau = 10-100 and 100-1000, for frozen,
    dt_write and full.

Outputs:
  - results/task_probe_control.json: checks and summaries;
  - results/task_probe_control_curves.npz: lag-resolved arrays and per-query class sums;
    gitignored.

Usage:
    python3 scripts/task_probe_control.py [--smoke]
Runtime: a few minutes on one GPU.
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.mqar_intervention as mq
import scripts.selective_copy_intervention as sc
from scripts.camera_ready_single_layer import forward_variant
from scripts.task_sensitivity_diagnostic import CONDS, DEV, SEEDS, SEQ_LEN, examples
from scripts.task_sensitivity_v2 import DCLASSES, DKEYS, PARTS, SUB, TOL, labels, readout
from src.curves.selectivity_decomposition import _inputs, part_gradients
from src.model.single_layer_mamba import SingleLayerMamba

PROBES = ['margin', 'sq_logit']
GROUPS = ['all', 'correct', 'incorrect']
BANDS = [(1, 4), (4, 16), (16, 64), (64, 256), (256, 1024), (1024, SEQ_LEN + 1)]
SLOPES = [(10, 100), (100, 1000)]
IX = {k: i for i, k in enumerate(DKEYS)}
V2_TOL = 1e-3


def probe_value(logits, T, r):
    return (logits[:, T] * r).sum() if r is not None else (logits[:, T] ** 2).sum()


def probe_parts(model, ids, inp, lf, ef, T, r):
    """Full autograd gradient and the five parts at past positions p < T, for one probe:
    the margin readout r = dm/dlogits_T, or ||logits_T||^2 if r is None."""
    g, = torch.autograd.grad(probe_value(lf, T, r), ef, retain_graph=True)
    parts = {c: [] for c in PARTS}
    with torch.no_grad():
        for s0 in range(0, ids.shape[0], SUB):
            sl = slice(s0, s0 + SUB)
            pg = part_gradients(model, ids[sl], T, T + 1, {k: v[sl] for k, v in inp.items()},
                                dl_dlogits=None if r is None else r[sl])
            for c in PARTS:                                            # lag order -> position order
                parts[c].append(pg[c][:, 1:].flip(1))
    return g[:, :T], {c: torch.cat(v) for c, v in parts.items()}


def per_position(parts):
    """[batch, T, len(DKEYS)]: part energies, pairwise terms and full energy per position."""
    keys = [(parts[c] ** 2).sum(-1) for c in PARTS]
    keys += [2 * (parts[a] * parts[b]).sum(-1) for i, a in enumerate(PARTS) for b in PARTS[i + 1:]]
    keys.append((sum(parts.values()) ** 2).sum(-1))
    return torch.stack(keys, -1).double().cpu().numpy()


def run(task, vocab, model, batches, v2_first=None):
    cls = {p: [] for p in PROBES}
    lag = {p: np.zeros((len(GROUPS), SEQ_LEN + 1, len(DKEYS))) for p in PROBES}
    cnt = np.zeros((len(GROUPS), SEQ_LEN + 1))
    correct, checks = [], {}
    for bi, (ids, tg, queries) in enumerate(batches):
        ids, tg = ids.to(DEV), tg.to(DEV)
        lf, ef = forward_variant(model, ids, 'full')
        with torch.no_grad():
            inp = _inputs(model, ids)
        if bi == 0:
            ld, ed = forward_variant(model, ids, 'detach_all')
        for qi in range(len(queries[0])):
            T = queries[0][qi][0]
            r = readout(task, tg[:, T], vocab)
            cor = (lf[:, T].argmax(-1) == tg[:, T]).cpu().numpy()
            labs = [np.where(lab <= 1, lab, 2) for lab in
                    (labels(task, T, queries[b][qi], queries[b]) for b in range(ids.shape[0]))]
            for p, rp in (('margin', r), ('sq_logit', None)):
                g, parts = probe_parts(model, ids, inp, lf, ef, T, rp)
                if bi == 0 and qi == 0:
                    gd, = torch.autograd.grad(probe_value(ld, T, rp), ed, retain_graph=True)
                    gd = gd[:, :T]
                    checks[p] = {
                        'frozen_vs_autograd_detached': float((parts['frozen'] - gd).abs().max() / gd.abs().max()),
                        'sum_of_parts_vs_autograd_full': float((sum(parts.values()) - g).abs().max() / g.abs().max())}
                    bad = {k: v for k, v in checks[p].items() if not np.isfinite(v) or v > TOL}
                    if bad:
                        raise RuntimeError(f'{task} {p}: identity checks failed {bad}')
                pp = per_position(parts)
                for b in range(ids.shape[0]):
                    cls[p].append([pp[b][labs[b] == i].sum(0) for i in range(len(DCLASSES))])
                    for gi in (0, 1 if cor[b] else 2):
                        lag[p][gi, 1:T + 1] += pp[b][::-1]                # lag 1 .. T
            for b in range(ids.shape[0]):
                correct.append(int(cor[b]))
                for gi in (0, 1 if cor[b] else 2):
                    cnt[gi, 1:T + 1] += 1
        if bi == 0 and v2_first is not None:
            mine = np.array(cls['margin'])
            dev = float(np.abs(mine - v2_first[:len(mine)]).max() / np.abs(v2_first[:len(mine)]).max())
            checks['margin_class_sums_vs_v2_records'] = dev
            if not np.isfinite(dev) or dev > V2_TOL:
                raise RuntimeError(f'{task}: margin class sums differ from v2 records ({dev})')
    return {p: np.array(v) for p, v in cls.items()}, lag, cnt, np.array(correct), checks


def pooled(dec):
    v = dec.sum(0)
    v = v / v[IX['E_frozen']]
    E = {c: float(v[IX[f'E_{c}']]) for c in PARTS}
    return {'n': int(dec.shape[0]), 'over_frozen': {k: float(v[i]) for k, i in IX.items()},
            'largest_part': max(E, key=E.get)}


def lag_summary(acc, n):
    tau = np.arange(1, SEQ_LEN + 1)
    ok = n[1:] > 0
    m = acc[1:] / np.maximum(n[1:], 1)[:, None]                            # per-lag means
    with np.errstate(invalid='ignore', divide='ignore'):                  # lags with no queries
        ratio = m[:, IX['full']] / m[:, IX['E_frozen']]
    out = {'full_over_frozen_tau1': float(ratio[0]), 'bands': {}, 'slopes': {}}
    for lo, hi in BANDS:
        sel = ok & (tau >= lo) & (tau < hi)
        if not sel.any():
            continue
        v = m[sel].sum(0)
        gm = float(np.exp(np.mean(np.log(ratio[sel]))))
        out['bands'][f'{lo}-{hi - 1}'] = {
            'n_lags': int(sel.sum()), 'min_queries': int(n[1:][sel].min()),
            'sum_over_frozen': {k: float(v[i] / v[IX['E_frozen']]) for k, i in IX.items()},
            'full_over_frozen_gm': gm, 'full_over_frozen_gm_normalized_tau1': gm / float(ratio[0])}
    for lo, hi in SLOPES:
        sel = ok & (tau >= lo) & (tau <= hi)
        out['slopes'][f'{lo}-{hi}'] = {k: float(np.polyfit(np.log(tau[sel]), np.log(m[sel, IX[k]]), 1)[0])
                                       for k in ['E_frozen', 'E_dt_write', 'full'] if (m[sel, IX[k]] > 0).all()}
    return out


def main():
    smoke = '--smoke' in sys.argv
    v2 = np.load('results/task_sensitivity_v2_records.npz')
    out = {'note': 'see module docstring', 'probes': PROBES, 'groups': GROUPS, 'dclasses': DCLASSES,
           'dkeys': DKEYS, 'runs': {}}
    arrays = {}
    for task, mod in (('mqar', mq), ('selcopy', sc)):
        batches = examples(task)[:1] if smoke else examples(task)
        for s in SEEDS[:1] if smoke else SEEDS:
            for c in CONDS[:1] if smoke else CONDS:
                t0 = time.time()
                key = f'{task}_{c}_seed{s}'
                ck = torch.load(f'checkpoints/{task}/rerun_seed{s}/{task}_{c}.pt', map_location=DEV,
                                weights_only=False)
                m = SingleLayerMamba(d_model=256, d_inner=512, d_state=16, vocab_size=mod.VOCAB_SIZE).to(DEV)
                m.load_state_dict(ck['model_state_dict'])
                m.eval()
                first = s == SEEDS[0] and c == CONDS[0]
                cls, lag, cnt, cor, checks = run(task, mod.VOCAB_SIZE, m, batches,
                                                 v2[f'{key}__dec'] if first else None)
                sel = {'all': np.ones(len(cor), bool), 'correct': cor == 1, 'incorrect': cor == 0}
                res = {'checks': checks, 'n': {g: int(v.sum()) for g, v in sel.items()}, 'probes': {}}
                for p in PROBES:
                    res['probes'][p] = {
                        'classes': {g: {cl: pooled(cls[p][v][:, i]) for i, cl in enumerate(DCLASSES)}
                                    for g, v in sel.items() if v.any()},
                        'lags': {g: lag_summary(lag[p][gi], cnt[gi]) for gi, g in enumerate(GROUPS)
                                 if sel[g].any()}}
                    arrays[f'{key}__{p}__lag'] = lag[p]
                    arrays[f'{key}__{p}__dec'] = cls[p]
                arrays[f'{key}__count'] = cnt
                arrays[f'{key}__correct'] = cor
                out['runs'][key] = res
                rel = {p: res['probes'][p]['classes']['all']['relevant']['over_frozen']['full'] for p in PROBES}
                print(key, json.dumps(checks), 'relevant full/frozen', json.dumps(rel),
                      '%.0fs' % (time.time() - t0), flush=True)
    if smoke:
        print(json.dumps(out['runs'], indent=0)[:3000])
        return
    with open('results/task_probe_control.json', 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed('results/task_probe_control_curves.npz', **arrays)
    print('wrote results/task_probe_control.json and results/task_probe_control_curves.npz')


if __name__ == '__main__':
    main()
