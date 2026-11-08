"""
Position classes and the margin
decomposition, in one pass over the same examples and all 18 task checkpoints. The probe
(the margin m) and the examples are those of scripts/task_sensitivity_diagnostic.py.

Position classes are mutually exclusive and exhaustive over the past positions p < T. The
current position T is excluded.
  relevant     the queried pair's key and value (MQAR); the output's marker and content
               (selective copy)
  other_pairs  the other 7 pairs or marks
  control      the query marker (MQAR); EOS (selective copy, once it is in the past)
  previous     earlier queries in the query block (MQAR); earlier teacher-forced outputs
               (selective copy)
  filler       all other past positions
For each query, the raw energies sum_{p in class} ||dm/du_p||^2 are saved, from autograd
through the training forward pass, together with identifiers (batch, sequence, output
index), target, prediction, correctness, distance and margin.

Margin decomposition. For lags >= 1, the margin gradient splits into the five parts of
src/curves/selectivity_decomposition.py (frozen, dt_ret, dt_write, B, C), using the
probe's readout direction dm/dlogits_T. Per query and class (relevant, other_pairs, rest),
it records each part's energy, every pairwise term 2<g_c, g_d>, and the full energy (the
sum of all of them). This measures contributions and interactions; it does not assign a
cause to an error.

Checks on the first batch of every run, enforced (above 1e-2 or nonfinite stops the run):
  - the frozen part against autograd with Delta, B, C detached;
  - the sum of the parts against the full autograd gradient.
Both compare vectors at the past positions.

Outputs:
  - results/task_sensitivity_v2.json: checks and per-run summaries;
  - results/task_sensitivity_v2_records.npz: per-query arrays; gitignored, about 5 min to
    regenerate.

Usage:
    python3 scripts/task_sensitivity_v2.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.mqar_intervention as mq
import scripts.selective_copy_intervention as sc
from scripts.camera_ready_single_layer import forward_variant
from scripts.task_sensitivity_diagnostic import CONDS, DEV, SEEDS, SEQ_LEN, examples
from src.curves.selectivity_decomposition import PARTS, _inputs, part_gradients
from src.model.single_layer_mamba import SingleLayerMamba

CLASSES = ['relevant', 'other_pairs', 'control', 'previous', 'filler']
DCLASSES = ['relevant', 'other_pairs', 'rest']
XPAIRS = [(a, b) for i, a in enumerate(PARTS) for b in PARTS[i + 1:]]
DKEYS = [f'E_{c}' for c in PARTS] + [f'X_{a}_{b}' for a, b in XPAIRS] + ['full']
SUB, TOL = 4, 1e-2


def labels(task, T, entry, queries_b):
    """Class index (into CLASSES) for every past position p < T of one query."""
    _, k, others = entry
    lab = np.full(T, 4)
    if task == 'mqar':
        q_start = max(int(SEQ_LEN * 0.78), int(SEQ_LEN * 0.75) + 4)
        if q_start < T:
            lab[q_start] = 2
        for q2, _, _ in queries_b:
            if q2 < T:
                lab[q2] = 3
    else:
        out_start = SEQ_LEN - 9
        if out_start < T:
            lab[out_start] = 2
        lab[out_start + 1:T] = 3
    for a, c in others:
        lab[a] = lab[c] = 1
    lab[k] = lab[k + 1] = 0
    return lab


def readout(task, tgt, vocab):
    valid = (torch.arange(mq.N_KEYS, mq.N_KEYS + mq.N_VALUES) if task == 'mqar'
             else torch.arange(sc.N_CONTENT)).to(DEV)
    r = torch.zeros(len(tgt), vocab, device=DEV)
    r[:, valid] = -1.0 / (len(valid) - 1)
    r[torch.arange(len(tgt)), tgt] = 1.0
    return r


def run(task, vocab, model, batches, check):
    rec = {k: [] for k in ['batch', 'seq', 'qi', 'T', 'target', 'pred', 'correct', 'distance', 'margin',
                           'E_class', 'dec']}
    checks = {}
    for bi, (ids, tg, queries) in enumerate(batches):
        ids, tg = ids.to(DEV), tg.to(DEV)
        lf, ef = forward_variant(model, ids, 'full')
        with torch.no_grad():
            inp = _inputs(model, ids)
        if check and bi == 0:
            ld, ed = forward_variant(model, ids, 'detach_all')
        for qi in range(len(queries[0])):
            T = queries[0][qi][0]
            r = readout(task, tg[:, T], vocab)
            margin = (lf[:, T] * r).sum(-1)
            g, = torch.autograd.grad(margin.sum(), ef, retain_graph=True)
            g = g[:, :T]                                               # past positions, [b, T, d]
            parts = {c: [] for c in PARTS}
            with torch.no_grad():
                for s0 in range(0, ids.shape[0], SUB):
                    sl = slice(s0, s0 + SUB)
                    pg = part_gradients(model, ids[sl], T, T + 1, {k: v[sl] for k, v in inp.items()},
                                        dl_dlogits=r[sl])
                    for c in PARTS:                                    # lag order -> position order
                        parts[c].append(pg[c][:, 1:].flip(1))
            parts = {c: torch.cat(v) for c, v in parts.items()}
            if check and bi == 0 and qi == 0:
                gd, = torch.autograd.grad((ld[:, T] * r).sum(), ed, retain_graph=True)
                gd = gd[:, :T]
                checks['frozen_vs_autograd_detached'] = float((parts['frozen'] - gd).abs().max() / gd.abs().max())
                total = sum(parts.values())
                checks['sum_of_parts_vs_autograd_full'] = float((total - g).abs().max() / g.abs().max())
                bad = {k: v for k, v in checks.items() if not np.isfinite(v) or v > TOL}
                if bad:
                    raise RuntimeError(f'{task}: identity checks failed {bad}')
            I = (g ** 2).sum(-1).cpu().numpy()                           # [b, T]
            en = {c: (parts[c] ** 2).sum(-1) for c in PARTS}
            xs = {(a, b): 2 * (parts[a] * parts[b]).sum(-1) for a, b in XPAIRS}
            fullp = (sum(parts.values()) ** 2).sum(-1)
            per_pos = torch.stack([en[c] for c in PARTS] + [xs[p] for p in XPAIRS] + [fullp], -1).cpu().numpy()
            pred = lf[:, T].argmax(-1).cpu().numpy()
            tgt = tg[:, T].cpu().numpy()
            mg = margin.detach().cpu().numpy()
            for b in range(ids.shape[0]):
                lab = labels(task, T, queries[b][qi], queries[b])
                q, k, _ = queries[b][qi]
                rec['batch'].append(bi)
                rec['seq'].append(b)
                rec['qi'].append(qi)
                rec['T'].append(T)
                rec['target'].append(int(tgt[b]))
                rec['pred'].append(int(pred[b]))
                rec['correct'].append(int(pred[b] == tgt[b]))
                rec['distance'].append(q - k - (1 if task == 'selcopy' else 0))
                rec['margin'].append(float(mg[b]))
                rec['E_class'].append([float(I[b][lab == i].sum()) for i in range(len(CLASSES))])
                dl = np.where(lab <= 1, lab, 2)
                rec['dec'].append([per_pos[b][dl == i].sum(0) for i in range(len(DCLASSES))])
    return {k: np.array(v) for k, v in rec.items()}, checks


def summarize(r):
    out = {}
    for name, sel in (('correct', r['correct'] == 1), ('incorrect', r['correct'] == 0)):
        n = int(sel.sum())
        if n == 0:
            out[name] = {'n': 0}
            continue
        E = r['E_class'][sel]
        tot = E.sum(1, keepdims=True)
        dec = r['dec'][sel].sum(0)                                         # [3 classes, 16]
        out[name] = {
            'n': n,
            'mean_share': dict(zip(CLASSES, (E / tot).mean(0).round(4).tolist())),
            'median_raw_energy': dict(zip(CLASSES, np.median(E, 0).tolist())),
            'median_total_energy': float(np.median(tot)),
            'decomposition_pooled_over_frozen': {
                c: dict(zip(DKEYS, (dec[i] / dec[i][0]).round(4).tolist())) for i, c in enumerate(DCLASSES)},
            'parts_full_vs_autograd_class_energy_max_rel_dev': float(np.max(np.abs(
                np.stack([r['dec'][sel][:, 0, -1], r['dec'][sel][:, 1, -1], r['dec'][sel][:, 2, -1]], 1) /
                np.stack([E[:, 0], E[:, 1], E[:, 2:].sum(1)], 1) - 1)))}
    return out


def main():
    smoke = '--smoke' in sys.argv
    out = {'note': 'see module docstring', 'classes': CLASSES, 'dclasses': DCLASSES, 'dkeys': DKEYS,
           'runs': {}}
    arrays = {}
    for task, mod in (('mqar', mq), ('selcopy', sc)):
        batches = examples(task)[:1] if smoke else examples(task)
        for s in SEEDS[:1] if smoke else SEEDS:
            for c in CONDS[:1] if smoke else CONDS:
                ck = torch.load(f'checkpoints/{task}/rerun_seed{s}/{task}_{c}.pt', map_location=DEV,
                                weights_only=False)
                m = SingleLayerMamba(d_model=256, d_inner=512, d_state=16, vocab_size=mod.VOCAB_SIZE).to(DEV)
                m.load_state_dict(ck['model_state_dict'])
                m.eval()
                r, checks = run(task, mod.VOCAB_SIZE, m, batches, check=True)
                key = f'{task}_{c}_seed{s}'
                out['runs'][key] = {'checks': checks, 'summary': summarize(r)}
                arrays.update({f'{key}__{k}': v for k, v in r.items()})
                sm = out['runs'][key]['summary']
                print(key, json.dumps(checks), 'incorrect n', sm['incorrect'].get('n'), flush=True)
    if smoke:
        print(json.dumps(out['runs'], indent=0)[:3000])
        return
    with open('results/task_sensitivity_v2.json', 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed('results/task_sensitivity_v2_records.npz', **arrays)
    print('wrote results/task_sensitivity_v2.json and results/task_sensitivity_v2_records.npz')


if __name__ == '__main__':
    main()
