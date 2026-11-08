"""
Task sensitivity diagnostic for the 18 task runs
(3 A-initializations x seeds 42-44 x MQAR and selective copy).

It describes which inputs each trained task model's answer is sensitive to. It is not a
mediation test: changing the training seed changes many learned weights.

Probe. At each query (MQAR) or output (selective copy) position T, the margin is
    m = logit_T[correct] - mean of logit_T over the other valid answers,
using the 16 value tokens (MQAR) or the 16 content tokens (selective copy). This avoids
cross-entropy's confidence scaling. It differs from the language-model squared-logit
probe, and its ERF answers a different question.

Influence of input position p: I_p = ||dm/du_p||^2, where u are the input embeddings and
the gradient runs through the training forward pass.
  - Relevant positions: the queried pair's key and value (MQAR), or the corresponding
    marker and content (selective copy).
  - Distractors: the other pairs or marks in the same sequence.

Per query:
  - distance and correctness;
  - the margin;
  - I_rel, the sum over the two relevant positions;
  - I_dis, the mean over the other pairs of their two-position sums;
  - share, I_rel / sum_{p<T} I_p.

Every run is evaluated on identical examples: the task generator with seed 12345,
10 batches of 16 sequences, and the training configuration (length 1536, 8 pairs,
4 queries; 8 marks).

Output: results/task_sensitivity_diagnostic.json, with per-query records and per-run
summaries by distance bucket and by correctness.

Usage:
    python3 scripts/task_sensitivity_diagnostic.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.mqar_intervention as mq
import scripts.selective_copy_intervention as sc
from src.model.single_layer_mamba import SingleLayerMamba

DEV = torch.device('cuda')
CONDS = ['slow', 'default', 'fast']
SEEDS = [42, 43, 44]
BUCKETS = [(1, 16), (16, 64), (64, 256), (256, 1024), (1024, 4096)]
SEQ_LEN, BATCH, N_BATCHES = 1536, 16, 10


def examples(task):
    """Identical evaluation batches for every run: (input_ids, target_ids, queries)."""
    rng = np.random.default_rng(seed=12345)
    out = []
    for _ in range(N_BATCHES):
        if task == 'mqar':
            ids, tg, d = mq.make_mqar_batch(BATCH, SEQ_LEN, 8, 4, rng)
            q_start = max(int(SEQ_LEN * 0.78), int(SEQ_LEN * 0.75) + 4)
            pairs = [[p for p in range(q_start) if ids[b, p] < mq.N_KEYS] for b in range(BATCH)]
            out.append((ids, tg, [[(q, k, [(p, p + 1) for p in pairs[b] if p != k]) for q, k in d[b]]
                                  for b in range(BATCH)]))
        else:
            ids, tg, d = sc.make_selcopy_batch(BATCH, SEQ_LEN, 8, rng)
            out.append((ids, tg, [[(o, c - 1, [(c2 - 1, c2) for _, c2 in d[b] if c2 != c]) for o, c in d[b]]
                                  for b in range(BATCH)]))
    return out


def run_model(task, model, batches):
    valid = (torch.arange(mq.N_KEYS, mq.N_KEYS + mq.N_VALUES) if task == 'mqar'
             else torch.arange(sc.N_CONTENT)).to(DEV)
    recs = []
    for ids, tg, queries in batches:
        ids, tg = ids.to(DEV), tg.to(DEV)
        emb = model.embedding(ids).detach().requires_grad_(True)
        logits, _ = model(embeddings=emb)
        for qi in range(len(queries[0])):
            T = queries[0][qi][0]
            lv = logits[:, T, valid]                                   # [batch, n_valid]
            corr = (valid[None, :] == tg[:, T, None])
            m = (lv * corr).sum(-1) - (lv * ~corr).sum(-1) / (len(valid) - 1)
            g, = torch.autograd.grad(m.sum(), emb, retain_graph=True)
            I = (g ** 2).sum(-1).cpu()                                 # [batch, L]
            pred, tgt, mm = logits[:, T].argmax(-1).cpu(), tg[:, T].cpu(), m.detach().cpu()
            for b in range(ids.shape[0]):
                q, k, others = queries[b][qi]
                Ib = I[b]
                rel = float(Ib[k] + Ib[k + 1])
                dis = float(np.mean([float(Ib[a] + Ib[c]) for a, c in others])) if others else float('nan')
                recs.append({'distance': q - k - (1 if task == 'selcopy' else 0),
                             'correct': int(pred[b] == tgt[b]), 'margin': float(mm[b]),
                             'I_rel': rel, 'I_dis': dis, 'share': rel / float(Ib[:q].sum())})
    return recs


def summarize(recs):
    def block(rs):
        if not rs:
            return {'n': 0}
        lr = np.log10([r['I_rel'] / r['I_dis'] for r in rs])
        return {'n': len(rs), 'accuracy': float(np.mean([r['correct'] for r in rs])),
                'mean_margin': float(np.mean([r['margin'] for r in rs])),
                'median_log10_rel_over_dis': float(np.median(lr)),
                'mean_share': float(np.mean([r['share'] for r in rs]))}
    out = {'all': block(recs),
           'by_bucket': {f'{lo}-{hi}': block([r for r in recs if lo <= r['distance'] < hi]) for lo, hi in BUCKETS},
           'correct': block([r for r in recs if r['correct']]),
           'incorrect': block([r for r in recs if not r['correct']])}
    return out


def main():
    out = {'note': 'see module docstring', 'runs': {}}
    for task, mod in (('mqar', mq), ('selcopy', sc)):
        batches = examples(task)
        for s in SEEDS:
            for c in CONDS:
                ck = torch.load(f'checkpoints/{task}/rerun_seed{s}/{task}_{c}.pt', map_location=DEV,
                                weights_only=False)
                m = SingleLayerMamba(d_model=256, d_inner=512, d_state=16, vocab_size=mod.VOCAB_SIZE).to(DEV)
                m.load_state_dict(ck['model_state_dict'])
                m.eval()
                recs = run_model(task, m, batches)
                key = f'{task}_{c}_seed{s}'
                out['runs'][key] = {'summary': summarize(recs), 'records': recs}
                a = out['runs'][key]['summary']['all']
                print(key, 'acc %.3f  median log10(I_rel/I_dis) %.2f  share %.3f' %
                      (a['accuracy'], a['median_log10_rel_over_dis'], a['mean_share']), flush=True)
    with open('results/task_sensitivity_diagnostic.json', 'w') as f:
        json.dump(out, f)
    print('wrote results/task_sensitivity_diagnostic.json')


if __name__ == '__main__':
    main()
