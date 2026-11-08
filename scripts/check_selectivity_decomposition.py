"""
Float64 check of the exact selectivity decomposition (src/curves/selectivity_decomposition.py).

The reference is autograd through the model's forward pass, using the reference scan in
float64 with Euler input weight, with some of Delta, B and C detached:
  - all detached          -> frozen
  - only Delta live       -> frozen + dt_ret + dt_write
  - only B live           -> frozen + B
  - only C live           -> frozen + C
  - nothing detached      -> all five parts (the full-model gradient)
Gradient vectors are compared at every source position t < T (lag >= 1). The input is
the first validation sequence, first 300 tokens, with T = 150 and 299, for the three
legacy checkpoints.

The 'only Delta live' case checks dt_ret + dt_write together. To check each Delta pathway
on its own, the reference is a copy of the scan in which Delta enters
the decay exp(A Delta) and the write Delta B xi as two separate tensors, either of which
can be detached (B and C detached throughout):
  - decay live  -> frozen + dt_ret
  - write live  -> frozen + dt_write
Each pathway, (decay live) - (all detached) or (write live) - (all detached), is compared
with that part alone, relative to the part's own largest entry. These split checks run on
the three legacy checkpoints (squared-logit probe, as above) and on two task checkpoints
(fast MQAR and fast selective copy, seed 42) with the margin probe of
scripts/task_sensitivity_v2.py at the first query of the first diagnostic sequence.
Output: results/selectivity_decomposition_check.json.

Usage:
    python3 scripts/check_selectivity_decomposition.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.camera_ready_single_layer import DEV, load
from src.curves.selectivity_decomposition import part_gradients
from src.data.wikitext import get_dataloader
from src.model.scan_ref import selective_scan_ref

CASES = {'all_detached': ('dbc', ['frozen']),
         'delta_live': ('bc', ['frozen', 'dt_ret', 'dt_write']),
         'B_live': ('dc', ['frozen', 'B']),
         'C_live': ('db', ['frozen', 'C']),
         'nothing_detached': ('', ['frozen', 'dt_ret', 'dt_write', 'B', 'C'])}


def reference_grad(m, ids, T, detach):
    emb = m.embedding(ids).detach().requires_grad_(True)
    L = ids.shape[1]
    xs, z = m.in_proj(emb).split(m.d_inner, dim=-1)
    xc = F.silu(m.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2))
    dt, Bt, Ct = F.softplus(m.dt_proj(xc)), m.B_proj(xc), m.C_proj(xc)
    dt = dt.detach() if 'd' in detach else dt
    Bt = Bt.detach() if 'b' in detach else Bt
    Ct = Ct.detach() if 'c' in detach else Ct
    y, _ = selective_scan_ref(m.A, Bt, Ct, dt, xc, m.D)
    logits = m.lm_head(m.out_proj(y * F.silu(z)) + emb)
    g, = torch.autograd.grad((logits[:, T] ** 2).sum(), emb)
    return g


def split_scan(A, Bt, Ct, dt_decay, dt_write, u, D):
    """The trained recurrence (Euler input weight) with separate copies of Delta in the
    decay and the write; equal to selective_scan_ref when the copies are equal."""
    h = torch.zeros(u.shape[0], u.shape[2], A.shape[1], dtype=u.dtype, device=u.device)
    ys = []
    for t in range(u.shape[1]):
        h = (torch.exp(dt_decay[:, t, :, None] * A) * h
             + dt_write[:, t, :, None] * Bt[:, t, None, :] * u[:, t, :, None])
        ys.append((Ct[:, t, None, :] * h).sum(-1))
    return torch.stack(ys, dim=1) + u * D


def split_reference_grad(m, ids, T, live, readout=None):
    """Autograd with B, C detached and only the named copy of Delta live ('decay', 'write'
    or '' for none). The probe is ||logits_T||^2, or <logits_T, readout> if given."""
    emb = m.embedding(ids).detach().requires_grad_(True)
    L = ids.shape[1]
    xs, z = m.in_proj(emb).split(m.d_inner, dim=-1)
    xc = F.silu(m.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2))
    dt = F.softplus(m.dt_proj(xc))
    y = split_scan(m.A, m.B_proj(xc).detach(), m.C_proj(xc).detach(),
                   dt if live == 'decay' else dt.detach(), dt if live == 'write' else dt.detach(),
                   xc, m.D)
    logits = m.lm_head(m.out_proj(y * F.silu(z)) + emb)
    probe = (logits[:, T] ** 2).sum() if readout is None else (logits[:, T] * readout).sum()
    g, = torch.autograd.grad(probe, emb)
    return g


def split_checks(m, ids, T, readout=None):
    """Each Delta pathway on its own against autograd, relative to the part's largest entry."""
    with torch.no_grad():
        parts = part_gradients(m, ids, T, T + 1, dl_dlogits=readout)
    te = torch.arange(1, T + 1, device=DEV)
    ref = {live: split_reference_grad(m, ids, T, live, readout)[:, T - te] for live in ['', 'decay', 'write']}
    res = {'frozen': float((ref[''] - parts['frozen'][:, 1:]).abs().max() / ref[''].abs().max())}
    for live, part in [('decay', 'dt_ret'), ('write', 'dt_write')]:
        mine = parts[part][:, 1:]
        res[f'{part}_alone'] = float(((ref[live] - ref['']) - mine).abs().max() / mine.abs().max())
        res[f'{part}_alone_scale_vs_frozen'] = float(mine.abs().max() / ref[''].abs().max())
    return res


def task_split_checks():
    import scripts.mqar_intervention as mq
    import scripts.selective_copy_intervention as sc
    from scripts.task_sensitivity_diagnostic import examples
    from scripts.task_sensitivity_v2 import readout
    from src.model.single_layer_mamba import SingleLayerMamba
    out = {}
    for task, mod in (('mqar', mq), ('selcopy', sc)):
        ck = torch.load(f'checkpoints/{task}/rerun_seed42/{task}_fast.pt', map_location=DEV,
                        weights_only=False)
        m = SingleLayerMamba(d_model=256, d_inner=512, d_state=16, vocab_size=mod.VOCAB_SIZE).to(DEV)
        m.load_state_dict(ck['model_state_dict'])
        m = m.eval().double()
        ids, tg, queries = examples(task)[0]
        ids, tg = ids[:1].to(DEV), tg[:1].to(DEV)
        T = queries[0][0][0]
        r = readout(task, tg[:, T], mod.VOCAB_SIZE).double()
        out[f'{task}_fast_seed42_T{T}'] = split_checks(m, ids, T, r)
        print(task, json.dumps(out[f'{task}_fast_seed42_T{T}']), flush=True)
    return out


def main():
    loader = get_dataloader(split='validation', batch_size=2, seq_len=2048, shuffle=False,
                            num_workers=0)
    ids = next(iter(loader))[0][:1, :300].to(DEV)
    out = {}
    for cond in ['slow', 'default', 'fast']:
        m = load(f'checkpoints/model_{cond}.pt').double()
        out[cond] = {}
        for T in [150, 299]:
            with torch.no_grad():
                parts = part_gradients(m, ids, T, T + 1)
            te = torch.arange(1, T + 1, device=DEV)
            res = {}
            for case, (det, keep) in CASES.items():
                ref = reference_grad(m, ids, T, det)[:, T - te]
                mine = sum(parts[k] for k in keep)[:, 1:]
                res[case] = float((ref - mine).abs().max() / ref.abs().max())
            res['split'] = split_checks(m, ids, T)
            out[cond][str(T)] = res
        print(cond, json.dumps(out[cond]), flush=True)
    out['task_split'] = task_split_checks()
    devs = ([v for c in ['slow', 'default', 'fast'] for t in out[c].values() for k, v in t.items() if k != 'split']
            + [v for c in ['slow', 'default', 'fast'] for t in out[c].values() for k, v in t['split'].items()
               if not k.endswith('scale_vs_frozen')]
            + [v for t in out['task_split'].values() for k, v in t.items() if not k.endswith('scale_vs_frozen')])
    out['max_over_all'] = max(devs)
    with open('results/selectivity_decomposition_check.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('max relative deviation over all checks:', out['max_over_all'])


if __name__ == '__main__':
    main()
