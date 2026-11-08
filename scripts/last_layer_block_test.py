"""
Block closed form for the last block of the two-layer model (App. J), trained recurrence.

The ERF is taken with respect to that block's incoming activations u2 (the layer-1
output), not the whole model's ERF with respect to the embeddings. Earlier layers would
need downstream gradients at several output times, which a single terminal readout
vector does not capture.

The readout is the gradient g_o of ||logits_T||^2 at the block output o_T, through the
final LayerNorm and the LM head. It is exact (autograd) and is a fixed vector per target.
So a_T = silu(z_T) * (g_o W_out) / 2, and the rest is the single-block closed form of
src/curves/block_erf.py: X = ||2 W_x^T v||^2.

Comparisons with autograd with respect to u2:
  - layer 2's Delta, B, C detached: should equal X (identity, first batch);
  - nothing detached: E2, the last block's actual ERF.
S2 is the scan probe of layer 2 (Eq. 15), the quantity App. J reports.

Every validation sequence x 128 targets (T = 1024, 1032, ..., 2040). Curves are pooled
and normalized at tau = 1. 95% CIs: bootstrap over sequences (500 resamples).
Output: results/last_layer_block_test.json.

Usage:
    python3 scripts/last_layer_block_test.py [--max_batches N]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from scripts.two_layer_experiment import TwoLayerMamba
from src.data.wikitext import get_dataloader
from src.model.scan_ref import selective_scan_ref

DEV = torch.device('cuda')
MAX_LAG = 1024
TARGETS = list(range(1024, 2048, 8))
BANDS = [(1, 1), (5, 15), (40, 60), (80, 120), (180, 220), (400, 600), (800, 1023)]


def layer2_head(model, u2, detach=''):
    """Layer 2 (training path), final norm and LM head from block input u2; logits."""
    L2, L = model.layer2, u2.shape[1]
    xs, z = L2.in_proj(u2).split(L2.d_inner, dim=-1)
    xc = F.silu(L2.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2))
    dt, Bt, Ct = F.softplus(L2.dt_proj(xc)), L2.B_proj(xc), L2.C_proj(xc)
    dt = dt.detach() if 'd' in detach else dt
    Bt = Bt.detach() if 'b' in detach else Bt
    Ct = Ct.detach() if 'c' in detach else Ct
    y = selective_scan_fn(xc.transpose(1, 2), dt.transpose(1, 2), L2.A, Bt.transpose(1, 2),
                          Ct.transpose(1, 2), D=L2.D).transpose(1, 2)
    o = L2.out_proj(y * F.silu(z)) + u2
    return model.lm_head(model.norm(o))


def autograd_erf(model, u2, targets, detach):
    u = u2.detach().requires_grad_(True)
    logits = layer2_head(model, u, detach)
    taus = torch.arange(MAX_LAG, device=DEV)
    acc = torch.zeros(u.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    for T in targets:
        g, = torch.autograd.grad((logits[:, T] ** 2).sum(), u, retain_graph=True)
        acc += (g[:, T - taus] ** 2).sum(-1).double()
    return acc / len(targets)


@torch.no_grad()
def closed_form(model, u2, targets):
    """Block closed form X and scan probe S2 for layer 2, per sequence, mean over targets."""
    L2, L = model.layer2, u2.shape[1]
    xs, z = L2.in_proj(u2).split(L2.d_inner, dim=-1)
    pre = L2.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2)
    xi = F.silu(pre)
    sig = torch.sigmoid(pre)
    dsilu = sig * (1 + pre * (1 - sig))
    dt, Bt, Ct = F.softplus(L2.dt_proj(xi)), L2.B_proj(xi), L2.C_proj(xi)
    y, inter = selective_scan_ref(L2.A, Bt, Ct, dt, xi, L2.D)
    b_bar = inter['B_bar']
    o = (L2.out_proj(y * F.silu(z)) + u2)
    cum = torch.cumsum((dt[..., None] * L2.A).double(), dim=1)
    W_x, w = L2.in_proj.weight[:L2.d_inner], L2.conv1d.weight[:, 0, :]
    taus = torch.arange(MAX_LAG, device=DEV)
    X = torch.zeros(u2.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    S = torch.zeros_like(X)
    with torch.enable_grad():
        o_leaf = o.detach().requires_grad_(True)
        logits = model.lm_head(model.norm(o_leaf))
        for T in targets:
            g_o, = torch.autograd.grad((logits[:, T] ** 2).sum(), o_leaf, retain_graph=True)
            g_o = g_o[:, T].detach()
            starts = T - taus
            ret = torch.exp(cum[:, T][:, None] - cum[:, starts]).float()
            K = (Ct[:, T][:, None, None, :] * b_bar[:, starts] * ret).sum(-1)
            K[:, 0] += L2.D
            S += 4 * ((y[:, T] ** 2)[:, None, :] * K ** 2).sum(-1).double()
            a = F.silu(z[:, T]) * (g_o @ L2.out_proj.weight) / 2
            Kw = dsilu[:, starts] * K
            G = w[:, 3] * Kw
            for k in range(1, 4):
                G[:, k:] += w[:, 3 - k] * Kw[:, :-k]
            X += ((2 * (a[:, None, :] * G) @ W_x) ** 2).sum(-1).double()
    return X / len(targets), S / len(targets)


def summarize(E, X, S):
    nE, nX, nS = (c.mean(0) / c.mean(0)[1] for c in (E, X, S))
    rr, nr = E.mean(0) / X.mean(0), nE / nX
    gm = lambda r: [float(np.exp(np.mean(np.log(r[a:b + 1])))) for a, b in BANDS]
    lsq = lambda c, a, b: float(np.polyfit(np.log10(np.arange(a, b + 1)), np.log10(c[a:b + 1]), 1)[0])
    return {'E_over_X_norm_range_2_500': [float(nr[2:501].min()), float(nr[2:501].max())],
            'E_over_X_norm_argmin_argmax_2_500': [int(2 + nr[2:501].argmin()), int(2 + nr[2:501].argmax())],
            'E_over_X_norm_range_4_500': [float(nr[4:501].min()), float(nr[4:501].max())],
            'E_over_X_raw_range_4_500': [float(rr[4:501].min()), float(rr[4:501].max())],
            'E_over_X_raw_range_2_500': [float(rr[2:501].min()), float(rr[2:501].max())],
            'E_over_X_norm_bands': gm(nr), 'E_over_X_raw_bands': gm(rr),
            'E_over_S_norm_range_2_500': [float((nE / nS)[2:501].min()), float((nE / nS)[2:501].max())],
            'slopes_lsq_10_100': {'E2': lsq(nE, 10, 100), 'X': lsq(nX, 10, 100), 'S2': lsq(nS, 10, 100)},
            'slopes_lsq_100_1000': {'E2': lsq(nE, 100, 1000), 'X': lsq(nX, 100, 1000), 'S2': lsq(nS, 100, 1000)}}


def with_ci(E, X, S, rng, n_boot=500):
    point = summarize(E, X, S)
    reps = [summarize(E[i], X[i], S[i]) for i in (rng.integers(0, len(E), len(E)) for _ in range(n_boot))]

    def ci(p, rs):
        if isinstance(p, dict):
            return {k: ci(v, [r[k] for r in rs]) for k, v in p.items()}
        lo, hi = np.percentile(np.array(rs, dtype=float), [2.5, 97.5], axis=0)
        return {'point': p, 'ci95': [np.asarray(lo).tolist(), np.asarray(hi).tolist()]}
    return ci(point, reps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_batches', type=int, default=0)
    args = p.parse_args()
    ck = torch.load('checkpoints/model_2layer.pt', map_location=DEV, weights_only=False)
    model = TwoLayerMamba().to(DEV)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    loader = get_dataloader(split='validation', batch_size=2, seq_len=2048, shuffle=False, num_workers=0)
    Es, Xs, Ss, out = [], [], [], {'note': 'see module docstring', 'n_targets': len(TARGETS)}
    for b, (ids, _) in enumerate(loader):
        if args.max_batches and b >= args.max_batches:
            break
        with torch.no_grad():
            u2, _ = model.layer1(model.embedding(ids.to(DEV)))
        X, S = closed_form(model, u2, TARGETS)
        if b == 0:
            sub = TARGETS[:16]
            Xsub, _ = closed_form(model, u2, sub)
            D = autograd_erf(model, u2, sub, 'dbc')
            out['identity_X_vs_autograd_layer2_detached_max_rel_dev'] = float(
                (Xsub[:, 1:] / D[:, 1:] - 1).abs().max())
            print('identity', out['identity_X_vs_autograd_layer2_detached_max_rel_dev'], flush=True)
        Es.append(autograd_erf(model, u2, TARGETS, '').cpu().numpy())
        Xs.append(X.cpu().numpy())
        Ss.append(S.cpu().numpy())
    E, X, S = np.concatenate(Es), np.concatenate(Xs), np.concatenate(Ss)
    out['n_sequences'] = len(E)
    out['values'] = with_ci(E, X, S, np.random.default_rng(0))
    out['pooled_normalized'] = {k: (c.mean(0) / c.mean(0)[1]).tolist() for k, c in (('E2', E), ('X', X), ('S2', S))}
    if not args.max_batches:
        with open('results/last_layer_block_test.json', 'w') as f:
            json.dump(out, f, indent=1)
    points = lambda d: {k: v['point'] if 'point' in v else points(v) for k, v in d.items()}
    print(json.dumps(points(out['values'])))


if __name__ == '__main__':
    main()
