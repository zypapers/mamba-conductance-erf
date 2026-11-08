"""
Check which input discretization the trained checkpoints compute.

Training and evaluation run the mamba_ssm
CUDA kernel (selective_scan_fn), whose input weight is Euler, B_bar = Delta B
(mamba_ssm selective_scan_ref: deltaB_u = delta * B * u). The exact ZOH input
weight is B_bar = (g/|A|) B; src/model/scan_ref.py (extract mode) implements
both. Both use A_bar = exp(A Delta), so the conductances
agree; only the input weight differs.

For the default, slow and fast checkpoints (checkpoints/model_<cond>.pt) this script
records:
  - validation loss over all validation sequences in both modes, against the
    loss stored in the checkpoint;
  - the relative L2 difference of the SSM output between a pure-PyTorch Euler
    scan, the ZOH scan and the CUDA kernel (first two sequences);
  - the frozen-parameter ERF prediction (Eq. 15) under ZOH and Euler: endpoint
    slope over tau = 10-100 on the intervention protocol (seed 42, 15 targets,
    first 10 sequences) and on all validation sequences x 64 targets with a
    bootstrap CI over sequences; the Euler identity ERF_frozen = 4 h_pred
    (autograd through the CUDA kernel, first batch);
  - for the default model, the full-model ERF (CUDA path, gradient of
    ||logits_T||^2 with respect to the embeddings) on the multiseed protocol
    and on the large sample, and its band ratios to both predictions.

scan_ref.py defaults to the Euler input weight; this script passes
input_weight='zoh' wherever it means the ZOH variant.

Usage:
    python3 scripts/check_discretization.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from src.model.single_layer_mamba import SingleLayerMamba
from src.model.scan_ref import selective_scan_ref
from src.data.wikitext import get_dataloader

DEV = torch.device('cuda')
MAX_LAG = 1024
BANDS = [(5, 15), (80, 120), (400, 600), (800, 1023)]
N_BOOT = 500


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    a = ck.get('args', {})
    m = SingleLayerMamba(d_model=a.get('d_model', 256), d_inner=a.get('d_inner', 512),
                         d_state=a.get('d_state', 16)).to(DEV)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, ck


def ssm_inputs(m, ids):
    xs, _ = m.in_proj(m.embedding(ids)).split(m.d_inner, dim=-1)
    xc = F.silu(m.conv1d(xs.transpose(1, 2))[:, :, :xs.shape[1]].transpose(1, 2))
    return xc, F.softplus(m.dt_proj(xc)), m.B_proj(xc), m.C_proj(xc)


def cuda_scan(m, xc, dt, Bt, Ct):
    return selective_scan_fn(xc.transpose(1, 2), dt.transpose(1, 2), m.A, Bt.transpose(1, 2),
                             Ct.transpose(1, 2), D=m.D).transpose(1, 2)


def h_pred(m, ids, targets, disc):
    """Eq. (15) per sequence, mean over targets: [batch, MAX_LAG]."""
    xc, dt, Bt, Ct = ssm_inputs(m, ids)
    if disc == 'zoh':
        y, inter = selective_scan_ref(m.A, Bt, Ct, dt, xc, m.D, input_weight='zoh')
        b_bar = inter['B_bar']
    else:
        y = cuda_scan(m, xc, dt, Bt, Ct)
        b_bar = dt[..., None] * Bt[:, :, None, :]
    g = 1 - torch.exp(dt[..., None] * m.A)
    cumret = torch.cumsum(torch.log1p(-g.clamp(max=1 - 1e-6)), dim=1)
    taus = torch.arange(MAX_LAG, device=DEV)
    acc = torch.zeros(ids.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    for T in targets:
        starts = T - taus
        term = Ct[:, T][:, None, None, :] * b_bar[:, starts] * torch.exp(
            cumret[:, T][:, None] - cumret[:, starts])
        acc += ((y[:, T] ** 2)[:, None, :] * term.sum(-1) ** 2).sum(-1).double()
    return acc / len(targets)


def frozen_erf_euler(m, ids, targets):
    """Frozen-parameter ERF of the CUDA (Euler) recurrence, mean over batch and targets."""
    with torch.no_grad():
        xc, dt, Bt, Ct = ssm_inputs(m, ids)
    u = xc.detach().requires_grad_(True)
    y = cuda_scan(m, u, dt, Bt, Ct)
    acc = torch.zeros(MAX_LAG, device=DEV, dtype=torch.float64)
    taus = torch.arange(MAX_LAG, device=DEV)
    for T in targets:
        grad, = torch.autograd.grad((y[:, T] ** 2).sum(), u, retain_graph=True)
        acc += (grad ** 2).sum(-1)[:, T - taus].double().mean(0)
    return acc / len(targets)


def full_erf(m, ids, targets):
    """Full-model ERF on the CUDA path, per sequence, mean over targets."""
    m.extract_mode = False
    emb = m.embedding(ids).detach().requires_grad_(True)
    logits, _ = m(embeddings=emb)
    taus = torch.arange(MAX_LAG, device=DEV)
    acc = torch.zeros(ids.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    for T in targets:
        grad, = torch.autograd.grad((logits[:, T] ** 2).sum(), emb, retain_graph=True)
        acc += (grad ** 2).sum(-1)[:, T - taus].double()
    return (acc / len(targets)).detach()


def norm(x):
    c = x.mean(0)
    return c / c[1]


def slope(c):
    return float(np.log10(c[100] / c[10]))


def band_gm(r):
    return [float(np.exp(np.mean(np.log(r[a:b + 1])))) for a, b in BANDS]


def boot(fn, arrays, rng):
    n = len(arrays[0])
    point = fn(*arrays)
    bs = np.array([fn(*[a[i] for a in arrays]) for i in (rng.integers(0, n, n)
                                                          for _ in range(N_BOOT))])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {'point': point, 'ci95': [lo.tolist(), hi.tolist()]}


def main():
    rng = np.random.default_rng(0)
    out = {'note': 'see module docstring; slopes are endpoint log10 slopes over tau = 10-100 of '
                   'curves normalized at tau = 1; bands are geometric means of the ratio'}
    val8 = get_dataloader(split='validation', batch_size=8, seq_len=2048, shuffle=False,
                          num_workers=0)
    val2 = get_dataloader(split='validation', batch_size=2, seq_len=2048, shuffle=False,
                          num_workers=0)
    t64 = np.linspace(1024, 2047, 64).round().astype(int).tolist()

    for cond in ['default', 'slow', 'fast']:
        m, ck = load(f'checkpoints/model_{cond}.pt')
        r = {'stored_val_loss': ck.get('val_loss')}
        tot = {'cuda': 0.0, 'zoh': 0.0}
        ntok = 0
        with torch.no_grad():
            for ids, tg in val8:
                ids, tg = ids.to(DEV), tg.to(DEV)
                for mode in tot:
                    m.extract_mode = (mode == 'zoh')
                    m.input_weight = 'zoh'  # the ZOH variant
                    logits, _ = m(input_ids=ids)
                    tot[mode] += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                                 tg.reshape(-1), reduction='sum').item()
                ntok += tg.numel()
            r['val_loss_cuda'] = tot['cuda'] / ntok
            r['val_loss_zoh'] = tot['zoh'] / ntok

            ids = next(iter(val2))[0].to(DEV)
            xc, dt, Bt, Ct = ssm_inputs(m, ids)
            y_cuda = cuda_scan(m, xc, dt, Bt, Ct)
            y_zoh, _ = selective_scan_ref(m.A, Bt, Ct, dt, xc, m.D, input_weight='zoh')
            x = torch.zeros(ids.shape[0], m.d_inner, m.A.shape[1], device=DEV)
            ys = []
            for t in range(ids.shape[1]):
                x = torch.exp(dt[:, t, :, None] * m.A) * x + \
                    dt[:, t, :, None] * Bt[:, t, None, :] * xc[:, t, :, None]
                ys.append((x * Ct[:, t, None, :]).sum(-1))
            y_euler = torch.stack(ys, 1) + xc * m.D
            rel = lambda a, b: float((a - b).norm() / b.norm())
            r['rel_l2_euler_ref_vs_cuda'] = rel(y_euler, y_cuda)
            r['rel_l2_zoh_ref_vs_cuda'] = rel(y_zoh, y_cuda)
            x_ad = (-m.A * dt[..., None]).flatten()
            factor = (1 - torch.exp(-x_ad)) / x_ad          # ZOH / Euler input weight
            q = torch.tensor([0.01, 0.5, 0.99], device=DEV)
            r['zoh_over_euler_factor_q01_q50_q99'] = torch.quantile(factor[::97], q).tolist()
            r['delta_median_max'] = [float(dt.median()), float(dt.max())]

        # Intervention protocol (scripts/intervention_experiment.py)
        torch.manual_seed(42)
        np.random.seed(42)
        t15 = sorted(np.random.choice(range(1024, 2048), size=15, replace=False).tolist())
        it = iter(val2)
        hz, he = [], []
        for b in range(5):
            ids = next(it)[0].to(DEV)
            with torch.no_grad():
                hz.append(h_pred(m, ids, t15, 'zoh').cpu().numpy())
                he.append(h_pred(m, ids, t15, 'euler').cpu().numpy())
            if b == 0:
                fe = frozen_erf_euler(m, ids, t15).cpu().numpy()
                r['euler_identity_max_abs_dev'] = float(np.max(np.abs(
                    fe[1:] / (4 * he[0].mean(0)[1:]) - 1)))
        r['slope_intervention_protocol_zoh'] = slope(norm(np.concatenate(hz)))
        r['slope_intervention_protocol_euler'] = slope(norm(np.concatenate(he)))

        # Which factor drives the small-protocol slope: sequences or targets (ZOH input weight)
        H15 = []
        with torch.no_grad():
            for ids, _ in val2:
                H15.append(h_pred(m, ids.to(DEV), t15, 'zoh').cpu().numpy())
        H15 = np.concatenate(H15)
        sub = [slope(norm(H15[rng.choice(len(H15), 10, replace=False)])) for _ in range(2000)]
        r['zoh_protocol_breakdown'] = {
            'first10_t15': slope(norm(H15[:10])),
            'first10_per_sequence_t15': [slope(norm(H15[i:i + 1])) for i in range(10)],
            'all_t15': slope(norm(H15)),
            'random10_subsets_t15_median_q025_q975': [float(np.median(sub)),
                                                      *np.percentile(sub, [2.5, 97.5]).tolist()],
        }

        # Large sample: all validation sequences x 64 targets
        HZ, HE = [], []
        with torch.no_grad():
            for ids, _ in val2:
                ids = ids.to(DEV)
                HZ.append(h_pred(m, ids, t64, 'zoh').cpu().numpy())
                HE.append(h_pred(m, ids, t64, 'euler').cpu().numpy())
        HZ, HE = np.concatenate(HZ), np.concatenate(HE)
        r['n_sequences_large'] = len(HZ)
        r['slope_large_zoh'] = boot(lambda a: slope(norm(a)), [HZ], rng)
        r['slope_large_euler'] = boot(lambda a: slope(norm(a)), [HE], rng)
        r['zoh_protocol_breakdown']['first10_t64'] = slope(norm(HZ[:10]))
        r['largest_sequence_share_tau1000'] = {'zoh': float(HZ[:, 1000].max() / HZ[:, 1000].sum()),
                                               'euler': float(HE[:, 1000].max() / HE[:, 1000].sum())}

        if cond == 'default':
            # Full-model ERF on the multiseed protocol (computed by an earlier script, not released)
            per_seed = []
            for s in range(5):
                torch.manual_seed(s * 42 + 7)
                np.random.seed(s * 42 + 7)
                tg = sorted(np.random.choice(list(range(1024, 2048)), size=15,
                                             replace=False).tolist())
                it = iter(val2)
                acc = np.concatenate([full_erf(m, next(it)[0].to(DEV), tg).cpu().numpy()
                                      for _ in range(10)])
                per_seed.append(norm(acc))
            r['full_erf_multiseed_protocol_slope'] = slope(np.mean(per_seed, 0))
            E = np.concatenate([full_erf(m, ids.to(DEV), t64).cpu().numpy() for ids, _ in val2])
            r['full_erf_large'] = boot(
                lambda e, hz_, he_: [slope(norm(e))] + band_gm(norm(he_) / norm(e))
                + band_gm(norm(hz_) / norm(e)), [E, HZ, HE], rng)
            r['full_erf_large_fields'] = ['full_erf_slope'] + \
                [f'euler_pred/full {a}-{b}' for a, b in BANDS] + \
                [f'zoh_pred/full {a}-{b}' for a, b in BANDS]
        out[cond] = r
        print(cond, json.dumps({k: v for k, v in r.items()
                                if not isinstance(v, dict)}), flush=True)

    os.makedirs('results', exist_ok=True)
    with open('results/discretization_check.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote results/discretization_check.json')


if __name__ == '__main__':
    main()
