"""
Scan-probe analyses for Apps. I and J under the trained recurrence, with larger samples
(see the note on the legacy scripts below).

For every layer of the chosen model, on natural text, it computes the scan probe of
Eq. (15) at lags tau = 0-1023:
  S(tau) = 4 E[ sum_j y_{T,j}^2 K_{T,T-tau,j}^2 ],
  K_{T,s,j} = sum_i C_{T,i} Bbar_{s,j,i} prod_{r=s+1}^{T} (1 - g_{r,j,i}) + D_j [s = T],
with the Euler input weight Bbar = Delta B that the trained models compute. Su is the
same with C_{T,i} Bbar_{s,j,i} -> |C_{T,i} Bbar_{s,j,i}|, the sign ablation.
Products are formed in log space: log(1 - g) = A Delta, accumulated in float64. y is the
reference-scan output (src/model/scan_ref.py).

  --model two_layer   checkpoints/model_2layer.pt (App. J). Layer 1 gets the token
                      embeddings, layer 2 the output of layer 1 (native forward). All
                      WikiText-103 validation (120) and test (138) sequences of 2048 tokens;
                      targets every 4th T = 1024-2047 (the main protocol's subset).
                      About 4 min.
  --model mamba370m   state-spaces/mamba-370m (App. I), all 48 layers, inputs from the
                      native forward. WikiText-103 validation text, tokenized with the
                      model's GPT-NeoX tokenizer, in every full chunk of 2048 tokens; targets
                      every 16th T = 1024-2047. About 21 min (the closed form costs about
                      27 ms per target per batch of 4 at d_inner = 2048).
--target_stride overrides the default spacing.
The legacy scripts scripts/pretrained_full_sweep.py and scripts/two_layer_experiment.py ran a
smaller version of this analysis with the ZOH input weight (Mamba-370M: one sentence repeated
30 times; two-layer model: 6 sequences; 10 targets each). Their outputs are not used in the
paper and are not released.

Per layer:
  - per-sequence curves S and Su (means over targets), in the npz;
  - least-squares slopes of log S on log tau over tau = 10-100 and 100-1000, from the mean
    curve, with 95% bootstrap intervals over sequences (1000 resamples); the endpoint slopes
    10->100 and 100->500 of the legacy definition, for comparison with it;
  - the sign ablation (Su/Su(1)) / (S/S(1)) at tau = 10, 100, 500, 1000;
  - conductance statistics over every position: fraction g < 0.01 (coast), g > 0.99, mean g.

Identity check, enforced on every layer (above 1e-3 or nonfinite stops the run). S for a
single target is compared with the frozen-parameter ERF computed by autograd through the
reference scan: Delta, B, C fixed, and the gradient of ||y_T||^2 taken with respect to the
scan input. It runs on the first sequence for T = 1023 (on its first 1024 positions; the
model is causal) and T = 2047. Two numbers are reported: the raw relative deviation, and
max over 1 <= tau <= 1023 of |(S/S(1)) / (ERF/ERF(1)) - 1|.

Outputs:
  - results/two_layer/scan_probe_large.json or results/pretrained/scan_probe_large_mamba-370m.json;
  - the matching *_curves.npz (per-sequence curves; gitignored).

Usage:
    python3 scripts/scan_probe_large_sample.py --model two_layer [--smoke]
    python3 scripts/scan_probe_large_sample.py --model mamba370m [--smoke]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.scan_ref import selective_scan_ref

DEV = torch.device('cuda')
MAX_LAG = 1024
SEQ_LEN = 2048
ID_TARGETS = [1023, 2047]
ID_TOL = 1e-3
TAUS_ABLATION = [10, 100, 500, 1000]
N_BOOT = 1000


@torch.no_grad()
def scan_probe(dt, A, Bt, Ct, y, D, targets):
    """Per-sequence means over targets of S and Su: two [batch, MAX_LAG] float64 tensors."""
    cum = torch.cumsum((dt[..., None] * A).double(), dim=1)
    taus = torch.arange(MAX_LAG, device=dt.device)
    S = torch.zeros(dt.shape[0], MAX_LAG, dtype=torch.float64, device=dt.device)
    Su = torch.zeros_like(S)
    for T in targets:
        s = T - taus
        ret = torch.exp(cum[:, T][:, None] - cum[:, s]).float()                  # [b, lag, j, i]
        cb = Ct[:, T][:, None, None, :] * (dt[:, s][..., None] * Bt[:, s][:, :, None, :])
        ysq = y[:, T] ** 2
        for acc, w in ((S, cb), (Su, cb.abs())):
            K = (w * ret).sum(-1)
            K[:, 0] += D
            acc += 4 * (ysq[:, None, :] * K ** 2).sum(-1).double()
        del ret, cb
    return S / len(targets), Su / len(targets)


def identity_check(xc, dt, A, Bt, Ct, D):
    """S against autograd on the first sequence, for each T in ID_TARGETS."""
    out = {}
    taus = torch.arange(MAX_LAG, device=xc.device)
    for T in ID_TARGETS:
        sl = (slice(0, 1), slice(0, T + 1))
        u = xc[sl].detach().requires_grad_(True)
        y, _ = selective_scan_ref(A, Bt[sl].detach(), Ct[sl].detach(), dt[sl].detach(), u, D.detach())
        g, = torch.autograd.grad((y[:, T] ** 2).sum(), u)
        ref = (g[:, T - taus] ** 2).sum(-1).double()[0]
        with torch.no_grad():
            S, _ = scan_probe(dt[sl], A, Bt[sl], Ct[sl], y.detach(), D, [T])
        S = S[0]
        out[f'T{T}'] = {
            'raw_max_rel_dev': float((S - ref).abs().max() / ref.abs().max()),
            'normalized_ratio_max_dev_tau_1_1023': float(((S[1:] / S[1]) / (ref[1:] / ref[1]) - 1).abs().max())}
    return out


@torch.no_grad()
def conductance_counts(dt, A):
    g = 1 - torch.exp(dt[..., None] * A)
    return np.array([float((g < 0.01).sum()), float((g > 0.99).sum()), float(g.double().sum()), g.numel()])


class Layer:
    """Per-layer accumulators."""

    def __init__(self):
        self.S, self.Su, self.counts, self.checks = [], [], np.zeros(4), None

    def add(self, xc, dt, A, Bt, Ct, D, targets, check):
        A, D = A.detach(), D.detach()
        with torch.no_grad():
            y, _ = selective_scan_ref(A, Bt, Ct, dt, xc, D)
            S, Su = scan_probe(dt, A, Bt, Ct, y, D, targets)
        self.S.append(S.cpu().numpy())
        self.Su.append(Su.cpu().numpy())
        self.counts += conductance_counts(dt, A)
        if check:
            self.checks = identity_check(xc, dt, A, Bt, Ct, D)
            bad = {k: v for k, v in self.checks.items()
                   if not np.isfinite(v['raw_max_rel_dev']) or v['raw_max_rel_dev'] > ID_TOL}
            if bad:
                raise RuntimeError(f'identity check failed: {bad}')

    def summary(self):
        S, Su = np.concatenate(self.S), np.concatenate(self.Su)
        mean = S.mean(0)
        rng = np.random.default_rng(0)
        boot = {r: [] for r in ['10-100', '100-1000']}
        for _ in range(N_BOOT):
            m = S[rng.integers(0, len(S), len(S))].mean(0)
            for r in boot:
                boot[r].append(ls_slope(m, *map(int, r.split('-'))))
        nS, nSu = mean / mean[1], Su.mean(0) / Su.mean(0)[1]
        c = self.counts
        return {
            'n_sequences': int(len(S)),
            'identity_check': self.checks,
            'slope_ls': {r: {'value': ls_slope(mean, *map(int, r.split('-'))),
                             'ci95': [float(x) for x in np.percentile(boot[r], [2.5, 97.5])]} for r in boot},
            'slope_endpoint': {'10-100': endpoint_slope(mean, 10, 100), '100-500': endpoint_slope(mean, 100, 500)},
            'sign_ablation_normalized_Su_over_S': {str(t): float(nSu[t] / nS[t]) for t in TAUS_ABLATION},
            'g_stats': {'frac_below_0.01': c[0] / c[3], 'frac_above_0.99': c[1] / c[3], 'mean': c[2] / c[3]},
            'S_mean_normalized': nS.tolist(),
            'Su_mean_normalized': nSu.tolist()}


def ls_slope(curve, lo, hi):
    t = np.arange(lo, hi + 1)
    return float(np.polyfit(np.log(t), np.log(curve[lo:hi + 1]), 1)[0])


def endpoint_slope(curve, t1, t2):
    return float(np.log10(curve[t2] / curve[t1]) / np.log10(t2 / t1))


def run_two_layer(smoke, stride):
    from scripts.two_layer_experiment import TwoLayerMamba
    from src.data.wikitext import get_dataloader
    m = TwoLayerMamba().to(DEV)
    m.load_state_dict(torch.load('checkpoints/model_2layer.pt', map_location=DEV, weights_only=False)['model_state_dict'])
    m.eval()
    targets = list(range(1024, 2048, 64 if smoke else stride))
    layers = {'layer1': Layer(), 'layer2': Layer()}
    first = True
    for split in ['validation'] if smoke else ['validation', 'test']:
        bs = 2 if smoke else 6                     # the loader drops a partial last batch; 6 divides 120 and 138
        loader = get_dataloader(split=split, batch_size=bs, seq_len=SEQ_LEN, shuffle=False, num_workers=0)
        if not smoke and len(loader.dataset) % bs:
            raise RuntimeError(f'{split}: {len(loader.dataset)} sequences do not divide into batches of {bs}')
        for bi, (ids, _) in enumerate(loader):
            if smoke and bi == 1:
                break
            with torch.no_grad():
                x = m.embedding(ids.to(DEV))
            for name, layer in (('layer1', m.layer1), ('layer2', m.layer2)):
                with torch.no_grad():
                    xs, _ = layer.in_proj(x).split(layer.d_inner, dim=-1)
                    xc = F.silu(layer.conv1d(xs.transpose(1, 2))[:, :, :SEQ_LEN].transpose(1, 2))
                    dt, Bt, Ct = F.softplus(layer.dt_proj(xc)), layer.B_proj(xc), layer.C_proj(xc)
                layers[name].add(xc, dt, layer.A, Bt, Ct, layer.D, targets, first)
                with torch.no_grad():
                    x, _ = layer(x)
            first = False
            print(split, bi, flush=True)
    return layers, {'targets': [targets[0], targets[-1], targets[1] - targets[0]], 'tokenizer': 'gpt2'}


def run_mamba370m(smoke, stride):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from scripts.pretrained_full_sweep import load_pretrained_mamba
    from scripts.preview_pretrained_euler import layer_inputs
    m = load_pretrained_mamba('state-spaces/mamba-370m', DEV)
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    raw = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    ids = tok.encode(''.join(raw['text']), return_tensors='pt')[0]
    chunks = ids[:len(ids) // SEQ_LEN * SEQ_LEN].view(-1, SEQ_LEN)
    if smoke:
        chunks = chunks[:2]
    n_layers = len(m.backbone.layers)
    which = [0, n_layers // 2, n_layers - 1] if smoke else list(range(n_layers))
    targets = list(range(1024, 2048, 64 if smoke else stride))
    layers = {f'L{i}': Layer() for i in which}
    bs = 2 if smoke else 4
    for bi in range(0, len(chunks), bs):
        t0 = time.time()
        with torch.no_grad():
            x = m.backbone.embedding(chunks[bi:bi + bs].to(DEV))
        for i, block in enumerate(m.backbone.layers):
            if i in which:
                with torch.no_grad():
                    xc, dt, Bt, Ct, A = layer_inputs(block.mixer, block, x)
                layers[f'L{i}'].add(xc, dt, A, Bt, Ct, block.mixer.D.float(), targets, bi == 0)
            if i == which[-1]:
                break
            with torch.no_grad():
                x = x + block.mixer(block.norm(x))
        print('batch', bi // bs, 'of', (len(chunks) + bs - 1) // bs, '%.0fs' % (time.time() - t0), flush=True)
    return layers, {'targets': [targets[0], targets[-1], targets[1] - targets[0]], 'tokenizer': 'EleutherAI/gpt-neox-20b',
                    'n_chunks': int(len(chunks))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['two_layer', 'mamba370m'], required=True)
    p.add_argument('--smoke', action='store_true', help='one small batch, few targets (and 3 layers for 370M); no outputs')
    p.add_argument('--target_stride', type=int, default=None, help='default 4 (two_layer) or 16 (mamba370m)')
    args = p.parse_args()
    stride = args.target_stride or (4 if args.model == 'two_layer' else 16)
    t0 = time.time()
    layers, meta = (run_two_layer if args.model == 'two_layer' else run_mamba370m)(args.smoke, stride)
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    out = {'note': 'see module docstring', 'model': args.model, 'commit': commit, 'max_lag': MAX_LAG,
           'seq_len': SEQ_LEN, 'input_weight': 'euler', **meta, 'layers': {}}
    curves = {}
    for name, acc in layers.items():
        out['layers'][name] = acc.summary()
        curves[f'{name}__S'] = np.concatenate(acc.S)
        curves[f'{name}__Su'] = np.concatenate(acc.Su)
        s = out['layers'][name]
        print(name, 'slope 10-100 %.3f [%.3f, %.3f]' % (s['slope_ls']['10-100']['value'], *s['slope_ls']['10-100']['ci95']),
              'coast %.4f' % s['g_stats']['frac_below_0.01'], json.dumps(s['identity_check']), flush=True)
    out['runtime_s'] = time.time() - t0
    print('runtime %.0fs' % out['runtime_s'])
    if args.smoke:
        return
    base = 'results/two_layer/scan_probe_large' if args.model == 'two_layer' else 'results/pretrained/scan_probe_large_mamba-370m'
    with open(base + '.json', 'w') as f:
        json.dump(out, f, indent=1)
    np.savez_compressed(base + '_curves.npz', **curves)
    print('wrote', base + '.json')


if __name__ == '__main__':
    main()
