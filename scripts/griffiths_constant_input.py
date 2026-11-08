"""
App. C (block-structured input) and App. D's constant-input check on the nine matched
language models, with the trained recurrence.

Inputs, one sequence of 2048 tokens each:
  block      blocks of 256 copies of token 500 and 256 copies of token 30000, alternating,
             starting with 500;
  shuffled   the block sequence's tokens in a fixed random order (numpy seed 0): the same 1024
             copies of each token, without the block structure. It also changes the
             convolution's 4-token windows, and with them the per-position Delta, B, C and
             readout, so it does not separate sustained regimes from variability alone (that is
             what block_order_permuted does, for the retention and write coefficients);
  block_order_permuted (a post-extraction kernel control with fixed readout, derived from the
             block input): the block input's extracted Delta_t and B_t (its retention and write
             coefficients) permuted jointly at random within positions 3-1023 and within
             1024-2047, with the readout weights C_T and y_T of the original forward pass in place,
             for 33 permutation seeds (0-2 and 100-129; three seeds alone
             understate the spread of random order). This preserves the coefficients' empirical distribution
             and leaves the static mixture unchanged (asserted); it changes the temporal
             organization of retention and writing and their alignment with the fixed readout.
             The token shuffle above also changes the convolution's 4-token windows, and with them
             the per-position values; this control does not.
  block_conductances_permuted, block_writes_permuted (the same 33 permutations, applied to one
             coefficient at a time): Delta_t permuted where it sets the decay (the conductances)
             with the writes Bbar_t in place, or Bbar_t permuted with the decay in place. With the
             joint permutation they separate the persistence of the conductances from the
             alignment of the writes with the fixed readout. The static mixture is unchanged in
             both (asserted).
  block_swapped (the same kind of control, deterministic): adjacent blocks of the extracted Delta_t
             and Bbar_t exchanged together (positions 3-255 <-> 259-511, 512-767 <-> 768-1023,
             1024-1279 <-> 1280-1535, 1536-1791 <-> 1792-2047; positions 0-2 and 256-258 stay), with
             the readout in place. The coefficients stay persistent and each write stays with its own
             Delta_t, but each sits in a block of the other token, out of line with the readout. A
             random permutation changes both the persistence of the conductances and their alignment
             with the readout; this control changes only the alignment. The static mixture is
             unchanged (asserted).
  const500   2048 copies of token 500;
  const30000 2048 copies of token 30000.

For each model and input, with the reference scan and the Euler input weight (Bbar = Delta B):
  - dynamic: the scan probe of Eq. (15),
        S(tau) = 4 mean_T sum_j y_{T,j}^2 ( sum_i C_{T,i} Bbar_{T-tau,j,i} prod_{r=T-tau+1}^{T} (1 - g_{r,j,i}) )^2,
    over every 16th target T = 1024-2047, tau = 1-1023, products in log space (log(1 - g) = A Delta);
  - static: the static mixture of App. C, each quantity averaged where it enters the probe: the
    readout quantities C and y^2 over the targets, the source and propagation quantities Bbar and g
    over the target window (positions 1024-2047):
        4 sum_j mean_T(y_{T,j}^2) ( sum_i mean_T(C_{T,i}) mean_t(Bbar_{t,j,i}) (1 - mean_t(g_{t,j,i}))^tau )^2;
  - both normalized at tau = 1, and their ratio dynamic/static: maximum and its lag, means over
    tau = 50-200 and 300-1000, and max |log10 ratio| over tau = 1-500, 1-1021 and 1-1023. For
    tau <= 1021 every source position is at least 3, past the width-4 convolution's start-up
    positions 0-2, so on constant input the two forms must agree there.
For the block and shuffled inputs, also the mean and standard deviation over the target window of
the channel-mean Delta per position; for the block input, the mean Delta over each block type in the
target window, skipping the three positions after each boundary that the width-4 convolution mixes.

Runs on CPU; about 2 s per probe evaluation, about 35 min in all (the 99 permutation
probes per model dominate).

Usage:
    python3 scripts/griffiths_constant_input.py
Output: results/camera_ready/griffiths_constant_input.json
"""

import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.single_layer_mamba import SingleLayerMamba

L, BLOCK, LO, HI = 2048, 256, 500, 30000
TARGETS = list(range(1024, 2048, 16))
PERM_SEEDS = [0, 1, 2] + list(range(100, 130))
MAX_LAG = 1024
CONDS = ['slow', 'default', 'fast']
SEEDS = [1, 2, 3]


def inputs():
    block = torch.tensor([LO if (t // BLOCK) % 2 == 0 else HI for t in range(L)])
    shuffled = block[torch.from_numpy(np.random.default_rng(0).permutation(L))]
    return {'block': block, 'shuffled': shuffled,
            'const500': torch.full((L,), LO), 'const30000': torch.full((L,), HI)}


def load(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = SingleLayerMamba(d_model=256, d_inner=512, d_state=16)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    m.extract_mode = True
    m.input_weight = 'euler'
    return m


@torch.no_grad()
def extract(m, ids):
    _, it = m(input_ids=ids[None])
    return {'A': it['A'].double(),                        # [d_inner, d_state]
            'delta': it['delta_t'][0].double(),           # [L, d_inner]
            'bbar': it['B_bar'][0].double(),              # [L, d_inner, d_state]
            'C': it['C_t'][0].double(),                   # [L, d_state]
            'y': it['y'][0].double()}                     # [L, d_inner]


@torch.no_grad()
def probe(A, delta, bbar, C, y):
    """Dynamic scan probe and static mixture from per-position quantities."""
    logret = delta[:, :, None] * A[None]                  # log(1 - g), [L, d_inner, d_state]
    cum = torch.cumsum(logret, 0)
    lags = torch.arange(1, MAX_LAG)
    dyn = torch.zeros(MAX_LAG, dtype=torch.float64)
    for T in TARGETS:
        s = T - lags                                      # source positions, tau = 1..1023
        k = (C[T][None, None, :] * bbar[s] * torch.exp(cum[T][None] - cum[s])).sum(-1)   # [tau, j]
        dyn[1:] += 4 * (y[T][None] ** 2 * k ** 2).sum(-1)
    dyn /= len(TARGETS)
    w = slice(1024, 2048)
    gbar = (1 - torch.exp(logret[w])).mean(0)
    h = (C[TARGETS].mean(0)[None, None, :] * bbar[w].mean(0)[None]
         * torch.exp(lags[:, None, None].double() * torch.log1p(-gbar)[None])).sum(-1)
    sta = torch.zeros(MAX_LAG, dtype=torch.float64)
    sta[1:] = 4 * ((y[TARGETS] ** 2).mean(0)[None] * h ** 2).sum(-1)
    return dyn.numpy(), sta.numpy()


def curves(m, ids):
    ex = extract(m, ids)
    dyn, sta = probe(**ex)
    return dyn, sta, ex['delta'].numpy(), ex


def order_permuted(ex, seed, which='joint'):
    """Post-extraction kernel control with fixed readout: the extracted retention and write
    coefficients (Delta_t with B_t) permuted within positions 3-1023 and within 1024-2047;
    the readout weights (C, y) of the original forward pass stay in place. The coefficients'
    empirical distribution and the static mixture are unchanged; the temporal organization of
    retention and writing, and their alignment with the readout, change.
    which: 'joint' permutes both (Delta_t in the decay and Bbar_t); 'conductances' permutes only
    Delta_t in the decay, leaving the writes Bbar_t in place; 'writes' permutes only Bbar_t.
    The same seed gives the same permutation in every arm."""
    rng = np.random.default_rng(seed)
    perm = np.arange(L)
    perm[3:1024] = 3 + rng.permutation(1021)
    perm[1024:] = 1024 + rng.permutation(1024)
    perm = torch.from_numpy(perm)
    delta = ex['delta'][perm] if which in ('joint', 'conductances') else ex['delta']
    bbar = ex['bbar'][perm] if which in ('joint', 'writes') else ex['bbar']
    return probe(ex['A'], delta, bbar, ex['C'], ex['y'])


def block_swapped(ex):
    """Post-extraction kernel control with fixed readout: adjacent blocks of the extracted retention and
    write coefficients (Delta_t with Bbar_t) exchanged together; positions 0-2 and 256-258 stay. The
    coefficients stay persistent and coupled, but each sits in a block of the other token, out of line
    with the readout weights (C, y) of the original forward pass, which stay in place."""
    perm = np.arange(L)
    perm[3:BLOCK], perm[BLOCK + 3:2 * BLOCK] = np.arange(BLOCK + 3, 2 * BLOCK), np.arange(3, BLOCK)
    for a in (2 * BLOCK, 4 * BLOCK, 6 * BLOCK):
        perm[a:a + BLOCK], perm[a + BLOCK:a + 2 * BLOCK] = np.arange(a + BLOCK, a + 2 * BLOCK), np.arange(a, a + BLOCK)
    perm = torch.from_numpy(perm)
    return probe(ex['A'], ex['delta'][perm], ex['bbar'][perm], ex['C'], ex['y'])


def summarize(dyn, sta):
    r = (dyn[1:] / dyn[1]) / (sta[1:] / sta[1])           # tau = 1..1023
    tau = np.arange(1, MAX_LAG)
    lr = np.abs(np.log10(r))
    return {'max_ratio': float(r.max()), 'argmax_tau': int(tau[r.argmax()]),
            'max_ratio_1_1021': float(r[:1021].max()), 'argmax_tau_1_1021': int(tau[r[:1021].argmax()]),
            'min_ratio': float(r.min()), 'argmin_tau': int(tau[r.argmin()]),
            'mean_ratio_50_200': float(r[49:200].mean()), 'mean_ratio_300_1000': float(r[299:1000].mean()),
            'max_abs_log10_ratio_1_500': float(lr[:500].max()),
            'max_abs_log10_ratio_1_1021': float(lr[:1021].max()),
            'max_abs_log10_ratio_1_1023': float(lr.max())}, r


def block_delta(delta, ids):
    pos = np.arange(1024, 2048)
    keep = pos[(pos % BLOCK) >= 3]                        # skip the convolution's mixed positions
    tok = ids.numpy()[keep]
    d = delta[keep].mean(1)                               # mean over channels per position
    lo, hi = float(d[tok == LO].mean()), float(d[tok == HI].mean())
    return {'mean_delta_token500': lo, 'mean_delta_token30000': hi, 'ratio_larger_over_smaller': max(lo, hi) / min(lo, hi)}


def main():
    torch.set_num_threads(8)
    t0 = time.time()
    ins = inputs()
    out = {'commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                    text=True).stdout.strip(),
           'input_weight': 'euler', 'targets': [TARGETS[0], TARGETS[-1], 16], 'max_lag': MAX_LAG,
           'block_size': BLOCK, 'tokens': [LO, HI], 'static_window': [1024, 2047], 'models': {}}
    ratio_curves = {}
    for s in SEEDS:
        for c in CONDS:
            name = f'{c}_seed{s}'
            m = load(f'checkpoints/matched/seed{s}/final_{c}.pt')
            res = {}
            for key, ids in ins.items():
                dyn, sta, delta, ex = curves(m, ids)
                res[key], r = summarize(dyn, sta)
                if key in ('block', 'shuffled'):
                    d = delta[1024:2048].mean(1)          # channel-mean Delta per position, target window
                    res[key].update({'delta_window_mean': float(d.mean()), 'delta_window_sd': float(d.std())})
                if key == 'block':
                    res[key].update(block_delta(delta, ids))
                    ratio_curves[name] = r.tolist()
                    for which, rkey in (('joint', 'block_order_permuted'),
                                        ('conductances', 'block_conductances_permuted'),
                                        ('writes', 'block_writes_permuted')):
                        perms = []
                        for seed in PERM_SEEDS:
                            pd, ps = order_permuted(ex, seed, which)
                            assert np.allclose(ps, sta, rtol=1e-12, atol=0), 'static mixture must not change'
                            perms.append(summarize(pd, ps)[0])
                        res[rkey] = {
                            'seeds': PERM_SEEDS, 'per_seed': perms,
                            **{k: max(x[k] for x in perms) for k in      # the largest over the seeds
                               ['max_ratio_1_1021', 'mean_ratio_50_200', 'mean_ratio_300_1000',
                                'max_abs_log10_ratio_1_1021']}}
                    pd, ps = block_swapped(ex)
                    assert np.allclose(ps, sta, rtol=1e-12, atol=0), 'static mixture must not change'
                    res['block_swapped'] = summarize(pd, ps)[0]
            out['models'][name] = res
            print(name, {k: (round(v['max_abs_log10_ratio_1_1021'], 6), round(v['max_ratio_1_1021'], 4))
                         for k, v in res.items()}, f'{time.time() - t0:.0f}s', flush=True)
    out['block_ratio_curves_tau1_to_1023'] = ratio_curves
    out['runtime_s'] = time.time() - t0
    with open('results/camera_ready/griffiths_constant_input.json', 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
