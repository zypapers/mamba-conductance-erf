"""
The ERF protocol for the single-layer models.

Trained recurrence: the reference scan with the Euler input weight, checked against the
trained models by scripts/check_scan_fix.py. For each of the slow, default and fast checkpoints,
each split (validation, test), and every sequence:
  - the closed forms of src/curves/block_erf.py (S, Su, R, P, X, Xu) at every
    target T in [1024, 2047];
  - the full-model ERF E (gradient of ||logits_T||^2 with respect to the
    embeddings, through the training forward pass) at every target;
  - on a 256-target subset (T = 1024, 1028, ..., 2044): E again (E_sub), the
    ERF with Delta, B, C detached together (Dall) and one at a time (Ddt, DB,
    DC), and the selectivity-gradient terms g_s = g_full - g_Dall:
    ||g_s||^2 (Gss) and 2 <g_Dall, g_s> (Gx).

  - on a 64-target subset (T = 1024, 1040, ..., 2032): the exact selectivity
    decomposition (src/curves/selectivity_decomposition.py), giving the energy of each
    part and every pairwise term (keys dec_*).

Two cohorts. --cohort legacy is the three legacy checkpoints (checkpoints/model_<cond>.pt), with per-sequence curves in
results/camera_ready/single_layer_<model>_<split>.npz. --cohort matched is the nine
matched-recipe final checkpoints, in
results/camera_ready/matched/single_layer_<model>_seed<S>_<split>.npz. Each file records
its checkpoint, seed, recurrence convention, target sets and code commit (key meta).
With --tag <tag>, --cohort matched reads the weight-decay control cohort
checkpoints/matched_<tag>/ instead, and writes to results/camera_ready/matched_<tag>/.
Both are gitignored. The summaries come from scripts/analyze_camera_ready_single_layer.py
(legacy) and scripts/analyze_camera_ready_matched.py (matched).

Checks on the first validation batch, saved to
results/camera_ready/single_layer_checks.json:
  - the training-path forward reproduces the model's own forward;
  - X against autograd with Delta, B, C detached (raw, tau = 1-1023);
  - S against the autograd frozen-parameter ERF of the scan (reference scan);
  - E through the reference scan against E through the training path.

Usage:
    python3 scripts/camera_ready_single_layer.py [--cohort legacy|matched] [--tag T] [--max_batches N]
Runtime: about 45 min for the legacy cohort and 2.3 h for the matched cohort, on one GPU.
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

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from src.curves.block_erf import block_erf_curves, VARIANTS
from src.curves.selectivity_decomposition import decomposition_curves, part_gradients
from src.data.wikitext import get_dataloader
from src.model.scan_ref import selective_scan_ref
from src.model.single_layer_mamba import SingleLayerMamba

DEV = torch.device('cuda')
MAX_LAG = 1024
TARGETS = list(range(1024, 2048))
SUBSET = list(range(1024, 2048, 4))
DECOMP_TARGETS = list(range(1024, 2048, 16))
CONDS = ['slow', 'default', 'fast']
DETACH = {'full': '', 'detach_dt': 'd', 'detach_B': 'b', 'detach_C': 'c', 'detach_all': 'dbc'}


def load(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    a = ck.get('args', {})
    m = SingleLayerMamba(d_model=a.get('d_model', 256), d_inner=a.get('d_inner', 512),
                         d_state=a.get('d_state', 16)).to(DEV)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m


def forward_variant(m, ids, variant, reference=False):
    """The model's forward pass with Delta, B and/or C detached; returns (logits, embeddings)."""
    emb = m.embedding(ids).detach().requires_grad_(True)
    L = ids.shape[1]
    xs, z = m.in_proj(emb).split(m.d_inner, dim=-1)
    xc = F.silu(m.conv1d(xs.transpose(1, 2))[:, :, :L].transpose(1, 2))
    dt, Bt, Ct = F.softplus(m.dt_proj(xc)), m.B_proj(xc), m.C_proj(xc)
    det = DETACH[variant]
    dt = dt.detach() if 'd' in det else dt
    Bt = Bt.detach() if 'b' in det else Bt
    Ct = Ct.detach() if 'c' in det else Ct
    if reference:
        y, _ = selective_scan_ref(m.A, Bt, Ct, dt, xc, m.D)
    else:
        y = selective_scan_fn(xc.transpose(1, 2), dt.transpose(1, 2), m.A, Bt.transpose(1, 2),
                              Ct.transpose(1, 2), D=m.D).transpose(1, 2)
    return m.lm_head(m.out_proj(y * F.silu(z)) + emb), emb


def lagged(g, T):
    return g[:, T - torch.arange(MAX_LAG, device=DEV)]       # [batch, MAX_LAG, d_model]


def erf(logits, emb, targets):
    acc = torch.zeros(emb.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    for T in targets:
        g, = torch.autograd.grad((logits[:, T] ** 2).sum(), emb, retain_graph=True)
        acc += (lagged(g, T) ** 2).sum(-1).double()
    return acc / len(targets)


def scan_frozen_erf(m, ids, targets):
    """Autograd ERF of ||y_T||^2 with respect to the scan input, Delta, B, C fixed (reference scan)."""
    with torch.no_grad():
        emb = m.embedding(ids)
        xs, _ = m.in_proj(emb).split(m.d_inner, dim=-1)
        xc = F.silu(m.conv1d(xs.transpose(1, 2))[:, :, :ids.shape[1]].transpose(1, 2))
        dt, Bt, Ct = F.softplus(m.dt_proj(xc)), m.B_proj(xc), m.C_proj(xc)
    u = xc.clone().requires_grad_(True)
    y, _ = selective_scan_ref(m.A, Bt, Ct, dt, u, m.D)
    return erf(y, u, targets)


def first_batch_checks(m, ids):
    out = {}
    with torch.no_grad():
        ref, _ = m(embeddings=m.embedding(ids))
        lf, _ = forward_variant(m, ids, 'full')
    out['training_path_vs_model_forward_rel_l2'] = float((lf - ref).norm() / ref.norm())
    sub = SUBSET[::16]
    cf = block_erf_curves(m, ids, sub, MAX_LAG)
    dall = erf(*forward_variant(m, ids, 'detach_all'), sub)
    out['X_vs_autograd_detach_all_max_rel_dev'] = float((cf['X'][:, 1:] / dall[:, 1:] - 1).abs().max())
    sf = scan_frozen_erf(m, ids, sub[:4])
    cf4 = block_erf_curves(m, ids, sub[:4], MAX_LAG)
    out['S_vs_autograd_scan_frozen_erf_max_rel_dev'] = float((cf4['S'][:, 1:] / sf[:, 1:] - 1).abs().max())
    e_train = erf(*forward_variant(m, ids, 'full'), sub[:4])
    e_ref = erf(*forward_variant(m, ids, 'full', reference=True), sub[:4])
    out['E_reference_scan_vs_training_path_max_rel_dev'] = float((e_ref[:, 1:] / e_train[:, 1:] - 1).abs().max())
    devs = []
    for T in sub[:2]:
        with torch.no_grad():
            g_sum = sum(part_gradients(m, ids, T, MAX_LAG).values())[:, 1:]
        lf, ef = forward_variant(m, ids, 'full')
        g, = torch.autograd.grad((lf[:, T] ** 2).sum(), ef)
        ref = lagged(g, T)[:, 1:]
        devs.append(float((g_sum - ref).abs().max() / ref.abs().max()))
    out['decomposition_sum_vs_autograd_full_max_rel_dev'] = float(np.max(devs))  # np.max keeps NaN
    return out


# Float32 deviations observed on the legacy checkpoints are at most 2e-4; implementation errors
# are O(1). A check above its tolerance, or a nonfinite one, stops the run, so a failed check
# cannot end in a successful exit.
TOLERANCES = {'training_path_vs_model_forward_rel_l2': 1e-5,
              'X_vs_autograd_detach_all_max_rel_dev': 1e-2,
              'S_vs_autograd_scan_frozen_erf_max_rel_dev': 1e-2,
              'E_reference_scan_vs_training_path_max_rel_dev': 1e-2,
              'decomposition_sum_vs_autograd_full_max_rel_dev': 1e-2}


def enforce(checks, name):
    bad = {k: v for k, v in checks.items() if not np.isfinite(v) or v > TOLERANCES[k]}
    if bad:
        raise RuntimeError(f'{name}: first-batch checks failed {bad}; tolerances {TOLERANCES}')


def run_batch(m, ids):
    res = {k: v for k, v in block_erf_curves(m, ids, TARGETS, MAX_LAG).items()}
    lf, ef = forward_variant(m, ids, 'full')
    la, ea = forward_variant(m, ids, 'detach_all')
    z = lambda: torch.zeros(ids.shape[0], MAX_LAG, device=DEV, dtype=torch.float64)
    E, Es, Da, Gss, Gx = z(), z(), z(), z(), z()
    sub = set(SUBSET)
    for T in TARGETS:
        gf, = torch.autograd.grad((lf[:, T] ** 2).sum(), ef, retain_graph=True)
        gf = lagged(gf, T)
        E += (gf ** 2).sum(-1).double()
        if T in sub:
            ga, = torch.autograd.grad((la[:, T] ** 2).sum(), ea, retain_graph=True)
            ga = lagged(ga, T)
            gs = gf - ga
            Es += (gf ** 2).sum(-1).double()
            Da += (ga ** 2).sum(-1).double()
            Gss += (gs ** 2).sum(-1).double()
            Gx += 2 * (ga * gs).sum(-1).double()
    del lf, ef, la, ea
    n = len(SUBSET)
    res.update({'E': E / len(TARGETS), 'E_sub': Es / n, 'Dall': Da / n, 'Gss': Gss / n, 'Gx': Gx / n})
    for v, key in (('detach_dt', 'Ddt'), ('detach_B', 'DB'), ('detach_C', 'DC')):
        res[key] = erf(*forward_variant(m, ids, v), SUBSET)
    res.update({f'dec_{k}': v for k, v in decomposition_curves(m, ids, DECOMP_TARGETS, MAX_LAG).items()})
    return {k: v.cpu().numpy() for k, v in res.items()}


def runs_for(args):
    """(condition, seed, checkpoint path, output name) for every model in the cohort."""
    if args.cohort == 'legacy':
        return [(c, None, f'checkpoints/model_{c}.pt', f'single_layer_{c}') for c in CONDS]
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    tmpl = args.ckpt_template or f'checkpoints/{cohort}/seed{{seed}}/final_{{cond}}.pt'
    return [(c, s, tmpl.format(seed=s, cond=c), f'single_layer_{c}_seed{s}')
            for s in [int(x) for x in args.seeds.split(',')] for c in CONDS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cohort', choices=['legacy', 'matched'], default='legacy')
    p.add_argument('--seeds', default='1,2,3', help='matched cohort only')
    p.add_argument('--ckpt_template', default=None,
                   help='matched cohort only; default checkpoints/matched/seed{seed}/final_{cond}.pt')
    p.add_argument('--tag', default='', help='matched cohort only: the control cohort matched_<tag>')
    p.add_argument('--max_batches', type=int, default=0)
    args = p.parse_args()
    if args.tag and args.cohort != 'matched':
        p.error('--tag needs --cohort matched')
    out_dir = 'results/camera_ready' + (f'/matched{"_" + args.tag if args.tag else ""}'
                                        if args.cohort == 'matched' else '')
    os.makedirs(out_dir, exist_ok=True)
    tag = '_partial' if args.max_batches else ''
    commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                            text=True).stdout.strip()
    checks = {}
    for cond, seed, path, name in runs_for(args):
        m = load(path)
        meta = {'checkpoint': path, 'condition': cond, 'seed': seed, 'input_weight': 'euler',
                'max_lag': MAX_LAG, 'targets': [TARGETS[0], TARGETS[-1], 1],
                'subset': [SUBSET[0], SUBSET[-1], 4], 'decomp_targets': [DECOMP_TARGETS[0],
                                                                        DECOMP_TARGETS[-1], 16],
                'loss': '||logits_T||^2', 'commit': commit}
        for split in ['validation', 'test']:
            loader = get_dataloader(split=split, batch_size=2, seq_len=2048, shuffle=False,
                                    num_workers=0)
            curves = {}
            t0 = time.time()
            for b, (ids, _) in enumerate(loader):
                if args.max_batches and b >= args.max_batches:
                    break
                ids = ids.to(DEV)
                if b == 0 and split == 'validation':
                    checks[name] = first_batch_checks(m, ids)
                    print(name, 'checks', json.dumps(checks[name]), flush=True)
                    enforce(checks[name], name)
                for k, v in run_batch(m, ids).items():
                    curves.setdefault(k, []).append(v)
                print(f'{name} {split} batch {b} {time.time() - t0:.0f}s', flush=True)
            np.savez_compressed(f'{out_dir}/{name}_{split}{tag}.npz', targets=np.array(TARGETS),
                                subset=np.array(SUBSET), decomp_targets=np.array(DECOMP_TARGETS),
                                meta=json.dumps({**meta, 'split': split}),
                                **{k: np.concatenate(v) for k, v in curves.items()})
    with open(f'{out_dir}/single_layer_checks{tag}.json', 'w') as f:
        json.dump({'commit': commit, 'cohort': args.cohort, 'tag': args.tag, 'checks': checks}, f, indent=1)
    print('done')


if __name__ == '__main__':
    main()
