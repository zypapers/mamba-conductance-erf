"""
Legacy diagnostic: per-layer scan-probe slopes of Mamba-370M under the ZOH and the Euler input
weights, on the small protocol of scripts/pretrained_full_sweep.py, whose output it reads (not
released). scripts/scan_probe_large_sample.py imports layer_inputs from here.

Same input and protocol as scripts/pretrained_full_sweep.py (one sequence of
1024 tokens, 10 targets in [512, 1024), seed 42 per layer, max lag 512; layer
inputs cached from the native forward). For every layer, the frozen-ERF
prediction (Eq. 15) is computed with the ZOH input weight (checked against
the slopes saved by scripts/pretrained_full_sweep.py) and with the native kernel's Euler input weight
B_bar = Delta B (the recurrence the model computes). Reports the endpoint
log-log slope over tau = 10-100 and the coast fraction g < 0.01.

Usage:
    python3 scripts/preview_pretrained_euler.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from scripts.pretrained_full_sweep import load_pretrained_mamba, cache_layer_inputs
from src.model.scan_ref import selective_scan_ref

MAX_LAG = 512


def layer_inputs(mixer, layer, x_input):
    L = x_input.shape[1]
    xz = mixer.in_proj(layer.norm(x_input))
    x_ssm, _ = xz.chunk(2, dim=-1)
    x_conv = F.silu(mixer.conv1d(x_ssm.transpose(1, 2))[:, :, :L].transpose(1, 2))
    dt_rank = mixer.dt_proj.in_features
    dt_raw, B_t, C_t = mixer.x_proj(x_conv).split([dt_rank, mixer.d_state, mixer.d_state], dim=-1)
    delta_t = F.softplus(mixer.dt_proj(dt_raw))
    return x_conv, delta_t, B_t, C_t, -torch.exp(mixer.A_log.float())


def slope(xc, dt, Bt, Ct, A, D, targets, disc):
    if disc == 'zoh':
        y, inter = selective_scan_ref(A, Bt, Ct, dt, xc, D, input_weight='zoh')
        b_bar = inter['B_bar']
    else:
        y = selective_scan_fn(xc.transpose(1, 2).contiguous(), dt.transpose(1, 2).contiguous(), A,
                              Bt.transpose(1, 2).contiguous(), Ct.transpose(1, 2).contiguous(),
                              D=D).transpose(1, 2)
        b_bar = dt[..., None] * Bt[:, :, None, :]
    g = 1 - torch.exp(dt[..., None] * A)
    cum = torch.cumsum(torch.log1p(-g.clamp(max=1 - 1e-6)), dim=1)
    taus = torch.arange(MAX_LAG, device=xc.device)
    acc = torch.zeros(MAX_LAG, device=xc.device, dtype=torch.float64)
    for T in targets:
        st = T - taus
        term = Ct[:, T][:, None, None, :] * b_bar[:, st] * torch.exp(cum[:, T][:, None] - cum[:, st])
        acc += ((y[:, T] ** 2)[:, None, :] * term.sum(-1) ** 2).sum(-1).double().mean(0)
    h = acc.cpu().numpy()
    return float(np.log10(h[100] / h[10])), y


def main():
    device = 'cuda'
    model = load_pretrained_mamba('state-spaces/mamba-370m', device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    text = ("The history of artificial intelligence began in antiquity, "
            "with myths and stories of artificial beings endowed with intelligence. ") * 30
    ids = tok.encode(text, return_tensors='pt')[:, :1024].to(device)
    n_layers = len(model.backbone.layers)
    cache = cache_layer_inputs(model, ids, list(range(n_layers)))
    saved = json.load(open('results/pretrained/full_sweep_state-spaces_mamba-370m.json'))['layers']
    out = {'note': 'see module docstring', 'layers': {}}
    with torch.no_grad():
        for i in range(n_layers):
            layer = model.backbone.layers[i]
            mixer = layer.mixer
            xc, dt, Bt, Ct, A = layer_inputs(mixer, layer, cache[i])
            D = mixer.D.float()
            torch.manual_seed(42)
            np.random.seed(42)
            L = xc.shape[1]
            targets = sorted(np.random.choice(range(L // 2, L), size=10, replace=False).tolist())
            s_zoh, _ = slope(xc, dt, Bt, Ct, A, D, targets, 'zoh')
            s_eul, _ = slope(xc, dt, Bt, Ct, A, D, targets, 'euler')
            g = 1 - torch.exp(dt[..., None] * A)
            out['layers'][str(i)] = {
                'slope_zoh': s_zoh, 'slope_euler': s_eul,
                'saved_slope': saved[str(i)]['erf_frozen_slope_10_100'],
                'coast_frac': float((g < 0.01).float().mean()),
                'saved_coast_frac': saved[str(i)]['g_stats']['frac_below_0.01']}
            print(i, json.dumps(out['layers'][str(i)]), flush=True)
    with open('results/preview_pretrained_euler.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('wrote results/preview_pretrained_euler.json')


if __name__ == '__main__':
    main()
