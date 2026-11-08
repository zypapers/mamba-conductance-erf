"""
Legacy script. The paper uses the two-layer model it trains (checkpoints/model_2layer.pt) and its
TwoLayerMamba class; App. J's analyses come from scripts/scan_probe_large_sample.py and
scripts/last_layer_block_test.py. Its own per-layer comparison with a spectral-measure prediction
is not used in the paper, and its outputs are not released.

Usage:
    python3 scripts/two_layer_experiment.py
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.scan_ref import selective_scan_ref
from src.extraction.conductances import compute_conductances
from src.utils.logspace import cumulative_log_retention
from src.data.wikitext import get_dataloader


class MambaLayer(nn.Module):
    """Single Mamba layer (no embedding/LM head)."""

    def __init__(self, d_model=256, d_inner=512, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner, bias=True)

        A_log = torch.log(torch.linspace(1, d_state, d_state)
                          .unsqueeze(0).expand(d_inner, -1).clone())
        self.A_log = nn.Parameter(A_log)

        self.dt_proj = nn.Linear(d_inner, d_inner, bias=True)
        # Init dt bias
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)

        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        self.extract_mode = False
        self.input_weight = 'euler'  # reference-scan input weight, see src/model/scan_ref.py

    @property
    def A(self):
        return -torch.exp(self.A_log)

    def forward(self, x):
        """x: [batch, L, d_model] -> [batch, L, d_model], intermediates"""
        residual = x
        xz = self.in_proj(x)
        x_ssm, z = xz.split(self.d_inner, dim=-1)

        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_ssm.shape[1]]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        delta_t = F.softplus(self.dt_proj(x_conv))
        B_t = self.B_proj(x_conv)
        C_t = self.C_proj(x_conv)

        if self.extract_mode:
            y_ssm, scan_inter = selective_scan_ref(
                self.A, B_t, C_t, delta_t, x_conv, self.D, input_weight=self.input_weight)
        else:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                y_ssm = selective_scan_fn(
                    x_conv.transpose(1, 2), delta_t.transpose(1, 2),
                    self.A, B_t.transpose(1, 2), C_t.transpose(1, 2),
                    D=self.D, z=None, return_last_state=False)
                y_ssm = y_ssm.transpose(1, 2)
                scan_inter = {}
            except ImportError:
                y_ssm, scan_inter = selective_scan_ref(
                    self.A, B_t, C_t, delta_t, x_conv, self.D, input_weight=self.input_weight)

        z_gate = F.silu(z)
        y = y_ssm * z_gate
        y = self.out_proj(y) + residual

        intermediates = {'A': self.A, 'delta_t': delta_t, 'B_t': B_t,
                         'C_t': C_t, **scan_inter}
        return y, intermediates


class TwoLayerMamba(nn.Module):
    """Two-layer Mamba for language modeling."""

    def __init__(self, d_model=256, d_inner=512, d_state=16, vocab_size=50257):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer1 = MambaLayer(d_model, d_inner, d_state)
        self.layer2 = MambaLayer(d_model, d_inner, d_state)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, inter1 = self.layer1(x)
        x, inter2 = self.layer2(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, inter1, inter2


def train_model(model, args, device):
    """Train 2-layer model."""
    model = model.to(device)
    train_loader = get_dataloader(split='train', seq_len=args.seq_len,
                                   batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(split='validation', seq_len=args.seq_len,
                                 batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.1, betas=(0.9, 0.95))
    total_steps = args.epochs * len(train_loader)
    warmup = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss()
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_loss_sum, train_tok = 0.0, 0

        for batch_idx, (ids, tgt) in enumerate(train_loader):
            ids, tgt = ids.to(device), tgt.to(device)
            logits, _, _ = model(ids)
            loss = loss_fn(logits.view(-1, 50257), tgt.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item() * tgt.numel()
            train_tok += tgt.numel()

        model.eval()
        val_loss_sum, val_tok = 0.0, 0
        with torch.no_grad():
            for ids, tgt in val_loader:
                ids, tgt = ids.to(device), tgt.to(device)
                logits, _, _ = model(ids)
                loss = loss_fn(logits.view(-1, 50257), tgt.view(-1))
                val_loss_sum += loss.item() * tgt.numel()
                val_tok += tgt.numel()

        train_loss = train_loss_sum / train_tok
        val_loss = val_loss_sum / val_tok
        val_ppl = min(math.exp(val_loss), 1e6)
        print(f"  Epoch {epoch+1}/{args.epochs} | train={train_loss:.4f} | "
              f"val={val_loss:.4f} | ppl={val_ppl:.1f} | {time.time()-t0:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_ppl': val_ppl,
                        'epoch': epoch + 1},
                       'checkpoints/model_2layer.pt')

    return best_val, min(math.exp(best_val), 1e6)


def evaluate_layer(layer, x_input, device, layer_name, max_lag=1024, n_targets=10, n_batches=3):
    """Evaluate spectral prediction for one layer."""
    layer.extract_mode = True
    targets = sorted(np.random.choice(range(1024, 2048), size=n_targets, replace=False).tolist())

    erf_frozen = torch.zeros(max_lag, device=device)
    h_dyn = torch.zeros(max_lag, device=device)
    count = 0

    for batch_x in x_input[:n_batches]:
        # Frozen-param ERF for this layer
        with torch.no_grad():
            xz = layer.in_proj(batch_x)
            x_ssm, z = xz.split(layer.d_inner, dim=-1)
            xc = x_ssm.transpose(1, 2)
            xc = layer.conv1d(xc)[:, :, :x_ssm.shape[1]].transpose(1, 2)
            xc = F.silu(xc)
            dt_f = F.softplus(layer.dt_proj(xc))
            Bt_f = layer.B_proj(xc)
            Ct_f = layer.C_proj(xc)

        u_live = xc.detach().requires_grad_(True)
        y_ssm, _ = selective_scan_ref(layer.A, Bt_f, Ct_f, dt_f, u_live, layer.D,
                                  input_weight=layer.input_weight)

        for T in targets:
            if u_live.grad is not None:
                u_live.grad.zero_()
            loss = (y_ssm[:, T, :] ** 2).sum()
            loss.backward(retain_graph=True)
            gsq = (u_live.grad ** 2).sum(dim=-1)
            for tau in range(min(max_lag, T)):
                erf_frozen[tau] += gsq[:, T - tau].mean()

        # h_dyn
        with torch.no_grad():
            _, inter = layer(batch_x)
            g = compute_conductances(inter['A'], inter['delta_t'])
            cumret = cumulative_log_retention(g, dim=1)

            for T in targets:
                c = inter['C_t'][:, T, :]
                yT = inter['y'][:, T, :]
                yTsq = yT ** 2
                log_end = cumret[:, T - 1, :, :]
                for tau in range(min(max_lag, T)):
                    start = T - tau
                    if tau == 0:
                        ret = torch.ones_like(log_end)
                    elif start == 0:
                        ret = torch.exp(cumret[:, T, :, :])
                    else:
                        ret = torch.exp(cumret[:, T, :, :] - cumret[:, start, :, :])
                    b = inter['B_bar'][:, start, :, :]
                    pch = (c[:, None, :] * b * ret).sum(dim=-1)
                    h_dyn[tau] += (yTsq * pch ** 2).sum(dim=-1).mean()

        count += 1

    n_total = count * len(targets)
    erf_frozen /= n_total
    h_dyn /= n_total
    erf_frozen = erf_frozen / erf_frozen[1].clamp(min=1e-30)
    h_dyn = h_dyn / h_dyn[1].clamp(min=1e-30)

    # Slopes and ratios
    results = {'layer': layer_name}
    for tau in [10, 50, 100, 500]:
        h, e = h_dyn[tau].item(), erf_frozen[tau].item()
        results[f'ratio_tau{tau}'] = h / e if e > 1e-15 else None

    for t1, t2 in [(10, 100), (100, 500)]:
        for name, arr in [('h_dyn', h_dyn), ('erf_frozen', erf_frozen)]:
            v1, v2 = arr[t1].item(), arr[t2].item()
            if v1 > 1e-15 and v2 > 1e-15:
                results[f'{name}_slope_{t1}_{t2}'] = float(
                    np.log10(v2 / v1) / np.log10(t2 / t1))

    results['h_dyn'] = h_dyn.cpu().tolist()
    results['erf_frozen'] = erf_frozen.cpu().tolist()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--results_path', default='results/two_layer/results.json')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/two_layer', exist_ok=True)

    print("=" * 60)
    print("Two-Layer Mamba Experiment")
    print("=" * 60)

    model = TwoLayerMamba(d_model=256, d_inner=512, d_state=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: 2-layer Mamba, {n_params:,} params")

    ckpt_path = 'checkpoints/model_2layer.pt'
    if args.skip_train and os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        val_ppl = ckpt['val_ppl']
    else:
        print("\nTraining...")
        val_loss, val_ppl = train_model(model, args, device)

    model.eval()
    print(f"Val PPL: {val_ppl:.1f}")

    # Collect layer inputs
    print("\nCollecting layer inputs...")
    val_loader = get_dataloader(split='validation', batch_size=2, seq_len=2048,
                                 shuffle=False, num_workers=0)
    layer1_inputs = []
    layer2_inputs = []

    model.layer1.extract_mode = True
    model.layer2.extract_mode = True

    with torch.no_grad():
        for batch_idx, (ids, _) in enumerate(val_loader):
            if batch_idx >= 3:
                break
            ids = ids.to(device)
            x = model.embedding(ids)
            layer1_inputs.append(x)
            x_out, _ = model.layer1(x)
            layer2_inputs.append(x_out)

    # Evaluate each layer
    torch.manual_seed(42)
    np.random.seed(42)

    print("\nEvaluating Layer 1...")
    r1 = evaluate_layer(model.layer1, layer1_inputs, device, "layer1")

    print("Evaluating Layer 2...")
    r2 = evaluate_layer(model.layer2, layer2_inputs, device, "layer2")

    # Print results
    print(f"\n{'':>10} | {'ratio@10':>10} | {'ratio@100':>10} | {'ratio@500':>10} | {'slope 10-100':>13}")
    print('-' * 65)
    for r in [r1, r2]:
        name = r['layer']
        r10 = r.get('ratio_tau10', 'N/A')
        r100 = r.get('ratio_tau100', 'N/A')
        r500 = r.get('ratio_tau500', 'N/A')
        s = r.get('h_dyn_slope_10_100', 'N/A')
        es = r.get('erf_frozen_slope_10_100', 'N/A')
        print(f'{name:>10} | {r10:>10.2f} | {r100:>10.2f} | {r500:>10.2f} | h:{s:>5.2f} e:{es:>5.2f}')

    # Save
    output = {
        'val_ppl': float(val_ppl),
        'n_params': n_params,
        'layer1': {k: v for k, v in r1.items() if k != 'h_dyn' and k != 'erf_frozen'},
        'layer2': {k: v for k, v in r2.items() if k != 'h_dyn' and k != 'erf_frozen'},
        'layer1_curves': {'h_dyn': r1['h_dyn'], 'erf_frozen': r1['erf_frozen']},
        'layer2_curves': {'h_dyn': r2['h_dyn'], 'erf_frozen': r2['erf_frozen']},
    }
    with open(args.results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.results_path}")


if __name__ == '__main__':
    main()
