"""
Check that the reference scan reproduces the trained models (src/model/scan_ref.py defaults to the Euler
input weight B_bar = Delta B, the recurrence the mamba_ssm CUDA kernel computes).

For every checkpoint: validation loss over all validation sequences in extract
mode (reference scan) against the loss stored at training time, and the
relative L2 difference between extract-mode and CUDA-path logits on the first
batch. Pass: |extract - stored| <= 1e-5 for every checkpoint. checkpoints/best_model.pt is a
byte-identical copy of checkpoints/model_default.pt and is not in the archive; when it is absent,
the script reads model_default.pt under the name best_model.

Usage:
    python3 scripts/check_scan_fix.py
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_discretization import DEV, load
from scripts.two_layer_experiment import TwoLayerMamba
from src.data.wikitext import get_dataloader

TOL = 1e-5


def set_extract(model, on):
    for mod in model.modules():
        if hasattr(mod, 'extract_mode'):
            mod.extract_mode = on


def logits_of(model, ids):
    out = model(ids) if isinstance(model, TwoLayerMamba) else model(input_ids=ids)
    return out[0]


def main():
    loader = get_dataloader(split='validation', batch_size=2, seq_len=2048, shuffle=False,
                            num_workers=0)
    out = {'tolerance': TOL}
    for name in ['best_model', 'model_default', 'model_slow', 'model_fast', 'model_2layer']:
        path = f'checkpoints/{name}.pt'
        if name == 'best_model' and not os.path.exists(path):
            path = 'checkpoints/model_default.pt'  # best_model.pt is a byte-identical copy (same sha256)
        if name == 'model_2layer':
            ck = torch.load(path, map_location=DEV, weights_only=False)
            m = TwoLayerMamba().to(DEV)
            m.load_state_dict(ck['model_state_dict'])
            m.eval()
        else:
            m, ck = load(path)
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for b, (ids, tg) in enumerate(loader):
                ids, tg = ids.to(DEV), tg.to(DEV)
                set_extract(m, True)
                lg = logits_of(m, ids)
                tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                                       reduction='sum').item()
                ntok += tg.numel()
                if b == 0:
                    set_extract(m, False)
                    ref = logits_of(m, ids)
                    rel = float((lg - ref).norm() / ref.norm())
        stored = ck.get('val_loss')
        r = {'stored_val_loss': stored, 'extract_mode_val_loss': tot / ntok,
             'abs_diff': abs(tot / ntok - stored), 'logits_rel_l2_extract_vs_cuda_batch0': rel}
        r['pass'] = r['abs_diff'] <= TOL
        out[name] = r
        print(name, json.dumps(r), flush=True)
    out['all_pass'] = all(out[k]['pass'] for k in out if isinstance(out[k], dict))
    with open('results/scan_fix_gate.json', 'w') as f:
        json.dump(out, f, indent=1)
    print('all_pass', out['all_pass'])


if __name__ == '__main__':
    main()
