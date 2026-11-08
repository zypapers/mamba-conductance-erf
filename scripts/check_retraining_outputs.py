"""
Check the outputs of scripts/launch_retraining.sh.

Task reruns, for each task (mqar, selcopy) and seed 42-44:
  - results/<task>/rerun/<task>_seed<S>.json has all three conditions, config seed S and
    paired_data True;
  - checkpoints/<task>/rerun_seed<S>/<task>_<condition>.pt exists, with seed S.
WikiText, for each seed 1-3 and condition:
  - checkpoints/matched/seed<S>/final_<c>.pt is at epoch 12, and best_<c>.pt exists, both
    recording seed S and condition c;
  - results/matched/training_<c>_seed<S>.json has 12 epochs.

Prints task accuracies and final and best validation perplexities. Exits with status 1
if anything is missing or inconsistent.

Usage:
    python3 scripts/check_retraining_outputs.py
"""

import json
import os
import sys

import torch

CONDS = ['slow', 'default', 'fast']


def ckpt_args(path):
    m = torch.load(path, map_location='cpu', weights_only=False)
    return m, m.get('args', {})


def main():
    problems, rows = [], []
    for task in ['mqar', 'selcopy']:
        for s in [42, 43, 44]:
            rp = f'results/{task}/rerun/{task}_seed{s}.json'
            if not os.path.exists(rp):
                problems.append(f'missing {rp}')
                continue
            d = json.load(open(rp))
            cfg = d.get('config', {})
            if cfg.get('seed') != s or not cfg.get('paired_data'):
                problems.append(f'{rp}: seed {cfg.get("seed")}, paired_data {cfg.get("paired_data")}')
            for c in CONDS:
                if c not in d.get('conditions', {}):
                    problems.append(f'{rp}: no condition {c}')
                    continue
                rows.append((task, s, c, f"overall_acc {d['conditions'][c]['overall_acc']:.3f}"))
                ck = f'checkpoints/{task}/rerun_seed{s}/{task}_{c}.pt'
                if not os.path.exists(ck):
                    problems.append(f'missing {ck}')
                    continue
                m, a = ckpt_args(ck)
                if a.get('seed') != s or m.get('condition') != c:
                    problems.append(f'{ck}: seed {a.get("seed")}, condition {m.get("condition")}')
    for s in [1, 2, 3]:
        for c in CONDS:
            for kind in ['final', 'best']:
                ck = f'checkpoints/matched/seed{s}/{kind}_{c}.pt'
                if not os.path.exists(ck):
                    problems.append(f'missing {ck}')
                    continue
                m, a = ckpt_args(ck)
                if a.get('seed') != s or a.get('condition') != c:
                    problems.append(f'{ck}: seed {a.get("seed")}, condition {a.get("condition")}')
                if kind == 'final' and m.get('epoch') != 12:
                    problems.append(f'{ck}: epoch {m.get("epoch")}')
            hp = f'results/matched/training_{c}_seed{s}.json'
            if not os.path.exists(hp):
                problems.append(f'missing {hp}')
                continue
            h = json.load(open(hp))
            if len(h.get('val_ppl', [])) != 12:
                problems.append(f'{hp}: {len(h.get("val_ppl", []))} epochs')
                continue
            rows.append(('wikitext', s, c, f"final ppl {h['val_ppl'][-1]:.2f}, best ppl "
                                           f"{min(h['val_ppl']):.2f} (epoch {h['best_epoch']})"))
    for r in rows:
        print(*r)
    if problems:
        print(f'{len(problems)} problem(s):')
        for p in problems:
            print('  ' + p)
        sys.exit(1)
    print('all outputs present and consistent')


if __name__ == '__main__':
    main()
