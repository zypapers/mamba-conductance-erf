"""
Validate the weight-decay control cohort written by scripts/launch_wd_control.sh.

For each seed 1-3 and condition:
  - checkpoints/matched_<tag>/seed<S>/final_<c>.pt is at epoch 12, and best_<c>.pt exists, both
    recording seed S, condition c and the expected exempt parameters;
  - results/matched_<tag>/training_<c>_seed<S>.json has 12 epochs.
Prints one line per run (best validation perplexity) and exits with status 1 on any problem.

Usage:
    python3 scripts/check_wd_control_outputs.py <tag> <param> [<param> ...]
"""

import json
import os
import sys

import torch


def main():
    tag, expected = sys.argv[1], sorted(set(sys.argv[2:]))
    problems = []
    for s in (1, 2, 3):
        for c in ('slow', 'default', 'fast'):
            for kind in ('final', 'best'):
                ck = f'checkpoints/matched_{tag}/seed{s}/{kind}_{c}.pt'
                if not os.path.exists(ck):
                    problems.append(f'{ck}: missing')
                    continue
                m = torch.load(ck, map_location='cpu', weights_only=False)
                a = m.get('args', {})
                if a.get('seed') != s or a.get('condition') != c:
                    problems.append(f'{ck}: seed {a.get("seed")}, condition {a.get("condition")}')
                if a.get('no_wd') != expected:
                    problems.append(f'{ck}: no_wd {a.get("no_wd")}, expected {expected}')
                if kind == 'final' and m.get('epoch') != 12:
                    problems.append(f'{ck}: epoch {m.get("epoch")}')
            hp = f'results/matched_{tag}/training_{c}_seed{s}.json'
            if not os.path.exists(hp):
                problems.append(f'{hp}: missing')
                continue
            with open(hp) as f:
                h = json.load(f)
            if len(h.get('val_ppl', [])) != 12:
                problems.append(f'{hp}: {len(h.get("val_ppl", []))} epochs')
            else:
                print(f'seed {s} {c:7s}: best val ppl {min(h["val_ppl"]):.2f} (epoch {h["best_epoch"]}), '
                      f'final {h["val_ppl"][-1]:.2f}')
    for line in problems:
        print('PROBLEM:', line)
    print('all expected outputs present' if not problems else f'{len(problems)} problems')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
