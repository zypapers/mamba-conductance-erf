"""
Matched-recipe WikiText-103 training for the A-initialization comparison.

The legacy default checkpoint, checkpoints/model_default.pt, is the best-validation
checkpoint at epoch 12 of a 20-epoch schedule, while the legacy slow and fast checkpoints
completed a 12-epoch schedule. Here
all three conditions use one recipe, scripts/intervention_experiment.py's train_model:
12 epochs, AdamW (learning rate 1e-3, weight decay 0.1, betas (0.9, 0.95)), cosine
schedule with 500-step warmup, gradient clipping 1.0, batch 8, length 2048. They differ
only in the initialization of A (A_CONFIGS) and in the seed, which sets torch, numpy,
random and the data order.

Saves two checkpoints under checkpoints/matched/seed<S>/:
final_<condition>.pt (the end of epoch 12; the paper analyzes these, since choosing them involves
no selection on the validation split that the ERF analysis also uses) and
best_<condition>.pt (best validation loss). The history goes to
results/matched/training_<condition>_seed<S>.json. No evaluation.

Weight-decay control: --no_wd names parameters to exempt from weight decay (everything
else keeps 0.1) and requires --tag, which sends all outputs to checkpoints/matched_<tag>/
and results/matched_<tag>/, so the matched cohort is never overwritten.

Usage:
    python3 scripts/train_matched.py --condition default --seed 1 [--smoke]
    python3 scripts/train_matched.py --condition default --seed 1 --no_wd A_log --tag <tag>
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.intervention_experiment as ie


class Head:
    """The first n batches of a loader, with a matching len (for --smoke)."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __len__(self):
        return self.n

    def __iter__(self):
        for i, batch in enumerate(self.loader):
            if i >= self.n:
                return
            yield batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', required=True, choices=list(ie.A_CONFIGS))
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--smoke', action='store_true', help='3 batches per split, 1 epoch, scratch output')
    p.add_argument('--no_wd', nargs='+', default=[], help='parameter names exempt from weight decay')
    p.add_argument('--tag', default='', help='output suffix; required with --no_wd')
    args = p.parse_args()
    if bool(args.no_wd) != bool(args.tag):
        p.error('--no_wd and --tag go together')
    cohort = f'matched_{args.tag}' if args.tag else 'matched'
    run = argparse.Namespace(condition=args.condition, seed=args.seed, d_model=256, d_inner=512,
                             d_state=16, seq_len=2048, batch_size=8, lr=1e-3,
                             epochs=1 if args.smoke else args.epochs, no_wd=sorted(set(args.no_wd)),
                             save_dir=f'checkpoints/{cohort}{"_smoke" if args.smoke else ""}/seed{args.seed}')
    if args.smoke:
        get = ie.get_dataloader
        ie.get_dataloader = lambda **kw: Head(get(**kw), 3)
    os.makedirs(run.save_dir, exist_ok=True)
    os.makedirs(f'results/{cohort}', exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = ie.A_CONFIGS[args.condition]
    model = ie.create_model_with_A_init(run.d_model, run.d_inner, run.d_state, cfg['low'], cfg['high'])
    history = ie.train_model(model, run, torch.device('cuda'))
    best = os.path.join(run.save_dir, f'best_{args.condition}.pt')
    final = os.path.join(run.save_dir, f'final_{args.condition}.pt')
    os.replace(os.path.join(run.save_dir, f'model_{args.condition}.pt'), best)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': run.epochs,
                'val_loss': history['val_loss'][-1], 'val_ppl': history['val_ppl'][-1],
                'args': vars(run)}, final)
    meta = {'args': vars(run), 'checkpoints': {'final': final, 'best': best},
            'best_epoch': int(np.argmin(history['val_loss']) + 1), **history}
    if not args.smoke:
        with open(f'results/{cohort}/training_{args.condition}_seed{args.seed}.json', 'w') as f:
            json.dump(meta, f, indent=1)
    print('done', args.condition, args.seed, 'best val ppl', min(history['val_ppl']))


if __name__ == '__main__':
    main()
