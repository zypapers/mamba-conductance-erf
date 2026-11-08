"""
Selective copy intervention experiment.

Companion to mqar_intervention.py. Trains the same three
A-initializations (slow/default/fast) on a different long-context
task: selective copy. The model sees a sequence of filler tokens
with N memorable content tokens placed at random positions, each
preceded by a MARKER. After END, the model must output the marked
tokens in order. This is the canonical "easier" SSM memory benchmark
and is known to be solvable by single-layer Mamba.

Distance for stratified eval = position of mark in input sequence to
position of corresponding output token.

Smoke test (--smoke_test): 3000 steps, all three conditions.
Full run: --steps (default 10000); scripts/launch_retraining.sh gives the paper's configuration.

Usage:
    python3 scripts/selective_copy_intervention.py --smoke_test
    python3 scripts/selective_copy_intervention.py
    python3 scripts/selective_copy_intervention.py --condition slow
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

from src.model.single_layer_mamba import SingleLayerMamba


# Vocab layout
N_CONTENT = 16        # content tokens: ids 0..15 (must be remembered)
PAD_ID = 16           # filler/noise
MARKER = 17           # marks: "the next token must be remembered"
EOS = 18              # signals output begin
VOCAB_SIZE = 19

A_CONFIGS = {
    'slow':    {'low': 0.1, 'high': 2.0},
    'default': {'low': 1.0, 'high': 16.0},
    'fast':    {'low': 5.0, 'high': 50.0},
}


def make_selcopy_batch(batch_size, seq_len, n_marks, rng,
                       mark_region_end_frac=0.85):
    """
    Build one selective-copy batch.

    Layout per sequence:
      [PAD ... MARKER content_1 PAD ... MARKER content_2 ... MARKER content_N
       PAD ... EOS  content_1 content_2 ... content_N]

    Marks are placed at random positions in the first 85% of the sequence.
    EOS lands just before the output block of N tokens.

    Model trained next-token: target[t] = expected content at output position.

    Returns:
        input_ids: [batch, seq_len]
        target_ids: [batch, seq_len] (PAD where ignored by loss)
        distances: per-batch list of (output_pos, mark_pos) tuples
    """
    mark_region_end = int(seq_len * mark_region_end_frac)
    output_block_len = n_marks + 1  # EOS + N content positions
    output_start = seq_len - output_block_len  # EOS goes here

    if output_start <= n_marks * 2:
        raise ValueError(
            f"seq_len={seq_len} too small for n_marks={n_marks}")

    # We need n_marks (MARKER, content) pairs in the first part of the
    # sequence; each pair takes 2 positions. Choose mark positions on a
    # stride to avoid collisions.
    slot_size = max(2, mark_region_end // n_marks)
    if slot_size < 2:
        raise ValueError(f"slot_size < 2 for given n_marks/seq_len")

    input_ids = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    target_ids = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    distances = []

    for b in range(batch_size):
        contents = rng.choice(N_CONTENT, size=n_marks, replace=True)

        per_seq = []  # list of (output_pos, mark_pos) — for distance
        # Place markers in slots
        for i, c in enumerate(contents):
            slot_start = i * slot_size
            offset = int(rng.integers(0, slot_size - 1))
            mark_pos = slot_start + offset
            input_ids[b, mark_pos] = MARKER
            input_ids[b, mark_pos + 1] = int(c)
            # The content position is mark_pos + 1; the model's "memory"
            # of this content forms there. For per-distance analysis we
            # use the content position.
            per_seq.append((None, mark_pos + 1))

        # EOS at output_start
        input_ids[b, output_start] = EOS

        # The output positions: output_start+1, output_start+2, ..., +n_marks
        # Per next-token convention: target[t] is what logits[t] predicts
        # next. Place input[output_start] = EOS, target[output_start] =
        # content_1 (so logits at EOS predict content_1). Then
        # input[output_start+1] = content_1 (the model has now produced it,
        # we provide it back), target[output_start+1] = content_2. Etc.
        for i, c in enumerate(contents):
            output_pos = output_start + i  # logits at this position predict content_i+1
            target_ids[b, output_pos] = int(c)
            # Update record with actual output position
            per_seq[i] = (output_pos, per_seq[i][1])
            # Provide the content as input at the next position (for
            # the next prediction in the sequence)
            if i < n_marks - 1:
                input_ids[b, output_pos + 1] = int(c)
        distances.append(per_seq)

    return input_ids, target_ids, distances


def make_model(d_model, d_inner, d_state, condition, device):
    cfg = A_CONFIGS[condition]
    model = SingleLayerMamba(
        d_model=d_model, d_inner=d_inner, d_state=d_state,
        vocab_size=VOCAB_SIZE,
    )
    with torch.no_grad():
        A_log = torch.log(
            torch.linspace(cfg['low'], cfg['high'], d_state)
            .unsqueeze(0).expand(d_inner, -1).clone()
        )
        model.A_log.copy_(A_log)
    return model.to(device)


def evaluate_by_distance(model, args, device, n_batches=20,
                         distance_buckets=None):
    if distance_buckets is None:
        distance_buckets = [(1, 16), (16, 64), (64, 256), (256, 1024),
                            (1024, 4096)]
    all_pairs = []  # (distance, correct)

    rng = np.random.default_rng(seed=12345)
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, target_ids, distances = make_selcopy_batch(
                batch_size=args.batch_size, seq_len=args.seq_len,
                n_marks=args.n_marks, rng=rng,
            )
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits, _ = model(input_ids=input_ids)
            preds = logits.argmax(dim=-1)
            for b in range(input_ids.shape[0]):
                for output_pos, mark_pos in distances[b]:
                    distance = output_pos - mark_pos
                    pred = preds[b, output_pos].item()
                    target = target_ids[b, output_pos].item()
                    correct = int(pred == target)
                    all_pairs.append((distance, correct))

    bucket_correct = [0] * len(distance_buckets)
    bucket_total = [0] * len(distance_buckets)
    total_correct = 0
    for distance, correct in all_pairs:
        total_correct += correct
        for bi, (lo, hi) in enumerate(distance_buckets):
            if lo <= distance < hi:
                bucket_correct[bi] += correct
                bucket_total[bi] += 1
                break

    per_bucket = {}
    for (lo, hi), c, t in zip(distance_buckets, bucket_correct, bucket_total):
        label = f"{lo}-{hi}"
        per_bucket[label] = {'correct': c, 'total': t,
                             'acc': c / t if t > 0 else None}
    overall = total_correct / max(len(all_pairs), 1)

    if all_pairs:
        ds = np.array([d for d, _ in all_pairs])
        print(f"    distance distribution: min={ds.min()}, "
              f"25%={int(np.percentile(ds, 25))}, "
              f"med={int(np.median(ds))}, "
              f"75%={int(np.percentile(ds, 75))}, "
              f"max={ds.max()}")

    return overall, per_bucket, all_pairs


def train_selcopy(condition, args, device):
    print(f"\n=== Training condition '{condition}' ===")
    model = make_model(args.d_model, args.d_inner, args.d_state, condition, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01, betas=(0.9, 0.95))
    warmup_steps = max(50, args.steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # --paired_data: one training stream for every condition within a seed. Without it,
    # the data seed is hash(condition) + seed, which Python salts per process.
    data_seed = args.seed if args.paired_data else (hash(condition) + args.seed) % (2**31)
    rng = np.random.default_rng(seed=data_seed)
    history = {'step': [], 'loss': [], 'acc': []}
    t0 = time.time()
    model.train()

    for step in range(args.steps):
        input_ids, target_ids, _ = make_selcopy_batch(
            batch_size=args.batch_size, seq_len=args.seq_len,
            n_marks=args.n_marks, rng=rng,
        )
        input_ids = input_ids.to(device); target_ids = target_ids.to(device)

        logits, _ = model(input_ids=input_ids)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.log_interval == 0 or step == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = (target_ids != PAD_ID)
                acc = (((preds == target_ids) & mask).sum().float() /
                       mask.sum().float().clamp(min=1)).item()
            history['step'].append(step + 1)
            history['loss'].append(loss.item())
            history['acc'].append(acc)
            elapsed = time.time() - t0
            est_remain = elapsed / (step + 1) * (args.steps - step - 1)
            print(f"  step {step+1}/{args.steps} | loss={loss.item():.4f} | "
                  f"batch_acc={acc:.3f} | {elapsed:.0f}s "
                  f"(est remain {est_remain:.0f}s)")

    print(f"  done in {time.time() - t0:.0f}s")
    return model, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', default='all',
                        help='slow, default, fast, or all')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_inner', type=int, default=512)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=1536)
    parser.add_argument('--n_marks', type=int, default=8,
                        help='Number of marked positions to remember')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--n_eval_batches', type=int, default=50)
    parser.add_argument('--save_dir', default='checkpoints/selcopy')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master seed for torch + numpy + data sampling')
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--paired_data', action='store_true',
                        help='same training-data stream for all conditions within a seed')
    parser.add_argument('--results_path', default=None,
                        help='write results here instead of the default path')
    args = parser.parse_args()

    if args.smoke_test:
        args.steps = 3000
        args.log_interval = 500
        args.d_model = 256
        args.d_inner = 512
        args.batch_size = 16
        args.seq_len = 768
        args.n_marks = 8
        args.n_eval_batches = 10
        print("[SMOKE TEST] WikiText-sized model, 3000 steps, all 3 conditions; ~5-8 min")
        print(f"  Random baseline = 1/{N_CONTENT} = {1/N_CONTENT:.3f}; "
              f"smoke success = overall_acc > 0.30 in at least one condition.")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results/selcopy', exist_ok=True)

    if args.condition == 'all':
        conditions = ['slow', 'default', 'fast']
    else:
        conditions = [args.condition]

    all_results = {'config': vars(args), 'conditions': {}}
    for cond in conditions:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        model, history = train_selcopy(cond, args, device)
        ckpt_path = os.path.join(args.save_dir, f'selcopy_{cond}.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'args': vars(args), 'condition': cond,
                    'history': history}, ckpt_path)
        print(f"  Saved {ckpt_path}")

        print(f"  Evaluating {cond} by distance...")
        overall_acc, per_bucket, all_pairs = evaluate_by_distance(
            model, args, device, n_batches=args.n_eval_batches,
        )
        print(f"  overall_acc={overall_acc:.3f} "
              f"(random baseline 1/{N_CONTENT} = {1/N_CONTENT:.3f})")
        print(f"  per-distance accuracy:")
        for label, stats in per_bucket.items():
            acc = stats['acc']
            acc_s = f"{acc:.3f}" if acc is not None else "n/a"
            print(f"    {label:>10}: {acc_s}  (n={stats['total']})")

        all_results['conditions'][cond] = {
            'overall_acc': overall_acc,
            'per_bucket': per_bucket,
            'history': history,
            'all_pairs': all_pairs,
        }

    suffix = f'_seed{args.seed}' if args.seed != 42 else ''
    out_path = ('results/selcopy/selcopy_smoke.json' if args.smoke_test
                else f'results/selcopy/selcopy_intervention{suffix}.json')
    if args.results_path:
        out_path = args.results_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    if len(conditions) > 1:
        print(f"\n{'='*70}")
        print(f"{'distance':>12} | " +
              " | ".join(f"{c:>10}" for c in conditions))
        print('-' * (15 + 13 * len(conditions)))
        bucket_labels = list(all_results['conditions'][conditions[0]]
                             ['per_bucket'].keys())
        for label in bucket_labels:
            row = [f"{label:>12}"]
            for c in conditions:
                acc = all_results['conditions'][c]['per_bucket'][label]['acc']
                row.append(f"{acc:>10.3f}" if acc is not None else f"{'n/a':>10}")
            print(' | '.join(row))


if __name__ == '__main__':
    main()
