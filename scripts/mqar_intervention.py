"""
MQAR (Multi-Query Associative Recall) intervention experiment.

Trains single-layer Mamba models with slow / default / fast A-initialization on
MQAR sequences and evaluates accuracy stratified by the distance between key
insertion and query. The paper reports these accuracies without attributing
them to the ERF.

Smoke test (--smoke_test): 3000 steps on a smaller task, all three conditions.
Full run: --steps (default 10000); scripts/launch_retraining.sh gives the paper's configuration.

Usage:
    python3 scripts/mqar_intervention.py --smoke_test
    python3 scripts/mqar_intervention.py --condition slow
    python3 scripts/mqar_intervention.py --condition default
    python3 scripts/mqar_intervention.py --condition fast
    python3 scripts/mqar_intervention.py --condition all
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


# Vocab layout (small synthetic vocab)
# Smaller vocab makes the random baseline (1/N_VALUES) larger and easier to
# beat, which gives us cleaner signal in distance-stratified accuracy.
N_KEYS = 16           # key tokens: ids 0..15
N_VALUES = 16         # value tokens: ids 16..31
PAD_ID = 32           # filler/noise token id
QUERY_MARKER = 33     # token marking start of queries section
EOS = 34              # end-of-sequence
VOCAB_SIZE = 35

A_CONFIGS = {
    'slow':    {'low': 0.1, 'high': 2.0},
    'default': {'low': 1.0, 'high': 16.0},
    'fast':    {'low': 5.0, 'high': 50.0},
}


def make_mqar_batch(batch_size, seq_len, n_pairs, n_queries, rng,
                    kv_region_end_frac=0.75, query_region_start_frac=0.78):
    """
    Build one MQAR batch with key positions spread across the sequence.

    Layout per sequence (PAD = filler everywhere except the listed positions):
      KV region [0, kv_end):  K_i, V_i pairs at K_i_pos and K_i_pos+1, K_i_pos
                              chosen randomly (within slot) so distances vary.
      Query region [query_start, seq_len): Q_j at q_j_pos, target at SAME pos
                                            (logits[q_j_pos] predicts next-token V_j).

    Convention: target[q_pos] = V is consistent with how SingleLayerMamba and
    the wikitext loader train (CE between logits[t] and target[t], where logits[t]
    predicts the token that would naturally follow input[0..t]).

    Returns:
        input_ids: [batch, seq_len] LongTensor
        target_ids: [batch, seq_len] LongTensor (PAD_ID where ignored by loss)
        distances: per-batch list of (q_pos, k_pos) tuples for eval stratification
    """
    kv_region_end = int(seq_len * kv_region_end_frac)
    query_region_start = max(int(seq_len * query_region_start_frac), kv_region_end + 4)
    if query_region_start >= seq_len - 2:
        raise ValueError(f"seq_len={seq_len} too small for query region")

    slot_size = (kv_region_end - 1) // n_pairs
    if slot_size < 2:
        raise ValueError(
            f"Need slot_size >= 2: seq_len={seq_len}, kv_region_end={kv_region_end}, "
            f"n_pairs={n_pairs}. Reduce n_pairs or increase seq_len.")

    input_ids = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    target_ids = torch.full((batch_size, seq_len), PAD_ID, dtype=torch.long)
    distances = []

    for b in range(batch_size):
        keys = rng.choice(N_KEYS, size=n_pairs, replace=False)
        values = rng.choice(N_VALUES, size=n_pairs, replace=False) + N_KEYS

        # Place K_i at slot_i_start + random offset within slot, V_i at K_i+1
        kv_map = {}  # key_id -> (k_pos, v_id)
        for i, (k, v) in enumerate(zip(keys, values)):
            slot_start = i * slot_size
            offset = int(rng.integers(0, slot_size - 1))
            k_pos = slot_start + offset
            input_ids[b, k_pos] = int(k)
            input_ids[b, k_pos + 1] = int(v)
            kv_map[int(k)] = (k_pos, int(v))

        # Choose n_queries distinct keys to query, place in query region
        if n_queries > n_pairs:
            raise ValueError(f"n_queries ({n_queries}) > n_pairs ({n_pairs})")
        query_keys = rng.choice(keys, size=n_queries, replace=False)
        per_seq = []
        # Place QUERY_MARKER once just before the queries (helps the model identify
        # the start of the query block; not strictly necessary)
        marker_pos = query_region_start
        if marker_pos < seq_len:
            input_ids[b, marker_pos] = QUERY_MARKER

        # Space queries 2 apart starting from marker_pos+1; ensure they fit
        for q_i, qk in enumerate(query_keys):
            q_pos = marker_pos + 1 + 2 * q_i
            if q_pos >= seq_len:
                break
            k_pos, v_id = kv_map[int(qk)]
            input_ids[b, q_pos] = int(qk)
            # Target convention: target[t] is what logits[t] should predict next.
            target_ids[b, q_pos] = v_id
            per_seq.append((q_pos, k_pos))
        distances.append(per_seq)

    return input_ids, target_ids, distances


def make_model(d_model, d_inner, d_state, condition, device):
    """Create a SingleLayerMamba with custom vocab and A-init."""
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


def evaluate_by_distance(model, args, device, n_batches=20, distance_buckets=None):
    """
    Eval accuracy stratified by key-query distance.

    Records (distance, correct?) for every query, then bins by distance buckets.
    Default buckets are log-spaced; empty buckets are reported as n=0 (visible).

    Returns:
        overall_acc: float
        per_bucket: dict {bucket_label: {'correct', 'total', 'acc'}}
        all_pairs: list of (distance, correct) tuples for plotting
    """
    if distance_buckets is None:
        distance_buckets = [(1, 16), (16, 64), (64, 256), (256, 1024),
                            (1024, 4096)]
    all_pairs = []  # (distance, correct) — for downstream plotting

    rng = np.random.default_rng(seed=12345)
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, target_ids, kv_distances = make_mqar_batch(
                batch_size=args.batch_size, seq_len=args.seq_len,
                n_pairs=args.n_pairs, n_queries=args.n_queries, rng=rng,
            )
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits, _ = model(input_ids=input_ids)
            preds = logits.argmax(dim=-1)

            for b in range(input_ids.shape[0]):
                for q_pos, k_pos in kv_distances[b]:
                    distance = q_pos - k_pos
                    # Target convention: logits[q_pos] predicts the next-token V,
                    # so we read prediction at q_pos.
                    pred = preds[b, q_pos].item()
                    target = target_ids[b, q_pos].item()
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

    # Distance distribution summary (helpful for diagnosing empty buckets)
    if all_pairs:
        ds = np.array([d for d, _ in all_pairs])
        print(f"    distance distribution: min={ds.min()}, "
              f"25%={int(np.percentile(ds, 25))}, "
              f"med={int(np.median(ds))}, "
              f"75%={int(np.percentile(ds, 75))}, "
              f"max={ds.max()}")

    return overall, per_bucket, all_pairs


def train_mqar(condition, args, device):
    """Train one model on MQAR. Returns trained model + history."""
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
        input_ids, target_ids, _ = make_mqar_batch(
            batch_size=args.batch_size, seq_len=args.seq_len,
            n_pairs=args.n_pairs, n_queries=args.n_queries, rng=rng,
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
                if mask.sum() > 0:
                    acc = ((preds == target_ids) & mask).sum().float() / mask.sum().float()
                    acc = acc.item()
                else:
                    acc = 0.0
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
    parser.add_argument('--d_model', type=int, default=256,
                        help='Match WikiText model for fairness')
    parser.add_argument('--d_inner', type=int, default=512)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--n_pairs', type=int, default=8,
                        help='Number of key-value pairs per sequence '
                             '(fewer = easier)')
    parser.add_argument('--n_queries', type=int, default=4,
                        help='Number of queries per sequence')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--n_eval_batches', type=int, default=50)
    parser.add_argument('--save_dir', default='checkpoints/mqar')
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
        # Smoke test: the WikiText model size, fewer key-value pairs and a
        # smaller vocabulary, for 3000 steps.
        args.steps = 3000
        args.log_interval = 500
        args.d_model = 256
        args.d_inner = 512
        args.batch_size = 16
        args.seq_len = 768
        args.n_pairs = 8
        args.n_queries = 4
        args.n_eval_batches = 10
        print("[SMOKE TEST] WikiText-sized model, 3000 steps, all 3 conditions; "
              "~5-8 min total")
        print(f"  Random baseline = 1/{N_VALUES} = {1/N_VALUES:.3f}; "
              f"smoke success = overall_acc > 0.15 in at least one condition.")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('results/mqar', exist_ok=True)

    if args.condition == 'all':
        conditions = ['slow', 'default', 'fast']
    else:
        conditions = [args.condition]

    all_results = {'config': vars(args), 'conditions': {}}
    for cond in conditions:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model, history = train_mqar(cond, args, device)
        ckpt_path = os.path.join(args.save_dir, f'mqar_{cond}.pt')
        torch.save({'model_state_dict': model.state_dict(),
                    'args': vars(args), 'condition': cond,
                    'history': history}, ckpt_path)
        print(f"  Saved {ckpt_path}")

        print(f"  Evaluating {cond} by distance...")
        overall_acc, per_bucket, all_pairs = evaluate_by_distance(
            model, args, device, n_batches=args.n_eval_batches,
        )
        print(f"  overall_acc={overall_acc:.3f} "
              f"(random baseline 1/{N_VALUES} = {1/N_VALUES:.3f})")
        print(f"  per-distance accuracy:")
        for label, stats in per_bucket.items():
            acc = stats['acc']
            acc_s = f"{acc:.3f}" if acc is not None else "n/a"
            print(f"    {label:>10}: {acc_s}  (n={stats['total']})")

        all_results['conditions'][cond] = {
            'overall_acc': overall_acc,
            'per_bucket': per_bucket,
            'history': history,
            # all_pairs lets the user re-bin distances after the fact for plots
            'all_pairs': all_pairs,
        }

    suffix = f'_seed{args.seed}' if args.seed != 42 else ''
    out_path = ('results/mqar/mqar_smoke.json' if args.smoke_test
                else f'results/mqar/mqar_intervention{suffix}.json')
    if args.results_path:
        out_path = args.results_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Cross-condition comparison
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
