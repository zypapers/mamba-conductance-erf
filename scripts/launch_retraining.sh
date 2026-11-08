#!/usr/bin/env bash
# Trains the 18 retrieval-task models and the nine main language models (about 9.6 h on one GPU).
#  1. MQAR and selective copy: seeds 42-44, all three A-inits, the configuration below
#     with --paired_data, which gives every condition the same training stream within a
#     seed (without it, the data seed is a per-process salted hash(condition) + seed).
#     18 runs, about 24 minutes in all. Outputs go to results/<task>/rerun/ and
#     checkpoints/<task>/rerun_seed<S>/.
#  2. WikiText-103 with the matched recipe (scripts/train_matched.py): 3 A-inits x seeds 1-3,
#     three runs at a time, about 1 h each when run alone. Within a seed, all non-A initial
#     weights and the data order are identical across conditions (checked). Each run saves
#     final_ (epoch 12) and best_ (best validation) checkpoints.
# Every job's exit status is checked; failures are listed in logs/failures.txt. At the end,
# scripts/check_retraining_outputs.py validates every expected file and its seed, condition
# and epoch metadata (report: logs/retraining_check.txt). Exits with status 1 on any failure.
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
mkdir -p logs
: > logs/failures.txt
fail=0
note_failure() { echo "FAILED: $1" | tee -a logs/failures.txt; fail=1; }

for s in 42 43 44; do
  python3 scripts/mqar_intervention.py --condition all --seq_len 1536 --seed "$s" --paired_data \
    --save_dir "checkpoints/mqar/rerun_seed$s" --results_path "results/mqar/rerun/mqar_seed$s.json" \
    > "logs/mqar_seed$s.log" 2>&1 || note_failure "mqar seed $s"
  python3 scripts/selective_copy_intervention.py --condition all --seq_len 1536 --seed "$s" --paired_data \
    --save_dir "checkpoints/selcopy/rerun_seed$s" --results_path "results/selcopy/rerun/selcopy_seed$s.json" \
    > "logs/selcopy_seed$s.log" 2>&1 || note_failure "selcopy seed $s"
done
for s in 1 2 3; do
  pids=()
  names=()
  for c in slow default fast; do
    python3 scripts/train_matched.py --condition "$c" --seed "$s" > "logs/matched_${c}_seed$s.log" 2>&1 &
    pids+=("$!")
    names+=("wikitext $c seed $s")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || note_failure "${names[$i]}"
  done
done
python3 scripts/check_retraining_outputs.py > logs/retraining_check.txt 2>&1 \
  || note_failure "output check (see logs/retraining_check.txt)"
if [ "$fail" -ne 0 ]; then
  echo "retraining finished WITH FAILURES; see logs/failures.txt"
  exit 1
fi
echo "retraining done; outputs checked (logs/retraining_check.txt)"
