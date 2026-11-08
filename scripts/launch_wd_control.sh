#!/usr/bin/env bash
# Weight-decay control (about 11.5 h on one GPU).
#  1. Training (about 9 h): the matched WikiText recipe (scripts/train_matched.py), 3 A-inits x
#     seeds 1-3, three runs at a time, with the named parameters exempt from weight decay and
#     everything else unchanged. Within a seed, the initial weights and the data order are those
#     of the matched cohort (the optimizer change consumes no randomness), so the two cohorts
#     pair by seed and condition. scripts/check_wd_control_outputs.py validates every file.
#  2. Only if training passed: the matched protocol on the new cohort (about 2.2 h,
#     scripts/camera_ready_single_layer.py --cohort matched --tag), then its summary, the
#     conductance and Delta statistics, the parameter drift, and the paired comparison with the
#     matched cohort (scripts/compare_wd_control.py; about 3 min in all).
# Outputs go to checkpoints/matched_<tag>/, results/matched_<tag>/ and
# results/camera_ready/matched_<tag>*; the matched cohort is never touched. Every job's exit
# status is checked; failures go to logs/failures_<tag>.txt. Exits with status 1 on any failure.
#
# Usage: bash scripts/launch_wd_control.sh <tag> <param> [<param> ...]
#   e.g. bash scripts/launch_wd_control.sh nowd_timescale A_log dt_proj.bias
cd "$(dirname "$0")/.."
tag="$1"
shift
if [ -z "$tag" ] || [ "$#" -eq 0 ]; then
  echo "usage: $0 <tag> <param> [<param> ...]"
  exit 2
fi
if [ -e "checkpoints/matched_$tag" ]; then
  echo "checkpoints/matched_$tag exists; choose a new tag"
  exit 2
fi
export PYTHONHASHSEED=0
mkdir -p logs
: > "logs/failures_$tag.txt"
fail=0
note_failure() { echo "FAILED: $1" | tee -a "logs/failures_$tag.txt"; fail=1; }

for s in 1 2 3; do
  pids=()
  names=()
  for c in slow default fast; do
    python3 scripts/train_matched.py --condition "$c" --seed "$s" --no_wd "$@" --tag "$tag" \
      > "logs/matched_${tag}_${c}_seed$s.log" 2>&1 &
    pids+=("$!")
    names+=("$tag $c seed $s")
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || note_failure "${names[$i]}"
  done
done
python3 scripts/check_wd_control_outputs.py "$tag" "$@" > "logs/check_$tag.txt" 2>&1 \
  || note_failure "output check (see logs/check_$tag.txt)"
if [ "$fail" -ne 0 ]; then
  echo "control training finished WITH FAILURES; see logs/failures_$tag.txt"
  exit 1
fi
echo "training done; outputs checked (logs/check_$tag.txt); running the protocol"

python3 scripts/camera_ready_single_layer.py --cohort matched --tag "$tag" \
  > "logs/protocol_$tag.log" 2>&1 || note_failure "protocol (see logs/protocol_$tag.log)"
if [ "$fail" -eq 0 ]; then
  python3 scripts/analyze_camera_ready_matched.py --tag "$tag" > "logs/analysis_$tag.log" 2>&1 \
    || note_failure "summary (see logs/analysis_$tag.log)"
  python3 scripts/conductance_stats.py --tag "$tag" >> "logs/analysis_$tag.log" 2>&1 \
    || note_failure "conductance statistics (see logs/analysis_$tag.log)"
  python3 scripts/check_weight_decay_drift.py --tag "$tag" >> "logs/analysis_$tag.log" 2>&1 \
    || note_failure "parameter drift (see logs/analysis_$tag.log)"
  python3 scripts/compare_wd_control.py --control "matched_$tag" >> "logs/analysis_$tag.log" 2>&1 \
    || note_failure "comparison with the matched cohort (see logs/analysis_$tag.log)"
fi
if [ "$fail" -ne 0 ]; then
  echo "control finished WITH FAILURES; see logs/failures_$tag.txt"
  exit 1
fi
echo "control done: training, protocol and analyses (logs/*_$tag*)"
