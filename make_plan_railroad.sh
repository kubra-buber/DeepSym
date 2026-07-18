#!/bin/bash
# DeepSym Railroad planning wrapper.
#
# Nominal deterministic Railroad:
#   ./make_plan_railroad.sh opts.yaml "(H3)" --nominal
#
# Exact finite-horizon expected reachability:
#   ./make_plan_railroad.sh opts.yaml "(H3)" --expected -max-steps 25
#
# Railroad MCTS:
#   ./make_plan_railroad.sh opts.yaml "(H3)" --mcts \
#       --iterations 10000 --max-depth 25 --mcts-runs 10
#
# Existing exact closed-loop implementation:
#   ./make_plan_railroad.sh opts.yaml "(H3)" --closed-loop \
#       --outcome-source argmax-progress

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 opts.yaml \"(H3)\" [--nominal|--expected|--mcts|--closed-loop] [planner args...]" >&2
  exit 2
fi

OPTS="$1"
GOAL="$2"
shift 2

RAILROAD_PYTHON="${RAILROAD_PYTHON:-/home/kubra/miniconda3/envs/deepsym_railroad/bin/python}"
RECOGNIZE_PYTHON="${RECOGNIZE_PYTHON:-python}"
RECOGNIZE_DEVICE="${RECOGNIZE_DEVICE:-auto}"

PLANNER="nominal"
CLOSED_LOOP=0
RECOGNIZE=1
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --nominal)
      PLANNER="nominal"
      shift
      ;;
    --expected)
      PLANNER="expected"
      shift
      ;;
    --mcts)
      PLANNER="mcts"
      shift
      ;;
    --closed-loop)
      CLOSED_LOOP=1
      PLANNER="expected"
      shift
      ;;
    --no-recognize)
      RECOGNIZE=0
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$RECOGNIZE" -eq 1 ]; then
  "$RECOGNIZE_PYTHON" recognize.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    -device "$RECOGNIZE_DEVICE"
fi

if [ "$CLOSED_LOOP" -eq 1 ]; then
  "$RAILROAD_PYTHON" closed_loop_railroad.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    "${EXTRA_ARGS[@]}"
elif [ "$PLANNER" = "expected" ]; then
  "$RAILROAD_PYTHON" make_plan_railroad_expected.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    "${EXTRA_ARGS[@]}"
elif [ "$PLANNER" = "mcts" ]; then
  "$RAILROAD_PYTHON" make_plan_railroad_mcts.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    "${EXTRA_ARGS[@]}"
else
  "$RAILROAD_PYTHON" make_plan_railroad.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    "${EXTRA_ARGS[@]}"
fi

loc="$(grep '^save:' "$OPTS" | sed 's/^.*: //')"
echo "Plan saved to $loc/plan.txt"