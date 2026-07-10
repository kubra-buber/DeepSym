#!/bin/bash
# Railroad planning wrapper with optional closed-loop probabilistic execution. Uses railroad_problem.json, not problem.pddl.
#
# Open-loop nominal (default):
#   ./make_plan_railroad.sh opts.yaml "(H3)"
#
# Open-loop expected/maxprob:
#   ./make_plan_railroad.sh opts.yaml "(H3)" --expected
#
# Closed-loop probabilistic planning, dry-run/simulated outcomes:
#   ./make_plan_railroad.sh opts.yaml "(H2) (S4)" --closed-loop --outcome-source argmax-progress
#
# Closed-loop probabilistic planning with physical execution and manual outcome feedback:
#   ./make_plan_railroad.sh opts.yaml "(H2) (S4)" --closed-loop --execute --outcome-source manual

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 opts.yaml \"(H3)\" [--expected|--nominal] [--closed-loop] [extra planner args...]" >&2
  exit 2
fi

OPTS="$1"
GOAL="$2"
shift 2

# Railroad conda environment (Python 3.12 required for Railroad).
RAILROAD_PYTHON="${RAILROAD_PYTHON:-/home/kubra/miniconda3/envs/deepsym_railroad/bin/python}"

# Recognition uses the repo/.venv Python because it needs ROS + your DeepSym model stack.
RECOGNIZE_PYTHON="${RECOGNIZE_PYTHON:-python}"
# Use auto by default: recognize.py will try opts.yaml device and fall back to CPU if CUDA init fails.
RECOGNIZE_DEVICE="${RECOGNIZE_DEVICE:-auto}"

# Save location from opts.yaml. Keeps the original simple convention.
loc="$(grep '^save:' "$OPTS" | sed 's/^.*: //')"

PLANNER="nominal"
CLOSED_LOOP=0
RECOGNIZE=1
EXTRA_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --expected)
      PLANNER="expected"
      shift
      ;;
    --nominal)
      PLANNER="nominal"
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
  "$RECOGNIZE_PYTHON" recognize.py -opts "$OPTS" -goal "$GOAL" -device "$RECOGNIZE_DEVICE"
fi

if [ "$CLOSED_LOOP" -eq 1 ]; then
  "$RAILROAD_PYTHON" closed_loop_railroad.py -opts "$OPTS" -goal "$GOAL" "${EXTRA_ARGS[@]}"
else
  if [ "$PLANNER" = "expected" ]; then
    "$RAILROAD_PYTHON" make_plan_railroad_expected.py -opts "$OPTS" -goal "$GOAL" "${EXTRA_ARGS[@]}"
  else
    "$RAILROAD_PYTHON" make_plan_railroad.py -opts "$OPTS" -goal "$GOAL" "${EXTRA_ARGS[@]}"
  fi
  echo "Plan saved to $loc/plan.txt"
fi