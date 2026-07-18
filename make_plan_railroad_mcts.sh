#!/bin/bash
# DeepSym scene recognition + Railroad MCTS planning.
#
# Full representative plan:
#   ./make_plan_railroad_mcts.sh opts.yaml "(H3)" \
#       --iterations 10000 --max-depth 25 --mcts-runs 10
#
# Next physical action for closed-loop execution:
#   ./make_plan_railroad_mcts.sh opts.yaml "(H3)" \
#       --output-mode next-physical-action --mcts-runs 10

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 opts.yaml \"(H3)\" [--no-recognize] [MCTS args...]" >&2
  exit 2
fi

OPTS="$1"
GOAL="$2"
shift 2

RAILROAD_PYTHON="${RAILROAD_PYTHON:-/home/kubra/miniconda3/envs/deepsym_railroad/bin/python}"
RECOGNIZE_PYTHON="${RECOGNIZE_PYTHON:-python}"
RECOGNIZE_DEVICE="${RECOGNIZE_DEVICE:-auto}"

RUN_RECOGNIZE=1
EXTRA_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-recognize)
      RUN_RECOGNIZE=0
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$RUN_RECOGNIZE" -eq 1 ]; then
  "$RECOGNIZE_PYTHON" recognize.py \
    -opts "$OPTS" \
    -goal "$GOAL" \
    -device "$RECOGNIZE_DEVICE"
fi

"$RAILROAD_PYTHON" make_plan_railroad_mcts.py \
  -opts "$OPTS" \
  -goal "$GOAL" \
  "${EXTRA_ARGS[@]}"