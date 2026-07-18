#!/usr/bin/env bash
# Run fixed-original-objective Railroad MCTS and preserve each result.
set -euo pipefail

OPTS="${1:-opts.yaml}"
ITERATIONS="${ITERATIONS:-10000}"
MAX_DEPTH="${MAX_DEPTH:-25}"
MCTS_RUNS="${MCTS_RUNS:-20}"

cd "$(dirname "$0")"

SAVE_DIR="$(
python - "$OPTS" <<'PY'
import sys, yaml
with open(sys.argv[1], "r") as f:
    print(yaml.safe_load(f)["save"])
PY
)"

mkdir -p "$SAVE_DIR/mcts_fixed_results"

run_and_save () {
  local label="$1"
  local goal="$2"

  echo
  echo "============================================================"
  echo "Fixed Railroad MCTS: $label, goal=$goal"
  echo "============================================================"

  ./make_plan_railroad.sh "$OPTS" "$goal" \
    --mcts \
    --no-recognize \
    --iterations "$ITERATIONS" \
    --max-depth "$MAX_DEPTH" \
    --mcts-runs "$MCTS_RUNS" \
    --trace

  cp "$SAVE_DIR/plan.txt" \
     "$SAVE_DIR/mcts_fixed_results/plan_${label}.txt"
  cp "$SAVE_DIR/mcts_result.json" \
     "$SAVE_DIR/mcts_fixed_results/mcts_result_${label}.json"
}

# Both runs use the already-recognized scene. recognize.py is intentionally
# not called, so the planners are compared on identical objects and relations.
run_and_save "H3" "(H3)"
run_and_save "H2_S4" "(H2) (S4)"

echo
echo "Saved results:"
find "$SAVE_DIR/mcts_fixed_results" -maxdepth 1 -type f -printf '  %p\n' | sort
