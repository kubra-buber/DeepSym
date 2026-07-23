#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 BASE_OPTS.yaml [OUTPUT_ROOT]" >&2
  exit 2
fi

BASE_OPTS="$1"
OUTPUT_ROOT="${2:-save/poster_runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-original vq dynamic}"
SEEDS="${SEEDS:-1 2 3 4 5}"
SKIP_DONE="${SKIP_DONE:-1}"
RUN_RULE_EXPORT="${RUN_RULE_EXPORT:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_ROOT"

for model in $MODELS; do
  for seed in $SEEDS; do
    run_dir="$OUTPUT_ROOT/$model/seed_$seed"
    mkdir -p "$run_dir"

    if [[ "$SKIP_DONE" == "1" && -f "$run_dir/poster_metrics.json" ]]; then
      echo "[skip] $model seed=$seed already complete"
      continue
    fi

    echo
    echo "================================================================"
    echo "Training model=$model seed=$seed -> $run_dir"
    echo "================================================================"
    "$PYTHON_BIN" "$SCRIPT_DIR/train.py" \
      -opts "$BASE_OPTS" \
      --model "$model" \
      --seed "$seed" \
      --save-dir "$run_dir" \
      --level both \
      2>&1 | tee "$run_dir/train.log"

    if [[ "$RUN_RULE_EXPORT" == "1" ]]; then
      if [[ "$model" == "original" ]]; then
        "$PYTHON_BIN" save_cat.py -opts "$run_dir/opts.yaml" \
          2>&1 | tee "$run_dir/save_cat.log"
        "$PYTHON_BIN" learn_rules.py -opts "$run_dir/opts.yaml" \
          2>&1 | tee "$run_dir/learn_rules.log"
      else
        "$PYTHON_BIN" "$SCRIPT_DIR/save_cat_vq.py" \
          -opts "$run_dir/opts.yaml" --model "$model" \
          2>&1 | tee "$run_dir/save_cat_vq.log"
        "$PYTHON_BIN" "$SCRIPT_DIR/learn_rules_vq.py" \
          -opts "$run_dir/opts.yaml" \
          2>&1 | tee "$run_dir/learn_rules_vq.log"
      fi
    fi
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_poster_results.py" \
  --root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/aggregate"

echo
echo "All requested runs complete. Aggregate results: $OUTPUT_ROOT/aggregate"