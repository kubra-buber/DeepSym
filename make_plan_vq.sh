#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OPTS_YAML \"(H3) (S4)\"" >&2
  exit 2
fi

OPTS="$1"
GOAL="$2"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_KIND="${MODEL_KIND:-dynamic}"
SCENE_SEED="${SCENE_SEED:-21}"

loc="$("$PYTHON_BIN" - "$OPTS" <<'PY'
import os
import sys
import yaml
with open(sys.argv[1], "r") as f:
    opts = yaml.safe_load(f)
print(os.path.abspath(opts["save"]))
PY
)"

"$PYTHON_BIN" recognize_vq.py \
  -opts "$OPTS" \
  -goal "$GOAL" \
  --model-kind "$MODEL_KIND" \
  --seed "$SCENE_SEED"

cat "$loc/domain.pddl" "$loc/problem.pddl" > "$loc/temp.pddl"

./mdpsim/mdpsim \
  --port=2322 \
  -R 100 \
  --time-limit=10000 \
  "$loc/temp.pddl" &
server_pid=$!

cleanup() {
  kill "$server_pid" 2>/dev/null || true
  rm -rf logs
}
trap cleanup EXIT

./mini-gpt/planner \
  -v 100 \
  -h ff \
  localhost:2322 \
  "$loc/temp.pddl" \
  dom1 > "$loc/planresult.txt"

"$PYTHON_BIN" parse_plan.py -opts "$OPTS"

cat "$loc/plan.txt" >> "$loc/objects.txt"
rm -f "$loc/planresult.txt" "$loc/plan.txt"
mv "$loc/objects.txt" "$loc/plan.txt"

echo "Plan written to: $loc/plan.txt"
cat "$loc/plan.txt"