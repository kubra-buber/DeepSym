#!/usr/bin/env bash
set -euo pipefail

# Resolve this script's directory BEFORE changing the working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_ROOT="${1:-$HOME/DeepSym/railroad_source}"
RR_PY="${RR_PY:-/home/kubra/miniconda3/envs/deepsym_railroad/bin/python}"
PATCH_SCRIPT="$SCRIPT_DIR/patch_railroad_mcts_correctness.py"

if [[ ! -f "$PATCH_SCRIPT" ]]; then
  echo "ERROR: Patch script not found: $PATCH_SCRIPT" >&2
  echo "Keep rebuild_railroad_mcts.sh and patch_railroad_mcts_correctness.py in the same directory." >&2
  exit 1
fi

if [[ ! -d "$SOURCE_ROOT/packages/railroad" ]]; then
  echo "ERROR: Railroad source directory not found: $SOURCE_ROOT/packages/railroad" >&2
  exit 1
fi

"$RR_PY" "$PATCH_SCRIPT" \
  --source-root "$SOURCE_ROOT"

cd "$SOURCE_ROOT"

"$RR_PY" -m pip install pybind11 pybind11-stubgen

rm -rf packages/railroad/build
find packages/railroad/src/railroad \
  -maxdepth 1 -type f -name '_bindings*.so' -delete

"$RR_PY" -m pip install \
  -e packages/railroad \
  --no-build-isolation \
  --no-cache-dir

"$RR_PY" - <<'PY'
import railroad
import railroad.planner
import railroad._bindings

print("Railroad package paths:")
for path in railroad.__path__:
    print(" ", path)
print("planner:", railroad.planner.__file__)
print("bindings:", railroad._bindings.__file__)
PY

echo
echo "Source checks:"
grep -n "HEURISTIC_CANNOT_FIND_GOAL_PENALTY" \
  packages/railroad/include/railroad/constants.hpp
grep -n "const auto &all_actions = all_actions_base" \
  packages/railroad/include/railroad/planner.hpp
grep -n -C 2 "Goal states are terminal" \
  packages/railroad/include/railroad/planner.hpp
grep -n -C 1 "depth < max_depth &&" \
  packages/railroad/include/railroad/planner.hpp

echo
echo "Railroad MCTS rebuilt successfully."
