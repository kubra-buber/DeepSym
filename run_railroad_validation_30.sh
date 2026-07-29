#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/DeepSym"

RR_PY=/home/kubra/miniconda3/envs/deepsym_railroad/bin/python
RECOG_PY="${RECOG_PY:-python}"

ROOT="experiments/railroad_mcts/validation_30runs"
RUNS=30
ITERATIONS=10000
MAX_DEPTH=25
MAX_SYMBOLIC_STEPS=25
LOW_PROB_THRESHOLD=0.80

mkdir -p "$ROOT/scenes"

# recognize.py deney sonunda eski haline döndürülsün.
RECOGNIZE_BACKUP="$ROOT/recognize.py.before_validation"
cp recognize.py "$RECOGNIZE_BACKUP"

restore_recognize() {
    cp "$RECOGNIZE_BACKUP" recognize.py
}
trap restore_recognize EXIT


# Format:
# label|seed|goal
SCENARIOS=(
    "seed33_H4|33|(H4)"
    "seed33_H3_S4|33|(H3) (S4)"
    "seed31_H4|31|(H4)"
    "seed25_H1_S2|25|(H1) (S2)"
    "seed25_H1_S3|25|(H1) (S3)"
    "seed24_S3|24|(S3)"
    "seed23_H1_S2|23|(H1) (S2)"
)


set_recognize_seed() {
    local seed="$1"

    "$RECOG_PY" - "$seed" <<'PY'
from pathlib import Path
import re
import sys

seed = int(sys.argv[1])
path = Path("recognize.py")
text = path.read_text()

new_text, count = re.subn(
    r"np\.random\.seed\(\s*\d+\s*\)",
    f"np.random.seed({seed})",
    text,
    count=1,
)

if count != 1:
    raise SystemExit(
        "ERROR: recognize.py içinde tek bir np.random.seed(...) "
        f"satırı bulunamadı. Eşleşme sayısı: {count}"
    )

path.write_text(new_text)
print(f"recognize.py seed set to {seed}")
PY
}


activate_or_create_scene() {
    local seed="$1"
    local goal="$2"

    local scene_cache="$ROOT/scenes/seed${seed}"
    mkdir -p "$scene_cache"

    if [[ -s "$scene_cache/railroad_problem.json" &&
          -s "$scene_cache/objects.txt" &&
          -s "$scene_cache/railroad_operators.json" ]]; then

        echo "Using cached scene for seed $seed"

        cp "$scene_cache/railroad_problem.json" \
           save/stable1/railroad_problem.json

        cp "$scene_cache/objects.txt" \
           save/stable1/objects.txt

        cp "$scene_cache/railroad_operators.json" \
           save/stable1/railroad_operators.json

        return
    fi

    echo "Generating new scene for seed $seed"

    set_recognize_seed "$seed"

    "$RECOG_PY" recognize.py \
        -opts opts.yaml \
        -goal "$goal" \
        -device cpu \
        2>&1 | tee "$scene_cache/recognize.log"

    test -s save/stable1/railroad_problem.json
    test -s save/stable1/objects.txt
    test -s save/stable1/railroad_operators.json

    cp save/stable1/railroad_problem.json \
       "$scene_cache/railroad_problem.json"

    cp save/stable1/objects.txt \
       "$scene_cache/objects.txt"

    cp save/stable1/railroad_operators.json \
       "$scene_cache/railroad_operators.json"

    cp recognize.py "$scene_cache/recognize_seed${seed}.py"

    sha256sum \
        "$scene_cache/railroad_problem.json" \
        "$scene_cache/objects.txt" \
        "$scene_cache/railroad_operators.json" \
        > "$scene_cache/SHA256SUMS"
}


run_expected() {
    local scenario_dir="$1"
    local goal="$2"

    local out="$scenario_dir/expected"
    mkdir -p "$out"

    if [[ -s "$out/plan.txt" && -s "$out/expected.log" ]]; then
        echo "Expected result already exists; skipping."
        return
    fi

    echo "Running expected planner: goal=$goal"

    "$RR_PY" make_plan_railroad_expected.py \
        -opts opts.yaml \
        -goal "$goal" \
        -max-steps "$MAX_SYMBOLIC_STEPS" \
        --plan-output maxprob-linear \
        --rollout-outcome progress \
        --debug-actions \
        2>&1 | tee "$out/expected.log"

    cp save/stable1/plan.txt "$out/plan.txt"
}


run_mcts_method() {
    local scenario_dir="$1"
    local goal="$2"
    local method="$3"
    local c="$4"
    local hm="$5"
    local ladd="$6"
    local lmax="$7"
    local lff="$8"

    local out="$scenario_dir/mcts/$method"
    mkdir -p "$out"

    if [[ -s "$out/summary.json" &&
          -s "$out/plan_frequency.csv" &&
          -s "$out/runs.csv" ]]; then
        echo "$method already exists; skipping."
        return
    fi

    echo
    echo "Running $method — goal=$goal, runs=$RUNS"

    "$RR_PY" run_mcts_100_plans.py \
        -opts opts.yaml \
        -goal "$goal" \
        --runs "$RUNS" \
        --iterations "$ITERATIONS" \
        --max-depth "$MAX_DEPTH" \
        --max-symbolic-steps "$MAX_SYMBOLIC_STEPS" \
        --c "$c" \
        --heuristic-multiplier "$hm" \
        --lambda-add "$ladd" \
        --lambda-max "$lmax" \
        --lambda-ff "$lff" \
        --rollout-outcome progress \
        --output-dir "$out"
}


compare_scenario() {
    local scenario_dir="$1"

    SCENARIO_DIR="$scenario_dir" \
    RUNS="$RUNS" \
    LOW_PROB_THRESHOLD="$LOW_PROB_THRESHOLD" \
    "$RR_PY" - <<'PY' \
        2>&1 | tee "$scenario_dir/comparison.txt"

import csv
import json
import os
from pathlib import Path

scenario = Path(os.environ["SCENARIO_DIR"])
expected_runs = int(os.environ["RUNS"])
low_threshold = float(os.environ["LOW_PROB_THRESHOLD"])

methods = [
    ("default", "Default"),
    ("ff_only", "FF-only"),
    ("max_only", "Max-only"),
    ("max_ff", "Max+FF"),
]


def normalize(plan):
    return " | ".join(
        part.strip() for part in str(plan).split("|")
    )


def read_expected_plan(path):
    physical = []

    for line in path.read_text().splitlines():
        parts = line.strip().split()

        # plan.txt format:
        # stack BELOW ABOVE
        if len(parts) == 3 and parts[0].lower() == "stack":
            below, above = parts[1], parts[2]
            physical.append(f"{above} on {below}")

    return normalize(" | ".join(physical))


def read_frequency(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    parsed = []

    for row in rows:
        plan = row.get("physical_plan") or row.get("plan")
        count = row.get("count")

        parsed.append({
            "plan": normalize(plan),
            "count": int(float(count)),
        })

    return sorted(parsed, key=lambda item: -item["count"])


def read_probabilities(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    probability_key = next(
        (
            field for field in fields
            if "representative_branch_probability" in field
        ),
        None,
    )

    if probability_key is None:
        probability_key = next(
            (
                field for field in fields
                if "probability" in field.lower()
            ),
            None,
        )

    if probability_key is None:
        raise RuntimeError(
            f"Probability column not found in {path}: {fields}"
        )

    values = []

    for row in rows:
        value = row.get(probability_key)
        if value not in ("", None):
            values.append(float(value))

    return values


expected_path = scenario / "expected" / "plan.txt"
expected_plan = read_expected_plan(expected_path)

print("=" * 120)
print("SCENARIO:", scenario.name)
print("=" * 120)
print("EXPECTED MAX-PROBABILITY LINEAR PLAN:")
print(" ", expected_plan or "<no physical actions>")

print(
    f"\nLow-probability definition: "
    f"representative p < {low_threshold:.2f}"
)

print("\n" + "=" * 120)
print(
    f"{'Method':12s}"
    f"{'Runs':>7s}"
    f"{'Unique':>9s}"
    f"{'Mean p':>11s}"
    f"{'Top-3':>16s}"
    f"{'Low-p':>16s}"
    f"{'Expected':>16s}"
)
print("=" * 120)

all_results = []

for folder, display in methods:
    base = scenario / "mcts" / folder

    summary = json.load((base / "summary.json").open())
    frequencies = read_frequency(base / "plan_frequency.csv")
    probabilities = read_probabilities(base / "runs.csv")

    total = sum(item["count"] for item in frequencies)
    unique = len(frequencies)
    top3 = sum(item["count"] for item in frequencies[:3])
    low = sum(value < low_threshold for value in probabilities)
    expected_count = sum(
        item["count"]
        for item in frequencies
        if item["plan"] == expected_plan
    )

    mean_probability = float(
        summary["mean_representative_branch_probability"]
    )

    all_results.append({
        "folder": folder,
        "display": display,
        "total": total,
        "unique": unique,
        "top3": top3,
        "low": low,
        "probability_count": len(probabilities),
        "expected_count": expected_count,
        "mean_probability": mean_probability,
        "frequencies": frequencies,
    })

    print(
        f"{display:12s}"
        f"{total:7d}"
        f"{unique:9d}"
        f"{mean_probability:11.4f}"
        f"{top3:6d}/{total:<3d}"
        f" {100 * top3 / total:5.1f}%"
        f"{low:6d}/{len(probabilities):<3d}"
        f" {100 * low / len(probabilities):5.1f}%"
        f"{expected_count:6d}/{total:<3d}"
        f" {100 * expected_count / total:5.1f}%"
    )


for result in all_results:
    print("\n" + "-" * 120)
    print(result["display"])
    print("-" * 120)

    for rank, item in enumerate(result["frequencies"], start=1):
        marker = (
            "  <-- EXPECTED"
            if item["plan"] == expected_plan
            else ""
        )

        print(
            f"{rank:2d}. "
            f"{item['count']:3d}/{result['total']} "
            f"({100 * item['count'] / result['total']:5.1f}%)  "
            f"{item['plan']}{marker}"
        )
PY
}


for scenario_spec in "${SCENARIOS[@]}"; do
    IFS='|' read -r label seed goal <<< "$scenario_spec"

    echo
    echo "################################################################################"
    echo "SCENARIO: $label"
    echo "SEED:     $seed"
    echo "GOAL:     $goal"
    echo "################################################################################"

    scenario_dir="$ROOT/$label"
    mkdir -p "$scenario_dir/scene" "$scenario_dir/mcts"

    activate_or_create_scene "$seed" "$goal"

    # Aktif sahneyi bu deney klasörüne de dondur.
    cp save/stable1/railroad_problem.json \
       "$scenario_dir/scene/railroad_problem.json"

    cp save/stable1/objects.txt \
       "$scenario_dir/scene/objects.txt"

    cp save/stable1/railroad_operators.json \
       "$scenario_dir/scene/railroad_operators.json"

    cp opts.yaml "$scenario_dir/scene/opts.yaml"

    {
        echo "label=$label"
        echo "seed=$seed"
        echo "goal=$goal"
        echo "runs=$RUNS"
        echo "iterations=$ITERATIONS"
    } > "$scenario_dir/experiment_info.txt"

    run_expected "$scenario_dir" "$goal"

    # Original/default Railroad
    run_mcts_method \
        "$scenario_dir" "$goal" \
        default \
        1.41421356237 5 \
        0.5 0 0.5

    # FF-only
    run_mcts_method \
        "$scenario_dir" "$goal" \
        ff_only \
        1.41421356237 5 \
        0 0 1

    # Tuned max-only
    run_mcts_method \
        "$scenario_dir" "$goal" \
        max_only \
        3 1 \
        0 1 0

    # Tuned max + FF
    run_mcts_method \
        "$scenario_dir" "$goal" \
        max_ff \
        5 1 \
        0 0.5 0.5

    compare_scenario "$scenario_dir"
done


ROOT_DIR="$ROOT" \
LOW_PROB_THRESHOLD="$LOW_PROB_THRESHOLD" \
"$RR_PY" - <<'PY' \
    2>&1 | tee "$ROOT/aggregate_summary.txt"

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
low_threshold = float(os.environ["LOW_PROB_THRESHOLD"])

methods = [
    ("default", "Default"),
    ("ff_only", "FF-only"),
    ("max_only", "Max-only"),
    ("max_ff", "Max+FF"),
]


def normalize(plan):
    return " | ".join(
        part.strip() for part in str(plan).split("|")
    )


def expected_plan(path):
    actions = []

    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0].lower() == "stack":
            below, above = parts[1], parts[2]
            actions.append(f"{above} on {below}")

    return normalize(" | ".join(actions))


def frequencies(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    result = []

    for row in rows:
        result.append({
            "plan": normalize(
                row.get("physical_plan") or row.get("plan")
            ),
            "count": int(float(row["count"])),
        })

    return sorted(result, key=lambda item: -item["count"])


def probabilities(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    key = next(
        (
            field for field in fields
            if "representative_branch_probability" in field
        ),
        None,
    )

    if key is None:
        key = next(
            (
                field for field in fields
                if "probability" in field.lower()
            ),
            None,
        )

    return [
        float(row[key])
        for row in rows
        if row.get(key) not in ("", None)
    ]


totals = defaultdict(lambda: {
    "scenarios": 0,
    "runs": 0,
    "expected_matches": 0,
    "low": 0,
    "top3": 0,
    "unique_sum": 0,
    "mean_prob_sum": 0.0,
})

scenario_dirs = sorted(
    path
    for path in root.iterdir()
    if path.is_dir() and (path / "expected" / "plan.txt").exists()
)

for scenario in scenario_dirs:
    exact = expected_plan(scenario / "expected" / "plan.txt")

    for folder, display in methods:
        base = scenario / "mcts" / folder

        if not (base / "summary.json").exists():
            continue

        freq = frequencies(base / "plan_frequency.csv")
        probs = probabilities(base / "runs.csv")
        summary = json.load((base / "summary.json").open())

        total_runs = sum(item["count"] for item in freq)
        exact_count = sum(
            item["count"]
            for item in freq
            if item["plan"] == exact
        )

        row = totals[display]
        row["scenarios"] += 1
        row["runs"] += total_runs
        row["expected_matches"] += exact_count
        row["low"] += sum(p < low_threshold for p in probs)
        row["top3"] += sum(item["count"] for item in freq[:3])
        row["unique_sum"] += len(freq)
        row["mean_prob_sum"] += float(
            summary["mean_representative_branch_probability"]
        )


print("=" * 122)
print("AGGREGATE VALIDATION SUMMARY")
print("=" * 122)
print(
    f"{'Method':12s}"
    f"{'Scenes':>8s}"
    f"{'Runs':>8s}"
    f"{'Mean p':>12s}"
    f"{'Avg unique':>14s}"
    f"{'Top-3':>16s}"
    f"{'Low-p':>16s}"
    f"{'Expected':>16s}"
)
print("=" * 122)

for _, display in methods:
    row = totals[display]

    scenes = row["scenarios"]
    runs = row["runs"]

    mean_p = (
        row["mean_prob_sum"] / scenes
        if scenes else 0.0
    )
    avg_unique = (
        row["unique_sum"] / scenes
        if scenes else 0.0
    )

    print(
        f"{display:12s}"
        f"{scenes:8d}"
        f"{runs:8d}"
        f"{mean_p:12.4f}"
        f"{avg_unique:14.2f}"
        f"{row['top3']:7d}/{runs:<3d}"
        f" {100 * row['top3'] / runs:5.1f}%"
        f"{row['low']:7d}/{runs:<3d}"
        f" {100 * row['low'] / runs:5.1f}%"
        f"{row['expected_matches']:7d}/{runs:<3d}"
        f" {100 * row['expected_matches'] / runs:5.1f}%"
    )
PY

echo
echo "All experiments completed."
echo "Results: $ROOT"
echo "Aggregate summary: $ROOT/aggregate_summary.txt"
