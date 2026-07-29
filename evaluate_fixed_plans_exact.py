#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import yaml
from railroad.planner import get_usable_actions

from make_plan_railroad_expected import (
    build_initial_state,
    load_operators_from_json,
    parse_deepsym_goal,
    parse_problem_railroad,
    progress_outcomes_for_plan,
    state_key,
    transition_safe,
)

PhysicalPair = Tuple[str, str]  # (below, above)


def normalize_plan(text: str) -> str:
    return " | ".join(part.strip() for part in str(text).split("|") if part.strip())


def parse_physical_plan(text: str) -> List[PhysicalPair]:
    pairs: List[PhysicalPair] = []
    normalized = normalize_plan(text)
    if not normalized:
        return pairs

    for part in normalized.split(" | "):
        match = re.fullmatch(r"(O\d+)\s+on\s+(O\d+)", part.strip(), flags=re.I)
        if not match:
            raise ValueError(f"Cannot parse physical-plan step: {part!r}")
        above = match.group(1).upper()
        below = match.group(2).upper()
        pairs.append((below, above))
    return pairs


def physical_plan_text(pairs: Sequence[PhysicalPair]) -> str:
    return " | ".join(f"{above} on {below}" for below, above in pairs)


def read_expected_plan(path: Path) -> Tuple[float, str]:
    probability = None
    pairs: List[PhysicalPair] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.lower().startswith("plan probability:"):
            probability = float(line.split(":", 1)[1].strip())
            continue

        parts = line.split()
        if len(parts) == 3 and parts[0].lower() == "stack":
            below, above = parts[1].upper(), parts[2].upper()
            pairs.append((below, above))

    if probability is None:
        raise RuntimeError(f"No plan probability found in {path}")

    return probability, physical_plan_text(pairs)


def action_kind(name: str):
    parts = name.split()
    if len(parts) == 2 and parts[0] == "makebase":
        return "makebase", parts[1].upper()
    if len(parts) == 3 and parts[0].startswith("stack"):
        return "stack", (parts[1].upper(), parts[2].upper())
    if parts and (parts[0].startswith("increase_height") or parts[0].startswith("increase_stack")):
        return "aux", None
    return "other", None


def fixed_physical_plan_probability(
    initial_state,
    goal,
    all_actions: Sequence,
    physical_plan: str,
    max_steps: int,
) -> float:
    """Maximum probability progress branch consistent with one physical plan.

    This uses the same linear-plan semantics as maxprob-linear: failure branches
    are plan failure, while deterministic bookkeeping actions may be inserted in
    whichever applicable order yields the best valid symbolic path.
    """
    pairs = parse_physical_plan(physical_plan)
    if not pairs:
        return 1.0 if goal.evaluate(initial_state.fluents) else 0.0

    base_object = pairs[0][0]
    sorted_actions = sorted(all_actions, key=lambda action: action.name)
    state_store = {state_key(initial_state): initial_state}

    # Node: (state key, number of physical stack actions consumed, symbolic depth)
    start_key = state_key(initial_state)
    start_node = (start_key, 0, 0)
    best_cost: Dict[Tuple[Tuple[str, ...], int, int], float] = {start_node: 0.0}
    queue: List[Tuple[float, int, Tuple[str, ...], int, int]] = []
    push_counter = 0
    heapq.heappush(queue, (0.0, push_counter, start_key, 0, 0))

    while queue:
        cost, _seq, key, physical_index, depth = heapq.heappop(queue)
        node = (key, physical_index, depth)
        if cost > best_cost.get(node, float("inf")) + 1e-12:
            continue

        state = state_store[key]
        if physical_index == len(pairs) and goal.evaluate(state.fluents):
            return math.exp(-cost)
        if depth >= max_steps:
            continue

        for action in sorted(get_usable_actions(state, sorted_actions), key=lambda item: item.name):
            kind, payload = action_kind(action.name)
            next_physical_index = physical_index

            if kind == "makebase":
                if payload != base_object or physical_index != 0:
                    continue
            elif kind == "stack":
                if physical_index >= len(pairs) or payload != pairs[physical_index]:
                    continue
                next_physical_index += 1
            elif kind == "aux":
                pass
            else:
                continue

            outcomes = transition_safe(state, action)
            if not outcomes:
                continue
            if kind == "stack":
                outcomes = progress_outcomes_for_plan(action.name, outcomes)
            if not outcomes:
                continue

            for next_state, probability in outcomes:
                probability = float(probability)
                if probability <= 0.0:
                    continue

                next_key = state_key(next_state)
                state_store.setdefault(next_key, next_state)
                next_depth = depth + 1
                next_node = (next_key, next_physical_index, next_depth)
                next_cost = cost - math.log(max(probability, 1e-15))

                if next_cost + 1e-12 < best_cost.get(next_node, float("inf")):
                    best_cost[next_node] = next_cost
                    push_counter += 1
                    heapq.heappush(
                        queue,
                        (next_cost, push_counter, next_key, next_physical_index, next_depth),
                    )

    return 0.0


def read_frequency(path: Path) -> List[dict]:
    if not path.exists():
        return []

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    result = []
    for row in rows:
        plan = row.get("physical_plan") or row.get("plan")
        count = row.get("count")
        if plan is None or count is None:
            continue
        result.append({"physical_plan": normalize_plan(plan), "count": int(float(count))})
    return sorted(result, key=lambda item: (-item["count"], item["physical_plan"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opts", required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--expected-plan", required=True)
    parser.add_argument("--plan-frequency", required=True)
    parser.add_argument("--requested-runs", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = yaml.safe_load(Path(args.opts).read_text())
    save_dir = Path(opts["save"])

    operators = load_operators_from_json(save_dir / "railroad_operators.json")
    objects, obj_types, relations, counters, _object_infos = parse_problem_railroad(
        save_dir / "railroad_problem.json",
        save_dir / "objects.txt",
    )
    initial_state, objects_by_type = build_initial_state(objects, obj_types, relations, counters)

    all_actions = []
    for operator in operators:
        all_actions.extend(operator.instantiate(objects_by_type))
    all_actions.sort(key=lambda action: action.name)

    goal = parse_deepsym_goal(args.goal)
    expected_file_probability, expected_physical_plan = read_expected_plan(Path(args.expected_plan))
    precise_optimum_probability = fixed_physical_plan_probability(
        initial_state,
        goal,
        all_actions,
        expected_physical_plan,
        args.max_steps,
    )

    # Use the precise constrained evaluation. The plan.txt probability is printed
    # to six decimals and is retained for diagnostics.
    optimum_probability = precise_optimum_probability
    if optimum_probability <= 0.0 and expected_file_probability > 0.0:
        raise RuntimeError(
            "Expected plan could not be re-evaluated under fixed-plan semantics: "
            f"{expected_physical_plan}"
        )

    frequencies = read_frequency(Path(args.plan_frequency))
    successful_runs = sum(item["count"] for item in frequencies)
    failed_runs = max(0, args.requested_runs - successful_runs)

    plan_rows = []
    all_probabilities: List[float] = []
    tolerance = max(1e-9, optimum_probability * 1e-7)

    for item in frequencies:
        probability = fixed_physical_plan_probability(
            initial_state,
            goal,
            all_actions,
            item["physical_plan"],
            args.max_steps,
        )
        regret = max(0.0, optimum_probability - probability)
        ratio = probability / optimum_probability if optimum_probability > 0.0 else 0.0
        is_optimal = probability >= optimum_probability - tolerance
        within_95 = probability + tolerance >= 0.95 * optimum_probability

        plan_rows.append({
            "physical_plan": item["physical_plan"],
            "count": item["count"],
            "exact_linear_probability": probability,
            "regret": regret,
            "relative_to_optimum": ratio,
            "is_optimal": int(is_optimal),
            "within_95_percent": int(within_95),
        })
        all_probabilities.extend([probability] * item["count"])

    # A planner failure has exact plan probability zero.
    all_probabilities.extend([0.0] * failed_runs)
    all_regrets = [max(0.0, optimum_probability - value) for value in all_probabilities]

    optimal_count = sum(
        row["count"] for row in plan_rows if bool(row["is_optimal"])
    )
    within_95_count = sum(
        row["count"] for row in plan_rows if bool(row["within_95_percent"])
    )

    successful_probabilities = [
        value
        for row in plan_rows
        for value in [row["exact_linear_probability"]] * row["count"]
    ]

    metrics = {
        "method": args.method,
        "goal": args.goal,
        "requested_runs": args.requested_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "planner_failure_rate": failed_runs / args.requested_runs if args.requested_runs else 0.0,
        "expected_plan_file_probability": expected_file_probability,
        "optimum_exact_linear_probability": optimum_probability,
        "optimum_physical_plan": expected_physical_plan,
        "mean_exact_plan_probability": statistics.mean(all_probabilities) if all_probabilities else 0.0,
        "mean_exact_plan_probability_successful_only": (
            statistics.mean(successful_probabilities) if successful_probabilities else 0.0
        ),
        "mean_regret": statistics.mean(all_regrets) if all_regrets else optimum_probability,
        "median_regret": statistics.median(all_regrets) if all_regrets else optimum_probability,
        "optimal_count": optimal_count,
        "optimal_plan_rate": optimal_count / args.requested_runs if args.requested_runs else 0.0,
        "within_95_count": within_95_count,
        "within_95_percent_rate": within_95_count / args.requested_runs if args.requested_runs else 0.0,
        "unique_successful_plans": len(plan_rows),
    }

    with (output_dir / "exact_plan_values.csv").open("w", newline="") as handle:
        fields = [
            "physical_plan",
            "count",
            "exact_linear_probability",
            "regret",
            "relative_to_optimum",
            "is_optimal",
            "within_95_percent",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(plan_rows)

    (output_dir / "exact_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("=" * 110)
    print(f"METHOD: {args.method}")
    print(f"GOAL: {args.goal}")
    print(f"OPTIMUM: {optimum_probability:.9f}  {expected_physical_plan}")
    print(f"SUCCESS/FAILURE: {successful_runs}/{failed_runs} of {args.requested_runs}")
    print("-" * 110)
    print(f"Mean exact plan probability       : {metrics['mean_exact_plan_probability']:.9f}")
    print(f"Mean exact probability (success) : {metrics['mean_exact_plan_probability_successful_only']:.9f}")
    print(f"Mean regret                       : {metrics['mean_regret']:.9f}")
    print(f"Median regret                     : {metrics['median_regret']:.9f}")
    print(f"Optimal-plan rate                 : {optimal_count}/{args.requested_runs} ({100*metrics['optimal_plan_rate']:.1f}%)")
    print(f"Within 95% of optimum             : {within_95_count}/{args.requested_runs} ({100*metrics['within_95_percent_rate']:.1f}%)")
    print(f"Planner failure rate              : {failed_runs}/{args.requested_runs} ({100*metrics['planner_failure_rate']:.1f}%)")
    print("-" * 110)
    for row in plan_rows:
        print(
            f"{row['count']:3d}x  p={row['exact_linear_probability']:.9f}  "
            f"regret={row['regret']:.9f}  ratio={row['relative_to_optimum']:.4f}  "
            f"{row['physical_plan']}"
        )


if __name__ == "__main__":
    main()