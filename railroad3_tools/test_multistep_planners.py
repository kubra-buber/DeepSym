#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators
from generic_expected_planner import ExactExpectedReachabilityPlanner

from railroad.planner import MCTSPlanner


def main():
    parser = argparse.ArgumentParser(
        "Compare exact expected reachability and Railroad MCTS."
    )
    parser.add_argument(
        "--case-dir",
        default="railroad3_multistep",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-mcts-rate", type=float, default=0.80)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    manifest = json.loads(
        (case_dir / "multistep_manifest.json").read_text()
    )

    domain = parse_domain(case_dir / manifest["domain"])
    parsed_problem = parse_problem(case_dir / manifest["problem"])
    rr_problem = build_problem(parsed_problem)

    built_operators = build_operators(domain)
    actions = ground_operators(
        built_operators,
        rr_problem.objects_by_type,
    )

    print("=== GROUNDED ACTIONS ===")
    for action in actions:
        print(f"  {action.name}")

    print("\n=== EXACT EXPECTED REACHABILITY ===")
    exact = ExactExpectedReachabilityPlanner(actions)
    selected, value, root_values = exact.solve(
        rr_problem.initial_state,
        rr_problem.goal,
        horizon=int(manifest["horizon"]),
    )

    for action_name, action_value in sorted(root_values.items()):
        print(f"{action_name}: V={action_value:.8f}")

    print(f"Selected action: {selected}")
    print(f"Goal probability: {value:.8f}")

    expected_action = str(manifest["expected_action"])
    expected_value = float(manifest["expected_value"])

    if selected != expected_action:
        raise AssertionError(
            f"Exact planner selected {selected!r}; expected "
            f"{expected_action!r}"
        )

    if not math.isclose(
        value,
        expected_value,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise AssertionError(
            f"Exact value {value} != expected {expected_value}"
        )

    for action_name, expected_action_value in (
        manifest["root_action_values"].items()
    ):
        actual = root_values.get(action_name)
        if actual is None or not math.isclose(
            actual,
            float(expected_action_value),
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise AssertionError(
                f"{action_name}: expected {expected_action_value}, "
                f"got {actual}"
            )

    print("PASS: exact finite-horizon planner chose the safer two-step path.")

    print("\n=== PATCHED RAILROAD MCTS ===")
    selections = Counter()
    last_planner = None

    for _ in range(args.runs):
        planner = MCTSPlanner(actions)
        selected_mcts = planner(
            rr_problem.initial_state,
            rr_problem.goal,
            max_iterations=args.iterations,
            max_depth=args.max_depth,
        )
        selections[str(selected_mcts)] += 1
        last_planner = planner

    for action_name, count in selections.most_common():
        print(f"{action_name!r}: {count}/{args.runs}")

    correct = selections.get(expected_action, 0)
    rate = correct / args.runs
    print(f"MCTS correct-selection rate: {rate:.3f}")

    if args.show_trace and last_planner is not None:
        print("\nLast MCTS trace:")
        print(last_planner.get_trace_from_last_mcts_tree())

    if rate < args.min_mcts_rate:
        raise AssertionError(
            f"MCTS selected {expected_action!r} only {correct}/"
            f"{args.runs} times; required rate is "
            f"{args.min_mcts_rate:.2f}"
        )

    print(
        "PASS: patched Railroad MCTS usually chose the safer "
        "higher-reachability policy."
    )


if __name__ == "__main__":
    main()