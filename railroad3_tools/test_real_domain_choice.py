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
        "Test exact planning and patched Railroad MCTS on a problem "
        "generated from the full real probabilistic domain."
    )
    parser.add_argument(
        "--case-dir",
        default="railroad3_real_choice",
    )
    parser.add_argument("--prob-domain", default="domain_prob.pddl")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--min-mcts-rate", type=float, default=0.80)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    manifest = json.loads(
        (case_dir / "real_choice_manifest.json").read_text()
    )

    domain = parse_domain(Path(args.prob_domain))
    parsed_problem = parse_problem(
        case_dir / manifest["problem"]
    )
    rr_problem = build_problem(parsed_problem)

    built = build_operators(domain)
    actions = ground_operators(
        built,
        rr_problem.objects_by_type,
    )

    expected_action = str(manifest["expected_action"])
    expected_value = float(manifest["expected_value"])
    horizon = int(manifest["horizon"])

    print("=== REAL-DOMAIN CONTROLLED PROBLEM ===")
    print(f"Goal: {manifest['goal']}")
    print(f"Grounded actions: {len(actions)}")
    print(f"Expected action: {expected_action}")
    print(f"Expected value: {expected_value:.8f}")

    print("\n=== EXACT EXPECTED REACHABILITY ===")
    exact = ExactExpectedReachabilityPlanner(actions)
    selected, value, root_values = exact.solve(
        rr_problem.initial_state,
        rr_problem.goal,
        horizon=horizon,
    )

    for name, probability in sorted(
        root_values.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        if probability > 0.0:
            print(f"{name}: V={probability:.8f}")

    print(f"Selected action: {selected}")
    print(f"Goal probability: {value:.8f}")

    if selected != expected_action:
        raise AssertionError(
            f"Exact planner selected {selected!r}; "
            f"expected {expected_action!r}"
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

    print(
        "PASS: exact planner reproduced the best action in the "
        "full real domain."
    )

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

    for name, count in selections.most_common():
        print(f"{name!r}: {count}/{args.runs}")

    correct = selections.get(expected_action, 0)
    rate = correct / args.runs
    print(f"MCTS correct-selection rate: {rate:.3f}")

    if args.show_trace and last_planner is not None:
        print("\nLast MCTS trace:")
        print(last_planner.get_trace_from_last_mcts_tree())

    if rate < args.min_mcts_rate:
        raise AssertionError(
            f"MCTS selected {expected_action!r} only "
            f"{correct}/{args.runs} times"
        )

    print(
        "PASS: patched Railroad MCTS selected the best real-domain "
        "action at the required rate."
    )


if __name__ == "__main__":
    main()