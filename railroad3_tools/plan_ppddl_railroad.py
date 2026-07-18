#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators
from generic_expected_planner import ExactExpectedReachabilityPlanner
from railroad.planner import MCTSPlanner


def run_exact(actions, rr_problem, horizon: int, show_values: bool):
    planner = ExactExpectedReachabilityPlanner(actions)
    selected, value, root_values = planner.solve(
        rr_problem.initial_state,
        rr_problem.goal,
        horizon=horizon,
    )

    if show_values:
        print("Root action values:")
        for name, action_value in sorted(
            root_values.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            print(f"  {name}: {action_value:.8f}")

    print(f"Exact selected action: {selected}")
    print(f"Exact goal probability: {value:.8f}")

    return {
        "selected_action": selected,
        "goal_probability": value,
        "root_action_values": root_values,
    }


def run_mcts(
    actions,
    rr_problem,
    *,
    runs: int,
    iterations: int,
    max_depth: int,
    show_trace: bool,
):
    selections = Counter()
    last_planner = None

    for _ in range(runs):
        planner = MCTSPlanner(actions)
        selected = planner(
            rr_problem.initial_state,
            rr_problem.goal,
            max_iterations=iterations,
            max_depth=max_depth,
        )
        selections[str(selected)] += 1
        last_planner = planner

    print("MCTS selections:")
    for name, count in selections.most_common():
        print(f"  {name!r}: {count}/{runs}")

    selected_action = selections.most_common(1)[0][0] if selections else None
    print(f"MCTS modal action: {selected_action}")

    if show_trace and last_planner is not None:
        print("Last MCTS trace:")
        print(last_planner.get_trace_from_last_mcts_tree())

    return {
        "selected_action": selected_action,
        "selection_counts": dict(selections),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(
        "Generic PPDDL-to-Railroad planning entry point."
    )
    parser.add_argument("--domain", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument(
        "--planner",
        choices=("exact", "mcts", "both"),
        default="both",
    )
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--show-values", action="store_true")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    domain = parse_domain(Path(args.domain))
    parsed_problem = parse_problem(Path(args.problem))
    rr_problem = build_problem(parsed_problem)

    built_operators = build_operators(domain)
    actions = ground_operators(
        built_operators,
        rr_problem.objects_by_type,
    )

    print("=== GENERIC RAILROAD PROBLEM ===")
    print(f"Domain: {domain.name}")
    print(f"Problem: {parsed_problem.name}")
    print(f"Action schemas: {len(domain.actions)}")
    print(f"Grounded actions: {len(actions)}")
    print(f"Horizon: {args.horizon}")

    results = {
        "domain": domain.name,
        "problem": parsed_problem.name,
        "action_schemas": len(domain.actions),
        "grounded_actions": len(actions),
        "horizon": args.horizon,
    }

    if args.planner in ("exact", "both"):
        print("\n=== EXACT EXPECTED REACHABILITY ===")
        results["exact"] = run_exact(
            actions,
            rr_problem,
            args.horizon,
            args.show_values,
        )

    if args.planner in ("mcts", "both"):
        print("\n=== RAILROAD MCTS ===")
        results["mcts"] = run_mcts(
            actions,
            rr_problem,
            runs=args.runs,
            iterations=args.iterations,
            max_depth=args.max_depth,
            show_trace=args.show_trace,
        )

    if args.planner == "both":
        exact_action = results["exact"]["selected_action"]
        mcts_action = results["mcts"]["selected_action"]
        results["agreement"] = exact_action == mcts_action
        print(f"\nExact/MCTS agreement: {results['agreement']}")

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()