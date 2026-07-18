#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators

from railroad.core import transition
from railroad.planner import MCTSPlanner, get_usable_actions


def state_key(state) -> Tuple[str, ...]:
    return tuple(sorted(str(fluent) for fluent in state.fluents))


def merge_outcomes(outcomes):
    merged: Dict[Tuple[str, ...], List] = {}
    for next_state, probability in outcomes:
        key = state_key(next_state)
        if key in merged:
            merged[key][1] += float(probability)
        else:
            merged[key] = [next_state, float(probability)]

    total = sum(probability for _, probability in merged.values())
    return [
        (state, probability / total)
        for state, probability in merged.values()
    ]


def goal_probability(problem, outcomes) -> float:
    return sum(
        probability
        for state, probability in outcomes
        if problem.goal.evaluate(state.fluents)
    )


def load_case(case_dir: Path):
    manifest = json.loads((case_dir / "choice_manifest.json").read_text())
    domain = parse_domain(case_dir / manifest["domain"])
    parsed_problem = parse_problem(case_dir / manifest["problem"])
    rr_problem = build_problem(parsed_problem)
    built = build_operators(domain)
    actions = ground_operators(built, rr_problem.objects_by_type)
    return manifest, rr_problem, actions


def run_variant(
    name: str,
    actions,
    rr_problem,
    *,
    runs: int,
    iterations: int,
    max_depth: int,
    c: float,
    heuristic_multiplier: float,
    lambda_add: float,
    lambda_max: float,
    lambda_ff: float,
    show_trace: bool,
):
    print(f"\n=== {name} ===")
    print("Action order:")
    for index, action in enumerate(actions):
        print(f"  {index}: {action.name}")

    selections = Counter()
    last_planner = None

    for _ in range(runs):
        planner = MCTSPlanner(
            actions,
            lambda_add=lambda_add,
            lambda_max=lambda_max,
            lambda_ff=lambda_ff,
        )
        selected = planner(
            rr_problem.initial_state,
            rr_problem.goal,
            max_iterations=iterations,
            max_depth=max_depth,
            c=c,
            heuristic_multiplier=heuristic_multiplier,
        )
        selections[str(selected)] += 1
        last_planner = planner

    print("Selections:")
    for selected, count in selections.most_common():
        print(f"  {selected!r}: {count}/{runs}")

    if show_trace and last_planner is not None:
        print("Last trace:")
        try:
            print(last_planner.get_trace_from_last_mcts_tree())
        except Exception as exc:
            print(f"  trace unavailable: {exc}")


def main():
    parser = argparse.ArgumentParser(
        "Diagnose Railroad MCTS choice on a 0.8-vs-0.2 controlled domain."
    )
    parser.add_argument("--case-dir", default="railroad3_choice")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    manifest, rr_problem, actions = load_case(Path(args.case_dir))

    print("=== EXACT TRANSITION CHECK ===")
    for action in actions:
        outcomes = merge_outcomes(transition(rr_problem.initial_state, action))
        print(
            f"{action.name}: "
            f"P(goal)={goal_probability(rr_problem, outcomes):.8f}"
        )
        for state, probability in sorted(
            outcomes, key=lambda item: -item[1]
        ):
            print(
                f"  p={probability:.8f}, "
                f"goal={rr_problem.goal.evaluate(state.fluents)}, "
                f"state={state_key(state)}"
            )

    variants = [
        {
            "name": "default / original order",
            "actions": list(actions),
            "heuristic_multiplier": 5.0,
            "lambda_add": 0.5,
            "lambda_max": 0.0,
            "lambda_ff": 0.5,
        },
        {
            "name": "default / reversed action order",
            "actions": list(reversed(actions)),
            "heuristic_multiplier": 5.0,
            "lambda_add": 0.5,
            "lambda_max": 0.0,
            "lambda_ff": 0.5,
        },
        {
            "name": "zero heuristic multiplier",
            "actions": list(actions),
            "heuristic_multiplier": 0.0,
            "lambda_add": 0.5,
            "lambda_max": 0.0,
            "lambda_ff": 0.5,
        },
        {
            "name": "h_add only",
            "actions": list(actions),
            "heuristic_multiplier": 5.0,
            "lambda_add": 1.0,
            "lambda_max": 0.0,
            "lambda_ff": 0.0,
        },
        {
            "name": "h_ff only",
            "actions": list(actions),
            "heuristic_multiplier": 5.0,
            "lambda_add": 0.0,
            "lambda_max": 0.0,
            "lambda_ff": 1.0,
        },
    ]

    for variant in variants:
        run_variant(
            variant["name"],
            variant["actions"],
            rr_problem,
            runs=args.runs,
            iterations=args.iterations,
            max_depth=args.max_depth,
            c=1.414,
            heuristic_multiplier=variant["heuristic_multiplier"],
            lambda_add=variant["lambda_add"],
            lambda_max=variant["lambda_max"],
            lambda_ff=variant["lambda_ff"],
            show_trace=args.show_trace,
        )

    print(
        "\nInterpretation: if reversing the action order reverses the selected "
        "action, the two actions are tied internally and selection is using "
        "an order-dependent tie break. If all variants still select the "
        "0.2 action, inspect the installed Railroad 0.2.0 MCTS implementation "
        "before relying on it for success-probability optimization."
    )


if __name__ == "__main__":
    main()