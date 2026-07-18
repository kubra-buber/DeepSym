#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
    if total <= 0.0:
        raise ValueError("Transition returned no positive probability mass")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        "Check whether Railroad MCTS prefers the higher-success action."
    )
    parser.add_argument("--case-dir", default="railroad3_choice")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    manifest = json.loads((case_dir / "choice_manifest.json").read_text())

    domain = parse_domain(case_dir / manifest["domain"])
    parsed_problem = parse_problem(case_dir / manifest["problem"])
    rr_problem = build_problem(parsed_problem)
    built = build_operators(domain)
    actions = ground_operators(built, rr_problem.objects_by_type)

    usable = get_usable_actions(rr_problem.initial_state, actions)
    applicable = []

    print("=== EXACT ONE-STEP TRANSITIONS ===")
    for action in sorted(usable, key=lambda item: item.name):
        try:
            outcomes = merge_outcomes(
                transition(rr_problem.initial_state, action)
            )
        except Exception as exc:
            if "precondition" in str(exc).lower():
                continue
            raise

        applicable.append(action)
        actual_probability = goal_probability(rr_problem, outcomes)
        expected_probability = float(
            manifest["goal_probabilities"][action.name]
        )

        print(
            f"{action.name}: goal probability "
            f"{actual_probability:.8f}"
        )

        if not math.isclose(
            actual_probability,
            expected_probability,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise AssertionError(
                f"{action.name}: expected {expected_probability}, "
                f"got {actual_probability}"
            )

    if len(applicable) != 2:
        raise AssertionError(
            f"Expected two applicable actions, got "
            f"{[action.name for action in applicable]}"
        )

    print("PASS: both PPDDL action probabilities were preserved.")

    print("\n=== MCTS ACTION CHOICE ===")
    selections = Counter()
    last_planner = None

    for _ in range(args.runs):
        planner = MCTSPlanner(actions)
        selected = planner(
            rr_problem.initial_state,
            rr_problem.goal,
            max_iterations=args.iterations,
            max_depth=args.max_depth,
        )
        selections[str(selected)] += 1
        last_planner = planner

    for action_name, count in selections.most_common():
        print(f"{action_name!r}: {count}/{args.runs}")

    expected_action = str(manifest["expected_action"])
    correct = selections.get(expected_action, 0)

    if args.show_trace and last_planner is not None:
        print("\nLast MCTS trace:")
        print(last_planner.get_trace_from_last_mcts_tree())

    if correct != args.runs:
        raise AssertionError(
            f"Expected {expected_action!r} in every run, but it was selected "
            f"{correct}/{args.runs}. All selections: {dict(selections)}"
        )

    print(
        "PASS: Railroad MCTS consistently preferred the action with "
        "0.80 goal probability over the action with 0.20."
    )


if __name__ == "__main__":
    main()