#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators

from railroad.core import transition
from railroad.planner import get_usable_actions


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


def load_case(case_dir: Path):
    manifest = json.loads((case_dir / "controlled_manifest.json").read_text())
    problem_path = case_dir / manifest["files"]["problem"]
    prob_domain_path = case_dir / manifest["files"]["probabilistic_domain"]
    nominal_domain_path = case_dir / manifest["files"]["nominal_domain"]
    return manifest, problem_path, prob_domain_path, nominal_domain_path


def build_and_ground(domain_path: Path, problem_path: Path):
    domain = parse_domain(domain_path)
    problem = parse_problem(problem_path)

    if problem.domain_name != domain.name:
        raise ValueError(
            f"Problem expects domain {problem.domain_name!r}, "
            f"but parsed domain is {domain.name!r}"
        )

    rr_problem = build_problem(problem)
    built_operators = build_operators(domain)
    actions = ground_operators(built_operators, rr_problem.objects_by_type)
    usable = get_usable_actions(rr_problem.initial_state, actions)

    # Railroad can be permissive in usable-action filtering, so transition()
    # remains the final applicability check.
    applicable = []
    for action in usable:
        try:
            outcomes = transition(rr_problem.initial_state, action)
        except Exception as exc:
            if "precondition" in str(exc).lower():
                continue
            raise
        applicable.append((action, merge_outcomes(outcomes)))

    return domain, problem, rr_problem, actions, applicable


def goal_probability(rr_problem, outcomes) -> float:
    return sum(
        probability
        for state, probability in outcomes
        if rr_problem.goal.evaluate(state.fluents)
    )


def print_outcomes(rr_problem, outcomes) -> None:
    for index, (state, probability) in enumerate(
        sorted(outcomes, key=lambda item: (-item[1], state_key(item[0])))
    ):
        reaches_goal = bool(rr_problem.goal.evaluate(state.fluents))
        print(
            f"  outcome {index}: p={probability:.8f}, "
            f"goal={reaches_goal}, state={state_key(state)}"
        )


def validate_probabilistic(case_dir: Path, manifest, problem_path, domain_path):
    print("\n=== PROBABILISTIC DOMAIN ===")
    domain, problem, rr_problem, actions, applicable = build_and_ground(
        domain_path, problem_path
    )

    print(f"Parsed actions: {len(domain.actions)}")
    print(f"Grounded actions: {len(actions)}")
    print(f"Applicable actions: {len(applicable)}")

    if len(applicable) != 1:
        raise AssertionError(
            f"Expected exactly one applicable action, got "
            f"{[action.name for action, _ in applicable]}"
        )

    action, outcomes = applicable[0]
    print(f"Selected Railroad action: {action.name}")
    print_outcomes(rr_problem, outcomes)

    actual = goal_probability(rr_problem, outcomes)
    expected = float(manifest["expected_one_step_goal_probability"])

    print(f"Expected goal probability: {expected:.8f}")
    print(f"Railroad goal probability: {actual:.8f}")

    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError(
            f"Probability mismatch: expected {expected}, Railroad produced {actual}"
        )

    print("PASS: probabilistic transition and goal probability match the PPDDL case.")


def validate_nominal(case_dir: Path, manifest, problem_path, domain_path):
    print("\n=== NOMINAL DOMAIN ===")
    domain, problem, rr_problem, actions, applicable = build_and_ground(
        domain_path, problem_path
    )

    print(f"Parsed actions: {len(domain.actions)}")
    print(f"Grounded actions: {len(actions)}")
    print(f"Applicable actions: {len(applicable)}")

    if len(applicable) != 1:
        raise AssertionError(
            f"Expected exactly one applicable nominal action, got "
            f"{[action.name for action, _ in applicable]}"
        )

    action, outcomes = applicable[0]
    print(f"Selected Railroad action: {action.name}")
    print_outcomes(rr_problem, outcomes)

    if len(outcomes) != 1:
        raise AssertionError(f"Nominal action should have one outcome, got {len(outcomes)}")

    actual = goal_probability(rr_problem, outcomes)
    nominal_branch = int(manifest["nominal_branch"])
    nominal_reaches_goal = bool(
        manifest["outcomes"][nominal_branch]["reaches_goal"]
    )
    expected = 1.0 if nominal_reaches_goal else 0.0

    print(f"Expected nominal goal probability: {expected:.8f}")
    print(f"Railroad nominal goal probability: {actual:.8f}")

    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError(
            f"Nominal mismatch: expected {expected}, Railroad produced {actual}"
        )

    print("PASS: nominal/argmax transition matches the controlled PPDDL case.")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Validate the controlled PPDDL case against generic Railroad operators."
    )
    parser.add_argument(
        "--case-dir",
        default="railroad3_controlled",
        help="Directory created by generate_controlled_ppddl_case.py",
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    manifest, problem_path, prob_domain_path, nominal_domain_path = load_case(case_dir)

    validate_probabilistic(case_dir, manifest, problem_path, prob_domain_path)
    validate_nominal(case_dir, manifest, problem_path, nominal_domain_path)

    print("\nALL CONTROLLED RAILROAD VALIDATIONS PASSED")


if __name__ == "__main__":
    main()