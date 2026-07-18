#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ppddl_parser import Equality, Literal, parse_domain
from railroad_builder import (
    build_operators,
    ground_operators,
    literal_to_fluent,
)
from railroad._bindings import State
from railroad.core import transition
from railroad.planner import get_usable_actions


def state_key(state: State) -> Tuple[str, ...]:
    return tuple(sorted(str(fluent) for fluent in state.fluents))


def action_binding(action_name: str, schema) -> Dict[str, str]:
    arguments = action_name.split()[1:]
    return {
        parameter.name: argument
        for parameter, argument in zip(schema.parameters, arguments)
    }


def minimal_precondition_state(schema, binding: Dict[str, str]) -> State:
    fluents = set()

    for condition in schema.preconditions:
        if isinstance(condition, Equality):
            continue

        grounded = condition.ground(binding)
        positive = literal_to_fluent(
            Literal(
                grounded.predicate,
                grounded.arguments,
                True,
            )
        )

        if grounded.positive:
            fluents.add(positive)
        else:
            fluents.discard(positive)

    return State(fluents=fluents)


def merged_outcomes(state: State, action):
    try:
        raw_outcomes = transition(state, action)
    except RuntimeError as exc:
        if "precondition not satisfied" in str(exc).lower():
            return []
        raise

    merged = defaultdict(float)
    states = {}

    for next_state, probability in raw_outcomes:
        key = state_key(next_state)
        merged[key] += float(probability)
        states[key] = next_state

    total = sum(merged.values())
    if total <= 0.0:
        return []

    return [
        (states[key], probability / total)
        for key, probability in merged.items()
    ]


def one_step_goal_probability(state: State, action, goal_fluent) -> float:
    return sum(
        probability
        for next_state, probability in merged_outcomes(state, action)
        if goal_fluent in next_state.fluents
    )


def candidate_goal_fluents(state: State, actions: Sequence):
    candidates = set()

    for action in actions:
        for next_state, _ in merged_outcomes(state, action):
            candidates.update(next_state.fluents - state.fluents)

    return candidates


def format_problem(domain_name: str, state: State, goal_fluent) -> str:
    init_lines = "\n".join(
        f"    {fluent}"
        for fluent in sorted(state.fluents, key=str)
    )

    return f"""(define (problem railroad-real-domain-choice)
  (:domain {domain_name})

  (:objects
    obj0 obj1
  )

  (:init
{init_lines}
  )

  (:goal
    {goal_fluent}
  )
)
"""


def main():
    parser = argparse.ArgumentParser(
        "Generate a controlled choice problem from the real probabilistic domain."
    )
    parser.add_argument("--prob-domain", default="domain_prob.pddl")
    parser.add_argument(
        "--output-dir",
        default="railroad3_real_choice",
    )
    parser.add_argument("--min-gap", type=float, default=0.05)
    args = parser.parse_args()

    domain_path = Path(args.prob_domain)
    domain = parse_domain(domain_path)

    built = build_operators(domain)
    actions = ground_operators(
        built,
        {"object": {"obj0", "obj1"}},
    )

    schemas = {schema.name: schema for schema in domain.actions}

    candidate_states = {}
    for action in actions:
        schema_name = action.name.split()[0]
        schema = schemas[schema_name]
        binding = action_binding(action.name, schema)
        state = minimal_precondition_state(schema, binding)
        candidate_states[state_key(state)] = state

    best_case = None

    for state in candidate_states.values():
        usable = [
            action
            for action in actions
            if merged_outcomes(state, action)
        ]
        if len(usable) < 2:
            continue

        for goal_fluent in candidate_goal_fluents(state, usable):
            values = {
                action.name: one_step_goal_probability(
                    state,
                    action,
                    goal_fluent,
                )
                for action in usable
            }
            values = {
                name: probability
                for name, probability in values.items()
                if probability > 0.0
            }

            if len(values) < 2:
                continue

            ranked = sorted(
                values.items(),
                key=lambda item: (-item[1], item[0]),
            )
            best_name, best_probability = ranked[0]
            second_probability = ranked[1][1]
            gap = best_probability - second_probability

            tied_best = sum(
                math.isclose(
                    probability,
                    best_probability,
                    rel_tol=0.0,
                    abs_tol=1e-10,
                )
                for probability in values.values()
            )

            if tied_best != 1 or gap < args.min_gap:
                continue

            score = (
                gap,
                best_probability,
                -len(usable),
            )

            if best_case is None or score > best_case["score"]:
                best_case = {
                    "score": score,
                    "state": state,
                    "goal": goal_fluent,
                    "usable": usable,
                    "values": values,
                    "expected_action": best_name,
                    "expected_value": best_probability,
                    "gap": gap,
                }

    if best_case is None:
        raise RuntimeError(
            "Could not find a real-domain state with at least two "
            "goal-reaching actions having distinct one-step probabilities."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_path = output_dir / "real_choice_problem.pddl"
    manifest_path = output_dir / "real_choice_manifest.json"

    problem_path.write_text(
        format_problem(
            domain.name,
            best_case["state"],
            best_case["goal"],
        )
    )

    manifest = {
        "prob_domain": str(domain_path),
        "problem": problem_path.name,
        "horizon": 1,
        "goal": str(best_case["goal"]),
        "expected_action": best_case["expected_action"],
        "expected_value": best_case["expected_value"],
        "probability_gap": best_case["gap"],
        "applicable_actions": [
            action.name for action in best_case["usable"]
        ],
        "positive_goal_probabilities": dict(
            sorted(best_case["values"].items())
        ),
        "initial_state": list(state_key(best_case["state"])),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote: {problem_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Goal: {best_case['goal']}")
    print(f"Applicable actions: {len(best_case['usable'])}")
    print("Positive one-step goal probabilities:")
    for name, probability in sorted(
        best_case["values"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {name}: {probability:.8f}")
    print(f"Expected action: {best_case['expected_action']}")
    print(f"Expected value: {best_case['expected_value']:.8f}")


if __name__ == "__main__":
    main()