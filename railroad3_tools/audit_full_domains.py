#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ppddl_parser import Equality, Literal, parse_domain
from railroad_builder import (
    build_operator,
    ground_operators,
    literal_to_fluent,
)

from railroad._bindings import State
from railroad.core import transition


def logical_action_key(name: str) -> str:
    """Ignore the final sample-count suffix, e.g. _c929."""
    return re.sub(r"_c\d+$", "", name)


def condition_key(condition):
    if isinstance(condition, Literal):
        return (
            "literal",
            condition.predicate,
            tuple(condition.arguments),
            bool(condition.positive),
        )
    if isinstance(condition, Equality):
        return (
            "equality",
            condition.left,
            condition.right,
            bool(condition.equal),
        )
    raise TypeError(type(condition))


def effects_key(effects: Sequence[Literal]):
    return tuple(
        sorted(
            (
                effect.predicate,
                tuple(effect.arguments),
                bool(effect.positive),
            )
            for effect in effects
        )
    )


def schema_signature(action):
    return {
        "parameters": tuple(
            (parameter.name, parameter.type_name)
            for parameter in action.parameters
        ),
        "preconditions": tuple(
            sorted(condition_key(condition) for condition in action.preconditions)
        ),
    }


def action_binding(action_name: str, schema) -> Dict[str, str]:
    parts = action_name.split()
    arguments = parts[1:]
    if len(arguments) != len(schema.parameters):
        raise ValueError(
            f"Grounded action {action_name!r} has the wrong arity"
        )
    return {
        parameter.name: argument
        for parameter, argument in zip(schema.parameters, arguments)
    }


def build_precondition_state(schema, binding: Dict[str, str]) -> State:
    """Construct a minimal state satisfying one grounded schema's literals."""
    fluents = set()

    for condition in schema.preconditions:
        if isinstance(condition, Equality):
            continue

        grounded = condition.ground(binding)
        if grounded.positive:
            fluents.add(literal_to_fluent(grounded))
        else:
            # Closed-world state: absence satisfies a negated precondition.
            fluents.discard(literal_to_fluent(
                Literal(grounded.predicate, grounded.arguments, True)
            ))

    return State(fluents=fluents)


def state_key(state: State) -> Tuple[str, ...]:
    return tuple(sorted(str(fluent) for fluent in state.fluents))


def apply_ppddl_effects(
    state: State,
    effects: Sequence[Literal],
    binding: Dict[str, str],
) -> Tuple[str, ...]:
    fluents = set(state.fluents)

    for effect in effects:
        grounded = effect.ground(binding)
        positive = literal_to_fluent(
            Literal(grounded.predicate, grounded.arguments, True)
        )
        if grounded.positive:
            fluents.add(positive)
        else:
            fluents.discard(positive)

    return tuple(sorted(str(fluent) for fluent in fluents))


def merge_expected_outcomes(
    state: State,
    schema,
    binding: Dict[str, str],
) -> Dict[Tuple[str, ...], float]:
    merged: Dict[Tuple[str, ...], float] = defaultdict(float)
    total = sum(float(outcome.probability) for outcome in schema.outcomes)

    for outcome in schema.outcomes:
        key = apply_ppddl_effects(state, outcome.effects, binding)
        merged[key] += float(outcome.probability) / total

    return dict(merged)


def merge_railroad_outcomes(outcomes) -> Dict[Tuple[str, ...], float]:
    merged: Dict[Tuple[str, ...], float] = defaultdict(float)

    for next_state, probability in outcomes:
        merged[state_key(next_state)] += float(probability)

    total = sum(merged.values())
    if total <= 0.0:
        raise AssertionError("Railroad transition returned no probability mass")

    return {
        key: probability / total
        for key, probability in merged.items()
    }


def assert_distributions_equal(
    expected: Dict[Tuple[str, ...], float],
    actual: Dict[Tuple[str, ...], float],
    *,
    action_name: str,
    tolerance: float,
):
    if set(expected) != set(actual):
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        raise AssertionError(
            f"{action_name}: outcome-state mismatch; "
            f"missing={missing}, extra={extra}"
        )

    for key in expected:
        if not math.isclose(
            expected[key],
            actual[key],
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise AssertionError(
                f"{action_name}: probability mismatch for {key}: "
                f"expected={expected[key]}, actual={actual[key]}"
            )


def audit_domain_pair(
    probabilistic_path: Path,
    nominal_path: Path,
    *,
    tolerance: float,
):
    probabilistic = parse_domain(probabilistic_path)
    nominal = parse_domain(nominal_path)

    if probabilistic.name != nominal.name:
        raise AssertionError(
            f"Domain names differ: {probabilistic.name} vs {nominal.name}"
        )

    if {
        (predicate.name, len(predicate.parameters))
        for predicate in probabilistic.predicates
    } != {
        (predicate.name, len(predicate.parameters))
        for predicate in nominal.predicates
    }:
        raise AssertionError("Predicate declarations differ")

    prob_by_key = {
        logical_action_key(action.name): action
        for action in probabilistic.actions
    }
    nominal_by_key = {
        logical_action_key(action.name): action
        for action in nominal.actions
    }

    if set(prob_by_key) != set(nominal_by_key):
        raise AssertionError(
            "Probabilistic and nominal domains have different logical actions"
        )

    outcome_counts = Counter()
    argmax_matches = 0

    for key in sorted(prob_by_key):
        prob_action = prob_by_key[key]
        nominal_action = nominal_by_key[key]

        if schema_signature(prob_action) != schema_signature(nominal_action):
            raise AssertionError(f"{key}: parameters/preconditions differ")

        probability_sum = sum(
            float(outcome.probability)
            for outcome in prob_action.outcomes
        )
        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise AssertionError(
                f"{prob_action.name}: probabilities sum to {probability_sum}"
            )

        if any(outcome.probability <= 0 for outcome in prob_action.outcomes):
            raise AssertionError(
                f"{prob_action.name}: non-positive outcome probability"
            )

        if len(nominal_action.outcomes) != 1:
            raise AssertionError(
                f"{nominal_action.name}: nominal action is not deterministic"
            )

        max_probability = max(
            float(outcome.probability)
            for outcome in prob_action.outcomes
        )
        argmax_outcomes = [
            outcome
            for outcome in prob_action.outcomes
            if math.isclose(
                float(outcome.probability),
                max_probability,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ]
        nominal_effect = effects_key(nominal_action.outcomes[0].effects)

        if nominal_effect not in {
            effects_key(outcome.effects)
            for outcome in argmax_outcomes
        }:
            raise AssertionError(
                f"{key}: nominal effect is not an argmax probabilistic effect"
            )

        outcome_counts[len(prob_action.outcomes)] += 1
        argmax_matches += 1

    return probabilistic, nominal, outcome_counts, argmax_matches


def audit_railroad_transitions(domain, *, tolerance: float):
    objects_by_type = {
        "object": {"obj0", "obj1"},
    }

    checked = 0
    grounded_total = 0

    for schema in domain.actions:
        built = build_operator(schema)
        grounded_actions = ground_operators([built], objects_by_type)
        grounded_total += len(grounded_actions)

        if not grounded_actions:
            raise AssertionError(f"{schema.name}: produced no valid grounding")

        # One representative grounding is sufficient for schema-level semantics.
        action = grounded_actions[0]
        binding = action_binding(action.name, schema)
        state = build_precondition_state(schema, binding)

        try:
            actual = merge_railroad_outcomes(transition(state, action))
        except Exception as exc:
            raise AssertionError(
                f"{action.name}: Railroad transition failed: {exc}"
            ) from exc

        expected = merge_expected_outcomes(state, schema, binding)
        assert_distributions_equal(
            expected,
            actual,
            action_name=action.name,
            tolerance=tolerance,
        )
        checked += 1

    return checked, grounded_total


def main():
    parser = argparse.ArgumentParser(
        "Audit the full Bilevel probabilistic/nominal domains and Railroad conversion."
    )
    parser.add_argument(
        "--prob-domain",
        default="domain_prob.pddl",
    )
    parser.add_argument(
        "--nominal-domain",
        default="domain.pddl",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
    )
    args = parser.parse_args()

    probabilistic_path = Path(args.prob_domain)
    nominal_path = Path(args.nominal_domain)

    (
        probabilistic,
        nominal,
        outcome_counts,
        argmax_matches,
    ) = audit_domain_pair(
        probabilistic_path,
        nominal_path,
        tolerance=args.tolerance,
    )

    print("=== FULL DOMAIN STRUCTURE ===")
    print(f"Domain: {probabilistic.name}")
    print(f"Predicates: {len(probabilistic.predicates)}")
    print(f"Probabilistic actions: {len(probabilistic.actions)}")
    print(f"Nominal actions: {len(nominal.actions)}")
    print(f"Outcome-count distribution: {dict(sorted(outcome_counts.items()))}")
    print(f"Nominal argmax matches: {argmax_matches}/{len(probabilistic.actions)}")
    print("PASS: full probabilistic and nominal domains are logically aligned.")

    checked, grounded_total = audit_railroad_transitions(
        probabilistic,
        tolerance=args.tolerance,
    )

    print("\n=== FULL RAILROAD TRANSITION AUDIT ===")
    print(f"Action schemas checked: {checked}")
    print(f"Total valid groundings with two objects: {grounded_total}")
    print(
        "PASS: every probabilistic action schema produced the same "
        "outcome states and probabilities in Railroad."
    )

    print("\nALL FULL-DOMAIN AUDITS PASSED")


if __name__ == "__main__":
    main()