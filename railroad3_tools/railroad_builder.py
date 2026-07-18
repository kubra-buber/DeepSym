from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from ppddl_parser import (
    ActionSchema,
    Condition,
    Domain,
    Equality,
    Literal,
    Problem,
)

from railroad._bindings import AndGoal, Fluent as F, LiteralGoal, State
from railroad.core import Effect, Operator


@dataclass
class BuiltOperator:
    """Railroad operator plus PPDDL constraints Railroad does not encode itself."""

    schema: ActionSchema
    operator: Operator
    equality_conditions: List[Equality]


@dataclass
class RailroadProblem:
    initial_state: State
    goal: object
    objects_by_type: Dict[str, set[str]]


def literal_to_fluent(literal: Literal):
    fluent = F(literal.predicate, *literal.arguments)
    return fluent if literal.positive else ~fluent


def build_operator(action: ActionSchema) -> BuiltOperator:
    """Convert one parsed PPDDL action schema into a Railroad Operator.

    Equality/inequality conditions are kept separately and checked after
    grounding because Railroad Operator preconditions are Fluent objects.
    """
    fluent_preconditions = [
        literal_to_fluent(condition)
        for condition in action.preconditions
        if isinstance(condition, Literal)
    ]
    equalities = [
        condition
        for condition in action.preconditions
        if isinstance(condition, Equality)
    ]

    if not action.outcomes:
        raise ValueError(f"Action {action.name} has no outcomes")

    if len(action.outcomes) == 1:
        outcome = action.outcomes[0]
        effects = [
            Effect(
                time=0,
                resulting_fluents={
                    literal_to_fluent(literal)
                    for literal in outcome.effects
                },
            )
        ]
    else:
        total = sum(float(outcome.probability) for outcome in action.outcomes)
        if total <= 0.0:
            raise ValueError(f"Action {action.name} has non-positive probability mass")

        branches = []
        for outcome in action.outcomes:
            probability = float(outcome.probability) / total
            branch = Effect(
                time=0,
                resulting_fluents={
                    literal_to_fluent(literal)
                    for literal in outcome.effects
                },
            )
            branches.append((probability, [branch]))

        effects = [
            Effect(
                time=0,
                resulting_fluents=set(),
                prob_effects=branches,
            )
        ]

    operator = Operator(
        name=action.name,
        parameters=[
            (parameter.name, parameter.type_name)
            for parameter in action.parameters
        ],
        preconditions=fluent_preconditions,
        effects=effects,
    )
    return BuiltOperator(action, operator, equalities)


def build_operators(domain: Domain) -> List[BuiltOperator]:
    return [build_operator(action) for action in domain.actions]


def build_objects_by_type(problem: Problem) -> Dict[str, set[str]]:
    """Build the mapping expected by Operator.instantiate()."""
    objects_by_type: Dict[str, set[str]] = {"object": set()}

    for obj in problem.objects:
        objects_by_type["object"].add(obj.name)
        objects_by_type.setdefault(obj.type_name, set()).add(obj.name)

    return objects_by_type


def _grounding_from_action_name(
    schema: ActionSchema,
    grounded_action_name: str,
) -> Dict[str, str]:
    """Recover parameter bindings from Railroad's '<name> arg1 arg2 ...' name."""
    parts = grounded_action_name.split()
    if not parts or parts[0] != schema.name:
        raise ValueError(
            f"Unexpected grounded action name {grounded_action_name!r} "
            f"for schema {schema.name!r}"
        )

    arguments = parts[1:]
    if len(arguments) != len(schema.parameters):
        raise ValueError(
            f"Ground action {grounded_action_name!r} has {len(arguments)} arguments; "
            f"expected {len(schema.parameters)}"
        )

    return {
        parameter.name: argument
        for parameter, argument in zip(schema.parameters, arguments)
    }


def equality_conditions_hold(
    conditions: Sequence[Equality],
    binding: Dict[str, str],
) -> bool:
    for condition in conditions:
        left = binding.get(condition.left, condition.left)
        right = binding.get(condition.right, condition.right)
        actual_equal = left == right
        if actual_equal != condition.equal:
            return False
    return True


def ground_operators(
    built_operators: Sequence[BuiltOperator],
    objects_by_type: Dict[str, set[str]],
):
    """Instantiate schemas and filter equality/inequality-invalid groundings."""
    grounded = []

    for built in built_operators:
        for action in built.operator.instantiate(objects_by_type):
            binding = _grounding_from_action_name(built.schema, action.name)
            if equality_conditions_hold(built.equality_conditions, binding):
                grounded.append(action)

    grounded.sort(key=lambda action: action.name)
    return grounded


def build_initial_state(problem: Problem) -> State:
    """Build a Railroad state using PPDDL closed-world semantics."""
    fluents = {
        literal_to_fluent(literal)
        for literal in problem.initial_literals
        if literal.positive
    }

    # Explicit negative literals are normally unnecessary in a PDDL initial
    # state. Reject them for now rather than silently changing their meaning.
    negative_init = [
        literal for literal in problem.initial_literals
        if not literal.positive
    ]
    if negative_init:
        raise NotImplementedError(
            "Explicit negated literals in :init are not supported yet. "
            "Use closed-world omission or positive predicates such as not_z0."
        )

    return State(fluents=fluents)


def build_goal(problem: Problem):
    goals = []

    for condition in problem.goal_conditions:
        if isinstance(condition, Equality):
            raise NotImplementedError("Equality conditions in goals are not supported yet")
        goals.append(LiteralGoal(literal_to_fluent(condition)))

    if not goals:
        raise ValueError("Problem goal is empty")
    return goals[0] if len(goals) == 1 else AndGoal(goals)


def build_problem(problem: Problem) -> RailroadProblem:
    return RailroadProblem(
        initial_state=build_initial_state(problem),
        goal=build_goal(problem),
        objects_by_type=build_objects_by_type(problem),
    )