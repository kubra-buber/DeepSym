from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from ppddl_parser import (
    ActionSchema,
    Condition,
    Domain,
    Equality,
    Literal,
    Outcome,
    TypedSymbol,
    condition_to_pddl,
    literal_to_pddl,
    parse_domain,
)


Atom = Tuple[str, Tuple[str, ...]]


def action_index(name: str) -> int | None:
    match = re.search(r"_i(\d+)_", name)
    return int(match.group(1)) if match else None


def select_action(domain: Domain, name: str | None, index: int | None) -> ActionSchema:
    if name is not None:
        matches = [action for action in domain.actions if action.name == name]
    elif index is not None:
        matches = [action for action in domain.actions if action_index(action.name) == index]
    else:
        # Default to the useful near-50/50 two-object action in the supplied
        # Bilevel domain when present. Otherwise choose a small probabilistic action.
        matches = [action for action in domain.actions if action_index(action.name) == 18]
        if not matches:
            candidates = [
                action
                for action in domain.actions
                if len(action.outcomes) > 1 and len(action.parameters) >= 1
            ]
            candidates.sort(
                key=lambda action: (
                    len(action.parameters) != 2,
                    len(action.outcomes),
                    abs(max(o.probability for o in action.outcomes) - 0.5),
                    action.name,
                )
            )
            matches = candidates[:1]

    if len(matches) != 1:
        raise ValueError(
            f"Could not select exactly one action: name={name!r}, index={index!r}, "
            f"matches={[a.name for a in matches]}"
        )
    return matches[0]


def build_binding(action: ActionSchema) -> Tuple[Dict[str, str], List[TypedSymbol]]:
    # Start with one distinct object per parameter.
    binding = {
        parameter.name: f"obj{i}"
        for i, parameter in enumerate(action.parameters)
    }

    # Support equality constraints by merging object names.
    changed = True
    while changed:
        changed = False
        for condition in action.preconditions:
            if isinstance(condition, Equality) and condition.equal:
                left = binding.get(condition.left, condition.left)
                right = binding.get(condition.right, condition.right)
                canonical = min(left, right)
                for key, value in list(binding.items()):
                    if value in {left, right} and value != canonical:
                        binding[key] = canonical
                        changed = True

    for condition in action.preconditions:
        if isinstance(condition, Equality) and not condition.equal:
            left = binding.get(condition.left, condition.left)
            right = binding.get(condition.right, condition.right)
            if left == right:
                raise ValueError(f"Cannot satisfy inequality {condition}")

    objects_by_name: Dict[str, TypedSymbol] = {}
    parameter_types = {parameter.name: parameter.type_name for parameter in action.parameters}
    for parameter_name, object_name in binding.items():
        objects_by_name.setdefault(
            object_name,
            TypedSymbol(object_name, parameter_types.get(parameter_name, "object")),
        )
    return binding, list(objects_by_name.values())


def specialize_action_for_unique_grounding(
    action: ActionSchema,
    binding: Dict[str, str],
) -> Tuple[ActionSchema, List[TypedSymbol], List[str]]:
    """Give each distinct bound object a private type.

    The selected Bilevel action has symmetric preconditions, so with two
    untyped objects both (?o0=obj0, ?o1=obj1) and the reversed grounding are
    applicable. For a controlled one-action test we preserve the selected
    grounded transition but constrain each argument role with a singleton type.
    """
    object_to_type: Dict[str, str] = {}
    type_names: List[str] = []

    for parameter in action.parameters:
        object_name = binding[parameter.name]
        if object_name not in object_to_type:
            type_name = f"controlled_role{len(object_to_type)}"
            object_to_type[object_name] = type_name
            type_names.append(type_name)

    typed_parameters = [
        TypedSymbol(parameter.name, object_to_type[binding[parameter.name]])
        for parameter in action.parameters
    ]
    typed_objects = [
        TypedSymbol(object_name, type_name)
        for object_name, type_name in object_to_type.items()
    ]

    specialized = ActionSchema(
        name=action.name,
        parameters=typed_parameters,
        preconditions=list(action.preconditions),
        outcomes=list(action.outcomes),
    )
    return specialized, typed_objects, type_names


def initial_state(action: ActionSchema, binding: Dict[str, str]) -> Set[Atom]:
    state: Set[Atom] = set()
    for condition in action.preconditions:
        if isinstance(condition, Literal):
            grounded = condition.ground(binding)
            if grounded.positive:
                state.add(grounded.atom())
            # A negative precondition is satisfied by absence under closed-world semantics.
    return state


def grounded_effects(outcome: Outcome, binding: Dict[str, str]) -> List[Literal]:
    return [effect.ground(binding) for effect in outcome.effects]


def apply_effects(state: Set[Atom], effects: Sequence[Literal]) -> Set[Atom]:
    result = set(state)
    deletes = {effect.atom() for effect in effects if not effect.positive}
    adds = {effect.atom() for effect in effects if effect.positive}
    result.difference_update(deletes)
    result.update(adds)
    return result


def relevant_atoms(init: Set[Atom], outcomes: Sequence[Sequence[Literal]]) -> Set[Atom]:
    atoms = set(init)
    for effects in outcomes:
        atoms.update(effect.atom() for effect in effects)
    return atoms


def distinguishing_goal(
    target_state: Set[Atom],
    other_states: Sequence[Set[Atom]],
    atoms: Set[Atom],
    init: Set[Atom],
) -> List[Literal]:
    """Find a small conjunction true in target and false in every other outcome."""
    if not other_states:
        changed = sorted(target_state.symmetric_difference(init))
        return [
            Literal(pred, args, positive=(pred, args) in target_state)
            for pred, args in changed
        ]

    candidates: List[Tuple[Literal, Set[int]]] = []
    for atom in sorted(atoms):
        target_truth = atom in target_state
        literal = Literal(atom[0], atom[1], positive=target_truth)
        excluded = {
            i
            for i, state in enumerate(other_states)
            if (atom in state) != target_truth
        }
        if excluded:
            candidates.append((literal, excluded))

    universe = set(range(len(other_states)))
    for size in range(1, min(len(candidates), 6) + 1):
        for combo in combinations(candidates, size):
            covered = set().union(*(excluded for _, excluded in combo))
            if covered == universe:
                return [literal for literal, _ in combo]

    # Greedy fallback.
    remaining = set(universe)
    selected: List[Literal] = []
    while remaining:
        best = max(candidates, key=lambda item: len(item[1] & remaining), default=None)
        if best is None or not (best[1] & remaining):
            raise ValueError("Target outcome cannot be distinguished from other outcomes")
        selected.append(best[0])
        remaining -= best[1]
    return selected


def serialize_parameters(parameters: Sequence[TypedSymbol]) -> str:
    if not parameters:
        return "()"
    groups: List[str] = []
    for parameter in parameters:
        if parameter.type_name == "object":
            groups.append(parameter.name)
        else:
            groups.extend([parameter.name, "-", parameter.type_name])
    return "(" + " ".join(groups) + ")"


def serialize_conditions(conditions: Sequence[Condition], indent: str = "        ") -> str:
    if not conditions:
        return "(and)"
    body = "\n".join(f"{indent}{condition_to_pddl(c)}" for c in conditions)
    return f"(and\n{body}\n{indent[:-4]})"


def serialize_effects(effects: Sequence[Literal], indent: str = "            ") -> str:
    if not effects:
        return "(and)"
    body = "\n".join(f"{indent}{literal_to_pddl(e)}" for e in effects)
    return f"(and\n{body}\n{indent[:-4]})"


def write_sliced_domain(
    path: Path,
    domain: Domain,
    action: ActionSchema,
    *,
    probabilistic: bool,
    controlled_types: Sequence[str] = (),
) -> None:
    requirements = list(domain.requirements)
    if probabilistic and ":probabilistic-effects" not in requirements:
        requirements.append(":probabilistic-effects")
    if not probabilistic:
        requirements = [r for r in requirements if r != ":probabilistic-effects"]
    if controlled_types and ":typing" not in requirements:
        requirements.append(":typing")

    pred_lines = []
    for predicate in domain.predicates:
        args = []
        for parameter in predicate.parameters:
            args.append(parameter.name)
            if parameter.type_name != "object":
                args.extend(["-", parameter.type_name])
        pred_lines.append(f"        ({predicate.name}{(' ' + ' '.join(args)) if args else ''})")

    pre = serialize_conditions(action.preconditions)

    if probabilistic:
        branch_lines = []
        for outcome in action.outcomes:
            branch_lines.append(
                f"            {outcome.probability:.8f} "
                + serialize_effects(outcome.effects, indent="                ")
            )
        effect = "(probabilistic\n" + "\n".join(branch_lines) + "\n        )"
    else:
        nominal = max(action.outcomes, key=lambda outcome: outcome.probability)
        effect = serialize_effects(nominal.effects, indent="            ")

    types_section = (
        f"\n    (:types {' '.join(controlled_types)})"
        if controlled_types else ""
    )

    text = f"""(define (domain {domain.name})
    (:requirements {' '.join(requirements)}){types_section}
    (:predicates
{chr(10).join(pred_lines)}
    )

    (:action {action.name}
        :parameters {serialize_parameters(action.parameters)}
        :precondition {pre}
        :effect {effect}
    )
)
"""
    path.write_text(text)


def write_problem(
    path: Path,
    domain_name: str,
    objects: Sequence[TypedSymbol],
    init: Set[Atom],
    goal: Sequence[Literal],
) -> None:
    object_tokens = []
    for obj in objects:
        object_tokens.append(obj.name)
        if obj.type_name != "object":
            object_tokens.extend(["-", obj.type_name])

    init_lines = [
        "        (" + " ".join((predicate, *args)) + ")"
        for predicate, args in sorted(init)
    ]
    goal_lines = [f"            {literal_to_pddl(literal)}" for literal in goal]

    text = f"""(define (problem controlled-{domain_name})
    (:domain {domain_name})
    (:objects {' '.join(object_tokens)})
    (:init
{chr(10).join(init_lines)}
    )
    (:goal
        (and
{chr(10).join(goal_lines)}
        )
    )
)
"""
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate a controlled one-action PPDDL/Railroad validation case."
    )
    parser.add_argument("--domain", required=True, help="Source probabilistic domain.")
    parser.add_argument("--output-dir", default="railroad3_controlled")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--action-name")
    selector.add_argument("--action-index", type=int)
    parser.add_argument(
        "--target-branch",
        type=int,
        default=1,
        help="Outcome branch used to build the distinguishing goal. Default: 1.",
    )
    args = parser.parse_args()

    domain = parse_domain(args.domain)
    action = select_action(domain, args.action_name, args.action_index)
    if len(action.outcomes) < 2:
        raise ValueError(f"Selected action {action.name} is not probabilistic")
    if not 0 <= args.target_branch < len(action.outcomes):
        raise ValueError(
            f"target branch {args.target_branch} outside 0..{len(action.outcomes)-1}"
        )

    binding, _ = build_binding(action)
    action, objects, controlled_types = specialize_action_for_unique_grounding(
        action, binding
    )
    init = initial_state(action, binding)
    grounded = [grounded_effects(outcome, binding) for outcome in action.outcomes]
    post_states = [apply_effects(init, effects) for effects in grounded]
    atoms = relevant_atoms(init, grounded)

    target_state = post_states[args.target_branch]
    other_states = [
        state for i, state in enumerate(post_states) if i != args.target_branch
    ]
    goal = distinguishing_goal(target_state, other_states, atoms, init)

    equivalent = [
        i for i, state in enumerate(post_states) if state == target_state
    ]
    expected_probability = sum(action.outcomes[i].probability for i in equivalent)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prob_path = output_dir / "controlled_domain_prob.pddl"
    nominal_path = output_dir / "controlled_domain_nominal.pddl"
    problem_path = output_dir / "controlled_problem.pddl"
    manifest_path = output_dir / "controlled_manifest.json"

    write_sliced_domain(
        prob_path,
        domain,
        action,
        probabilistic=True,
        controlled_types=controlled_types,
    )
    write_sliced_domain(
        nominal_path,
        domain,
        action,
        probabilistic=False,
        controlled_types=controlled_types,
    )
    write_problem(problem_path, domain.name, objects, init, goal)

    manifest = {
        "source_domain": str(Path(args.domain).resolve()),
        "domain_name": domain.name,
        "selected_action": action.name,
        "selected_action_index": action_index(action.name),
        "binding": binding,
        "controlled_types": controlled_types,
        "expected_ground_action": action.name + " " + " ".join(
            binding[p.name] for p in action.parameters
        ),
        "target_branch": args.target_branch,
        "goal": [literal_to_pddl(literal) for literal in goal],
        "expected_one_step_goal_probability": expected_probability,
        "nominal_branch": max(
            range(len(action.outcomes)),
            key=lambda i: action.outcomes[i].probability,
        ),
        "outcomes": [
            {
                "branch": i,
                "probability": outcome.probability,
                "effects": [literal_to_pddl(effect) for effect in grounded[i]],
                "reaches_goal": post_states[i] == target_state,
            }
            for i, outcome in enumerate(action.outcomes)
        ],
        "files": {
            "probabilistic_domain": prob_path.name,
            "nominal_domain": nominal_path.name,
            "problem": problem_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Selected action: {action.name}")
    print(f"Grounding: {binding}")
    print(f"Controlled singleton types: {controlled_types}")
    print(f"Expected grounded action: {manifest['expected_ground_action']}")
    print(f"Target branch: {args.target_branch}")
    print(f"Goal: {' '.join(manifest['goal'])}")
    print(f"Expected one-step goal probability: {expected_probability:.8f}")
    print(f"Nominal/argmax branch: {manifest['nominal_branch']}")
    print(f"Wrote: {prob_path}")
    print(f"Wrote: {nominal_path}")
    print(f"Wrote: {problem_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()