"""Railroad Option A planner for DeepSym: open-loop nominal-probability planning.

Drop-in replacement for the old make_plan.sh + mdpsim + mini-gpt pipeline when
DeepSym still executes a fixed plan.txt with execute_plan.py.

This planner does not execute a closed-loop contingency policy. It searches for
a fixed nominal symbolic plan and maximizes the product of the learned
probabilities of the selected successful symbolic effects.

Neuro-symbolic structure preserved:
    learned object/relation categories
        -> learned decision-tree symbolic rules
        -> learned effect probabilities
        -> symbolic planning with probability costs

No hard-coded object-size/stability rule is used.
"""

import argparse
import json
import math
import os
import re
from heapq import heappop, heappush
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from railroad._bindings import AndGoal, Fluent as F, LiteralGoal, State
from railroad.core import Effect, Operator, transition
from railroad.planner import get_usable_actions


# Learned distribution for each original stack rule, e.g.
# OPERATOR_EFFECT_DISTS["stack9"] = {"stacked": 0.984, "inserted": 0.003, ...}
OPERATOR_EFFECT_DISTS: Dict[str, Dict[str, float]] = {}

# Cost/probability for each nominal deterministic Railroad operator, e.g.
# NOMINAL_ACTION_PROBS["stack9__stacked"] = 0.984
NOMINAL_ACTION_PROBS: Dict[str, float] = {}
NOMINAL_ACTION_EFFECTS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Railroad operator reconstruction
# ---------------------------------------------------------------------------

def _success_effect_fluents(effect_name: str) -> set:
    """Map a successful DeepSym effect to Railroad fluents."""
    effect_name = str(effect_name)

    if effect_name == "stacked":
        # Original DeepSym PPDDL semantics: stacked also implies inserted.
        return {
            F("stacked"),
            F("inserted"),
            F("instack", "?above"),
            F("stackloc", "?above"),
            ~F("stackloc", "?below"),
        }

    if effect_name == "inserted":
        return {
            F("inserted"),
            F("instack", "?above"),
            F("stackloc", "?above"),
            ~F("stackloc", "?below"),
        }

    raise ValueError(f"{effect_name!r} is not a nominal success effect.")


def _is_nominal_success_effect(effect_name: str) -> bool:
    """Effects that can intentionally advance a DeepSym stack plan."""
    return str(effect_name) in {"stacked", "inserted"}


def build_nominal_stack_operators(spec: Dict) -> List[Operator]:
    """Build deterministic nominal Railroad operators from one learned rule.

    A learned probabilistic action like:
        stack9: 0.984 stacked, 0.003 inserted, 0.003 roll1, 0.010 tumble2

    becomes nominal deterministic planning actions:
        stack9__stacked   cost=-log(0.984)
        stack9__inserted  cost=-log(0.003)

    roll/tumble branches are not executable plan branches; they reduce success
    probability because they are not included in the chosen nominal branch.
    """
    below_type = spec.get("below_type", spec.get("obj1_type"))
    above_type = spec.get("above_type", spec.get("obj2_type"))
    if below_type is None or above_type is None:
        raise ValueError(f"Malformed stack operator spec: {spec}")

    original_name = spec["name"]

    effect_dist: Dict[str, float] = {}
    for pe in spec["prob_effects"]:
        effect_name = str(pe["effect_name"])
        p = float(pe["probability"])
        if p <= 0.0:
            continue
        effect_dist[effect_name] = effect_dist.get(effect_name, 0.0) + p

    OPERATOR_EFFECT_DISTS[original_name] = effect_dist

    preconditions = [
        ~F("stacked"),
        ~F("inserted"),
        F("pickloc", "?above"),
        F("stackloc", "?below"),
        F(below_type, "?below"),
        F(above_type, "?above"),
        F(spec["relation"], "?below", "?above"),
    ]

    operators: List[Operator] = []

    success_items = [
        (effect_name, p)
        for effect_name, p in effect_dist.items()
        if _is_nominal_success_effect(effect_name) and p > 0.0
    ]
    # Deterministic and reproducible order. If tied, prefer stacked because it
    # progresses both H and S counters.
    success_items.sort(key=lambda item: (item[0] != "stacked", item[0]))

    for effect_name, p in success_items:
        nominal_name = f"{original_name}__{effect_name}"
        NOMINAL_ACTION_PROBS[nominal_name] = p
        NOMINAL_ACTION_EFFECTS[nominal_name] = effect_name

        outcome_effect = Effect(
            time=0,
            resulting_fluents=_success_effect_fluents(effect_name),
        )

        # Original PPDDL places (not (pickloc ?above)) outside the probabilistic
        # branch; a successful nominal stack consumes the above object.
        pickloc_removal = Effect(
            time=0,
            resulting_fluents={~F("pickloc", "?above")},
        )

        operators.append(Operator(
            name=nominal_name,
            parameters=[("?below", "object"), ("?above", "object")],
            preconditions=preconditions,
            effects=[outcome_effect, pickloc_removal],
        ))

    return operators


def build_auxiliary_operator(spec: Dict) -> Operator:
    """Build deterministic helper operators used by DeepSym."""
    if spec["type"] == "increase_height":
        return Operator(
            name=spec["name"],
            parameters=[],
            preconditions=[F("stacked"), F(spec["from_counter"])],
            effects=[
                Effect(
                    time=0,
                    resulting_fluents={
                        ~F(spec["from_counter"]),
                        F(spec["to_counter"]),
                        ~F("stacked"),
                    },
                )
            ],
        )

    if spec["type"] == "increase_stack":
        return Operator(
            name=spec["name"],
            parameters=[],
            preconditions=[F("inserted"), F(spec["from_counter"])],
            effects=[
                Effect(
                    time=0,
                    resulting_fluents={
                        ~F(spec["from_counter"]),
                        F(spec["to_counter"]),
                        ~F("inserted"),
                    },
                )
            ],
        )

    if spec["type"] == "makebase":
        return Operator(
            name="makebase",
            parameters=[("?obj", "object")],
            preconditions=[~F("base")],
            effects=[
                Effect(
                    time=0,
                    resulting_fluents={
                        F("base"),
                        F("stacked"),
                        F("inserted"),
                        ~F("pickloc", "?obj"),
                        F("stackloc", "?obj"),
                    },
                )
            ],
        )

    raise ValueError(f"Unknown auxiliary operator type: {spec['type']}")


def load_operators_from_json(json_path: str) -> List[Operator]:
    """Load learned specs and reconstruct nominal Railroad operators."""
    with open(json_path, "r") as f:
        all_specs = json.load(f)

    OPERATOR_EFFECT_DISTS.clear()
    NOMINAL_ACTION_PROBS.clear()
    NOMINAL_ACTION_EFFECTS.clear()

    operators: List[Operator] = []
    for spec in all_specs["stack_operators"]:
        operators.extend(build_nominal_stack_operators(spec))
    for spec in all_specs["auxiliary_operators"]:
        operators.append(build_auxiliary_operator(spec))
    return operators


# ---------------------------------------------------------------------------
# PDDL-ish problem parsing
# ---------------------------------------------------------------------------

def parse_pddl_goal(goal_str: str):
    """Parse a small PDDL-style goal string into a Railroad goal expression."""
    negative_tokens = set(re.findall(r"\(not\s*\((\w+)\)\)", goal_str))

    without_not_clauses = re.sub(r"\(not\s*\(\w+\)\)", " ", goal_str)
    positive_tokens = [
        tok for tok in re.findall(r"\((\w+)\)", without_not_clauses)
        if tok not in {"and", "goal"}
    ]

    if not positive_tokens and not negative_tokens:
        raise ValueError(f"No goal fluents found in: {goal_str}")

    goal_literals = [LiteralGoal(F(token)) for token in positive_tokens]

    # DeepSym goals should end after auxiliary predicates are cleared.
    negative_tokens.add("stacked")
    negative_tokens.add("inserted")

    for token in sorted(negative_tokens):
        goal_literals.append(LiteralGoal(~F(token)))

    return AndGoal(goal_literals)


def _unique_preserving_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _extract_between_sections(content: str, start_section: str, next_section: str) -> str:
    """Extract text between two top-level DeepSym PDDL sections."""
    pattern = rf"\(\s*:{start_section}\b(.*?)\)\s*\(\s*:{next_section}\b"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def parse_problem_pddl(problem_path: str) -> Tuple[List[str], Dict[str, str], List[Tuple[str, str, str]]]:
    """Parse DeepSym's generated PDDL problem file."""
    with open(problem_path, "r") as f:
        content = f.read()

    objects_section = _extract_between_sections(content, "objects", "init")
    if not objects_section:
        obj_match = re.search(r"\(\s*:objects\b(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        objects_section = obj_match.group(1) if obj_match else ""

    objects = _unique_preserving_order(re.findall(r"\b\w+\b", objects_section))

    init_section = _extract_between_sections(content, "init", "goal")
    if not init_section:
        init_match = re.search(
            r"\(\s*:init\b(.*?)(?:\)\s*\(\s*:goal\b|\)\s*\)\s*$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        init_section = init_match.group(1) if init_match else ""

    predicates = re.findall(r"\(([^()]+)\)", init_section)

    obj_types: Dict[str, str] = {}
    relations: List[Tuple[str, str, str]] = []

    for pred_str in predicates:
        parts = pred_str.split()
        if not parts:
            continue

        if parts[0].startswith("objtype") and len(parts) == 2:
            obj_types[parts[1]] = parts[0]
        elif parts[0].startswith("relation") and len(parts) == 3:
            relations.append((parts[0], parts[1], parts[2]))

    if not objects:
        raise ValueError(f"No objects parsed from {problem_path}. Check PDDL formatting.")

    # Safety filter: in DeepSym, real object names are exactly those that have
    # objtype predicates.
    typed_objects = [obj for obj in objects if obj in obj_types]
    if typed_objects:
        objects = typed_objects

    return objects, obj_types, relations


def parse_objects_txt(objects_path: str) -> Dict[str, Dict[str, float]]:
    obj_infos = {}
    with open(objects_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    n = int(lines[0])
    for line in lines[1:n + 1]:
        parts = line.split()
        name = parts[0]
        x, y, size = float(parts[1]), float(parts[2]), float(parts[3])
        obj_infos[name] = {"x": x, "y": y, "size": size}

    return obj_infos


def build_initial_state(
    objects: Sequence[str],
    obj_types: Dict[str, str],
    relations: Sequence[Tuple[str, str, str]],
) -> Tuple[State, Dict[str, set]]:
    fluents = set()

    for obj in objects:
        fluents.add(F("pickloc", obj))

    for obj, type_name in obj_types.items():
        fluents.add(F(type_name, obj))

    for rel_name, obj1, obj2 in relations:
        fluents.add(F(rel_name, obj1, obj2))

    fluents.add(F("H0"))
    fluents.add(F("S0"))

    state = State(fluents=fluents)
    objects_by_type = {"object": set(objects)}
    return state, objects_by_type


# ---------------------------------------------------------------------------
# Open-loop nominal A* / Dijkstra planning
# ---------------------------------------------------------------------------

def state_key(state: State) -> frozenset:
    return frozenset(str(f) for f in state.fluents)


def action_operator_name(action_name: str) -> str:
    return action_name.split()[0]


def action_probability(action_name: str) -> float:
    """Return learned nominal success probability for a grounded action."""
    op_name = action_operator_name(action_name)
    return float(NOMINAL_ACTION_PROBS.get(op_name, 1.0))


def action_cost(action_name: str, eps: float = 1e-12) -> float:
    p = max(action_probability(action_name), eps)
    return -math.log(p)


def action_debug_string(action_name: str) -> str:
    op_name = action_operator_name(action_name)

    if op_name in NOMINAL_ACTION_PROBS:
        # stack9__stacked -> stack9
        original_name = op_name.split("__", 1)[0]
        branch = NOMINAL_ACTION_EFFECTS.get(op_name, "?")
        p = NOMINAL_ACTION_PROBS[op_name]
        dist = OPERATOR_EFFECT_DISTS.get(original_name, {})
        dist_str = ", ".join(
            f"{name}:{prob:.3f}"
            for name, prob in sorted(dist.items(), key=lambda item: (-item[1], item[0]))
        )
        return f"nominal={branch}, p={p:.3f}, full_dist=[{dist_str}]"

    return "auxiliary/deterministic"


def deterministic_successor(state: State, action) -> Optional[State]:
    """Apply a deterministic nominal Railroad action."""
    outcomes = transition(state, action)
    if not outcomes:
        return None
    return outcomes[0][0]


def open_loop_nominal_plan(
    start_state: State,
    goal,
    all_actions,
    max_steps: int,
    debug: bool = False,
) -> Tuple[List[str], float, float]:
    """Find a fixed nominal plan minimizing -log(success probability)."""
    start_key = state_key(start_state)

    heap = []
    tie_id = 0
    heappush(heap, (0.0, 0, tie_id, start_state, []))

    # Best known lexicographic cost for each state: (prob_cost, steps)
    best_cost: Dict[frozenset, Tuple[float, int]] = {start_key: (0.0, 0)}

    while heap:
        g_prob_cost, steps, _tie, state, history = heappop(heap)
        key = state_key(state)

        # Skip stale heap entries.
        if best_cost.get(key) != (g_prob_cost, steps):
            continue

        if goal.evaluate(state.fluents):
            plan_probability = math.exp(-g_prob_cost)
            return history, plan_probability, g_prob_cost

        if steps >= max_steps:
            continue

        usable = sorted(get_usable_actions(state, all_actions), key=lambda a: a.name)

        if debug:
            print(f"\nExpanded state at depth {steps}, usable actions: {len(usable)}")

        for action in usable:
            try:
                next_state = deterministic_successor(state, action)
            except Exception as exc:
                if debug:
                    print(f"  skip {action.name}: transition error {exc}")
                continue

            if next_state is None:
                continue

            step_cost = action_cost(action.name)
            new_prob_cost = g_prob_cost + step_cost
            new_steps = steps + 1
            next_key = state_key(next_state)

            old = best_cost.get(next_key)
            if old is not None:
                old_prob_cost, old_steps = old
                if (
                    new_prob_cost > old_prob_cost + 1e-12
                    or (
                        abs(new_prob_cost - old_prob_cost) <= 1e-12
                        and new_steps >= old_steps
                    )
                ):
                    continue

            best_cost[next_key] = (new_prob_cost, new_steps)
            tie_id += 1
            new_history = list(history)
            new_history.append(action.name)
            heappush(heap, (new_prob_cost, new_steps, tie_id, next_state, new_history))

    return [], 0.0, math.inf


# ---------------------------------------------------------------------------
# Plan output
# ---------------------------------------------------------------------------

def extract_plan_actions(action_history: Sequence[str]) -> List[Tuple[str, str]]:
    """Extract physical stack commands as (below_obj, above_obj)."""
    stack_actions = []

    for action_name in action_history:
        parts = action_name.split()
        if len(parts) != 3:
            continue

        op_name = parts[0]
        if op_name.startswith("stack"):
            below = parts[1]
            above = parts[2]
            stack_actions.append((below, above))

    return stack_actions


def write_plan_file(
    plan_path: str,
    objects_path: str,
    stack_actions: Sequence[Tuple[str, str]],
    plan_probability: float,
) -> None:
    """Write plan.txt in the format expected by DeepSym execute_plan.py."""
    if os.path.exists(objects_path):
        obj_infos = parse_objects_txt(objects_path)
        with open(plan_path, "w") as f:
            f.write(f"{len(obj_infos)}\n")
            for name, info in obj_infos.items():
                f.write(f"{name} {info['x']:.5f} {info['y']:.5f} {info['size']:.5f}\n")
            f.write(f"plan probability: {plan_probability:.6f}\n")
            if stack_actions and plan_probability > 0.0:
                for below, above in stack_actions:
                    f.write(f"stack {below.upper()} {above.upper()}\n")
            else:
                f.write("not found.\n")
        return

    with open(plan_path, "w") as f:
        f.write(f"plan probability: {plan_probability:.6f}\n")
        if stack_actions and plan_probability > 0.0:
            for below, above in stack_actions:
                f.write(f"stack {below.upper()} {above.upper()}\n")
        else:
            f.write("not found.\n")


def main() -> None:
    parser = argparse.ArgumentParser("Railroad Option A open-loop planner for DeepSym.")
    parser.add_argument("-opts", help="option file", type=str, required=True)
    parser.add_argument("-goal", help="goal state", type=str, default="(H3) (S0)")
    parser.add_argument(
        "-max-steps",
        help="maximum symbolic planning steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="print selected action probabilities and learned distributions",
    )
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    operators_path = os.path.join(save_dir, "railroad_operators.json")
    if not os.path.exists(operators_path):
        raise FileNotFoundError(
            f"Railroad operators not found at {operators_path}. "
            "Run learn_rules_railroad.py first."
        )

    all_operators = load_operators_from_json(operators_path)
    print(f"Loaded and reconstructed {len(all_operators)} nominal/auxiliary operators")

    problem_path = os.path.join(save_dir, "problem.pddl")
    objects, obj_types, relations = parse_problem_pddl(problem_path)
    print(f"Objects: {objects}")
    print(f"Types: {obj_types}")
    print(f"Relations: {len(relations)}")

    state, objects_by_type = build_initial_state(objects, obj_types, relations)
    print(f"Initial state built with {len(state.fluents)} fluents")

    all_actions = []
    for op in all_operators:
        all_actions.extend(op.instantiate(objects_by_type))
    all_actions.sort(key=lambda a: a.name)
    print(f"Grounded {len(all_actions)} actions from {len(all_operators)} operators")

    goal = parse_pddl_goal(args.goal)
    print(f"Goal: {goal}")

    print("\n=== Starting Railroad Option A open-loop nominal planning ===")
    action_history, plan_probability, prob_cost = open_loop_nominal_plan(
        start_state=state,
        goal=goal,
        all_actions=all_actions,
        max_steps=args.max_steps,
        debug=False,
    )

    if action_history:
        print(
            f"Goal reached with fixed-plan probability {plan_probability:.6f} "
            f"(cost={prob_cost:.6f})"
        )
        for i, act in enumerate(action_history):
            print(f"Step {i}: {act}")
            if args.debug_actions:
                print(f"        {action_debug_string(act)}")
    else:
        print("Planner returned NONE: no positive-probability fixed nominal plan found.")
        plan_probability = 0.0

    stack_actions = extract_plan_actions(action_history)

    plan_path = os.path.join(save_dir, "plan.txt")
    objects_path = os.path.join(save_dir, "objects.txt")
    write_plan_file(
        plan_path=plan_path,
        objects_path=objects_path,
        stack_actions=stack_actions,
        plan_probability=plan_probability,
    )

    print(f"\n=== Plan written to {plan_path} ===")
    print(f"Plan probability: {plan_probability:.6f}")
    print(f"Symbolic actions taken: {len(action_history)}")
    print(f"Physical stack actions: {len(stack_actions)}")
    for below, above in stack_actions:
        print(f"  stack {above} on {below}")


if __name__ == "__main__":
    main()