"""Deterministic/Nominal Railroad planner for DeepSym.

This is the non-expected-reachability Railroad planner. It is intended to be the
Railroad-side counterpart of the original PDDL/PPDDL planning pipeline for
comparing plans.

Input files, all under opts["save"]:
    railroad_operators.json   learned operator specs from learn_rules_railroad.py
    railroad_problem.json     current recognized symbolic scene
    objects.txt               object positions/sizes for execute_plan.py plan header

What this planner does:
    - reconstructs learned stack operators from railroad_operators.json
    - determinizes each probabilistic stack operator by selecting one nominal
      effect, by default the most probable effect
    - skips stack actions whose nominal effect is roll/tumble, because they do
      not progress DeepSym's H/S goals and only consume an object
    - performs deterministic best-first search over Railroad symbolic states
    - writes save/plan.txt in the same simple stack-command format used by the
      existing execute_plan.py pipeline

This file intentionally does NOT do expected reachability or MCTS. Use
make_plan_railroad_expected.py for policy/value planning over all probabilistic
branches.

Usage:
    python make_plan_railroad.py -opts opts.yaml -goal "(H3)"
    python make_plan_railroad.py -opts opts.yaml -goal "(S4)" --debug-actions
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from railroad._bindings import AndGoal, Fluent as F, LiteralGoal, State
from railroad.core import Effect, Operator, transition
from railroad.planner import get_usable_actions


PROGRESS_EFFECTS = {"stacked", "inserted"}
FAILURE_EFFECTS = {"roll1", "roll2", "tumble1", "tumble2"}

# Stored for debugging / action ordering.
OPERATOR_NOMINAL_EFFECT: Dict[str, str] = {}
OPERATOR_NOMINAL_PROB: Dict[str, float] = {}
OPERATOR_EFFECT_DISTS: Dict[str, Dict[str, float]] = {}


# ---------------------------------------------------------------------------
# Learned operator reconstruction
# ---------------------------------------------------------------------------

def _normalize_effect_distribution(prob_effects: Sequence[Dict], *, op_name: str) -> List[Dict]:
    merged: Dict[str, float] = {}
    for pe in prob_effects:
        name = str(pe["effect_name"])
        p = float(pe["probability"])
        if p <= 0.0:
            continue
        merged[name] = merged.get(name, 0.0) + p

    total = sum(merged.values())
    if total <= 0.0:
        raise ValueError(f"Operator {op_name} has an empty effect distribution")

    return [
        {"effect_name": name, "probability": p / total}
        for name, p in sorted(merged.items())
    ]


def choose_nominal_effect(prob_effects: Sequence[Dict], mode: str) -> Tuple[str, float]:
    """Choose one deterministic effect from a learned distribution.

    mode="argmax": choose the highest-probability effect, regardless of whether
    it is progress or failure. This is the closest nominal deterministic version.

    mode="progress": choose the highest-probability progress effect
    (stacked/inserted) if any exists; otherwise fall back to argmax. This is
    useful for debugging, but is less faithful to the learned distribution.
    """
    if not prob_effects:
        raise ValueError("Empty prob_effects")

    if mode == "progress":
        progress = [pe for pe in prob_effects if str(pe["effect_name"]) in PROGRESS_EFFECTS]
        if progress:
            best = max(progress, key=lambda pe: (float(pe["probability"]), str(pe["effect_name"])))
            return str(best["effect_name"]), float(best["probability"])

    best = max(prob_effects, key=lambda pe: (float(pe["probability"]), str(pe["effect_name"])))
    return str(best["effect_name"]), float(best["probability"])


def _effect_fluents(effect_name: str) -> set:
    """Map one DeepSym effect label to Railroad fluents."""
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

    if effect_name in FAILURE_EFFECTS:
        return {F(effect_name)}

    # Preserve unknown labels as fluents rather than crashing.
    return {F(effect_name)}


def build_nominal_stack_operator(spec: Dict, nominal_mode: str) -> Operator:
    below_type = spec.get("below_type", spec.get("obj1_type"))
    above_type = spec.get("above_type", spec.get("obj2_type"))
    if below_type is None or above_type is None:
        raise ValueError(f"Malformed stack operator spec: {spec}")

    op_name = str(spec["name"])
    dist = _normalize_effect_distribution(spec["prob_effects"], op_name=op_name)
    nominal_effect, nominal_prob = choose_nominal_effect(dist, nominal_mode)

    OPERATOR_EFFECT_DISTS[op_name] = {
        str(pe["effect_name"]): float(pe["probability"])
        for pe in dist
    }
    OPERATOR_NOMINAL_EFFECT[op_name] = nominal_effect
    OPERATOR_NOMINAL_PROB[op_name] = nominal_prob

    preconditions = [
        ~F("stacked"),
        ~F("inserted"),
        F("pickloc", "?above"),
        F("stackloc", "?below"),
        F(str(below_type), "?below"),
        F(str(above_type), "?above"),
        F(str(spec["relation"]), "?below", "?above"),
    ]

    # Deterministic version of the selected branch. The original PPDDL also
    # removes pickloc ?above outside the probabilistic effect, so we include it
    # for every nominal outcome.
    resulting = set(_effect_fluents(nominal_effect))
    resulting.add(~F("pickloc", "?above"))

    return Operator(
        name=op_name,
        parameters=[("?below", "object"), ("?above", "object")],
        preconditions=preconditions,
        effects=[Effect(time=0, resulting_fluents=resulting)],
    )


def build_auxiliary_operator(spec: Dict) -> Operator:
    if spec["type"] == "increase_height":
        return Operator(
            name=spec["name"],
            parameters=[],
            preconditions=[F("stacked"), F(spec["from_counter"])],
            effects=[Effect(
                time=0,
                resulting_fluents={
                    ~F(spec["from_counter"]),
                    F(spec["to_counter"]),
                    ~F("stacked"),
                },
            )],
        )

    if spec["type"] == "increase_stack":
        return Operator(
            name=spec["name"],
            parameters=[],
            preconditions=[F("inserted"), F(spec["from_counter"])],
            effects=[Effect(
                time=0,
                resulting_fluents={
                    ~F(spec["from_counter"]),
                    F(spec["to_counter"]),
                    ~F("inserted"),
                },
            )],
        )

    if spec["type"] == "makebase":
        return Operator(
            name="makebase",
            parameters=[("?obj", "object")],
            preconditions=[~F("base")],
            effects=[Effect(
                time=0,
                resulting_fluents={
                    F("base"),
                    F("stacked"),
                    F("inserted"),
                    ~F("pickloc", "?obj"),
                    F("stackloc", "?obj"),
                },
            )],
        )

    raise ValueError(f"Unknown auxiliary operator type: {spec['type']}")


def load_operators_from_json(json_path: str, nominal_mode: str) -> List[Operator]:
    with open(json_path, "r") as f:
        all_specs = json.load(f)

    OPERATOR_NOMINAL_EFFECT.clear()
    OPERATOR_NOMINAL_PROB.clear()
    OPERATOR_EFFECT_DISTS.clear()

    operators: List[Operator] = []
    for spec in all_specs["stack_operators"]:
        operators.append(build_nominal_stack_operator(spec, nominal_mode=nominal_mode))
    for spec in all_specs["auxiliary_operators"]:
        operators.append(build_auxiliary_operator(spec))
    return operators


# ---------------------------------------------------------------------------
# Problem parsing
# ---------------------------------------------------------------------------

def parse_deepsym_goal(goal_str: str):
    negative_tokens = set(re.findall(r"\(not\s*\((\w+)\)\)", goal_str))

    without_not_clauses = re.sub(r"\(not\s*\(\w+\)\)", " ", goal_str)
    positive_tokens = [
        tok for tok in re.findall(r"\((\w+)\)", without_not_clauses)
        if tok.lower() not in {"and", "goal"}
    ]

    goals = [LiteralGoal(F(tok)) for tok in positive_tokens]
    goals.extend(LiteralGoal(~F(tok)) for tok in sorted(negative_tokens))

    # DeepSym generated goals include these; add them for command line goals.
    if "stacked" not in negative_tokens:
        goals.append(LiteralGoal(~F("stacked")))
    if "inserted" not in negative_tokens:
        goals.append(LiteralGoal(~F("inserted")))

    if not goals:
        raise ValueError(f"No goal fluents found in: {goal_str}")
    return goals[0] if len(goals) == 1 else AndGoal(goals)


# Backward-compatible name for older imports.
parse_pddl_goal = parse_deepsym_goal


def _extract_parenthesized_section(content: str, start_token: str, end_token: str) -> str:
    start = content.find(start_token)
    if start < 0:
        return ""
    start += len(start_token)
    end = content.find(end_token, start)
    if end < 0:
        end = len(content)
    return content[start:end]


def parse_objects_txt(objects_path: str) -> Dict[str, Dict[str, float]]:
    infos: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(objects_path):
        return infos

    with open(objects_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return infos

    try:
        n = int(lines[0].split()[0])
    except Exception:
        return infos

    for line in lines[1:1 + n]:
        parts = line.split()
        if len(parts) < 4:
            continue
        infos[parts[0]] = {
            "x": float(parts[1]),
            "y": float(parts[2]),
            "size": float(parts[3]),
        }
    return infos


def parse_problem_railroad(scene_path: str, objects_path: Optional[str] = None):
    """Parse DeepSym/Railroad scene JSON for objects, types, relations, counters.

    This replaces the previous use of problem.pddl in the Railroad path.  The
    JSON is written by recognize.py as save_dir/railroad_problem.json.

    Expected format:
        {
          "objects": [
             {"name": "O1", "type": "objtype3", "loc": [x, y], "size": s}, ...
          ],
          "relations": [
             {"name": "relation0", "below": "O1", "above": "O2"}, ...
          ],
          "counters": {"H": "H0", "S": "S0"}
        }
    """
    if not os.path.exists(scene_path):
        raise FileNotFoundError(
            f"Missing {scene_path}. Run recognize.py first; it should write railroad_problem.json."
        )

    with open(scene_path, "r") as f:
        data = json.load(f)

    raw_objects = data.get("objects", [])
    if not raw_objects:
        raise ValueError(f"{scene_path} has no objects")

    objects: List[str] = []
    obj_types: Dict[str, str] = {}
    object_infos: Dict[str, Dict[str, float]] = {}

    for obj in raw_objects:
        name = str(obj["name"]).upper()
        typ = str(obj["type"])
        loc = obj.get("loc", obj.get("location", [0.0, 0.0]))
        size = float(obj.get("size", 0.1))
        objects.append(name)
        obj_types[name] = typ
        object_infos[name] = {"x": float(loc[0]), "y": float(loc[1]), "size": size}

    # objects.txt is still allowed only as execution geometry fallback.  It is
    # not a symbolic planning input and can be removed once plan writing uses
    # railroad_problem.json directly everywhere.
    if objects_path and os.path.exists(objects_path):
        txt_infos = parse_objects_txt(objects_path)
        if txt_infos:
            object_infos.update({k.upper(): v for k, v in txt_infos.items()})

    relations: List[Tuple[str, str, str]] = []
    for rel in data.get("relations", []):
        if isinstance(rel, dict):
            r = str(rel.get("name", rel.get("relation")))
            below = str(rel.get("below", rel.get("obj1"))).upper()
            above = str(rel.get("above", rel.get("obj2"))).upper()
        else:
            r, below, above = rel
            r = str(r)
            below = str(below).upper()
            above = str(above).upper()
        relations.append((r, below, above))

    counters_data = data.get("counters", {})
    if isinstance(counters_data, dict):
        counters = set(str(v) for v in counters_data.values())
    else:
        counters = set(str(c) for c in counters_data)
    if not any(c.startswith("H") for c in counters):
        counters.add("H0")
    if not any(c.startswith("S") for c in counters):
        counters.add("S0")

    missing_types = [obj for obj in objects if obj not in obj_types]
    if missing_types:
        raise ValueError(f"Objects missing types in railroad_problem.json: {missing_types}")

    return objects, obj_types, relations, counters, object_infos


# Backward-compatible name for older imports.  In the Railroad path this no
# longer parses PDDL; it expects railroad_problem.json.
def parse_problem_pddl(problem_path: str, objects_path: Optional[str] = None):
    return parse_problem_railroad(problem_path, objects_path)


def build_initial_state(objects: Sequence[str],
                        obj_types: Dict[str, str],
                        relations: Sequence[Tuple[str, str, str]],
                        counters: Iterable[str]):
    fluents = set()

    for obj in objects:
        fluents.add(F("pickloc", obj))

    for obj, typ in obj_types.items():
        fluents.add(F(typ, obj))

    for rel, obj1, obj2 in relations:
        fluents.add(F(rel, obj1, obj2))

    counter_set = set(counters)
    h_counters = sorted(c for c in counter_set if c.startswith("H"))
    s_counters = sorted(c for c in counter_set if c.startswith("S"))
    fluents.add(F(h_counters[0] if h_counters else "H0"))
    fluents.add(F(s_counters[0] if s_counters else "S0"))

    return State(fluents=fluents), {"object": set(objects)}


# ---------------------------------------------------------------------------
# Deterministic planner
# ---------------------------------------------------------------------------

def state_key(state: State) -> Tuple[str, ...]:
    return tuple(sorted(str(f) for f in state.fluents))


def action_base_name(action_name: str) -> str:
    return action_name.split()[0]


def is_stack_action(action_name: str) -> bool:
    return action_base_name(action_name).startswith("stack")


def transition_one(state: State, action) -> Optional[State]:
    """Apply a deterministic action safely."""
    try:
        outcomes = transition(state, action)
    except Exception as exc:
        msg = str(exc)
        if "Precondition not satisfied" in msg or "precondition" in msg.lower():
            return None
        raise

    if not outcomes:
        return None

    # Deterministic operators should produce one outcome. If Railroad returns
    # multiple identical/weighted outcomes for any reason, pick the highest p.
    next_state, _p = max(outcomes, key=lambda item: float(item[1]))
    return next_state


def _has_fluent(state: State, name: str) -> bool:
    return any(re.search(rf"\b{name}\b", str(f)) for f in state.fluents)


def _counter_value(state: State, prefix: str) -> int:
    best = 0
    pattern = re.compile(rf"\b{prefix}(\d+)\b")
    for fl in state.fluents:
        m = pattern.search(str(fl))
        if m:
            best = max(best, int(m.group(1)))
    return best


def _target_counter(goal_str: str, prefix: str) -> int:
    vals = [int(x) for x in re.findall(rf"\({prefix}(\d+)\)", goal_str)]
    return max(vals) if vals else 0


def heuristic(state: State, target_h: int, target_s: int) -> int:
    h = _counter_value(state, "H")
    s = _counter_value(state, "S")
    remaining = max(0, target_h - h) + max(0, target_s - s)
    # stacked/inserted flags must be consumed by auxiliary actions before the
    # next physical stack action is possible.
    if _has_fluent(state, "stacked"):
        remaining += 1
    if _has_fluent(state, "inserted"):
        remaining += 1
    return remaining


def action_sort_key(action) -> Tuple[int, float, str]:
    name = action.name
    base = action_base_name(name)

    if base.startswith("increase_height") or base.startswith("increase_stack"):
        return (0, 0.0, name)
    if base == "makebase":
        return (1, 0.0, name)
    if base.startswith("stack"):
        prob = OPERATOR_NOMINAL_PROB.get(base, 0.0)
        return (2, -prob, name)
    return (3, 0.0, name)


def should_skip_action(action, skip_nonprogress_stack: bool) -> bool:
    base = action_base_name(action.name)
    if not base.startswith("stack"):
        return False
    if not skip_nonprogress_stack:
        return False
    return OPERATOR_NOMINAL_EFFECT.get(base) not in PROGRESS_EFFECTS


def deterministic_plan(initial_state: State,
                       goal,
                       goal_str: str,
                       all_actions: Sequence,
                       max_steps: int,
                       skip_nonprogress_stack: bool,
                       debug: bool = False) -> Tuple[Optional[List[str]], Optional[State]]:
    target_h = _target_counter(goal_str, "H")
    target_s = _target_counter(goal_str, "S")

    start_key = state_key(initial_state)
    best_depth: Dict[Tuple[str, ...], int] = {start_key: 0}

    # heap entries: (heuristic, depth, tie_id, state, history)
    heap = []
    tie = 0
    heapq.heappush(heap, (heuristic(initial_state, target_h, target_s), 0, tie, initial_state, []))

    expansions = 0
    while heap:
        _h, depth, _tie, state, history = heapq.heappop(heap)
        key = state_key(state)
        if depth != best_depth.get(key, depth):
            continue

        if goal.evaluate(state.fluents):
            return history, state

        if depth >= max_steps:
            continue

        expansions += 1
        usable = list(get_usable_actions(state, all_actions))
        usable.sort(key=action_sort_key)

        for action in usable:
            if should_skip_action(action, skip_nonprogress_stack):
                if debug:
                    base = action_base_name(action.name)
                    print(f"skip non-progress {action.name}: nominal={OPERATOR_NOMINAL_EFFECT.get(base)}")
                continue

            next_state = transition_one(state, action)
            if next_state is None:
                continue

            next_key = state_key(next_state)
            next_depth = depth + 1
            if best_depth.get(next_key, 10**9) <= next_depth:
                continue

            best_depth[next_key] = next_depth
            tie += 1
            prio = heuristic(next_state, target_h, target_s)
            heapq.heappush(heap, (prio, next_depth, tie, next_state, history + [action.name]))

    if debug:
        print(f"Search exhausted after {expansions} expanded states")
    return None, None


# ---------------------------------------------------------------------------
# Output / debug
# ---------------------------------------------------------------------------

def extract_stack_actions(action_history: Sequence[str]) -> List[Tuple[str, str]]:
    """Extract physical stack commands as (below, above)."""
    result: List[Tuple[str, str]] = []
    for action_name in action_history:
        parts = action_name.split()
        if len(parts) == 3 and parts[0].startswith("stack"):
            result.append((parts[1], parts[2]))
    return result


def plan_probability(action_history: Sequence[str]) -> float:
    p = 1.0
    for action_name in action_history:
        base = action_base_name(action_name)
        if base.startswith("stack"):
            p *= float(OPERATOR_NOMINAL_PROB.get(base, 1.0))
    return p


def write_plan(plan_path: str,
               object_infos: Dict[str, Dict[str, float]],
               probability: float,
               stack_actions: Sequence[Tuple[str, str]],
               found: bool) -> None:
    with open(plan_path, "w") as f:
        if object_infos:
            f.write(f"{len(object_infos)}\n")
            for name, info in object_infos.items():
                f.write(f"{name} {info['x']:.5f} {info['y']:.5f} {info['size']:.5f}\n")

        if found:
            f.write(f"plan probability: {probability:.6f}\n")
            for below, above in stack_actions:
                f.write(f"stack {below.upper()} {above.upper()}\n")
        else:
            f.write("not found.\n")


def debug_action(action_name: str) -> None:
    base = action_base_name(action_name)
    if not base.startswith("stack"):
        print("        auxiliary/deterministic")
        return
    dist = OPERATOR_EFFECT_DISTS.get(base, {})
    dist_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(dist.items()))
    nominal = OPERATOR_NOMINAL_EFFECT.get(base)
    p = OPERATOR_NOMINAL_PROB.get(base, 0.0)
    print(f"        nominal={nominal}:{p:.3f}, learned_dist=[{dist_str}]")


def main() -> None:
    parser = argparse.ArgumentParser("Nominal deterministic Railroad planner for DeepSym.")
    parser.add_argument("-opts", type=str, required=True, help="option file")
    parser.add_argument("-goal", type=str, default="(H3)", help="DeepSym goal, e.g. '(H3)' or '(S4)'")
    parser.add_argument("-max-steps", type=int, default=25, help="maximum deterministic symbolic search depth")
    parser.add_argument("--debug-actions", action="store_true", help="print selected action distributions")
    parser.add_argument(
        "--nominal-mode",
        choices=["argmax", "progress"],
        default="argmax",
        help="how to determinize probabilistic stack operators",
    )
    parser.add_argument(
        "--allow-nonprogress-stack",
        action="store_true",
        help="do not skip stack actions whose nominal effect is roll/tumble",
    )
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    operators_path = os.path.join(save_dir, "railroad_operators.json")
    if not os.path.exists(operators_path):
        print(f"ERROR: Railroad operators not found at {operators_path}")
        print("Run learn_rules_railroad.py first.")
        sys.exit(1)

    all_operators = load_operators_from_json(operators_path, nominal_mode=args.nominal_mode)
    print(f"Loaded and reconstructed {len(all_operators)} nominal/auxiliary operators")

    scene_path = os.path.join(save_dir, "railroad_problem.json")
    objects_path = os.path.join(save_dir, "objects.txt")
    objects, obj_types, relations, counters, object_infos = parse_problem_railroad(scene_path, objects_path)

    print(f"Objects: {objects}")
    print(f"Types: {obj_types}")
    print(f"Relations: {len(relations)}")

    initial_state, objects_by_type = build_initial_state(objects, obj_types, relations, counters)
    print(f"Initial state built with {len(initial_state.fluents)} fluents")

    all_actions = []
    for op in all_operators:
        all_actions.extend(op.instantiate(objects_by_type))
    all_actions.sort(key=lambda a: a.name)
    print(f"Grounded {len(all_actions)} actions from {len(all_operators)} operators")

    goal = parse_deepsym_goal(args.goal)
    print(f"Goal: {goal}")

    print("\n=== Starting Railroad nominal deterministic planning ===")
    history, final_state = deterministic_plan(
        initial_state=initial_state,
        goal=goal,
        goal_str=args.goal,
        all_actions=all_actions,
        max_steps=args.max_steps,
        skip_nonprogress_stack=not args.allow_nonprogress_stack,
        debug=args.debug_actions,
    )

    found = history is not None and final_state is not None and bool(goal.evaluate(final_state.fluents))
    if found:
        prob = plan_probability(history)
        print(f"Plan found with nominal branch probability: {prob:.6f}")
        for i, act in enumerate(history):
            print(f"Step {i}: {act}")
            if args.debug_actions:
                debug_action(act)
        stack_actions = extract_stack_actions(history)
    else:
        prob = 0.0
        stack_actions = []
        print("No plan found.")

    plan_path = os.path.join(save_dir, "plan.txt")
    write_plan(plan_path, object_infos, prob, stack_actions, found)

    print(f"\n=== Plan written to {plan_path} ===")
    print(f"Physical stack actions: {len(stack_actions)}")
    for below, above in stack_actions:
        print(f"  stack {above} on {below}")


if __name__ == "__main__":
    main()