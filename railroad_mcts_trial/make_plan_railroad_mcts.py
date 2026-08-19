#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from railroad._bindings import AndGoal, Fluent as F, LiteralGoal, State
from railroad.core import Effect, Operator, transition
from railroad.planner import MCTSPlanner


OPERATOR_EFFECT_DISTS: Dict[str, Dict[str, float]] = {}


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

    if abs(total - 1.0) > 1e-6:
        print(
            f"WARNING: probabilities for {op_name} sum to {total:.6f}; normalizing.",
            file=sys.stderr,
        )

    return [
        {"probability": p / total, "effect_name": name}
        for name, p in sorted(merged.items())
    ]


def _effect_fluents(effect_name: str) -> set:
    """Map a DeepSym effect label to Railroad fluents."""
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


    if effect_name in {"roll1", "roll2", "tumble1", "tumble2"}:
        return {F(effect_name)}
    return {F(effect_name)}


def build_probabilistic_stack_operator(spec: Dict) -> Operator:
    """Build one probabilistic Railroad Operator from one learned stack rule."""
    below_type = spec.get("below_type", spec.get("obj1_type"))
    above_type = spec.get("above_type", spec.get("obj2_type"))
    if below_type is None or above_type is None:
        raise ValueError(f"Malformed stack operator spec: {spec}")

    op_name = str(spec["name"])
    normalized_effects = _normalize_effect_distribution(spec["prob_effects"], op_name=op_name)
    OPERATOR_EFFECT_DISTS[op_name] = {
        str(pe["effect_name"]): float(pe["probability"])
        for pe in normalized_effects
    }

    preconditions = [
        ~F("stacked"),
        ~F("inserted"),
        F("pickloc", "?above"),
        F("stackloc", "?below"),
        F(below_type, "?below"),
        F(above_type, "?above"),
        F(spec["relation"], "?below", "?above"),
    ]

    prob_branches = []
    for pe in normalized_effects:
        p = float(pe["probability"])
        effect_name = str(pe["effect_name"])
        branch_effect = Effect(
            time=0,
            resulting_fluents=_effect_fluents(effect_name),
        )
        prob_branches.append((p, [branch_effect]))

    probabilistic_effect = Effect(
        time=0,
        resulting_fluents=set(),
        prob_effects=prob_branches,
    )

    pickloc_removal = Effect(
        time=0,
        resulting_fluents={~F("pickloc", "?above")},
    )

    return Operator(
        name=op_name,
        parameters=[("?below", "object"), ("?above", "object")],
        preconditions=preconditions,
        effects=[probabilistic_effect, pickloc_removal],
    )


def build_auxiliary_operator(spec: Dict) -> Operator:
    """Build deterministic DeepSym bookkeeping operators."""
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


def load_operators_from_json(json_path: str) -> List[Operator]:
    """Load learned operator specs and reconstruct Railroad Operators."""
    with open(json_path, "r") as f:
        all_specs = json.load(f)

    OPERATOR_EFFECT_DISTS.clear()

    operators: List[Operator] = []
    for spec in all_specs["stack_operators"]:
        operators.append(build_probabilistic_stack_operator(spec))
    for spec in all_specs["auxiliary_operators"]:
        operators.append(build_auxiliary_operator(spec))
    return operators


def parse_deepsym_goal(goal_str: str):
    """Parse a small DeepSym goal string into a Railroad goal expression."""
    negative_tokens = set(re.findall(r"\(not\s*\((\w+)\)\)", goal_str))

    without_not_clauses = re.sub(r"\(not\s*\(\w+\)\)", " ", goal_str)
    positive_tokens = [
        tok for tok in re.findall(r"\((\w+)\)", without_not_clauses)
        if tok.lower() not in {"and", "goal"}
    ]

    goals = [LiteralGoal(F(tok)) for tok in positive_tokens]
    goals.extend(LiteralGoal(~F(tok)) for tok in sorted(negative_tokens))

    # DeepSym's generated PDDL goals include these; add them for command-line
    # goal strings like "(H3)" or "(S4)".
    if "stacked" not in negative_tokens:
        goals.append(LiteralGoal(~F("stacked")))
    if "inserted" not in negative_tokens:
        goals.append(LiteralGoal(~F("inserted")))

    if not goals:
        raise ValueError(f"No goal fluents found in: {goal_str}")
    return goals[0] if len(goals) == 1 else AndGoal(goals)


def parse_objects_txt(objects_path: str) -> Dict[str, Dict[str, float]]:
    """Parse DeepSym objects.txt/plan header.

    Format:
        5
        O1 x y size
        ...
    """
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
    fluents.add(F("H0" if not any(c.startswith("H") for c in counter_set) else next(c for c in counter_set if c.startswith("H"))))
    fluents.add(F("S0" if not any(c.startswith("S") for c in counter_set) else next(c for c in counter_set if c.startswith("S"))))

    return State(fluents=fluents), {"object": set(objects)}


def state_key(state: State) -> Tuple[str, ...]:
    """Hashable state key for DeepSym's zero-time symbolic states."""
    return tuple(sorted(str(f) for f in state.fluents))


def is_progress_state(state: State) -> bool:
    """A progress stack outcome creates the transient flags consumed by aux operators."""
    return F("stacked") in state.fluents or F("inserted") in state.fluents


def transition_safe(state: State, action) -> List[Tuple[State, float]]:
    """Call Railroad transition and normalize/merge identical states.

    `get_usable_actions` can be permissive with generated operators and negated
    fluents. Treat Railroad transition() as the final precondition check. If an
    action is rejected by transition(), skip it in the expected-reachability DP
    instead of crashing.
    """
    try:
        outcomes = transition(state, action)
    except Exception as exc:  # Railroad may raise bindings-side RuntimeError.
        msg = str(exc)
        if "Precondition not satisfied" in msg or "precondition" in msg.lower():
            return []
        raise

    merged: Dict[Tuple[str, ...], Tuple[State, float]] = {}

    for next_state, p in outcomes:
        p = float(p)
        if p <= 0.0:
            continue
        key = state_key(next_state)
        if key in merged:
            old_state, old_p = merged[key]
            merged[key] = (old_state, old_p + p)
        else:
            merged[key] = (next_state, p)

    total = sum(p for _, p in merged.values())
    if total <= 0.0:
        return []
    return [(s, p / total) for s, p in merged.values()]


def extract_stack_actions(action_history: Sequence[str]) -> List[Tuple[str, str]]:
    """Extract physical stack commands as (below, above)."""
    result: List[Tuple[str, str]] = []
    for action_name in action_history:
        parts = action_name.split()
        if len(parts) == 3 and parts[0].startswith("stack"):
            result.append((parts[1], parts[2]))
    return result


def write_plan(plan_path: str,
               object_infos: Dict[str, Dict[str, float]],
               expected_probability: float,
               stack_actions: Sequence[Tuple[str, str]],
               goal_reached_in_rollout: bool) -> None:
    with open(plan_path, "w") as f:
        if object_infos:
            f.write(f"{len(object_infos)}\n")
            for name, info in object_infos.items():
                f.write(f"{name} {info['x']:.5f} {info['y']:.5f} {info['size']:.5f}\n")

        f.write(f"plan probability: {expected_probability:.6f}\n")
        if expected_probability > 0.0 and stack_actions:
            for below, above in stack_actions:
                f.write(f"stack {below.upper()} {above.upper()}\n")
        elif goal_reached_in_rollout:
            # No physical stack may be needed for a trivial goal.
            pass
        else:
            f.write("not found.\n")


def debug_action(action_name: str) -> None:
    parts = action_name.split()
    base = parts[0]
    dist = OPERATOR_EFFECT_DISTS.get(base)
    if dist is None:
        print("        auxiliary/deterministic")
        return
    dist_str = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(dist.items()))
    print(f"        learned_dist=[{dist_str}]")


# ---------------------------------------------------------------------------
# Railroad MCTS wrapper
# ---------------------------------------------------------------------------

Action = object


def canonical_action_name(value: object) -> str:
    """Return a whitespace-normalized Railroad action name.

    Railroad 0.2.0 may instantiate a parameterless action with a trailing space
    (for example ``"increase_stack1 "``), while MCTS returns the same action
    as ``"increase_stack1"``.  Action identity should not depend on that
    formatting difference.
    """
    return " ".join(str(value).split())


def selected_action_name(selected: object) -> Optional[str]:
    """Return a canonical grounded action name from MCTS's return value."""
    if selected is None:
        return None
    name = getattr(selected, "name", None)
    text = canonical_action_name(name if name is not None else selected)

    # Railroad C++ returns the sentinel string "NONE" when MCTS has no
    # visited root action. It is not a grounded action name.
    if text.upper() == "NONE":
        return None

    return text or None


def resolve_selected_action(selected: object, actions_by_name: Dict[str, Action]) -> Optional[Action]:
    """Map MCTS output back to the grounded action object used by transition()."""
    if selected is None:
        return None

    # Some Railroad builds return the actual grounded action object.
    direct_name = getattr(selected, "name", None)
    if direct_name is not None:
        canonical = canonical_action_name(direct_name)
        if canonical in actions_by_name:
            return actions_by_name[canonical]

    name = selected_action_name(selected)
    if name is None:
        return None
    return actions_by_name.get(name)


def run_mcts_vote(
    state,
    goal,
    all_actions: Sequence[Action],
    *,
    runs: int,
    iterations: int,
    max_depth: int,
    c: float,
    heuristic_multiplier: float,
    lambda_add: float,
    lambda_max: float,
    lambda_ff: float,
    collect_trace: bool,
):

    if runs < 1:
        raise ValueError("mcts-runs must be at least 1")

    actions_by_name: Dict[str, Action] = {}
    for action in all_actions:
        key = canonical_action_name(action.name)
        previous = actions_by_name.get(key)
        if previous is not None and previous is not action:
            raise RuntimeError(
                "Two grounded Railroad actions collapse to the same canonical "
                f"name {key!r}: {previous.name!r} and {action.name!r}"
            )
        actions_by_name[key] = action
    counts: Counter[str] = Counter()
    elapsed: Dict[str, float] = {}

    selected_objects: Dict[str, object] = {}
    last_trace: Optional[str] = None

    for _ in range(runs):
        planner = MCTSPlanner(
            list(all_actions),
            lambda_add=lambda_add,
            lambda_max=lambda_max,
            lambda_ff=lambda_ff,
        )

        started = time.perf_counter()
        selected = planner(
            state,
            goal,
            max_iterations=iterations,
            max_depth=max_depth,
            c=c,
            heuristic_multiplier=heuristic_multiplier,
        )
        duration = time.perf_counter() - started

        name = selected_action_name(selected)
        if name is not None:
            counts[name] += 1
            elapsed[name] = elapsed.get(name, 0.0) + duration

            if not isinstance(selected, str):
                selected_objects[name] = selected

        if collect_trace:
            try:
                last_trace = str(planner.get_trace_from_last_mcts_tree())
            except Exception as exc:  # Trace is diagnostic, not required.
                last_trace = f"Trace unavailable: {exc}"

    if not counts:
        return None, counts, elapsed, last_trace


    winner_name = sorted(counts, key=lambda name: (-counts[name], name))[0]

    winner = selected_objects.get(winner_name)
    if winner is None:
        winner = actions_by_name.get(winner_name)

    if winner is None:

        available = sorted(actions_by_name)
        close = [name for name in available if name.replace("-", "_") == winner_name.replace("-", "_")]
        raise RuntimeError(
            "Could not map the Railroad MCTS result back to a grounded action. "
            f"MCTS returned {winner_name!r}; close grounded names={close[:10]!r}; "
            f"first grounded names={available[:20]!r}"
        )

    return winner, counts, elapsed, last_trace


def choose_outcome(
    action_name: str,
    outcomes,
    *,
    mode: str,
    rng: random.Random,
):
    """Choose one outcome only for constructing a representative rollout."""
    if not outcomes:
        raise ValueError("Cannot choose an outcome from an empty transition")

    if mode == "sample":
        r = rng.random()
        cumulative = 0.0
        for state, probability in outcomes:
            cumulative += float(probability)
            if r <= cumulative + 1e-12:
                return state, float(probability)
        state, probability = outcomes[-1]
        return state, float(probability)

    if mode == "progress" and action_name.startswith("stack"):
        progress = [
            (state, float(probability))
            for state, probability in outcomes
            if is_progress_state(state)
        ]
        if progress:
            return max(progress, key=lambda item: (item[1], state_key(item[0])))

    # "most-likely", or progress mode when no progress branch exists.
    return max(
        ((state, float(probability)) for state, probability in outcomes),
        key=lambda item: (item[1], state_key(item[0])),
    )


def plan_with_repeated_mcts(
    initial_state,
    goal,
    all_actions: Sequence[Action],
    *,
    max_symbolic_steps: int,
    output_mode: str,
    outcome_mode: str,
    mcts_runs: int,
    iterations: int,
    max_depth: int,
    c: float,
    heuristic_multiplier: float,
    lambda_add: float,
    lambda_max: float,
    lambda_ff: float,
    trace: bool,
    random_seed: Optional[int],
    debug_actions: bool,
):
    """Replan at every symbolic state and build a diagnostic rollout."""
    state = initial_state
    history: List[str] = []
    branch_probability = 1.0
    decision_log: List[Dict] = []
    rng = random.Random(random_seed)
    visited = set()

    for step in range(max_symbolic_steps):
        if goal.evaluate(state.fluents):
            break

        key = state_key(state)
        if key in visited:
            print(
                "WARNING: representative MCTS rollout revisited an identical "
                "state; stopping to avoid a loop.",
                file=sys.stderr,
            )
            break
        visited.add(key)

        selected, votes, elapsed, last_trace = run_mcts_vote(
            state,
            goal,
            all_actions,
            runs=mcts_runs,
            iterations=iterations,
            max_depth=max_depth,
            c=c,
            heuristic_multiplier=heuristic_multiplier,
            lambda_add=lambda_add,
            lambda_max=lambda_max,
            lambda_ff=lambda_ff,
            collect_trace=trace,
        )

        if selected is None:
            decision_log.append(
                {
                    "step": step,
                    "state": list(key),
                    "selected_action": None,
                    "votes": dict(votes),
                    "reason": "MCTS returned no action",
                }
            )
            break

        action_name = canonical_action_name(selected.name)
        outcomes = transition_safe(state, selected)
        if not outcomes:
            raise RuntimeError(
                "MCTS selected an action that Railroad transition() rejected. "
                "This is consistent with the known get_usable_actions()/"
                f"transition() applicability mismatch. Action: {action_name}"
            )

        next_state, chosen_probability = choose_outcome(
            action_name,
            outcomes,
            mode=outcome_mode,
            rng=rng,
        )

        history.append(action_name)
        branch_probability *= chosen_probability

        record = {
            "step": step,
            "state": list(key),
            "selected_action": action_name,
            "votes": dict(sorted(votes.items())),
            "mean_search_seconds_by_action": {
                name: elapsed[name] / votes[name]
                for name in sorted(elapsed)
                if votes[name] > 0
            },
            "outcomes": [
                {
                    "probability": float(probability),
                    "goal": bool(goal.evaluate(outcome_state.fluents)),
                    "progress": bool(is_progress_state(outcome_state)),
                    "state": list(state_key(outcome_state)),
                }
                for outcome_state, probability in outcomes
            ],
            "representative_outcome_probability": chosen_probability,
            "trace": last_trace if trace else None,
        }
        decision_log.append(record)

        print(f"Step {step}: {action_name}")
        print(
            "  votes: "
            + ", ".join(
                f"{name}={count}/{mcts_runs}"
                for name, count in sorted(
                    votes.items(), key=lambda item: (-item[1], item[0])
                )
            )
        )
        print(f"  representative outcome probability: {chosen_probability:.6f}")
        if debug_actions:
            debug_action(action_name)
        if trace and last_trace is not None:
            print("  last MCTS trace:")
            print(last_trace)

        state = next_state

        if output_mode == "next-physical-action" and action_name.startswith("stack"):
            break

    goal_reached = bool(goal.evaluate(state.fluents))
    return history, state, branch_probability, goal_reached, decision_log


def main() -> None:
    parser = argparse.ArgumentParser(
        "Railroad MCTS planner for the DeepSym stacking domain."
    )
    parser.add_argument("-opts", type=str, required=True, help="DeepSym option file")
    parser.add_argument(
        "-goal",
        type=str,
        default="(H3)",
        help="DeepSym goal, e.g. '(H3)', '(S4)', or '(H3) (S4)'",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="MCTS iterations per independent search",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=25,
        help="maximum depth inside each Railroad MCTS search",
    )
    parser.add_argument(
        "--max-symbolic-steps",
        type=int,
        default=25,
        help="maximum number of actions in the representative rollout",
    )
    parser.add_argument(
        "--mcts-runs",
        type=int,
        default=1,
        help=(
            "independent MCTS searches per state; the modal action is selected. "
            "Use 1 for the raw planner and 10-20 for stability experiments."
        ),
    )
    parser.add_argument("--c", type=float, default=1.41421356237, help="UCT exploration constant")
    parser.add_argument(
        "--heuristic-multiplier",
        type=float,
        default=5.0,
        help="weight applied to Railroad's heuristic estimate",
    )
    parser.add_argument("--lambda-add", type=float, default=0.5)
    parser.add_argument("--lambda-max", type=float, default=0.0)
    parser.add_argument("--lambda-ff", type=float, default=0.5)

    parser.add_argument(
        "--output-mode",
        choices=["representative-plan", "next-physical-action"],
        default="representative-plan",
        help=(
            "write a full diagnostic rollout, or only the next physical stack "
            "action for closed-loop execution"
        ),
    )
    parser.add_argument(
        "--rollout-outcome",
        choices=["progress", "most-likely", "sample"],
        default="progress",
        help=(
            "outcome used only to advance the diagnostic rollout; real closed-loop "
            "execution must use the observed outcome"
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="seed for --rollout-outcome sample (not Railroad's internal MCTS RNG)",
    )
    parser.add_argument("--debug-actions", action="store_true")
    parser.add_argument("--trace", action="store_true", help="print/store the last MCTS tree trace")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.max_depth < 1:
        parser.error("--max-depth must be at least 1")
    if args.max_symbolic_steps < 1:
        parser.error("--max-symbolic-steps must be at least 1")
    if args.mcts_runs < 1:
        parser.error("--mcts-runs must be at least 1")

    opts_path = os.path.abspath(args.opts)
    with open(opts_path, "r", encoding="utf-8") as f:
        opts = yaml.safe_load(f) or {}

    if "save" not in opts:
        parser.error(f"Missing required 'save' key in {opts_path}")

    configured_save = os.path.expanduser(str(opts["save"]))
    if os.path.isabs(configured_save):
        save_dir = configured_save
    else:
        save_dir = os.path.normpath(
            os.path.join(os.path.dirname(opts_path), configured_save)
        )

    operators_path = os.path.join(save_dir, "railroad_operators.json")
    scene_path = os.path.join(save_dir, "railroad_problem.json")
    objects_path = os.path.join(save_dir, "objects.txt")

    if not os.path.exists(operators_path):
        print(f"ERROR: Railroad operators not found: {operators_path}", file=sys.stderr)
        print("Copy railroad_operators.json into the save directory or regenerate it.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(scene_path):
        print(f"ERROR: Railroad scene not found: {scene_path}", file=sys.stderr)
        print("Copy railroad_problem.json into the save directory or regenerate it.", file=sys.stderr)
        sys.exit(1)

    all_operators = load_operators_from_json(operators_path)
    objects, obj_types, relations, counters, object_infos = parse_problem_railroad(
        scene_path,
        objects_path,
    )
    initial_state, objects_by_type = build_initial_state(
        objects,
        obj_types,
        relations,
        counters,
    )

    all_actions: List[Action] = []
    for operator in all_operators:
        all_actions.extend(operator.instantiate(objects_by_type))
    all_actions.sort(key=lambda action: canonical_action_name(action.name))

    goal = parse_deepsym_goal(args.goal)

    print(f"Loaded {len(all_operators)} Railroad operators")
    print(f"Grounded {len(all_actions)} actions")
    print(f"Objects: {objects}")
    print(f"Types: {obj_types}")
    print(f"Relations: {len(relations)}")
    print(f"Initial fluents: {len(initial_state.fluents)}")
    print(f"Goal: {goal}")
    print("\n=== Starting Railroad MCTS planning ===")

    history, final_state, branch_probability, goal_reached, decision_log = (
        plan_with_repeated_mcts(
            initial_state,
            goal,
            all_actions,
            max_symbolic_steps=args.max_symbolic_steps,
            output_mode=args.output_mode,
            outcome_mode=args.rollout_outcome,
            mcts_runs=args.mcts_runs,
            iterations=args.iterations,
            max_depth=args.max_depth,
            c=args.c,
            heuristic_multiplier=args.heuristic_multiplier,
            lambda_add=args.lambda_add,
            lambda_max=args.lambda_max,
            lambda_ff=args.lambda_ff,
            trace=args.trace,
            random_seed=args.random_seed,
            debug_actions=args.debug_actions,
        )
    )

    stack_actions = extract_stack_actions(history)
    plan_path = os.path.join(save_dir, "plan.txt")

    # This value is the probability of the selected representative outcome
    # sequence.  It is not an exact MCTS policy success probability.
    plan_probability = branch_probability if history else 0.0
    write_plan(
        plan_path,
        object_infos,
        plan_probability,
        stack_actions,
        goal_reached,
    )

    result = {
        "planner": "railroad_mcts",
        "goal": args.goal,
        "parameters": {
            "iterations": args.iterations,
            "max_depth": args.max_depth,
            "max_symbolic_steps": args.max_symbolic_steps,
            "mcts_runs": args.mcts_runs,
            "c": args.c,
            "heuristic_multiplier": args.heuristic_multiplier,
            "lambda_add": args.lambda_add,
            "lambda_max": args.lambda_max,
            "lambda_ff": args.lambda_ff,
            "output_mode": args.output_mode,
            "rollout_outcome": args.rollout_outcome,
        },
        "symbolic_action_history": history,
        "physical_stack_actions": [
            {"below": below, "above": above}
            for below, above in stack_actions
        ],
        "representative_branch_probability": plan_probability,
        "representative_rollout_goal_reached": goal_reached,
        "warning": (
            "representative_branch_probability is not an exact closed-loop "
            "MCTS policy value; compare empirical closed-loop success separately"
        ),
        "decisions": decision_log,
        "final_state": list(state_key(final_state)),
    }

    result_path = os.path.join(save_dir, "mcts_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Railroad MCTS result ===")
    print(f"Symbolic actions: {len(history)}")
    print(f"Physical stack actions: {len(stack_actions)}")
    print(f"Representative branch probability: {plan_probability:.6f}")
    print(f"Representative rollout reaches goal: {goal_reached}")
    print(f"Plan written to: {plan_path}")
    print(f"Detailed result written to: {result_path}")
    for below, above in stack_actions:
        print(f"  stack {above} on {below}")


if __name__ == "__main__":
    main()