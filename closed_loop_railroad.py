"""Closed-loop probabilistic Railroad controller for DeepSym.

This script implements the loop that the expected-reachability planner was
missing:

    current symbolic state
    -> compute expected-reachability policy
    -> take the next useful action
    -> execute at most one physical stack action
    -> observe/choose the realised symbolic outcome
    -> update the symbolic state
    -> replan

It intentionally separates three modes:

  1. Dry-run/simulation:
       --outcome-source argmax-progress | argmax | sample
     No physical robot motion is executed unless --execute is given.

  2. Real closed-loop with automatic observer:
       --execute --outcome-source observer
     execute_plan.py records raw object positions before/after the one-step
     action, observe_outcome.py classifies the actual symbolic outcome, and the
     controller replans from that observed outcome.

  3. Real closed-loop with manual symbolic feedback:
       --execute --outcome-source manual
     After each physical stack command, you type the observed symbolic outcome:
       stacked, inserted, roll1, tumble1, roll2, tumble2.

  4. Policy debugging:
       --debug-actions
     Prints the expected value and learned effect distribution for each chosen
     stack action.

Important limitation
--------------------
This is closed-loop at the symbolic planning level.  With --outcome-source
observer, physical outcomes are classified from simulator object positions.  The
attached recognize.py is still used only for the initial scene because it resets
and regenerates objects; this controller maintains the symbolic state internally
after each observed outcome.

Conservative failure / recovery policy:
  - tumble2/roll2 consume the moved object and allow replanning on the same base
  - tumble1/roll1 mean the base/tower was disturbed. By default the controller
    stops. With --base-failure-policy continue-same-base it continues despite the base failure.
    No hand-coded new-base recovery is used in this version.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from railroad._bindings import Fluent as F, State

# Reuse the existing expected planner implementation.  Keep this file in the
# same directory as make_plan_railroad_expected.py.
from make_plan_railroad_expected import (  # noqa: E402
    OPERATOR_EFFECT_DISTS,
    action_base_name,
    expected_reachability_plan,
    extract_stack_actions,
    is_stack_action,
    load_operators_from_json,
    parse_deepsym_goal,
    parse_problem_railroad,
    build_initial_state,
    state_key,
    transition_safe,
    write_plan,
)

PROGRESS_EFFECTS = {"stacked", "inserted"}
FAILURE_EFFECTS = {"roll1", "roll2", "tumble1", "tumble2"}
ALL_EFFECTS = ["stacked", "inserted", "roll1", "tumble1", "roll2", "tumble2"]


def has_fluent(state: State, name: str) -> bool:
    return F(name) in state.fluents


def outcome_name_from_state(state: State) -> str:
    # stacked branch contains both stacked and inserted in the original DeepSym
    # semantics, so check stacked first.
    if has_fluent(state, "stacked"):
        return "stacked"
    if has_fluent(state, "inserted"):
        return "inserted"
    for effect in FAILURE_EFFECTS:
        if has_fluent(state, effect):
            return effect
    return "unknown"


def is_progress_outcome(state: State) -> bool:
    return outcome_name_from_state(state) in PROGRESS_EFFECTS


def choose_outcome(action_name: str,
                   outcomes: Sequence[Tuple[State, float]],
                   source: str,
                   rng: random.Random) -> Tuple[State, float, str]:
    if not outcomes:
        raise RuntimeError(f"No outcomes for action {action_name}")

    if not is_stack_action(action_name):
        next_state, p = max(outcomes, key=lambda item: float(item[1]))
        return next_state, float(p), outcome_name_from_state(next_state)

    labelled = [(state, float(p), outcome_name_from_state(state)) for state, p in outcomes]

    if source == "manual":
        print("Observed outcome required.")
        print("Available model outcomes:")
        for _state, p, label in labelled:
            print(f"  {label:8s} p={p:.6f}")
        while True:
            ans = input("Type observed outcome [stacked/inserted/roll1/tumble1/roll2/tumble2]: ").strip()
            if ans in ALL_EFFECTS:
                matches = [(s, p, lab) for s, p, lab in labelled if lab == ans]
                if matches:
                    # If the distribution had that outcome, use its branch.
                    return matches[0]
                print(f"Outcome {ans!r} was not present for this operator; valid model branches are above.")
            else:
                print("Invalid outcome name.")

    if source == "argmax-progress":
        progress = [(s, p, lab) for s, p, lab in labelled if lab in PROGRESS_EFFECTS]
        if progress:
            return max(progress, key=lambda item: item[1])
        return max(labelled, key=lambda item: item[1])

    if source == "argmax":
        return max(labelled, key=lambda item: item[1])

    if source == "sample":
        total = sum(p for _s, p, _lab in labelled)
        r = rng.random() * total
        acc = 0.0
        for item in labelled:
            acc += item[1]
            if r <= acc:
                return item
        return labelled[-1]

    raise ValueError(source)


def select_outcome_by_label(action_name: str,
                            outcomes: Sequence[Tuple[State, float]],
                            observed_label: str) -> Tuple[State, float, str]:
    """Select the Railroad transition branch matching an observed label.

    The learned operator may not contain every possible effect label.  If the
    observer reports a label absent from this operator, use a conservative
    fallback that keeps the symbolic state consistent with the closest available
    branch.
    """
    labelled = [(state, float(p), outcome_name_from_state(state)) for state, p in outcomes]
    matches = [(s, p, lab) for s, p, lab in labelled if lab == observed_label]
    if matches:
        return matches[0]

    progress = [(s, p, lab) for s, p, lab in labelled if lab in PROGRESS_EFFECTS]
    failures = [(s, p, lab) for s, p, lab in labelled if lab in FAILURE_EFFECTS]

    print(
        f"    observer outcome {observed_label!r} not present in model branches; "
        f"available={[lab for _s, _p, lab in labelled]}"
    )

    if observed_label in PROGRESS_EFFECTS and progress:
        print("    using highest-probability available progress branch")
        return max(progress, key=lambda item: item[1])

    if observed_label in FAILURE_EFFECTS and failures:
        print("    using highest-probability available failure branch")
        return max(failures, key=lambda item: item[1])

    print("    using highest-probability branch as fallback")
    return max(labelled, key=lambda item: item[1])


def read_object_infos_as_top_z(object_infos: Dict[str, Dict[str, float]], table_z: float) -> Dict[str, float]:
    """Initial support surface for each standalone object/tower top."""
    top_z = {}
    for name, info in object_infos.items():
        top_z[name.upper()] = table_z + float(info["size"])
    return top_z


def copy_object_infos_upper(object_infos: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        name.upper(): {"x": float(info["x"]), "y": float(info["y"]), "size": float(info["size"])}
        for name, info in object_infos.items()
    }


def update_geometry_after_progress(object_infos: Dict[str, Dict[str, float]],
                                   top_z: Dict[str, float],
                                   below: str,
                                   above: str) -> float:
    """Update 2D object/tower bookkeeping after a successful progress effect.

    Returns the support height that should be used for the executed placement.
    """
    below = below.upper()
    above = above.upper()
    support_z = top_z[below]
    object_infos[above]["x"] = object_infos[below]["x"]
    object_infos[above]["y"] = object_infos[below]["y"]
    top_z[above] = support_z + float(object_infos[above]["size"])
    return support_z


def support_z_for_action(object_infos: Dict[str, Dict[str, float]],
                         top_z: Dict[str, float],
                         below: str,
                         table_z: float) -> float:
    below = below.upper()
    if below in top_z:
        return top_z[below]
    return table_z + float(object_infos[below]["size"])


def is_makebase_action(action_name: str) -> bool:
    return action_base_name(action_name) == "makebase"


def action_object_arg(action_name: str) -> Optional[str]:
    parts = action_name.split()
    if len(parts) >= 2:
        return parts[1].upper()
    return None


def filter_actions_for_available_makebase(state: State, all_actions: Sequence) -> List:
    """Restrict makebase to objects that are currently pickloc.

    The original DeepSym/Railroad makebase operator has only (not base) as a
    precondition. That is fine at the initial state where every object is
    pickloc, but after a recovery reset it would allow choosing an already-used
    tower object as a new base. Closed-loop recovery needs stricter semantics:
    a new base must be selected from remaining free/pickable objects.
    """
    filtered = []
    for action in all_actions:
        if not is_makebase_action(action.name):
            filtered.append(action)
            continue
        obj = action_object_arg(action.name)
        if obj is not None and F("pickloc", obj) in state.fluents:
            filtered.append(action)
    return filtered


def _remove_counter_fluent(fluent) -> bool:
    text = str(fluent)
    return bool(__import__("re").search(r"\b[HS]\d+\b", text))


def reset_state_for_new_base_after_failure(state: State, objects: Sequence[str]) -> Tuple[State, List[str]]:
    """Abandon disturbed base/tower and allow makebase on remaining objects.

    This is the conservative recovery model for roll1/tumble1. It does not try
    to salvage the old tower. Objects already consumed by makebase/stack actions
    remain unavailable because their pickloc fluents are absent. Remaining
    pickloc objects can become a new base.
    """
    remove = {
        F("base"), F("stacked"), F("inserted"),
        F("roll1"), F("roll2"), F("tumble1"), F("tumble2"),
    }
    for obj in objects:
        obju = str(obj).upper()
        remove.add(F("stackloc", obju))
        remove.add(F("instack", obju))

    new_fluents = set()
    for fl in state.fluents:
        if fl in remove:
            continue
        if _remove_counter_fluent(fl):
            continue
        new_fluents.add(fl)
    new_fluents.add(F("H0"))
    new_fluents.add(F("S0"))

    available = [str(obj).upper() for obj in objects if F("pickloc", str(obj).upper()) in new_fluents]
    return State(fluents=new_fluents), available


def reset_top_z_for_available_objects(top_z: Dict[str, float],
                                      object_infos: Dict[str, Dict[str, float]],
                                      available_objects: Sequence[str],
                                      table_z: float) -> None:
    top_z.clear()
    for obj in available_objects:
        obju = str(obj).upper()
        if obju in object_infos:
            top_z[obju] = table_z + float(object_infos[obju]["size"])


def write_trace(trace_path: str, trace: List[Dict]) -> None:
    with open(trace_path, "w") as f:
        for row in trace:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_one_action_plan(plan_path: str,
                          object_infos: Dict[str, Dict[str, float]],
                          probability: float,
                          action_name: str) -> Tuple[str, str]:
    stack_actions = extract_stack_actions([action_name])
    if not stack_actions:
        raise ValueError(f"Expected one stack action, got {action_name}")
    write_plan(plan_path, object_infos, probability, stack_actions, True)
    return stack_actions[0]


def run_execute_plan(execute_python: str,
                     execute_script: str,
                     plan_path: str,
                     uri: str,
                     support_z: float,
                     executed_action_file: Optional[str]) -> None:
    cmd = [
        execute_python,
        execute_script,
        "-p",
        plan_path,
        "-uri",
        uri,
        "--one-step",
        "--support-z",
        f"{support_z:.8f}",
    ]
    if executed_action_file:
        cmd += ["--executed-action-file", executed_action_file]
    print("Executing command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_observer(observer_python: str,
                 observer_script: str,
                 executed_action_file: str,
                 output_file: str,
                 extra_args: Sequence[str]) -> Dict:
    cmd = [
        observer_python,
        observer_script,
        "--executed-action-file",
        executed_action_file,
        "--output-file",
        output_file,
    ]
    cmd.extend(extra_args)
    print("Observing command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(output_file, "r") as f:
        return json.load(f)


def apply_observed_geometry(object_infos: Dict[str, Dict[str, float]],
                            observation: Optional[Dict]) -> None:
    """Update 2D object bookkeeping from observer's after positions if available."""
    if not observation:
        return
    after = observation.get("after_positions_by_name") or {}
    for name, pos in after.items():
        key = str(name).upper()
        if key not in object_infos:
            continue
        if pos.get("x") is not None and pos.get("y") is not None:
            object_infos[key]["x"] = float(pos["x"])
            object_infos[key]["y"] = float(pos["y"])


def action_distribution_string(action_name: str) -> str:
    dist = OPERATOR_EFFECT_DISTS.get(action_base_name(action_name), {})
    if not dist:
        return "deterministic"
    return ", ".join(f"{k}:{v:.3f}" for k, v in sorted(dist.items()))


def main() -> None:
    parser = argparse.ArgumentParser("Closed-loop probabilistic Railroad planner for DeepSym.")
    parser.add_argument("-opts", type=str, required=True, help="opts.yaml")
    parser.add_argument("-goal", type=str, default="(H3)", help="goal such as '(H2) (S4)'")
    parser.add_argument("-max-steps", type=int, default=25, help="expected-reachability horizon per replanning call")
    parser.add_argument("--max-loop-steps", type=int, default=40, help="safety limit for symbolic closed-loop iterations")
    parser.add_argument("--execute", action="store_true", help="execute each selected physical stack action")
    parser.add_argument(
        "--outcome-source",
        choices=["manual", "observer", "argmax-progress", "argmax", "sample"],
        default="manual",
        help="how to obtain the realised outcome after a stack action",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for --outcome-source sample")
    parser.add_argument("--table-z", type=float, default=0.7, help="table/base z used by execute_plan.py")
    parser.add_argument("--uri", type=str, default="http://localhost:11311", help="ROS master URI for execution")
    parser.add_argument(
        "--execute-python",
        type=str,
        default=sys.executable,
        help="Python executable used to run execute_plan.py. Use repo .venv python if Railroad env lacks ROS.",
    )
    parser.add_argument("--execute-script", type=str, default="execute_plan.py")
    parser.add_argument(
        "--observer-python",
        type=str,
        default=sys.executable,
        help="Python executable used to run observe_outcome.py. It does not need ROS.",
    )
    parser.add_argument("--observer-script", type=str, default="observe_outcome.py")
    parser.add_argument(
        "--observer-arg",
        action="append",
        default=[],
        help="extra argument passed to observe_outcome.py; repeat for value pairs, e.g. --observer-arg --near-xy --observer-arg 0.10",
    )
    parser.add_argument(
        "--observer-fallback",
        choices=["error", "argmax-progress", "argmax", "sample", "manual"],
        default="argmax-progress",
        help=(
            "what to do if observe_outcome.py cannot classify the result. "
            "argmax-progress lets development continue, but it is model-assumed, not true observation."
        ),
    )
    parser.add_argument(
        "--base-failure-policy",
        choices=["stop", "continue-same-base"],
        default="stop",
        help=(
            "how to handle observed/model roll1 or tumble1. "
            "stop is most conservative; continue-same-base keeps the old behavior."
        ),
    )
    # Backward-compatible alias. Prefer --base-failure-policy continue-same-base.
    parser.add_argument(
        "--continue-after-base-failure",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--debug-actions", action="store_true")
    args = parser.parse_args()
    if args.continue_after_base_failure and args.base_failure_policy == "stop":
        args.base_failure_policy = "continue-same-base"

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    operators_path = os.path.join(save_dir, "railroad_operators.json")
    scene_path = os.path.join(save_dir, "railroad_problem.json")
    objects_path = os.path.join(save_dir, "objects.txt")
    plan_path = os.path.join(save_dir, "plan.txt")
    trace_path = os.path.join(save_dir, "closed_loop_trace.jsonl")
    executed_action_file = os.path.join(save_dir, "last_executed_action.json")
    observed_outcome_file = os.path.join(save_dir, "last_observed_outcome.json")

    if not os.path.exists(operators_path):
        raise FileNotFoundError(f"Missing {operators_path}. Run learn_rules_railroad.py first.")
    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"Missing {scene_path}. Run recognize.py first; it should write railroad_problem.json.")
    if args.outcome_source == "observer" and not args.execute:
        raise ValueError("--outcome-source observer requires --execute so before/after object positions are recorded.")

    all_operators = load_operators_from_json(operators_path)
    objects, obj_types, relations, counters, parsed_object_infos = parse_problem_railroad(scene_path, objects_path)
    current_state, type_dict = build_initial_state(objects, obj_types, relations, counters)
    all_actions = []
    for op in all_operators:
        all_actions.extend(op.instantiate(type_dict))
    all_actions.sort(key=lambda a: a.name)
    goal = parse_deepsym_goal(args.goal)

    object_infos = copy_object_infos_upper(parsed_object_infos)
    if not object_infos:
        # Plan writing/execution needs object coordinates.  Closed-loop planning
        # can still run symbolically, but execution cannot.
        object_infos = {obj.upper(): {"x": 0.0, "y": 0.0, "size": 0.1} for obj in objects}
        print("WARNING: objects.txt missing or empty; using dummy coordinates. Do not execute physically.")
    top_z = read_object_infos_as_top_z(object_infos, args.table_z)

    rng = random.Random(args.seed)
    trace: List[Dict] = []
    cumulative_branch_probability = 1.0

    print(f"Loaded {len(all_operators)} operators, grounded {len(all_actions)} actions")
    print(f"Objects: {objects}")
    print(f"Goal: {args.goal}")
    print("\n=== Starting closed-loop probabilistic Railroad planning ===")

    for loop_idx in range(args.max_loop_steps):
        if goal.evaluate(current_state.fluents):
            print(f"Goal reached after {loop_idx} closed-loop iterations.")
            break

        actions_for_planning = filter_actions_for_available_makebase(current_state, all_actions)
        expected_p, policy, _state_store, outcome_cache, value_fn = expected_reachability_plan(
            current_state,
            goal,
            actions_for_planning,
            args.max_steps,
            debug=args.debug_actions,
        )
        key = state_key(current_state)
        action_name, action_value = policy.get((key, args.max_steps), (None, 0.0))

        if action_name is None or action_value <= 0.0:
            print("No positive-value action found from current state.")
            write_plan(plan_path, object_infos, 0.0, [], False)
            break

        action_by_name = {a.name: a for a in actions_for_planning}
        action = action_by_name[action_name]
        outcomes = outcome_cache.get((key, action.name)) or transition_safe(current_state, action)
        if not outcomes:
            print(f"Selected action had no valid transition: {action.name}")
            break

        print(f"\n[{loop_idx}] policy_value={expected_p:.6f} action={action.name}")
        if args.debug_actions:
            print("    dist:", action_distribution_string(action.name))

        if not is_stack_action(action.name):
            next_state, p, outcome_label = choose_outcome(action.name, outcomes, "argmax", rng)
            current_state = next_state
            cumulative_branch_probability *= p
            trace.append({
                "step": loop_idx,
                "action": action.name,
                "type": "symbolic_auxiliary",
                "outcome": outcome_label,
                "probability": p,
                "policy_value": expected_p,
                "cumulative_branch_probability": cumulative_branch_probability,
            })
            print(f"    applied symbolic action, outcome={outcome_label}, p={p:.6f}")
            continue

        stack_actions = extract_stack_actions([action.name])
        if len(stack_actions) != 1:
            raise RuntimeError(f"Could not parse stack action: {action.name}")
        below, above = stack_actions[0]
        below = below.upper()
        above = above.upper()

        support_z = support_z_for_action(object_infos, top_z, below, args.table_z)
        write_one_action_plan(plan_path, object_infos, expected_p, action.name)
        print(f"    wrote one-action plan: stack {below} {above}")
        print(f"    support_z={support_z:.5f}")

        observation = None
        if args.execute:
            run_execute_plan(
                args.execute_python,
                args.execute_script,
                plan_path,
                args.uri,
                support_z,
                executed_action_file,
            )
            if args.outcome_source == "observer":
                try:
                    observation = run_observer(
                        args.observer_python,
                        args.observer_script,
                        executed_action_file,
                        observed_outcome_file,
                        args.observer_arg,
                    )
                    observed_label = str(observation["outcome"])
                    print(f"    observer outcome={observed_label}: {observation.get('reason', '')}")
                    next_state, p, outcome_label = select_outcome_by_label(action.name, outcomes, observed_label)
                except Exception as exc:
                    if args.observer_fallback == "error":
                        raise
                    observation = {
                        "outcome": None,
                        "observer_failed": True,
                        "observer_error": str(exc),
                        "fallback": args.observer_fallback,
                    }
                    print(f"    WARNING: observer failed: {exc}")
                    print(f"    falling back to --observer-fallback {args.observer_fallback}")
                    next_state, p, outcome_label = choose_outcome(action.name, outcomes, args.observer_fallback, rng)
            else:
                next_state, p, outcome_label = choose_outcome(action.name, outcomes, args.outcome_source, rng)
        else:
            print("    dry-run: physical execution skipped")
            next_state, p, outcome_label = choose_outcome(action.name, outcomes, args.outcome_source, rng)

        cumulative_branch_probability *= p
        print(f"    observed/selected outcome={outcome_label}, p={p:.6f}")

        if observation is not None:
            apply_observed_geometry(object_infos, observation)

        if outcome_label in PROGRESS_EFFECTS:
            update_geometry_after_progress(object_infos, top_z, below, above)
        else:
            print("    non-progress outcome: geometry was not updated as a successful tower extension")

        current_state = next_state
        trace.append({
            "step": loop_idx,
            "action": action.name,
            "type": "physical_stack",
            "below": below,
            "above": above,
            "outcome": outcome_label,
            "probability": p,
            "policy_value": expected_p,
            "cumulative_branch_probability": cumulative_branch_probability,
            "executed": bool(args.execute),
            "observer": observation,
        })
        write_trace(trace_path, trace)

        observed_base_failure = False
        if observation is not None:
            observed_base_failure = str(observation.get("outcome")) in {"roll1", "tumble1"}
        model_base_failure = outcome_label in {"roll1", "tumble1"}
        if observed_base_failure or model_base_failure:
            fail_label = str(observation.get("outcome")) if observation is not None else outcome_label
            if args.base_failure_policy == "stop":
                print(
                    f"    conservative stop: {fail_label} means the base/tower was disturbed. "
                    "Reset/re-recognize before continuing."
                )
                break
            # continue-same-base: keep current_state exactly as the model branch produced it.
            print("    WARNING: continuing after base/tower failure without resetting base/tower state.")

    else:
        print(f"Stopped after max loop limit {args.max_loop_steps} without satisfying goal.")

    write_trace(trace_path, trace)
    print(f"\nTrace written to {trace_path}")
    print(f"Last one-action plan is at {plan_path}")
    print(f"Cumulative realised/selected branch probability: {cumulative_branch_probability:.6f}")


if __name__ == "__main__":
    main()