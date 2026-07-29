#!/usr/bin/env python3
"""Railroad MCTS planner wrapper for the DeepSym stacking domain.

This script intentionally reuses the same probabilistic Railroad operators and
DeepSym scene representation as make_plan_railroad_expected.py:

    railroad_operators.json  <- learn_rules_railroad.py
    railroad_problem.json    <- recognize.py

No MCTS-specific rule-learning/export step is required.  MCTS, exact expected
reachability, and closed-loop planning should all consume the same operator
model so their results remain directly comparable.

Two output modes are supported:

1. representative-plan (default)
   Replan with MCTS at every symbolic state and follow one selected outcome
   branch to produce a complete, representative linear plan.txt.  This is for
   comparison/debugging only; it is not the full closed-loop MCTS policy.

2. next-physical-action
   Resolve deterministic bookkeeping actions internally and stop after the
   next physical stack action.  Use this mode inside an observe -> plan ->
   execute one action -> observe loop.

Important Railroad 0.2.0 note
-----------------------------
The tested Railroad source was patched so that states for which the heuristic
cannot find a path to the goal receive a non-zero dead-end penalty.  Without
that patch, MCTS may fail to distinguish actions with different success
probabilities in domains containing terminal failure states.

Example:
    python make_plan_railroad_mcts.py \
        -opts opts.yaml -goal "(H3)" \
        --iterations 10000 --max-depth 25 --mcts-runs 10 \
        --debug-actions --trace
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from railroad.planner import MCTSPlanner

# Keep a single source of truth for DeepSym -> Railroad semantics.  These
# helpers reconstruct the learned probabilistic operators, parse the recognized
# scene, build the initial state, apply transitions safely, and write plan.txt.
from make_plan_railroad_expected import (
    OPERATOR_EFFECT_DISTS,
    build_initial_state,
    debug_action,
    extract_stack_actions,
    is_progress_state,
    load_operators_from_json,
    parse_deepsym_goal,
    parse_problem_railroad,
    state_key,
    transition_safe,
    write_plan,
)


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
    """Run independent MCTS searches and select the modal root action.

    `runs=1` is the unmodified single Railroad MCTS decision.  Larger values
    are useful for evaluation because Railroad MCTS is stochastic.  Ties are
    broken deterministically by action name to make experiment output stable.
    """
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
    # Keep the concrete object returned by MCTS for each voted name. This is
    # more robust than reconstructing parameterless actions from strings,
    # because Railroad 0.2.0 may format zero-argument grounded names
    # differently in the planner result and in the grounded-action list.
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
            # In normal Railroad builds this is one of the exact grounded
            # action objects passed to MCTS, so preserve it for later use.
            if not isinstance(selected, str):
                selected_objects[name] = selected

        if collect_trace:
            try:
                last_trace = str(planner.get_trace_from_last_mcts_tree())
            except Exception as exc:  # Trace is diagnostic, not required.
                last_trace = f"Trace unavailable: {exc}"

    if not counts:
        return None, counts, elapsed, last_trace

    # Highest vote count first; lexical tie break avoids action-order artifacts
    # in this Python wrapper.  It does not change MCTS's internal tie handling.
    winner_name = sorted(counts, key=lambda name: (-counts[name], name))[0]

    # Prefer the actual object returned by MCTS. This avoids the Railroad 0.2.0
    # zero-parameter action-name mismatch (e.g. increase_stack1). Fall back to
    # canonical string lookup for builds that return only a textual name.
    winner = selected_objects.get(winner_name)
    if winner is None:
        winner = actions_by_name.get(winner_name)

    if winner is None:
        # Last-resort diagnostic aliases. Keep this explicit rather than
        # silently selecting a different action.
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

        # In closed-loop use, write only the next physical action.  Auxiliary
        # actions have no robot command, so they are advanced symbolically until
        # the first stack action is selected.
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

    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)
    save_dir = str(opts["save"])

    operators_path = os.path.join(save_dir, "railroad_operators.json")
    scene_path = os.path.join(save_dir, "railroad_problem.json")
    objects_path = os.path.join(save_dir, "objects.txt")

    if not os.path.exists(operators_path):
        print(f"ERROR: Railroad operators not found: {operators_path}", file=sys.stderr)
        print("Run learn_rules_railroad.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(scene_path):
        print(f"ERROR: Railroad scene not found: {scene_path}", file=sys.stderr)
        print("Run recognize.py first.", file=sys.stderr)
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