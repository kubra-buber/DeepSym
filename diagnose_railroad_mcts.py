#!/usr/bin/env python3
"""Compare Railroad MCTS votes with exact finite-horizon Q values at one rollout state.

The script replays the representative action history in save/.../mcts_result.json
up to --step, using the same representative outcome rule as the MCTS wrapper.
It then computes exact expected-reachability Q values and runs repeated Railroad
MCTS searches from that exact state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List

import yaml

from make_plan_railroad_expected import (
    build_initial_state,
    expected_reachability_plan,
    is_progress_state,
    load_operators_from_json,
    parse_deepsym_goal,
    parse_problem_railroad,
    state_key,
    transition_safe,
)
from railroad_mcts_trial.make_plan_railroad_mcts import (
    canonical_action_name,
    choose_outcome,
    run_mcts_vote,
)


def normalize_goal(text: str) -> str:
    """Accept '(H2) (S4)', '(H2 S4)', or '(and (H2) (S4))'."""
    negative_clauses = re.findall(r"\(not\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\)", text)
    without_negative = re.sub(r"\(not\s*\(\s*[A-Za-z_]\w*\s*\)\s*\)", " ", text)
    counters = re.findall(r"\b[HS]\d+\b", without_negative, flags=re.IGNORECASE)
    counters = list(dict.fromkeys(token.upper() for token in counters))
    if not counters and not negative_clauses:
        raise ValueError(
            f"No DeepSym H/S goal found in {text!r}. Use '(H2) (S4)' or '(H2 S4)'."
        )
    parts = [f"({token})" for token in counters]
    parts.extend(f"(not ({token}))" for token in negative_clauses)
    return " ".join(parts)


def action_lookup(actions) -> Dict[str, object]:
    return {canonical_action_name(action.name): action for action in actions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("-goal", required=True)
    ap.add_argument(
        "--result-json",
        default=None,
        help="MCTS result JSON to replay. Default: <save>/mcts_result.json",
    )
    ap.add_argument("--step", type=int, default=6, help="state before this MCTS decision")
    ap.add_argument("--horizon", type=int, default=25)
    ap.add_argument("--mcts-runs", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--c", type=float, default=1.41421356237)
    ap.add_argument("--heuristic-multiplier", type=float, default=5.0)
    ap.add_argument("--lambda-add", type=float, default=0.5)
    ap.add_argument("--lambda-max", type=float, default=0.0)
    ap.add_argument("--lambda-ff", type=float, default=0.5)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = str(opts["save"])
    operators = load_operators_from_json(os.path.join(save_dir, "railroad_operators.json"))
    objects, obj_types, relations, counters, _ = parse_problem_railroad(
        os.path.join(save_dir, "railroad_problem.json"),
        os.path.join(save_dir, "objects.txt"),
    )
    state, objects_by_type = build_initial_state(objects, obj_types, relations, counters)
    actions: List[object] = []
    for operator in operators:
        actions.extend(operator.instantiate(objects_by_type))
    actions.sort(key=lambda a: canonical_action_name(a.name))
    by_name = action_lookup(actions)

    normalized_goal = normalize_goal(args.goal)
    goal = parse_deepsym_goal(normalized_goal)

    result_path = args.result_json or os.path.join(save_dir, "mcts_result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"MCTS result JSON not found: {result_path}")
    if os.path.getsize(result_path) == 0:
        raise ValueError(
            f"MCTS result JSON is empty: {result_path}. "
            "Pass a preserved result with --result-json."
        )
    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {result_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    print(f"Replaying MCTS result: {result_path}")
    history = result.get("symbolic_action_history", [])
    outcome_mode = result.get("parameters", {}).get("rollout_outcome", "progress")

    if args.step < 0 or args.step > len(history):
        raise ValueError(f"--step must be between 0 and {len(history)}")

    import random
    rng = random.Random(0)
    for i, raw_name in enumerate(history[: args.step]):
        name = canonical_action_name(raw_name)
        action = by_name.get(name)
        if action is None:
            raise RuntimeError(f"Action from mcts_result.json not found: {name}")
        outcomes = transition_safe(state, action)
        if not outcomes:
            raise RuntimeError(f"Replay transition rejected at step {i}: {name}")
        state, _ = choose_outcome(name, outcomes, mode=outcome_mode, rng=rng, current_state=state)

    print(f"Diagnostic state: before step {args.step}")
    print(f"Goal: {goal}")
    print(f"Goal already true: {goal.evaluate(state)}")
    print(f"State fluents: {len(state.fluents)}")

    exact_value, policy, _store, outcome_cache, value_fn = expected_reachability_plan(
        state, goal, actions, args.horizon
    )
    root_key = state_key(state)
    exact_action, _ = policy.get((root_key, args.horizon), (None, 0.0))

    rows = []
    for (key, action_name), outcomes in outcome_cache.items():
        if key != root_key or not outcomes:
            continue
        q = sum(float(p) * value_fn(state_key(next_state), args.horizon - 1)
                for next_state, p in outcomes)
        progress_p = sum(float(p) for next_state, p in outcomes if is_progress_state(next_state, state))
        rows.append((q, progress_p, canonical_action_name(action_name)))
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))

    print(f"\nExact root value, horizon {args.horizon}: {exact_value:.9f}")
    print(f"Exact selected action: {canonical_action_name(exact_action) if exact_action else None}")
    print("Top exact Q values:")
    for q, progress_p, name in rows[:20]:
        print(f"  Q={q:.9f}  immediate_progress_p={progress_p:.6f}  {name}")

    selected, votes, elapsed, trace = run_mcts_vote(
        state,
        goal,
        actions,
        runs=args.mcts_runs,
        iterations=args.iterations,
        max_depth=args.max_depth,
        c=args.c,
        heuristic_multiplier=args.heuristic_multiplier,
        lambda_add=args.lambda_add,
        lambda_max=args.lambda_max,
        lambda_ff=args.lambda_ff,
        collect_trace=True,
    )
    print(f"\nMCTS selected after voting: {canonical_action_name(selected.name) if selected else None}")
    print("MCTS votes:")
    for name, count in sorted(votes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count}/{args.mcts_runs}: {name}")
    print("\nLast MCTS trace:")
    print(trace)


if __name__ == "__main__":
    main()