#!/usr/bin/env python3
"""Compare exact reachability Q values with detailed Railroad MCTS root statistics."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from typing import Dict, List

import yaml
from railroad.planner import MCTSPlanner

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
from make_plan_railroad_mcts import (
    canonical_action_name,
    choose_outcome,
    selected_action_name,
)


def normalize_goal(text: str) -> str:
    negative_clauses = re.findall(r"\(not\s*\(\s*([A-Za-z_]\w*)\s*\)\s*\)", text)
    without_negative = re.sub(r"\(not\s*\(\s*[A-Za-z_]\w*\s*\)\s*\)", " ", text)
    counters = re.findall(r"\b[HS]\d+\b", without_negative, flags=re.IGNORECASE)
    counters = list(dict.fromkeys(token.upper() for token in counters))
    if not counters and not negative_clauses:
        raise ValueError(f"No DeepSym H/S goal found in {text!r}")
    parts = [f"({token})" for token in counters]
    parts.extend(f"(not ({token}))" for token in negative_clauses)
    return " ".join(parts)


def action_lookup(actions) -> Dict[str, object]:
    return {canonical_action_name(action.name): action for action in actions}


def parse_kv_trace_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if not parts or parts[0] not in {"ROOT_STATS", "ROOT_ACTION", "ROOT_OUTCOME"}:
        return None
    values = {}
    for field in parts[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            values[key] = value
    return parts[0], values


def parse_root_trace(trace: str):
    actions = {}
    outcomes = []

    for line in trace.splitlines():
        parsed = parse_kv_trace_line(line)
        if parsed is None:
            continue
        kind, values = parsed

        if kind == "ROOT_ACTION":
            name = canonical_action_name(values["name"])
            actions[name] = {
                "visits": int(values["visits"]),
                "total_reward": float(values["total_reward"]),
                "mean_q": float(values["mean_q"]),
                "ucb": float(values["ucb"]),
            }
        elif kind == "ROOT_OUTCOME":
            outcomes.append(
                {
                    "action": canonical_action_name(values["action"]),
                    "index": int(values["index"]),
                    "probability": float(values["probability"]),
                    "visits": int(values["visits"]),
                    "total_reward": float(values["total_reward"]),
                    "mean_q": float(values["mean_q"]),
                }
            )

    return actions, outcomes


def print_last_run_table(root_stats):
    print("\nRoot action statistics — last MCTS run:")
    print(
        f"{'action':28s} {'visits':>9s} {'total reward':>15s} "
        f"{'mean Q':>12s} {'final UCB':>12s}"
    )
    print("-" * 82)
    for name, row in sorted(
        root_stats.items(),
        key=lambda item: (-item[1]["visits"], item[0]),
    ):
        print(
            f"{name:28s} {row['visits']:9d} "
            f"{row['total_reward']:15.3f} "
            f"{row['mean_q']:12.3f} {row['ucb']:12.3f}"
        )


def print_last_outcomes(outcomes):
    print("\nRoot outcome statistics — last MCTS run:")
    print(
        f"{'action':28s} {'i':>2s} {'model p':>9s} {'visits':>9s} "
        f"{'mean Q':>12s}"
    )
    print("-" * 72)
    for row in sorted(outcomes, key=lambda r: (r["action"], r["index"])):
        print(
            f"{row['action']:28s} {row['index']:2d} "
            f"{row['probability']:9.6f} {row['visits']:9d} "
            f"{row['mean_q']:12.3f}"
        )


def print_aggregate(votes, aggregate, runs):
    print(f"\nAggregated root statistics over {runs} independent MCTS runs:")
    print(
        f"{'action':28s} {'votes':>7s} {'total visits':>13s} "
        f"{'mean visits':>12s} {'total reward':>15s} "
        f"{'weighted Q':>12s} {'mean run Q':>12s}"
    )
    print("-" * 112)

    names = sorted(
        aggregate,
        key=lambda name: (-votes.get(name, 0), -aggregate[name]["visits"], name),
    )
    for name in names:
        row = aggregate[name]
        total_visits = row["visits"]
        total_reward = row["total_reward"]
        weighted_q = total_reward / total_visits if total_visits else 0.0
        mean_run_q = statistics.fmean(row["run_q"]) if row["run_q"] else 0.0
        print(
            f"{name:28s} {votes.get(name, 0):3d}/{runs:<3d} "
            f"{total_visits:13d} {total_visits / runs:12.2f} "
            f"{total_reward:15.3f} {weighted_q:12.3f} {mean_run_q:12.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("-goal", required=True)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--step", type=int, default=6)
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

    with open(args.opts, "r") as f:
        opts = yaml.safe_load(f)
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

    goal = parse_deepsym_goal(normalize_goal(args.goal))

    result_path = args.result_json or os.path.join(save_dir, "mcts_result.json")
    if not os.path.exists(result_path):
        raise FileNotFoundError(result_path)
    if os.path.getsize(result_path) == 0:
        raise ValueError(f"Empty JSON file: {result_path}")
    with open(result_path, "r") as f:
        result = json.load(f)

    print(f"Replaying MCTS result: {result_path}")
    history = result.get("symbolic_action_history", [])
    outcome_mode = result.get("parameters", {}).get("rollout_outcome", "progress")

    if not 0 <= args.step <= len(history):
        raise ValueError(f"--step must be between 0 and {len(history)}")

    rng = random.Random(0)
    for i, raw_name in enumerate(history[: args.step]):
        name = canonical_action_name(raw_name)
        action = by_name.get(name)
        if action is None:
            raise RuntimeError(f"Replay action not found: {name}")
        outcomes = transition_safe(state, action)
        if not outcomes:
            raise RuntimeError(f"Replay transition rejected at step {i}: {name}")
        state, _ = choose_outcome(name, outcomes, mode=outcome_mode, rng=rng)

    print(f"Diagnostic state: before step {args.step}")
    print(f"Goal: {goal}")
    print(f"Goal already true: {goal.evaluate(state.fluents)}")
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
        q = sum(
            float(p) * value_fn(state_key(next_state), args.horizon - 1)
            for next_state, p in outcomes
        )
        progress_p = sum(
            float(p)
            for next_state, p in outcomes
            if is_progress_state(next_state)
        )
        rows.append((q, progress_p, canonical_action_name(action_name)))
    rows.sort(key=lambda row: (-row[0], -row[1], row[2]))

    print(f"\nExact root value, horizon {args.horizon}: {exact_value:.9f}")
    print(
        "Exact selected action: "
        f"{canonical_action_name(exact_action) if exact_action else None}"
    )
    print("Top exact Q values:")
    for q, progress_p, name in rows[:20]:
        print(f"  Q={q:.9f}  immediate_progress_p={progress_p:.6f}  {name}")

    votes = Counter()
    aggregate = defaultdict(lambda: {"visits": 0, "total_reward": 0.0, "run_q": []})
    last_trace = ""
    last_stats = {}
    last_outcomes = []

    for run_index in range(args.mcts_runs):
        planner = MCTSPlanner(
            list(actions),
            lambda_add=args.lambda_add,
            lambda_max=args.lambda_max,
            lambda_ff=args.lambda_ff,
        )

        selected = planner(
            state,
            goal,
            max_iterations=args.iterations,
            max_depth=args.max_depth,
            c=args.c,
            heuristic_multiplier=args.heuristic_multiplier,
        )

        name = selected_action_name(selected)
        if name is not None:
            votes[canonical_action_name(name)] += 1

        trace = str(planner.get_trace_from_last_mcts_tree())
        root_stats, outcome_stats = parse_root_trace(trace)
        if not root_stats:
            raise RuntimeError(
                "No ROOT_ACTION records found in the MCTS trace. "
                "Apply patch_railroad_mcts_root_stats.py and rebuild Railroad."
            )

        for action_name, row in root_stats.items():
            aggregate[action_name]["visits"] += row["visits"]
            aggregate[action_name]["total_reward"] += row["total_reward"]
            aggregate[action_name]["run_q"].append(row["mean_q"])

        if run_index == args.mcts_runs - 1:
            last_trace = trace
            last_stats = root_stats
            last_outcomes = outcome_stats

    winner_name = sorted(votes, key=lambda name: (-votes[name], name))[0]

    print(f"\nMCTS selected after voting: {winner_name}")
    print("MCTS votes:")
    for name, count in sorted(votes.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count}/{args.mcts_runs}: {name}")

    print_last_run_table(last_stats)
    print_last_outcomes(last_outcomes)
    print_aggregate(votes, aggregate, args.mcts_runs)

    print("\nLast MCTS trace:")
    print(last_trace)


if __name__ == "__main__":
    main()
