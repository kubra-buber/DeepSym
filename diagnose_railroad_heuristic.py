#!/usr/bin/env python3
from __future__ import annotations
"""Explain why selected successor states receive different Railroad FF heuristic values.

The script replays a preserved mcts_result.json up to --step, then evaluates every
outcome of selected root actions using:
- exact finite-horizon reachability continuation value,
- the actual mixed Railroad heuristic,
- derived h_add, h_max, and h_ff contributions,
- the approximate one-step surrogate reward used when that successor is first evaluated.

For a single non-disjunctive goal such as H3, subtracting the zero-weight baseline
cleanly exposes the components. With OR goals, different lambda settings can select
different DNF branches, so component differences should be treated as diagnostics.
"""
from asyncio import taskgroups

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
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
)


def action_lookup(actions) -> Dict[str, object]:
    return {canonical_action_name(action.name): action for action in actions}


def fmt_fluent_set(values) -> str:
    return " ; ".join(sorted(str(item) for item in values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("-goal", required=True)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument(
        "--actions", nargs="+",
        default=["stack17 O2 O3", "stack21 O2 O5", "stack27 O2 O1"],
    )
    ap.add_argument("--heuristic-multiplier", type=float, default=5.0)
    ap.add_argument("--lambda-add", type=float, default=0.5)
    ap.add_argument("--lambda-max", type=float, default=0.0)
    ap.add_argument("--lambda-ff", type=float, default=0.5)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    if args.horizon < 1:
        ap.error("--horizon must be at least 1")

    opts_path = Path(args.opts).expanduser().resolve()
    with opts_path.open("r") as f:
        opts = yaml.safe_load(f)
    save_dir = Path(str(opts["save"])).expanduser()
    if not save_dir.is_absolute():
        save_dir = (Path.cwd() / save_dir).resolve()

    operators = load_operators_from_json(str(save_dir / "railroad_operators.json"))
    objects, obj_types, relations, counters, _ = parse_problem_railroad(
        str(save_dir / "railroad_problem.json"),
        str(save_dir / "objects.txt"),
    )
    state, objects_by_type = build_initial_state(
        objects, obj_types, relations, counters
    )

    actions: List[object] = []
    for operator in operators:
        actions.extend(operator.instantiate(objects_by_type))
    actions.sort(key=lambda action: canonical_action_name(action.name))
    by_name = action_lookup(actions)
    goal = parse_deepsym_goal(args.goal)

    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else save_dir / "mcts_result.json"
    )
    result = json.loads(result_path.read_text())
    history = result.get("symbolic_action_history", [])
    if not 0 <= args.step <= len(history):
        raise ValueError(f"--step must be between 0 and {len(history)}")

    outcome_mode = result.get("parameters", {}).get("rollout_outcome", "progress")
    rng = random.Random(0)
    for index, raw_name in enumerate(history[:args.step]):
        name = canonical_action_name(raw_name)
        action = by_name.get(name)
        if action is None:
            raise RuntimeError(f"Replay action not found: {name}")
        outcomes = transition_safe(state, action)
        if not outcomes:
            raise RuntimeError(f"Replay transition rejected at step {index}: {name}")
        state, _ = choose_outcome(name, outcomes, mode=outcome_mode, rng=rng)

    # Match MCTS: its root node is a copy whose absolute time is reset.
    root_state = state.copy_and_zero_out_time()

    planners = {
        "zero": MCTSPlanner(list(actions), lambda_add=0.0, lambda_max=0.0, lambda_ff=0.0),
        "add": MCTSPlanner(list(actions), lambda_add=1.0, lambda_max=0.0, lambda_ff=0.0),
        "max": MCTSPlanner(list(actions), lambda_add=0.0, lambda_max=1.0, lambda_ff=0.0),
        "ff": MCTSPlanner(list(actions), lambda_add=0.0, lambda_max=0.0, lambda_ff=1.0),
        "mixed": MCTSPlanner(
            list(actions),
            lambda_add=args.lambda_add,
            lambda_max=args.lambda_max,
            lambda_ff=args.lambda_ff,
        ),
    }
    for label, planner in planners.items():
        if not hasattr(planner, "heuristic"):
            raise RuntimeError(
                f"Installed Railroad MCTSPlanner lacks heuristic(); failed at {label}. "
                "Rebuild the Railroad source version pinned by this project."
            )

    exact_root_value, policy, _store, _outcome_cache, value_fn = (
        expected_reachability_plan(root_state, goal, actions, args.horizon)
    )
    exact_selected, _ = policy.get(
        (state_key(root_state), args.horizon), (None, 0.0)
    )

    print(f"Replayed: {result_path}")
    print(f"State before symbolic step: {args.step}")
    print(f"Goal: {goal}")
    print(f"Exact root value at horizon {args.horizon}: {exact_root_value:.9f}")
    print(
        "Exact root action: "
        f"{canonical_action_name(exact_selected) if exact_selected else None}"
    )
    print(
        "\nColumns:\n"
        "  base = dtime + probabilistic retry delta (all lambda weights are zero)\n"
        "  add/max/ff = derived component contribution above base\n"
        "  leaf_reward = -g - heuristic_multiplier * mixed_h - action_extra_cost\n"
    )

    rows: list[dict] = []
    summaries = defaultdict(lambda: {
        "p": 0.0,
        "exact_q": 0.0,
        "leaf_reward": 0.0,
        "mixed_h": 0.0,
        "base": 0.0,
        "h_add": 0.0,
        "h_max": 0.0,
        "h_ff": 0.0,
    })

    parent_fluents = set(root_state.fluents)
    for requested_name in args.actions:
        name = canonical_action_name(requested_name)
        action = by_name.get(name)
        if action is None:
            close = [
                candidate for candidate in sorted(by_name)
                if candidate.startswith(name.split()[0])
            ]
            raise KeyError(f"Action not found: {name}. Same schema: {close[:20]}")

        outcomes = transition_safe(root_state, action)
        if not outcomes:
            print(f"\n{name}: no valid transition")
            continue

        print(f"\n{name}  extra_cost={float(action.extra_cost):.6f}")
        for outcome_index, (next_state, probability) in enumerate(outcomes):
            probability = float(probability)
            h0 = float(planners["zero"].heuristic(next_state, goal))
            h100 = float(planners["add"].heuristic(next_state, goal))
            h010 = float(planners["max"].heuristic(next_state, goal))
            h001 = float(planners["ff"].heuristic(next_state, goal))
            hmixed = float(planners["mixed"].heuristic(next_state, goal))

            # For a fixed DNF branch these differences isolate each component.
            derived_add = h100 - h0
            derived_max = h010 - h0
            derived_ff = h001 - h0
            reconstructed = (
                h0
                + args.lambda_add * derived_add
                + args.lambda_max * derived_max
                + args.lambda_ff * derived_ff
            )
            residual = hmixed - reconstructed

            g = float(next_state.time)
            h_for_reward = hmixed
            if not math.isfinite(h_for_reward) or h_for_reward > 1e10:
                h_for_reward = 100.0

            leaf_reward = (
                -g
                - args.heuristic_multiplier * h_for_reward
                - float(action.extra_cost)
            )
            exact_continuation = float(
                value_fn(state_key(next_state), args.horizon - 1)
            )
            added = set(next_state.fluents) - parent_fluents
            removed = parent_fluents - set(next_state.fluents)

            row = {
                "action": name,
                "outcome_index": outcome_index,
                "probability": probability,
                "goal": bool(goal.evaluate(next_state.fluents)),
                "progress": bool(is_progress_state(next_state)),
                "g_time": g,
                "h_base_dtime_plus_retry": h0,
                "derived_h_add": derived_add,
                "derived_h_max": derived_max,
                "derived_h_ff": derived_ff,
                "mixed_h": hmixed,
                "reconstruction_residual": residual,
                "leaf_reward": leaf_reward,
                "exact_continuation_value": exact_continuation,
                "exact_weighted_contribution": probability * exact_continuation,
                "added_fluents": fmt_fluent_set(added),
                "removed_fluents": fmt_fluent_set(removed),
            }
            rows.append(row)

            summary = summaries[name]
            summary["p"] += probability
            summary["exact_q"] += probability * exact_continuation
            summary["leaf_reward"] += probability * leaf_reward
            summary["mixed_h"] += probability * hmixed
            summary["base"] += probability * h0
            summary["h_add"] += probability * derived_add
            summary["h_max"] += probability * derived_max
            summary["h_ff"] += probability * derived_ff

            print(
                f"  outcome={outcome_index:2d} p={probability:.6f} "
                f"goal={row['goal']} progress={row['progress']} "
                f"g={g:.3f} base={h0:.3f} "
                f"h_add={derived_add:.3f} h_max={derived_max:.3f} "
                f"h_ff={derived_ff:.3f} mixed={hmixed:.3f} "
                f"leaf_reward={leaf_reward:.3f} "
                f"exact_cont={exact_continuation:.6f}"
            )

    print("\nProbability-weighted action summaries:")
    print(
        f"{'action':28s} {'exact Q':>10s} {'E[leaf R]':>12s} "
        f"{'E[mixed h]':>12s} {'E[base]':>10s} "
        f"{'E[h_add]':>10s} {'E[h_max]':>10s} {'E[h_ff]':>10s}"
    )
    print("-" * 118)
    for name, summary in sorted(
        summaries.items(), key=lambda item: (-item[1]["exact_q"], item[0])
    ):
        print(
            f"{name:28s} {summary['exact_q']:10.6f} "
            f"{summary['leaf_reward']:12.3f} {summary['mixed_h']:12.3f} "
            f"{summary['base']:10.3f} {summary['h_add']:10.3f} "
            f"{summary['h_max']:10.3f} {summary['h_ff']:10.3f}"
        )

    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else save_dir / f"heuristic_successors_step{args.step}.csv"
    )
    if rows:
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV written: {output_csv}")


if __name__ == "__main__":
    main()
