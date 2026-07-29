#!/usr/bin/env python3
"""Sweep UCB c and heuristic multiplier at one replayed Railroad MCTS root state.

This is a screening experiment. It reports:
- action vote distribution over independent MCTS searches,
- exact finite-horizon Q of every selected action,
- optimal-action vote rate,
- exact-Q regret of the stochastic MCTS selections,
- root visits and surrogate Q when the root-statistics patch is installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Dict, List

import yaml
from railroad.planner import MCTSPlanner

from diagnose_railroad_mcts_root_stats import parse_root_trace
from make_plan_railroad_expected import (
    build_initial_state,
    expected_reachability_plan,
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


def parse_float_list(text: str) -> list[float]:
    values = [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return values


def action_lookup(actions) -> Dict[str, object]:
    return {canonical_action_name(action.name): action for action in actions}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("-goal", required=True)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--step", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument(
        "--c-values", type=parse_float_list,
        default=parse_float_list("1.41421356237,10,20,30,50"),
    )
    ap.add_argument(
        "--heuristic-multipliers", type=parse_float_list,
        default=parse_float_list("5,2,1,0.5"),
    )
    ap.add_argument("--lambda-add", type=float, default=0.5)
    ap.add_argument("--lambda-max", type=float, default=0.0)
    ap.add_argument("--lambda-ff", type=float, default=0.5)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.horizon < 1:
        ap.error("--horizon must be at least 1")
    if args.runs < 1 or args.iterations < 1:
        ap.error("--runs and --iterations must be positive")

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

    root_state = state.copy_and_zero_out_time()

    exact_value, policy, _store, _outcome_cache, value_fn = (
        expected_reachability_plan(root_state, goal, actions, args.horizon)
    )
    root_key = state_key(root_state)
    exact_selected, _ = policy.get((root_key, args.horizon), (None, 0.0))
    exact_selected_name = (
        canonical_action_name(exact_selected) if exact_selected else None
    )

    exact_q: dict[str, float] = {}
    for action in actions:
        name = canonical_action_name(action.name)
        outcomes = transition_safe(root_state, action)
        if not outcomes:
            continue
        exact_q[name] = sum(
            float(probability)
            * float(value_fn(state_key(next_state), args.horizon - 1))
            for next_state, probability in outcomes
        )
    exact_ranking = {
        name: rank + 1
        for rank, (name, _) in enumerate(
            sorted(exact_q.items(), key=lambda item: (-item[1], item[0]))
        )
    }
    best_exact_q = max(exact_q.values()) if exact_q else float("nan")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else save_dir / "mcts_experiments" / f"root_sweep_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config_rows: list[dict] = []
    action_rows: list[dict] = []
    raw_runs: list[dict] = []

    for c in args.c_values:
        for heuristic_multiplier in args.heuristic_multipliers:
            votes: Counter[str] = Counter()
            stats = defaultdict(lambda: {
                "visits": 0,
                "total_reward": 0.0,
                "run_q": [],
            })
            selected_exact_values: list[float] = []
            optimal_selections = 0

            print(
                f"\n=== c={c:g}, heuristic_multiplier={heuristic_multiplier:g} ==="
            )
            for run_index in range(args.runs):
                planner = MCTSPlanner(
                    list(actions),
                    lambda_add=args.lambda_add,
                    lambda_max=args.lambda_max,
                    lambda_ff=args.lambda_ff,
                )
                selected = planner(
                    root_state,
                    goal,
                    max_iterations=args.iterations,
                    max_depth=args.max_depth,
                    c=c,
                    heuristic_multiplier=heuristic_multiplier,
                )
                raw_name = selected_action_name(selected)
                name = canonical_action_name(raw_name) if raw_name else "<none>"
                votes[name] += 1

                q = exact_q.get(name, 0.0)
                selected_exact_values.append(q)
                if exact_selected_name is not None and name == exact_selected_name:
                    optimal_selections += 1

                trace = str(planner.get_trace_from_last_mcts_tree())
                root_stats, _ = parse_root_trace(trace)
                for action_name, row in root_stats.items():
                    stats[action_name]["visits"] += row["visits"]
                    stats[action_name]["total_reward"] += row["total_reward"]
                    stats[action_name]["run_q"].append(row["mean_q"])

                raw_runs.append({
                    "c": c,
                    "heuristic_multiplier": heuristic_multiplier,
                    "run": run_index,
                    "selected_action": name,
                    "selected_exact_q": q,
                    "exact_regret": best_exact_q - q,
                })

            modal_action, modal_count = votes.most_common(1)[0]
            mean_selected_q = fmean(selected_exact_values)
            config_row = {
                "c": c,
                "heuristic_multiplier": heuristic_multiplier,
                "runs": args.runs,
                "iterations": args.iterations,
                "modal_action": modal_action,
                "modal_fraction": modal_count / args.runs,
                "modal_exact_q": exact_q.get(modal_action, 0.0),
                "exact_optimal_action": exact_selected_name,
                "exact_optimal_q": best_exact_q,
                "optimal_vote_rate": optimal_selections / args.runs,
                "mean_exact_q_of_selected_action": mean_selected_q,
                "mean_exact_regret": best_exact_q - mean_selected_q,
                "unique_selected_actions": len(votes),
            }
            config_rows.append(config_row)
            print(json.dumps(config_row, indent=2))

            all_names = set(votes) | set(stats)
            for name in sorted(
                all_names,
                key=lambda value: (-votes.get(value, 0), value),
            ):
                row = stats[name]
                total_visits = row["visits"]
                action_rows.append({
                    "c": c,
                    "heuristic_multiplier": heuristic_multiplier,
                    "action": name,
                    "votes": votes.get(name, 0),
                    "vote_fraction": votes.get(name, 0) / args.runs,
                    "exact_q": exact_q.get(name, 0.0),
                    "exact_rank": exact_ranking.get(name, ""),
                    "total_root_visits": total_visits,
                    "mean_root_visits": total_visits / args.runs,
                    "weighted_surrogate_q": (
                        row["total_reward"] / total_visits
                        if total_visits else ""
                    ),
                    "mean_run_surrogate_q": (
                        fmean(row["run_q"]) if row["run_q"] else ""
                    ),
                })

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "config_summary.csv", config_rows)
    write_csv(output_dir / "action_stats.csv", action_rows)
    write_csv(output_dir / "raw_runs.csv", raw_runs)
    (output_dir / "experiment.json").write_text(json.dumps({
        "result_json": str(result_path),
        "step": args.step,
        "goal": args.goal,
        "horizon": args.horizon,
        "exact_root_value": exact_value,
        "exact_selected_action": exact_selected_name,
        "exact_q": exact_q,
        "parameters": vars(args),
        "note": (
            "Root visits and surrogate-Q fields are populated only when "
            "patch_railroad_mcts_root_stats.py has been applied and Railroad rebuilt."
        ),
    }, indent=2, default=str))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
