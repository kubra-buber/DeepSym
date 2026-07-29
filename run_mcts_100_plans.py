#!/usr/bin/env python3
"""Run 100 independent full Railroad MCTS representative plans and aggregate them.

Important:
- Every subprocess uses --mcts-runs 1. This measures raw Railroad MCTS variability.
- The representative rollout outcome rule is separate from Railroad's internal RNG.
- representative_branch_probability is not a closed-loop policy success probability.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

import yaml


def resolve_save_dir(opts_path: Path, repo_root: Path) -> Path:
    with opts_path.open("r") as f:
        opts = yaml.safe_load(f)
    raw = Path(str(opts["save"])).expanduser()
    return raw if raw.is_absolute() else (repo_root / raw).resolve()


def physical_plan_key(actions: list[dict]) -> str:
    if not actions:
        return "<no physical stack action>"
    return " | ".join(
        f"{str(item.get('above'))} on {str(item.get('below'))}"
        for item in actions
    )


def first_physical_key(actions: list[dict]) -> str:
    if not actions:
        return "<none>"
    item = actions[0]
    return f"{str(item.get('above'))} on {str(item.get('below'))}"


def write_counter_csv(path: Path, counter: Counter[str], total: int, label: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[label, "count", "fraction"])
        writer.writeheader()
        for key, count in counter.most_common():
            writer.writerow({label: key, "count": count, "fraction": count / total})


def git_value(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("-goal", required=True)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--max-depth", type=int, default=25)
    ap.add_argument("--max-symbolic-steps", type=int, default=25)
    ap.add_argument("--c", type=float, default=1.41421356237)
    ap.add_argument("--heuristic-multiplier", type=float, default=5.0)
    ap.add_argument("--lambda-add", type=float, default=0.5)
    ap.add_argument("--lambda-max", type=float, default=0.0)
    ap.add_argument("--lambda-ff", type=float, default=0.5)
    ap.add_argument(
        "--rollout-outcome",
        choices=["progress", "most-likely", "sample"],
        default="progress",
    )
    ap.add_argument("--trace", action="store_true")
    ap.add_argument(
        "--planner-script", default="make_plan_railroad_mcts.py",
        help="Path relative to the DeepSym repository root, or an absolute path.",
    )
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs must be at least 1")

    repo_root = Path(__file__).resolve().parent
    opts_path = Path(args.opts).expanduser()
    if not opts_path.is_absolute():
        opts_path = (repo_root / opts_path).resolve()
    planner_script = Path(args.planner_script).expanduser()
    if not planner_script.is_absolute():
        planner_script = (repo_root / planner_script).resolve()

    if not opts_path.exists():
        raise FileNotFoundError(opts_path)
    if not planner_script.exists():
        raise FileNotFoundError(planner_script)

    save_dir = resolve_save_dir(opts_path, repo_root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (save_dir / "mcts_experiments" / f"full_plans_{timestamp}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result_source = save_dir / "mcts_result.json"
    plan_source = save_dir / "plan.txt"

    manifest = {
        "created_at": timestamp,
        "repo_root": str(repo_root),
        "git_commit": git_value(repo_root, "rev-parse", "HEAD"),
        "git_status": git_value(repo_root, "status", "--short"),
        "opts": str(opts_path),
        "goal": args.goal,
        "runs": args.runs,
        "parameters": {
            "iterations": args.iterations,
            "max_depth": args.max_depth,
            "max_symbolic_steps": args.max_symbolic_steps,
            "mcts_runs_per_state": 1,
            "c": args.c,
            "heuristic_multiplier": args.heuristic_multiplier,
            "lambda_add": args.lambda_add,
            "lambda_max": args.lambda_max,
            "lambda_ff": args.lambda_ff,
            "rollout_outcome": args.rollout_outcome,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    rows: list[dict] = []
    for run_index in range(args.runs):
        run_dir = output_dir / f"run_{run_index:03d}"
        run_dir.mkdir()

        cmd = [
            sys.executable,
            str(planner_script),
            "-opts", str(opts_path),
            "-goal", args.goal,
            "--iterations", str(args.iterations),
            "--max-depth", str(args.max_depth),
            "--max-symbolic-steps", str(args.max_symbolic_steps),
            "--mcts-runs", "1",
            "--c", str(args.c),
            "--heuristic-multiplier", str(args.heuristic_multiplier),
            "--lambda-add", str(args.lambda_add),
            "--lambda-max", str(args.lambda_max),
            "--lambda-ff", str(args.lambda_ff),
            "--rollout-outcome", args.rollout_outcome,
        ]
        if args.rollout_outcome == "sample":
            # This seed controls only the representative rollout successor choice.
            cmd += ["--random-seed", str(run_index)]
        if args.trace:
            cmd.append("--trace")

        started = time.perf_counter()
        completed = subprocess.run(
            cmd, cwd=repo_root, text=True, capture_output=True
        )
        elapsed = time.perf_counter() - started
        (run_dir / "stdout.txt").write_text(completed.stdout)
        (run_dir / "stderr.txt").write_text(completed.stderr)
        (run_dir / "command.json").write_text(json.dumps(cmd, indent=2))

        if completed.returncode != 0:
            rows.append({
                "run": run_index,
                "ok": False,
                "elapsed_seconds": elapsed,
                "first_symbolic_action": "",
                "first_physical_action": "",
                "physical_plan": "",
                "representative_branch_probability": "",
                "representative_goal_reached": "",
                "symbolic_action_count": "",
                "physical_action_count": "",
                "error": f"planner exited with code {completed.returncode}",
            })
            continue

        if not result_source.exists():
            raise FileNotFoundError(
                f"{result_source} was not produced by run {run_index}"
            )

        shutil.copy2(result_source, run_dir / "mcts_result.json")
        if plan_source.exists():
            shutil.copy2(plan_source, run_dir / "plan.txt")

        result = json.loads(result_source.read_text())
        history = result.get("symbolic_action_history", [])
        physical = result.get("physical_stack_actions", [])
        row = {
            "run": run_index,
            "ok": True,
            "elapsed_seconds": elapsed,
            "first_symbolic_action": history[0] if history else "<none>",
            "first_physical_action": first_physical_key(physical),
            "physical_plan": physical_plan_key(physical),
            "representative_branch_probability": float(
                result.get("representative_branch_probability", 0.0)
            ),
            "representative_goal_reached": bool(
                result.get("representative_rollout_goal_reached", False)
            ),
            "symbolic_action_count": len(history),
            "physical_action_count": len(physical),
            "error": "",
        }
        rows.append(row)
        print(
            f"[{run_index + 1:3d}/{args.runs}] "
            f"{row['first_symbolic_action']} | "
            f"p_repr={row['representative_branch_probability']:.6f} | "
            f"goal={row['representative_goal_reached']}"
        )

    fieldnames = [
        "run", "ok", "elapsed_seconds",
        "first_symbolic_action", "first_physical_action", "physical_plan",
        "representative_branch_probability", "representative_goal_reached",
        "symbolic_action_count", "physical_action_count", "error",
    ]
    with (output_dir / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    good = [row for row in rows if row["ok"]]
    if not good:
        raise RuntimeError(f"All runs failed. Inspect {output_dir}")

    first_symbolic = Counter(str(row["first_symbolic_action"]) for row in good)
    first_physical = Counter(str(row["first_physical_action"]) for row in good)
    plans = Counter(str(row["physical_plan"]) for row in good)

    write_counter_csv(
        output_dir / "first_symbolic_action_frequency.csv",
        first_symbolic, len(good), "first_symbolic_action",
    )
    write_counter_csv(
        output_dir / "first_physical_action_frequency.csv",
        first_physical, len(good), "first_physical_action",
    )
    write_counter_csv(
        output_dir / "plan_frequency.csv",
        plans, len(good), "physical_plan",
    )

    probabilities = [
        float(row["representative_branch_probability"]) for row in good
    ]
    summary = {
        "successful_process_runs": len(good),
        "failed_process_runs": len(rows) - len(good),
        "representative_rollout_goal_rate": (
            sum(bool(row["representative_goal_reached"]) for row in good) / len(good)
        ),
        "mean_representative_branch_probability": fmean(probabilities),
        "most_common_first_symbolic_actions": first_symbolic.most_common(),
        "most_common_first_physical_actions": first_physical.most_common(),
        "most_common_physical_plans": plans.most_common(),
        "warning": (
            "representative_branch_probability and representative rollout goal rate "
            "describe the chosen diagnostic branch, not the closed-loop MCTS policy."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
