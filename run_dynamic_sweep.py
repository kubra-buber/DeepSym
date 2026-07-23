#!/usr/bin/env python3
"""Run named Dynamic EMA-VQ ablations from a single YAML suite file.

Each experiment is stored as::

    <output_root>/<experiment_name>/
        experiment_opts.yaml
        experiment_manifest.json
        seed_1/
        seed_2/
        ...
        aggregate/

Cross-experiment comparisons are stored under::

    <output_root>/comparisons/<comparison_name>/

The runner does not hard-code model hyperparameters. It merges each
experiment's ``overrides`` into the base opts file, so the resolved opts.yaml
saved inside every seed folder remains the authoritative run configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml


SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
BEST_FILES = [
    "encoder1_best.ckpt",
    "decoder1_best.ckpt",
    "encoder2_best.ckpt",
    "decoder2_best.ckpt",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def validate_name(name: str) -> str:
    if not SAFE_NAME.match(name):
        raise ValueError(
            f"Invalid experiment/comparison name {name!r}. "
            "Use letters, numbers, dot, underscore, or hyphen."
        )
    return name


def normalize_experiments(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("experiments")
    if isinstance(raw, dict):
        experiments = []
        for name, body in raw.items():
            body = {} if body is None else dict(body)
            body["name"] = name
            experiments.append(body)
        return experiments
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    raise ValueError("config.experiments must be a list or mapping")


def normalize_comparisons(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = config.get("comparisons", [])
    if isinstance(raw, dict):
        comparisons = []
        for name, members in raw.items():
            if isinstance(members, dict):
                item = dict(members)
                item["name"] = name
            else:
                item = {"name": name, "experiments": members}
            comparisons.append(item)
        return comparisons
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    raise ValueError("config.comparisons must be a list or mapping")


def command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def run_and_tee(command: Sequence[str], log_path: Path, dry_run: bool) -> None:
    print("$ " + command_text(command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(part) for part in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def is_complete(run_dir: Path, level: str) -> bool:
    if level == "1":
        required = [
            "encoder1_best.ckpt",
            "decoder1_best.ckpt",
            "best_level1.json",
            "metrics.csv",
            "growth_events_level1.json",
        ]
        return all((run_dir / filename).exists() for filename in required)

    if level == "2":
        required = [
            "encoder2_best.ckpt",
            "decoder2_best.ckpt",
            "best_level2.json",
            "metrics.csv",
            "growth_events_level2.json",
        ]
        return all((run_dir / filename).exists() for filename in required)

    return (run_dir / "poster_metrics.json").exists()


def checkpoints_complete(run_dir: Path) -> bool:
    return all((run_dir / filename).exists() for filename in BEST_FILES)


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def source_argument(name: str, path: Path) -> str:
    return f"{name}={path}"


def aggregate(
    python_bin: str,
    aggregate_script: Path,
    sources: Sequence[Tuple[str, Path, Sequence[int]]],
    output_dir: Path,
    dry_run: bool,
) -> None:
    command: List[str] = [python_bin, str(aggregate_script)]
    for name, path, seeds in sources:
        command.extend(["--source", source_argument(name, path)])
        if seeds:
            command.extend(["--seed-filter", f"{name}={','.join(str(int(seed)) for seed in seeds)}"])
    command.extend(["--experiments", *[name for name, _, _ in sources]])
    command.extend(["--output", str(output_dir)])
    run_and_tee(command, output_dir / "aggregate.log", dry_run)


def main() -> None:
    parser = argparse.ArgumentParser("Run Dynamic EMA-VQ hyperparameter sweeps.")
    parser.add_argument("-c", "--config", required=True, help="Sweep YAML")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment names")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--rerun", action="store_true", help="Ignore completed-run markers")
    parser.add_argument("--no-aggregate", action="store_true", help="Skip per-experiment and comparison aggregation")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    repo_root = Path(config.get("repo_root", config_path.parent)).expanduser().resolve()
    base_opts_path = Path(config["base_opts"])
    if not base_opts_path.is_absolute():
        base_opts_path = (repo_root / base_opts_path).resolve()
    output_root = Path(config.get("output_root", "save/dynamic_sweeps"))
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()

    python_bin = str(config.get("python_bin", sys.executable))
    train_script = Path(config.get("train_script", "train.py"))
    if not train_script.is_absolute():
        train_script = (repo_root / train_script).resolve()
    aggregate_script = Path(config.get("aggregate_script", "aggregate_dynamic_sweeps.py"))
    if not aggregate_script.is_absolute():
        aggregate_script = (repo_root / aggregate_script).resolve()
    poster_eval_script = Path(config.get("poster_eval_script", "poster_eval.py"))
    if not poster_eval_script.is_absolute():
        poster_eval_script = (repo_root / poster_eval_script).resolve()

    for required in (base_opts_path, train_script, aggregate_script):
        if not required.exists():
            raise FileNotFoundError(required)

    base_opts = load_yaml(base_opts_path)
    default_seeds = [int(seed) for seed in config.get("seeds", [1, 2, 3])]
    default_num_workers = int(config.get("num_workers", 0))
    default_device = config.get("device")
    skip_done = bool(config.get("skip_done", True))
    repair_eval = bool(config.get("repair_missing_evaluation", True))
    continue_on_error = bool(config.get("continue_on_error", False))

    selected = set(args.only) if args.only else None
    experiments = normalize_experiments(config)
    comparisons = normalize_comparisons(config)
    names = [validate_name(str(exp["name"])) for exp in experiments]
    if len(names) != len(set(names)):
        raise ValueError("Experiment names must be unique")
    if selected:
        unknown = selected.difference(names)
        if unknown:
            raise ValueError(f"Unknown --only experiments: {sorted(unknown)}")

    output_root.mkdir(parents=True, exist_ok=True)
    suite_manifest = {
        "config": str(config_path),
        "base_opts": str(base_opts_path),
        "repo_root": str(repo_root),
        "git_commit": git_commit(repo_root),
        "experiments": names,
    }
    if not args.dry_run:
        (output_root / "suite_manifest.json").write_text(json.dumps(suite_manifest, indent=2) + "\n")
        (output_root / "suite_config_snapshot.yaml").write_text(config_path.read_text())

    experiment_sources: Dict[str, Path] = {}
    experiment_seeds: Dict[str, List[int]] = {}
    failed: List[str] = []

    for exp in experiments:
        name = validate_name(str(exp["name"]))
        enabled = bool(exp.get("enabled", True))
        if not enabled or (selected is not None and name not in selected):
            continue

        existing_root = exp.get("existing_root")
        if existing_root:
            source = Path(existing_root).expanduser()
            if not source.is_absolute():
                source = (repo_root / source).resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            experiment_sources[name] = source
            experiment_seeds[name] = [int(seed) for seed in exp.get("seeds", default_seeds)]
            print(f"[external] {name} -> {source}")
            if not args.no_aggregate:
                aggregate(
                    python_bin,
                    aggregate_script,
                    [(name, source, experiment_seeds[name])],
                    output_root / name / "aggregate",
                    args.dry_run,
                )
            continue

        exp_dir = output_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        overrides = dict(exp.get("overrides", {}))
        resolved = deep_merge(base_opts, overrides)
        resolved["experiment_name"] = name
        resolved["experiment_description"] = str(exp.get("description", ""))
        resolved["sweep_name"] = str(config.get("sweep_name", output_root.name))
        resolved["sweep_overrides"] = overrides
        resolved["sweep_config"] = str(config_path)
        resolved["sweep_git_commit"] = git_commit(repo_root)

        experiment_opts = exp_dir / "experiment_opts.yaml"
        manifest = {
            "name": name,
            "description": exp.get("description", ""),
            "seeds": [int(seed) for seed in exp.get("seeds", default_seeds)],
            "overrides": overrides,
            "base_opts": str(base_opts_path),
            "source_git_commit": git_commit(repo_root),
        }
        if not args.dry_run:
            experiment_opts.write_text(yaml.safe_dump(resolved, sort_keys=False))
            (exp_dir / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        seeds = manifest["seeds"]
        level = str(exp.get("level", "both"))
        num_workers = int(exp.get("num_workers", default_num_workers))
        device = exp.get("device", default_device)
        skip_poster_eval = bool(exp.get("skip_poster_eval", False))

        for seed in seeds:
            run_dir = exp_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                if not args.rerun and skip_done and is_complete(run_dir, level):
                    print(f"[skip] {name} seed={seed} already complete")
                    continue

                # A training run may have finished while poster evaluation was
                # interrupted. Repair that case without retraining.
                if (
                    not args.rerun
                    and repair_eval
                    and checkpoints_complete(run_dir)
                    and (run_dir / "opts.yaml").exists()
                    and not is_complete(run_dir)
                    and poster_eval_script.exists()
                ):
                    print(f"[repair-eval] {name} seed={seed}")
                    command = [
                        python_bin,
                        str(poster_eval_script),
                        "-opts",
                        str(run_dir / "opts.yaml"),
                        "--model",
                        "dynamic",
                    ]
                    run_and_tee(command, run_dir / "poster_eval_repair.log", args.dry_run)
                    continue

                command = [
                    python_bin,
                    str(train_script),
                    "-opts",
                    str(experiment_opts),
                    "--model",
                    "dynamic",
                    "--seed",
                    str(seed),
                    "--save-dir",
                    str(run_dir),
                    "--level",
                    level,
                    "--num-workers",
                    str(num_workers),
                ]
                if device:
                    command.extend(["--device", str(device)])
                if skip_poster_eval:
                    command.append("--skip-poster-eval")
                run_and_tee(command, run_dir / "train.log", args.dry_run)
            except Exception as exc:
                failed.append(f"{name}/seed_{seed}: {exc}")
                print(f"ERROR: {failed[-1]}", file=sys.stderr)
                if not continue_on_error:
                    raise

        experiment_sources[name] = exp_dir
        experiment_seeds[name] = list(seeds)
        if not args.no_aggregate:
            aggregate(
                python_bin,
                aggregate_script,
                [(name, exp_dir, experiment_seeds[name])],
                exp_dir / "aggregate",
                args.dry_run,
            )

    if not args.no_aggregate:
        for comparison in comparisons:
            name = validate_name(str(comparison["name"]))
            members = [str(item) for item in comparison.get("experiments", [])]
            if selected is not None:
                # Keep explicitly selected external baseline if it is listed,
                # but do not require disabled experiments.
                members = [member for member in members if member in experiment_sources]
            missing = [member for member in members if member not in experiment_sources]
            if missing:
                print(f"[skip comparison {name}] unavailable experiments: {missing}")
                continue
            sources = [(member, experiment_sources[member], experiment_seeds.get(member, [])) for member in members]
            if sources:
                aggregate(
                    python_bin,
                    aggregate_script,
                    sources,
                    output_root / "comparisons" / name,
                    args.dry_run,
                )

        if bool(config.get("aggregate_all", True)) and experiment_sources:
            sources = [(name, experiment_sources[name], experiment_seeds.get(name, [])) for name in names if name in experiment_sources]
            aggregate(
                python_bin,
                aggregate_script,
                sources,
                output_root / "comparisons" / "all_experiments",
                args.dry_run,
            )

    if failed:
        print("\nFailed runs:")
        for item in failed:
            print("  - " + item)
        raise SystemExit(1)

    print(f"\nSweep outputs: {output_root}")


if __name__ == "__main__":
    main()
