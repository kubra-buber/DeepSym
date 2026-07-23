#!/usr/bin/env python3
"""Small Level-1 sweep runner for the grow/prune Dynamic EMA-VQ model."""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def display(command) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def complete(run_dir: Path) -> bool:
    required = [
        "encoder1_best.ckpt",
        "decoder1_best.ckpt",
        "best_level1.json",
        "metrics.csv",
        "growth_events_level1.json",
    ]
    return all((run_dir / name).exists() for name in required)


def run_and_log(command, log_path: Path, dry_run: bool) -> None:
    print("$ " + display(command))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as handle:
        process = subprocess.Popen(
            [str(item) for item in command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    root = Path(config.get("repo_root", ".")).expanduser().resolve()
    base_opts_path = Path(config["base_opts"])
    if not base_opts_path.is_absolute():
        base_opts_path = root / base_opts_path
    output_root = Path(config["output_root"])
    if not output_root.is_absolute():
        output_root = root / output_root

    python_bin = str(config.get("python_bin", sys.executable))
    train_script = root / str(config.get("train_script", "train.py"))
    seeds = [int(value) for value in config.get("seeds", [1, 2, 3])]
    device = str(config.get("device", "auto"))
    num_workers = int(config.get("num_workers", 0))
    skip_done = bool(config.get("skip_done", True))

    base_opts = load_yaml(base_opts_path)
    experiments = config.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("experiments must be a YAML mapping")

    selected = set(args.only) if args.only else None
    unknown = (
        selected.difference(experiments)
        if selected is not None
        else set()
    )
    if unknown:
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "suite_config_snapshot.yaml").write_text(
        config_path.read_text()
    )

    for name, body in experiments.items():
        body = {} if body is None else dict(body)
        if not bool(body.get("enabled", True)):
            continue
        if selected is not None and name not in selected:
            continue

        experiment_dir = output_root / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        overrides = dict(body.get("overrides", {}))
        resolved = merge(base_opts, overrides)
        resolved["experiment_name"] = name
        resolved["experiment_description"] = str(
            body.get("description", "")
        )
        resolved["sweep_config"] = str(config_path)
        resolved["sweep_overrides"] = overrides

        experiment_opts = experiment_dir / "experiment_opts.yaml"
        experiment_opts.write_text(
            yaml.safe_dump(resolved, sort_keys=False)
        )
        (experiment_dir / "experiment_manifest.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "description": body.get("description", ""),
                    "seeds": seeds,
                    "overrides": overrides,
                },
                indent=2,
            )
            + "\n"
        )

        for seed in seeds:
            run_dir = experiment_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if (
                not args.rerun
                and skip_done
                and complete(run_dir)
            ):
                print(f"[skip] {name} seed={seed}")
                continue

            command = [
                python_bin,
                str(train_script),
                "-opts",
                str(experiment_opts),
                "--model",
                "dynamic_prune",
                "--seed",
                str(seed),
                "--save-dir",
                str(run_dir),
                "--level",
                "1",
                "--skip-poster-eval",
                "--device",
                device,
                "--num-workers",
                str(num_workers),
            ]
            run_and_log(
                command,
                run_dir / "train.log",
                args.dry_run,
            )

    print(f"\nSweep outputs: {output_root}")


if __name__ == "__main__":
    main()