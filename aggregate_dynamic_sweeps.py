#!/usr/bin/env python3
"""Aggregate Dynamic EMA-VQ hyperparameter experiments across seeds.

The script understands experiment folders of the form::

    save/dynamic_sweeps/<experiment>/seed_<N>/poster_metrics.json

It can also compare external experiment roots, which is useful for reusing the
existing poster baseline without retraining it::

    --source baseline=save/poster_5seed/dynamic
    --source kobj6=save/dynamic_sweeps/kobj6_krel2

For every aggregation it writes CSV/JSON/LaTeX summaries and one PDF+PNG per
metric. Growth curves are reconstructed from the per-run metrics.csv files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


TRACKED_HPARAMS = [
    "vq_num_embeddings1",
    "vq_num_embeddings2",
    "dynamic_initial_embeddings",
    "dynamic_warmup_steps",
    "dynamic_growth_interval",
    "dynamic_min_support",
    "dynamic_min_support_fraction",
    "dynamic_required_checks",
    "surprise_threshold_1",
    "surprise_threshold_2",
    "vq_commitment_cost",
    "vq_decay",
    "vq_epsilon",
    "batch_size1",
    "batch_size2",
    "epoch1",
    "epoch2",
    "learning_rate1",
    "learning_rate2",
]

METRICS: Mapping[str, Tuple[str, str, str]] = {
    "level1_weighted_mse": (
        "Level-1 weighted MSE",
        "Weighted MSE",
        "lower",
    ),
    "level2_weighted_mse": (
        "Level-2 weighted MSE",
        "Weighted MSE",
        "lower",
    ),
    "active_object_codes": (
        "Active object codes at selected checkpoint",
        "Active codes",
        "context",
    ),
    "object_used_codes": (
        "Used object codes",
        "Used codes",
        "context",
    ),
    "active_relation_codes": (
        "Active relation codes at selected checkpoint",
        "Active codes",
        "context",
    ),
    "relation_used_codes": (
        "Used relation codes",
        "Used codes",
        "context",
    ),
    "object_perplexity": (
        "Object-code perplexity",
        "Perplexity",
        "context",
    ),
    "relation_perplexity": (
        "Relation-code perplexity",
        "Perplexity",
        "context",
    ),
    "object_affordance_purity": (
        "Affordance-relevant object-code purity",
        "Purity",
        "higher",
    ),
    "tuple_effect_purity": (
        "Effect predictiveness of symbolic tuples",
        "Purity",
        "higher",
    ),
    "tuple_effect_entropy": (
        "Conditional effect entropy",
        "Entropy (nats)",
        "lower",
    ),
    "growth_events_level1": (
        "Number of Level-1 growth events",
        "Growth events",
        "context",
    ),
    "growth_events_level2": (
        "Number of Level-2 growth events",
        "Growth events",
        "context",
    ),
    "first_growth_step_level1": (
        "First Level-1 growth step",
        "Optimization step",
        "context",
    ),
    "first_growth_step_level2": (
        "First Level-2 growth step",
        "Optimization step",
        "context",
    ),
}


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(stem) + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_seed_filter(value: str) -> Tuple[str, set[int]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--seed-filter must be NAME=1,2,3")
    name, raw = value.split("=", 1)
    name = name.strip()
    try:
        seeds = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid seed filter: {value}") from exc
    if not name or not seeds:
        raise argparse.ArgumentTypeError("--seed-filter requires a name and at least one seed")
    return name, seeds


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    path = Path(path).expanduser().resolve()
    if not name:
        raise argparse.ArgumentTypeError("Source name cannot be empty")
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Source path does not exist: {path}")
    return name, path


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def growth_stats(events: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    steps = [safe_float(event.get("step")) for event in events]
    steps = [step for step in steps if math.isfinite(step)]
    return {
        "count": float(len(events)),
        "first": min(steps) if steps else float("nan"),
        "last": max(steps) if steps else float("nan"),
    }


def flatten_run(
    metrics: Mapping[str, Any],
    opts: Mapping[str, Any],
    experiment: str,
    run_dir: Path,
) -> Dict[str, Any]:
    tuple_effect = metrics.get("tuple_effect") or {}
    g1 = growth_stats(metrics.get("growth_events_level1") or [])
    g2 = growth_stats(metrics.get("growth_events_level2") or [])

    row: Dict[str, Any] = {
        "experiment": experiment,
        "seed": int(metrics.get("seed", opts.get("seed", -1))),
        "run_dir": str(run_dir),
        "level1_weighted_mse": safe_float((metrics.get("level1") or {}).get("weighted_mse")),
        "level2_weighted_mse": safe_float((metrics.get("level2") or {}).get("weighted_mse")),
        "level1_unweighted_mse": safe_float((metrics.get("level1") or {}).get("unweighted_mse")),
        "level2_unweighted_mse": safe_float((metrics.get("level2") or {}).get("unweighted_mse")),
        "active_object_codes": safe_float(metrics.get("active_object_codes")),
        "active_relation_codes": safe_float(metrics.get("active_relation_codes")),
        "object_used_codes": safe_float((metrics.get("object_codes") or {}).get("used_codes_threshold")),
        "relation_used_codes": safe_float((metrics.get("relation_codes") or {}).get("used_codes_threshold")),
        "object_perplexity": safe_float((metrics.get("object_codes") or {}).get("perplexity")),
        "relation_perplexity": safe_float((metrics.get("relation_codes") or {}).get("perplexity")),
        "object_shape_purity": safe_float((metrics.get("object_semantics") or {}).get("shape_purity")),
        "object_affordance_purity": safe_float((metrics.get("object_semantics") or {}).get("affordance_purity")),
        "object_affordance_nmi": safe_float((metrics.get("object_semantics") or {}).get("affordance_nmi")),
        "tuple_effect_purity": safe_float(tuple_effect.get("weighted_tuple_purity")),
        "tuple_effect_entropy": safe_float(tuple_effect.get("conditional_effect_entropy_nats")),
        "growth_events_level1": g1["count"],
        "growth_events_level2": g2["count"],
        "first_growth_step_level1": g1["first"],
        "last_growth_step_level1": g1["last"],
        "first_growth_step_level2": g2["first"],
        "last_growth_step_level2": g2["last"],
    }
    for key in TRACKED_HPARAMS:
        row[key] = opts.get(key)
    return row


def discover_source(experiment: str, source_root: Path) -> List[Dict[str, Any]]:
    metric_paths = sorted(source_root.glob("**/poster_metrics.json"))
    rows: List[Dict[str, Any]] = []
    for metric_path in metric_paths:
        run_dir = metric_path.parent
        with metric_path.open("r") as handle:
            metrics = json.load(handle)
        opts = load_yaml(run_dir / "opts.yaml")
        rows.append(flatten_run(metrics, opts, experiment, run_dir))
    return rows


def infer_sources_from_root(root: Path) -> List[Tuple[str, Path]]:
    sources: List[Tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name in {"comparisons", "aggregate"}:
            continue
        if list(child.glob("**/poster_metrics.json")):
            sources.append((child.name, child))
    return sources


def finite(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray([safe_float(value) for value in values], dtype=float)
    return array[np.isfinite(array)]


def mean_std(values: Iterable[Any]) -> Tuple[float, float]:
    array = finite(values)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1) if array.size > 1 else 0.0)


def constant_or_blank(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, float) and math.isnan(value):
            continue
        if value not in values:
            values.append(value)
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return json.dumps(values)


def summarize(rows: Sequence[Mapping[str, Any]], order: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["experiment"])].append(row)

    summary: List[Dict[str, Any]] = []
    for experiment in order:
        exp_rows = grouped.get(experiment, [])
        if not exp_rows:
            continue
        item: Dict[str, Any] = {
            "experiment": experiment,
            "n": len(exp_rows),
            "seeds": ",".join(str(int(row["seed"])) for row in sorted(exp_rows, key=lambda x: int(x["seed"]))),
        }
        for key in TRACKED_HPARAMS:
            item[key] = constant_or_blank(exp_rows, key)
        for metric in METRICS:
            mean, std = mean_std(row.get(metric) for row in exp_rows)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
        summary.append(item)
    return summary


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_pm(mean: Any, std: Any, digits: int = 4) -> str:
    m = safe_float(mean)
    s = safe_float(std)
    if not (math.isfinite(m) and math.isfinite(s)):
        return "--"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def write_latex(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Experiment & L1 MSE $\downarrow$ & L2 MSE $\downarrow$ & Active obj. & Used obj. & Aff. purity $\uparrow$ & Tuple purity $\uparrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "%s & %s & %s & %s & %s & %s & %s \\\\" % (
                latex_escape(str(row["experiment"])),
                format_pm(row.get("level1_weighted_mse_mean"), row.get("level1_weighted_mse_std")),
                format_pm(row.get("level2_weighted_mse_mean"), row.get("level2_weighted_mse_std")),
                format_pm(row.get("active_object_codes_mean"), row.get("active_object_codes_std"), 2),
                format_pm(row.get("object_used_codes_mean"), row.get("object_used_codes_std"), 2),
                format_pm(row.get("object_affordance_purity_mean"), row.get("object_affordance_purity_std"), 3),
                format_pm(row.get("tuple_effect_purity_mean"), row.get("tuple_effect_purity_std"), 3),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def metric_plot(summary: Sequence[Mapping[str, Any]], metric: str, output_dir: Path) -> None:
    title, ylabel, direction = METRICS[metric]
    labels = [str(row["experiment"]) for row in summary]
    means = np.asarray([safe_float(row.get(f"{metric}_mean")) for row in summary])
    stds = np.asarray([safe_float(row.get(f"{metric}_std"), 0.0) for row in summary])
    valid = np.isfinite(means)
    if not valid.any():
        return

    width = max(7.2, 1.05 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 5.1), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x[valid], means[valid], yerr=stds[valid], capsize=5, width=0.68, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28 if len(labels) > 4 else 0, ha="right" if len(labels) > 4 else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    finite_tops = means[valid] + np.nan_to_num(stds[valid])
    offset = 0.025 * max(1e-12, float(np.max(np.abs(finite_tops))))
    for bar, mean in zip(bars, means[valid]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + offset, f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    if direction == "lower":
        ax.text(0.995, 0.98, "lower is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    elif direction == "higher":
        ax.text(0.995, 0.98, "higher is better", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    save_figure(fig, output_dir / metric)


def read_epoch_growth(run_dir: Path, level: int) -> Dict[int, float]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return {}
    output: Dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                if int(float(row.get("level", -1))) != level:
                    continue
                epoch = int(float(row["epoch"]))
                active = float(row["active_codes"])
            except (KeyError, TypeError, ValueError):
                continue
            output[epoch] = active
    return output


def growth_comparison(raw_rows: Sequence[Mapping[str, Any]], order: Sequence[str], level: int, output_dir: Path) -> None:
    grouped: MutableMapping[str, List[Dict[int, float]]] = defaultdict(list)
    for row in raw_rows:
        curve = read_epoch_growth(Path(str(row["run_dir"])), level)
        if curve:
            grouped[str(row["experiment"])].append(curve)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    plotted = 0
    for experiment in order:
        curves = grouped.get(experiment, [])
        if not curves:
            continue
        epochs = sorted(set().union(*(curve.keys() for curve in curves)))
        means, stds = [], []
        for epoch in epochs:
            values = [curve[epoch] for curve in curves if epoch in curve]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
        epochs_arr = np.asarray(epochs)
        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)
        line = ax.plot(epochs_arr, means_arr, label=experiment, linewidth=2)[0]
        ax.fill_between(epochs_arr, means_arr - stds_arr, means_arr + stds_arr, alpha=0.16, color=line.get_color())
        plotted += 1
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active codes")
    ax.set_title(f"Dynamic codebook growth — Level {level}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2 if plotted > 4 else 1)
    save_figure(fig, output_dir / f"growth_comparison_level{level}")


def best_runs(raw_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["experiment"])].append(row)
    output = []
    for experiment, rows in grouped.items():
        valid = [row for row in rows if math.isfinite(safe_float(row.get("level2_weighted_mse")))]
        if not valid:
            continue
        best = min(valid, key=lambda row: safe_float(row.get("level2_weighted_mse")))
        output.append({
            "experiment": experiment,
            "best_seed": int(best["seed"]),
            "best_level2_weighted_mse": best["level2_weighted_mse"],
            "run_dir": best["run_dir"],
        })
    return sorted(output, key=lambda row: row["experiment"])


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate Dynamic EMA-VQ sweep results.")
    parser.add_argument("--root", action="append", default=[], help="Root containing experiment folders. May be repeated.")
    parser.add_argument("--source", action="append", default=[], help="Explicit experiment source as NAME=PATH. May be repeated.")
    parser.add_argument("--experiments", nargs="*", default=None, help="Optional experiment-name filter and display order.")
    parser.add_argument("--seed-filter", action="append", default=[], help="Limit one experiment to seeds, e.g. NAME=1,2,3. May be repeated.")
    parser.add_argument("--output", required=True, help="Output directory for the comparison.")
    args = parser.parse_args()

    sources: List[Tuple[str, Path]] = []
    for value in args.source:
        sources.append(parse_source(value))
    for root_value in args.root:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        sources.extend(infer_sources_from_root(root))

    # Preserve first occurrence and reject ambiguous duplicate names pointing to
    # different paths.
    dedup: Dict[str, Path] = {}
    for name, path in sources:
        if name in dedup and dedup[name] != path:
            raise ValueError(f"Experiment {name!r} was mapped to two paths: {dedup[name]} and {path}")
        dedup[name] = path

    seed_filters: Dict[str, set[int]] = {}
    for value in args.seed_filter:
        name, seeds = parse_seed_filter(value)
        seed_filters[name] = seeds

    if args.experiments:
        missing = [name for name in args.experiments if name not in dedup]
        if missing:
            raise ValueError(f"Requested experiments not found: {missing}")
        order = list(args.experiments)
    else:
        order = list(dedup)

    raw_rows: List[Dict[str, Any]] = []
    for name in order:
        rows = discover_source(name, dedup[name])
        if name in seed_filters:
            rows = [row for row in rows if int(row["seed"]) in seed_filters[name]]
        if not rows:
            print(f"WARNING: no poster_metrics.json files found for {name}: {dedup[name]}")
        raw_rows.extend(rows)

    if not raw_rows:
        raise FileNotFoundError("No poster_metrics.json files were found in the selected experiment sources")

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(raw_rows, order)

    write_csv(raw_rows, output_dir / "all_runs.csv")
    write_csv(summary, output_dir / "summary.csv")
    write_csv(best_runs(raw_rows), output_dir / "best_runs.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    write_latex(summary, output_dir / "summary.tex")

    source_manifest = {name: str(dedup[name]) for name in order}
    (output_dir / "sources.json").write_text(json.dumps(source_manifest, indent=2) + "\n")

    for metric in METRICS:
        metric_plot(summary, metric, output_dir / "figures")
    growth_comparison(raw_rows, order, 1, output_dir / "figures")
    growth_comparison(raw_rows, order, 2, output_dir / "figures")

    print(f"Aggregated {len(raw_rows)} runs from {len(summary)} experiments")
    for row in summary:
        print(
            f"{row['experiment']}: n={row['n']} "
            f"L2={format_pm(row.get('level2_weighted_mse_mean'), row.get('level2_weighted_mse_std'), 6)} "
            f"active_obj={format_pm(row.get('active_object_codes_mean'), row.get('active_object_codes_std'), 2)}"
        )
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
