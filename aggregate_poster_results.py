"""Aggregate multiple DeepSym poster runs into tables and comparison figures."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["original", "vq", "dynamic"]
METHOD_LABELS = {
    "original": "Original DeepSym",
    "vq": "Fixed EMA-VQ",
    "dynamic": "Dynamic EMA-VQ",
}


def save_figure(fig, stem: str) -> None:
    fig.savefig(stem + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def flatten_metrics(metrics: Dict) -> Dict[str, float]:
    tuple_effect = metrics.get("tuple_effect") or {}
    return {
        "seed": int(metrics.get("seed", -1)),
        "level1_weighted_mse": float(metrics["level1"]["weighted_mse"]),
        "level2_weighted_mse": float(metrics["level2"]["weighted_mse"]),
        "level1_unweighted_mse": float(metrics["level1"]["unweighted_mse"]),
        "level2_unweighted_mse": float(metrics["level2"]["unweighted_mse"]),
        "object_used_codes": float(metrics["object_codes"]["used_codes_threshold"]),
        "relation_used_codes": float(metrics["relation_codes"]["used_codes_threshold"]),
        "object_perplexity": float(metrics["object_codes"]["perplexity"]),
        "relation_perplexity": float(metrics["relation_codes"]["perplexity"]),
        "active_object_codes": float(metrics.get("active_object_codes", metrics["object_codes"]["capacity"])),
        "active_relation_codes": float(metrics.get("active_relation_codes", metrics["relation_codes"]["capacity"])),
        "object_shape_purity": float(metrics["object_semantics"]["shape_purity"]),
        "object_affordance_purity": float(metrics["object_semantics"]["affordance_purity"]),
        "object_affordance_nmi": float(metrics["object_semantics"]["affordance_nmi"]),
        "tuple_effect_purity": float(tuple_effect.get("weighted_tuple_purity", float("nan"))),
        "tuple_effect_entropy": float(tuple_effect.get("conditional_effect_entropy_nats", float("nan"))),
    }


def mean_std(values: List[float]):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def summarize(records: Dict[str, List[Dict[str, float]]]) -> List[Dict[str, object]]:
    fields = [
        "level1_weighted_mse", "level2_weighted_mse",
        "object_used_codes", "relation_used_codes",
        "object_perplexity", "relation_perplexity",
        "active_object_codes", "active_relation_codes",
        "object_shape_purity", "object_affordance_purity", "object_affordance_nmi",
        "tuple_effect_purity", "tuple_effect_entropy",
    ]
    rows = []
    for method in METHOD_ORDER:
        if method not in records:
            continue
        row = {"method": method, "label": METHOD_LABELS[method], "n": len(records[method])}
        for field in fields:
            mean, std = mean_std([record[field] for record in records[method]])
            row[field + "_mean"] = mean
            row[field + "_std"] = std
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "--"
    return ("%.*f $\\pm$ %.*f" % (decimals, mean, decimals, std))


def write_latex(rows: List[Dict[str, object]], path: str) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & L1 MSE $\downarrow$ & L2 MSE $\downarrow$ & Obj. codes & Rel. codes & Aff. purity $\uparrow$ & Tuple purity $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "%s & %s & %s & %s & %s & %s & %s \\\\" % (
                row["label"],
                fmt_pm(row["level1_weighted_mse_mean"], row["level1_weighted_mse_std"], 4),
                fmt_pm(row["level2_weighted_mse_mean"], row["level2_weighted_mse_std"], 4),
                fmt_pm(row["object_used_codes_mean"], row["object_used_codes_std"], 2),
                fmt_pm(row["relation_used_codes_mean"], row["relation_used_codes_std"], 2),
                fmt_pm(row["object_affordance_purity_mean"], row["object_affordance_purity_std"], 3),
                fmt_pm(row["tuple_effect_purity_mean"], row["tuple_effect_purity_std"], 3),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def comparison_plot(rows: List[Dict[str, object]], field: str, ylabel: str, title: str, stem: str) -> None:
    labels = [row["label"] for row in rows]
    means = [row[field + "_mean"] for row in rows]
    stds = [row[field + "_std"] for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, stem)


def read_metrics_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = {}
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def plot_dynamic_trajectories(metric_files: List[str], level: int, stem: str) -> None:
    trajectories = []
    for metrics_json in metric_files:
        metrics = json.load(open(metrics_json))
        if metrics.get("model") != "dynamic":
            continue
        rows = read_metrics_csv(os.path.join(os.path.dirname(metrics_json), "metrics.csv"))
        selected = [row for row in rows if int(row.get("level", -1)) == level and "active_codes" in row]
        if selected:
            trajectories.append((metrics.get("seed", -1), selected))
    if not trajectories:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for seed, rows in sorted(trajectories):
        ax.step(
            [row["epoch"] for row in rows],
            [row["active_codes"] for row in rows],
            where="post",
            label="seed %s" % seed,
            alpha=0.85,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active codebook size")
    ax.set_title("Dynamic VQ stability across seeds — level %d" % level)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_figure(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser("Aggregate poster metrics across seeds.")
    parser.add_argument("--root", required=True, help="Root containing method/seed_* run folders")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output = os.path.abspath(args.output or os.path.join(root, "aggregate"))
    os.makedirs(output, exist_ok=True)
    metric_files = sorted(glob.glob(os.path.join(root, "**", "poster_metrics.json"), recursive=True))
    if not metric_files:
        raise FileNotFoundError("No poster_metrics.json files found under %s" % root)

    records = defaultdict(list)
    raw_rows = []
    for path in metric_files:
        metrics = json.load(open(path))
        method = metrics["model"]
        record = flatten_metrics(metrics)
        record["method"] = method
        record["run_dir"] = os.path.dirname(path)
        records[method].append(record)
        raw_rows.append(record)

    summary = summarize(records)
    write_csv(raw_rows, os.path.join(output, "poster_all_runs.csv"))
    write_csv(summary, os.path.join(output, "poster_summary.csv"))
    with open(os.path.join(output, "poster_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    write_latex(summary, os.path.join(output, "poster_summary.tex"))

    plots = [
        ("level1_weighted_mse", "Weighted MSE", "Level-1 effect prediction", "comparison_level1_mse"),
        ("level2_weighted_mse", "Weighted MSE", "Level-2 effect prediction", "comparison_level2_mse"),
        ("object_used_codes", "Used symbols", "Object-symbol utilization", "comparison_object_codes"),
        ("relation_used_codes", "Used symbols", "Relation-symbol utilization", "comparison_relation_codes"),
        ("object_affordance_purity", "Purity", "Affordance-relevant object-symbol purity", "comparison_object_purity"),
        ("tuple_effect_purity", "Purity", "Effect predictiveness of symbolic tuples", "comparison_tuple_purity"),
    ]
    for field, ylabel, title, filename in plots:
        comparison_plot(summary, field, ylabel, title, os.path.join(output, filename))

    plot_dynamic_trajectories(metric_files, 1, os.path.join(output, "dynamic_growth_all_seeds_level1"))
    plot_dynamic_trajectories(metric_files, 2, os.path.join(output, "dynamic_growth_all_seeds_level2"))

    print("Aggregated %d runs into %s" % (len(metric_files), output))
    for row in summary:
        print(
            "%s: n=%d L2=%.6f±%.6f objectK=%.2f±%.2f relationK=%.2f±%.2f"
            % (
                row["label"], row["n"],
                row["level2_weighted_mse_mean"], row["level2_weighted_mse_std"],
                row["object_used_codes_mean"], row["object_used_codes_std"],
                row["relation_used_codes_mean"], row["relation_used_codes_std"],
            )
        )


if __name__ == "__main__":
    main()