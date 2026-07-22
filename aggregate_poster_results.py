#!/usr/bin/env python3
"""Aggregate DeepSym poster runs into tables and poster-quality figures.

This version also supports regenerating a non-overlapping relational-symbol
grid from a selected run's poster_assignments.npz file.

Example:
    python aggregate_poster_results.py \
        --root save/poster_5seed \
        --output save/poster_5seed/aggregate \
        --relation-run save/poster_5seed/dynamic/seed_1
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
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch


plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


METHOD_ORDER = ["original", "vq", "dynamic"]

METHOD_LABELS = {
    "original": "Original DeepSym",
    "vq": "Fixed EMA-VQ",
    "dynamic": "Dynamic EMA-VQ",
}

DEFAULT_OBJECT_NAMES = [
    "Sphere",
    "Cube",
    "Vertical cylinder",
    "Horizontal cylinder",
    "Cup",
]


def save_figure(fig, stem: str, *, dpi: int = 300) -> None:
    """Save vector PDF and high-resolution PNG without clipping labels."""
    output = Path(stem)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        str(output) + ".png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        str(output) + ".pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def flatten_metrics(metrics: Dict) -> Dict[str, float]:
    tuple_effect = metrics.get("tuple_effect") or {}

    return {
        "seed": int(metrics.get("seed", -1)),
        "level1_weighted_mse": float(metrics["level1"]["weighted_mse"]),
        "level2_weighted_mse": float(metrics["level2"]["weighted_mse"]),
        "level1_unweighted_mse": float(metrics["level1"]["unweighted_mse"]),
        "level2_unweighted_mse": float(metrics["level2"]["unweighted_mse"]),
        "object_used_codes": float(
            metrics["object_codes"]["used_codes_threshold"]
        ),
        "relation_used_codes": float(
            metrics["relation_codes"]["used_codes_threshold"]
        ),
        "object_perplexity": float(metrics["object_codes"]["perplexity"]),
        "relation_perplexity": float(
            metrics["relation_codes"]["perplexity"]
        ),
        "active_object_codes": float(
            metrics.get(
                "active_object_codes",
                metrics["object_codes"]["capacity"],
            )
        ),
        "active_relation_codes": float(
            metrics.get(
                "active_relation_codes",
                metrics["relation_codes"]["capacity"],
            )
        ),
        "object_shape_purity": float(
            metrics["object_semantics"]["shape_purity"]
        ),
        "object_affordance_purity": float(
            metrics["object_semantics"]["affordance_purity"]
        ),
        "object_affordance_nmi": float(
            metrics["object_semantics"]["affordance_nmi"]
        ),
        "tuple_effect_purity": float(
            tuple_effect.get("weighted_tuple_purity", float("nan"))
        ),
        "tuple_effect_entropy": float(
            tuple_effect.get(
                "conditional_effect_entropy_nats",
                float("nan"),
            )
        ),
    }


def mean_std(values: List[float]):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return float("nan"), float("nan")

    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return mean, std


def summarize(
    records: Dict[str, List[Dict[str, float]]]
) -> List[Dict[str, object]]:
    fields = [
        "level1_weighted_mse",
        "level2_weighted_mse",
        "object_used_codes",
        "relation_used_codes",
        "object_perplexity",
        "relation_perplexity",
        "active_object_codes",
        "active_relation_codes",
        "object_shape_purity",
        "object_affordance_purity",
        "object_affordance_nmi",
        "tuple_effect_purity",
        "tuple_effect_entropy",
    ]

    rows = []

    for method in METHOD_ORDER:
        if method not in records:
            continue

        row = {
            "method": method,
            "label": METHOD_LABELS[method],
            "n": len(records[method]),
        }

        for field in fields:
            mean, std = mean_std(
                [record[field] for record in records[method]]
            )
            row[field + "_mean"] = mean
            row[field + "_std"] = std

        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    if not (math.isfinite(mean) and math.isfinite(std)):
        return "--"

    return (
        "%.*f $\\pm$ %.*f"
        % (decimals, mean, decimals, std)
    )


def write_latex(rows: List[Dict[str, object]], path: str) -> None:
    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (
            r"Method & L1 MSE $\downarrow$ & L2 MSE $\downarrow$ "
            r"& Obj. codes & Rel. codes & Aff. purity $\uparrow$ "
            r"& Tuple purity $\uparrow$ \\"
        ),
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            "%s & %s & %s & %s & %s & %s & %s \\\\"
            % (
                row["label"],
                fmt_pm(
                    row["level1_weighted_mse_mean"],
                    row["level1_weighted_mse_std"],
                    4,
                ),
                fmt_pm(
                    row["level2_weighted_mse_mean"],
                    row["level2_weighted_mse_std"],
                    4,
                ),
                fmt_pm(
                    row["object_used_codes_mean"],
                    row["object_used_codes_std"],
                    2,
                ),
                fmt_pm(
                    row["relation_used_codes_mean"],
                    row["relation_used_codes_std"],
                    2,
                ),
                fmt_pm(
                    row["object_affordance_purity_mean"],
                    row["object_affordance_purity_std"],
                    3,
                ),
                fmt_pm(
                    row["tuple_effect_purity_mean"],
                    row["tuple_effect_purity_std"],
                    3,
                ),
            )
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def comparison_plot(
    rows: List[Dict[str, object]],
    field: str,
    ylabel: str,
    title: str,
    stem: str,
) -> None:
    labels = [
        row["label"]
        .replace("Original DeepSym", "Original\nDeepSym")
        .replace("Fixed EMA-VQ", "Fixed\nEMA-VQ")
        .replace("Dynamic EMA-VQ", "Dynamic\nEMA-VQ")
        for row in rows
    ]

    means = [row[field + "_mean"] for row in rows]
    stds = [row[field + "_std"] for row in rows]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.7),
        constrained_layout=True,
    )

    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        width=0.68,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Value labels above bars. These make the poster figure readable
    # without requiring the summary table.
    finite_tops = [
        mean + std
        for mean, std in zip(means, stds)
        if math.isfinite(mean) and math.isfinite(std)
    ]
    offset = 0.025 * max(finite_tops) if finite_tops else 0.02

    for bar, mean in zip(bars, means):
        if math.isfinite(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + offset,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

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


def plot_dynamic_trajectories(
    metric_files: List[str],
    level: int,
    stem: str,
) -> None:
    trajectories = []

    for metrics_json in metric_files:
        with open(metrics_json, "r") as handle:
            metrics = json.load(handle)

        if metrics.get("model") != "dynamic":
            continue

        rows = read_metrics_csv(
            os.path.join(
                os.path.dirname(metrics_json),
                "metrics.csv",
            )
        )

        selected = [
            row
            for row in rows
            if int(row.get("level", -1)) == level
            and "active_codes" in row
        ]

        if selected:
            trajectories.append(
                (metrics.get("seed", -1), selected)
            )

    if not trajectories:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.9))

    for seed, rows in sorted(trajectories):
        ax.step(
            [row["epoch"] for row in rows],
            [row["active_codes"] for row in rows],
            where="post",
            label=f"Seed {seed}",
            alpha=0.9,
            linewidth=2.0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active codebook size")
    ax.set_title(
        f"Dynamic EMA-VQ growth across seeds — Level {level}",
        pad=18,
    )
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    # Put the legend above the axes instead of over the trajectories.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(5, len(labels)),
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
    )

    # Reserve a dedicated top margin for the external legend.
    fig.subplots_adjust(
        left=0.14,
        right=0.98,
        bottom=0.16,
        top=0.78,
    )

    save_figure(fig, stem)


def load_object_names(run_dir: Path) -> Sequence[str]:
    opts_path = run_dir / "opts.yaml"

    if not opts_path.exists():
        return DEFAULT_OBJECT_NAMES

    try:
        with opts_path.open("r") as handle:
            opts = yaml.safe_load(handle)

        names = opts.get(
            "poster_object_names",
            DEFAULT_OBJECT_NAMES,
        )

        if isinstance(names, list) and len(names) == 5:
            return [str(name) for name in names]
    except Exception as exc:
        print(
            f"WARNING: could not read object names from "
            f"{opts_path}: {exc}"
        )

    return DEFAULT_OBJECT_NAMES


def relation_palette(capacity: int) -> ListedColormap:
    """Paper-like discrete palette."""
    base = [
        "#3B2EFF",  # blue
        "#FF1E1E",  # red
        "#19A974",  # optional extra colors
        "#FFB000",
        "#7A3EFF",
        "#666666",
    ]

    if capacity <= len(base):
        colors = base[:capacity]
    else:
        tab20 = plt.get_cmap("tab20", capacity)
        colors = [tab20(i) for i in range(capacity)]

    return ListedColormap(colors)


def plot_relation_symbol_grid(
    grids: np.ndarray,
    object_names: Sequence[str],
    capacity: int,
    output_stem: str,
) -> None:
    """
    Cleaner paper-like relational symbol grid.
    Expects shape: [row_object_type, col_object_type, row_size, col_size]
    """
    grids = np.asarray(grids)

    if grids.shape != (5, 5, 10, 10):
        raise ValueError(
            "Expected relation_grids shape (5, 5, 10, 10), "
            f"received {grids.shape}"
        )

    # Paper-like short labels
    short_names = [
        "Sphere",
        "Cube",
        "V. cylinder",
        "H. cylinder",
        "Cup",
    ]

    capacity = max(int(capacity), int(np.nanmax(grids)) + 1, 1)
    cmap = relation_palette(capacity)
    norm = BoundaryNorm(np.arange(-0.5, capacity + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(
        5, 5,
        figsize=(8.2, 7.3),
        facecolor="white"
    )

    plt.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.88,
        bottom=0.15,
        wspace=0.07,
        hspace=0.10,
    )

    for r in range(5):
        for c in range(5):
            ax = axes[r, c]

            ax.imshow(
                grids[r, c],
                origin="lower",
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )

            # Remove clutter
            ax.set_xticks([])
            ax.set_yticks([])

            # Thin white border between cells
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("white")

            if r == 0:
                ax.set_title(
                    short_names[c],
                    fontsize=10,
                    fontweight="bold",
                    pad=6,
                )

            if c == 0:
                ax.set_ylabel(
                    short_names[r],
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=28,
                    va="center",
                )

    fig.suptitle(
        "Learned relational symbol by ordered object pair and relative size",
        fontsize=12,
        fontweight="bold",
        y=0.955,
    )

    # Global axes labels (paper-like)
    fig.text(
        0.52, 0.06,
        "Size of the object below",
        ha="center",
        va="center",
        fontsize=12,
    )
    fig.text(
        0.03, 0.50,
        "Size of the object above",
        ha="center",
        va="center",
        rotation=90,
        fontsize=12,
    )

    # Arrows placed OUTSIDE the 5x5 grid
    x_arrow = FancyArrowPatch(
        (0.15, 0.12),   # start
        (0.92, 0.12),   # end
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.4,
        color="black",
    )
    y_arrow = FancyArrowPatch(
        (0.05, 0.17),   # start
        (0.05, 0.90),   # end
        transform=fig.transFigure,
        arrowstyle="->",
        mutation_scale=16,
        linewidth=1.4,
        color="black",
    )

    fig.add_artist(x_arrow)
    fig.add_artist(y_arrow)

    save_figure(fig, output_stem)


def regenerate_relation_grid(
    relation_run: str,
    output_dir: str,
    output_name: str,
) -> None:
    run_dir = Path(relation_run).resolve()
    assignments_path = run_dir / "poster_assignments.npz"
    metrics_path = run_dir / "poster_metrics.json"

    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Missing {assignments_path}. Run poster_eval.py for this "
            "checkpoint first."
        )

    assignments = np.load(assignments_path)

    if "relation_grids" not in assignments:
        raise KeyError(
            f"{assignments_path} does not contain 'relation_grids'."
        )

    relation_grids = assignments["relation_grids"]

    if metrics_path.exists():
        with metrics_path.open("r") as handle:
            metrics = json.load(handle)

        capacity = int(
            metrics.get(
                "active_relation_codes",
                metrics.get(
                    "relation_codes",
                    {},
                ).get(
                    "capacity",
                    int(np.max(relation_grids)) + 1,
                ),
            )
        )
    else:
        capacity = int(np.max(relation_grids)) + 1

    object_names = load_object_names(run_dir)

    output_stem = os.path.join(
        output_dir,
        output_name,
    )

    plot_relation_symbol_grid(
        relation_grids,
        object_names,
        capacity,
        output_stem,
    )

    print(
        "Regenerated non-overlapping relation grid from "
        f"{assignments_path}"
    )
    print(f"Output: {output_stem}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        "Aggregate poster metrics across seeds."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root containing method/seed_* run folders",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--relation-run",
        default=None,
        help=(
            "Optional run directory whose poster_assignments.npz is "
            "used to regenerate the relational-symbol grid."
        ),
    )
    parser.add_argument(
        "--relation-output-name",
        default="dynamic_vq_relation_symbol_grid",
        help="Output stem for the regenerated relation grid.",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    output = os.path.abspath(
        args.output
        or os.path.join(root, "aggregate")
    )
    os.makedirs(output, exist_ok=True)

    metric_files = sorted(
        glob.glob(
            os.path.join(
                root,
                "**",
                "poster_metrics.json",
            ),
            recursive=True,
        )
    )

    if not metric_files:
        raise FileNotFoundError(
            f"No poster_metrics.json files found under {root}"
        )

    records = defaultdict(list)
    raw_rows = []

    for path in metric_files:
        with open(path, "r") as handle:
            metrics = json.load(handle)

        method = metrics["model"]
        record = flatten_metrics(metrics)
        record["method"] = method
        record["run_dir"] = os.path.dirname(path)

        records[method].append(record)
        raw_rows.append(record)

    summary = summarize(records)

    write_csv(
        raw_rows,
        os.path.join(output, "poster_all_runs.csv"),
    )
    write_csv(
        summary,
        os.path.join(output, "poster_summary.csv"),
    )

    with open(
        os.path.join(output, "poster_summary.json"),
        "w",
    ) as handle:
        json.dump(summary, handle, indent=2)

    write_latex(
        summary,
        os.path.join(output, "poster_summary.tex"),
    )

    plots = [
        (
            "level1_weighted_mse",
            "Weighted MSE",
            "Level-1 effect prediction",
            "comparison_level1_mse",
        ),
        (
            "level2_weighted_mse",
            "Weighted MSE",
            "Level-2 effect prediction",
            "comparison_level2_mse",
        ),
        (
            "object_used_codes",
            "Used symbols",
            "Object-symbol utilization",
            "comparison_object_codes",
        ),
        (
            "relation_used_codes",
            "Used symbols",
            "Relation-symbol utilization",
            "comparison_relation_codes",
        ),
        (
            "object_affordance_purity",
            "Purity",
            "Affordance-relevant object-symbol purity",
            "comparison_object_purity",
        ),
        (
            "tuple_effect_purity",
            "Purity",
            "Effect predictiveness of symbolic tuples",
            "comparison_tuple_purity",
        ),
    ]

    for field, ylabel, title, filename in plots:
        comparison_plot(
            summary,
            field,
            ylabel,
            title,
            os.path.join(output, filename),
        )

    plot_dynamic_trajectories(
        metric_files,
        1,
        os.path.join(
            output,
            "dynamic_growth_all_seeds_level1",
        ),
    )
    plot_dynamic_trajectories(
        metric_files,
        2,
        os.path.join(
            output,
            "dynamic_growth_all_seeds_level2",
        ),
    )

    if args.relation_run:
        regenerate_relation_grid(
            args.relation_run,
            output,
            args.relation_output_name,
        )

    print(
        "Aggregated %d runs into %s"
        % (len(metric_files), output)
    )

    for row in summary:
        print(
            (
                "%s: n=%d L2=%.6f±%.6f "
                "objectK=%.2f±%.2f relationK=%.2f±%.2f"
            )
            % (
                row["label"],
                row["n"],
                row["level2_weighted_mse_mean"],
                row["level2_weighted_mse_std"],
                row["object_used_codes_mean"],
                row["object_used_codes_std"],
                row["relation_used_codes_mean"],
                row["relation_used_codes_std"],
            )
        )


if __name__ == "__main__":
    main()