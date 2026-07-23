#!/usr/bin/env python3
"""Analyze Level-1 Dynamic EMA-VQ sweeps.

Expected folder layout:
    save/dynamic_prune_level1_sweeps/
      l1_kobj3/seed_1/
      l1_kobj3/seed_2/
      ...

For every run this script writes:
  analysis/
    canonical_assignments.csv
    object_symbol_map.pdf/png
    latent_space.pdf/png
    code_usage.pdf/png

For every experiment it writes:
  aggregate/
    seed_pair_stability.csv
    coassignment_matrix.pdf/png
    growth_active_codes.pdf/png
    validation_mse_curves.pdf/png

At the sweep root it writes:
  analysis/
    level1_run_metrics.csv
    level1_experiment_summary.csv

The analysis uses the same 50 canonical central-view objects as
test_first_poster.py: 5 families x 10 sizes.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import data

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install it in the active environment."
    ) from exc


FAMILY_NAMES = [
    "Sphere",
    "Cube",
    "V. cylinder",
    "H. cylinder",
    "Cup",
]

# Affordance grouping used by the poster evaluation:
# sphere; cube + vertical cylinder; horizontal cylinder; cup.
AFFORDANCE_GROUP_BY_FAMILY = np.asarray([0, 1, 1, 2, 3], dtype=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Level-1 Dynamic EMA-VQ sweep runs."
    )
    parser.add_argument(
        "--root",
        default="save/dynamic_level1_sweeps",
        help="Sweep root containing experiment/seed_* directories.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment names. Default: discover all l1_* folders.",
    )
    parser.add_argument(
        "--checkpoint-ext",
        default="_best",
        help="Checkpoint suffix, normally _best or _last.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_model(run_dir: Path, checkpoint_ext: str):
    opts = load_yaml(run_dir / "opts.yaml")
    opts["device"] = "cpu"
    opts["save"] = str(run_dir)

    model_name = str(opts.get("poster_model", "dynamic"))
    module_name = {
        "dynamic": "models_vq_dynamic",
        "dynamic_prune": "models_vq_dynamic_prune",
        "vq": "models_vq",
    }.get(model_name)
    if module_name is None:
        raise ValueError(
            f"Unsupported poster_model={model_name!r} in {run_dir / 'opts.yaml'}"
        )
    module = importlib.import_module(module_name)
    model = module.EffectRegressorMLP(opts)
    model.load(str(run_dir), checkpoint_ext, 1)
    model.encoder1.eval()
    return model, opts


def run_before_vq(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    modules = list(encoder.children())
    if len(modules) < 2 or not hasattr(modules[-1], "get_indices"):
        raise TypeError(
            "Expected encoder1 to end in a VQ layer with get_indices()."
        )
    h = x
    for layer in modules[:-1]:
        h = layer(h)
    return h


def canonical_objects(opts: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return transformed and raw 50 canonical central-view observations."""
    raw_path = Path("data/img/obs_prev_z.pt")
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing canonical tensor: {raw_path}")

    raw_all = torch.load(raw_path, map_location="cpu")
    raw_all = raw_all.reshape(5, 10, 3, 4, 4, 42, 42)
    raw = raw_all[:, :, 0, 2, 2].reshape(50, 1, 42, 42).float()

    size = int(opts["size"])
    transform = data.default_transform(
        size=size,
        affine=False,
        mean=0.279,
        std=0.0094,
    )
    transformed = torch.empty(50, 1, size, size)
    for index in range(50):
        transformed[index] = transform(raw[index])
    return transformed, raw


def cluster_purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    clusters = np.asarray(clusters, dtype=int)
    if labels.size == 0:
        return float("nan")
    correct = 0
    for code in np.unique(clusters):
        subset = labels[clusters == code]
        counts = np.bincount(subset)
        correct += int(counts.max())
    return correct / labels.size


def assignment_perplexity(indices: np.ndarray) -> float:
    counts = np.bincount(indices)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -(probabilities * np.log(probabilities)).sum()
    return float(np.exp(entropy))


def read_best_mse(run_dir: Path) -> Tuple[float, int]:
    path = run_dir / "best_level1.json"
    with path.open("r") as handle:
        value = json.load(handle)
    return float(value["eval_effect"]), int(value["epoch"])


def read_structure_events(run_dir: Path) -> List[Dict]:
    path = run_dir / "growth_events_level1.json"
    if not path.exists():
        return []
    with path.open("r") as handle:
        value = json.load(handle)
    return value if isinstance(value, list) else []


def ema_occupancy(layer, active_k: int) -> np.ndarray:
    if not hasattr(layer, "cluster_size"):
        return np.full(active_k, np.nan, dtype=float)
    mass = layer.cluster_size[:active_k].detach().cpu().numpy().astype(float)
    mass = np.clip(mass, 0.0, None)
    total = float(mass.sum())
    if total <= 0.0:
        return np.full(active_k, np.nan, dtype=float)
    return mass / total


def save_figure(fig: plt.Figure, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_symbol_map(indices: np.ndarray, stem: Path, dpi: int) -> None:
    matrix = indices.reshape(5, 10)
    capacity = max(1, int(matrix.max()) + 1)

    fig, ax = plt.subplots(figsize=(10.5, 4.9), constrained_layout=True)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="tab20",
        vmin=-0.5,
        vmax=max(capacity - 0.5, 0.5),
    )
    ax.set_yticks(np.arange(5))
    ax.set_yticklabels(FAMILY_NAMES)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(np.arange(1, 11))
    ax.set_xlabel("Canonical size index")
    ax.set_title("Object-code assignment on 50 canonical objects")

    for family in range(5):
        for size in range(10):
            ax.text(
                size,
                family,
                str(int(matrix[family, size])),
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    ticks = np.arange(capacity)
    colorbar = fig.colorbar(image, ax=ax, ticks=ticks, pad=0.02)
    colorbar.set_label("Object code")
    save_figure(fig, stem, dpi)


def plot_latent(
    latent: np.ndarray,
    indices: np.ndarray,
    code_vectors: np.ndarray,
    stem: Path,
    dpi: int,
) -> None:
    if latent.shape[1] != 2:
        return

    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    markers = ["o", "s", "^", "D", "P"]

    for family in range(5):
        selection = np.arange(family * 10, (family + 1) * 10)
        scatter = ax.scatter(
            latent[selection, 0],
            latent[selection, 1],
            c=indices[selection],
            cmap="tab20",
            marker=markers[family],
            s=62,
            edgecolors="black",
            linewidths=0.5,
            label=FAMILY_NAMES[family],
        )
        for idx in selection:
            ax.annotate(
                str(idx % 10 + 1),
                (latent[idx, 0], latent[idx, 1]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )

    ax.scatter(
        code_vectors[:, 0],
        code_vectors[:, 1],
        marker="X",
        s=210,
        c=np.arange(len(code_vectors)),
        cmap="tab20",
        edgecolors="black",
        linewidths=1.2,
        label="Code vectors",
    )
    for code, vector in enumerate(code_vectors):
        ax.annotate(
            f"C{code}",
            (vector[0], vector[1]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Pre-VQ latent dimension 1")
    ax.set_ylabel("Pre-VQ latent dimension 2")
    ax.set_title("Canonical objects and active code vectors")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    save_figure(fig, stem, dpi)


def plot_usage(indices: np.ndarray, active_k: int, stem: Path, dpi: int) -> None:
    counts = np.bincount(indices, minlength=active_k)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    bars = ax.bar(np.arange(active_k), counts, edgecolor="black")
    ax.set_xticks(np.arange(active_k))
    ax.set_xlabel("Object code")
    ax.set_ylabel("Canonical objects assigned")
    ax.set_title("Code usage on 50 canonical objects")
    ax.set_ylim(0, max(counts.max() * 1.18, 1))
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            str(int(count)),
            ha="center",
            va="bottom",
        )
    save_figure(fig, stem, dpi)


def plot_ema_occupancy(
    occupancy: np.ndarray,
    stem: Path,
    dpi: int,
) -> None:
    if occupancy.size == 0 or not np.isfinite(occupancy).any():
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    positions = np.arange(len(occupancy))
    bars = ax.bar(positions, occupancy * 100.0, edgecolor="black")
    ax.set_xticks(positions)
    ax.set_xlabel("Object code")
    ax.set_ylabel("EMA training-stream occupancy (%)")
    ax.set_title("Code occupancy used by the pruning criterion")
    upper = max(float(np.nanmax(occupancy) * 118.0), 1.0)
    ax.set_ylim(0, upper)
    for bar, fraction in zip(bars, occupancy):
        if np.isfinite(fraction):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.015,
                f"{100.0 * fraction:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    save_figure(fig, stem, dpi)


def analyze_run(
    run_dir: Path,
    experiment: str,
    seed: int,
    checkpoint_ext: str,
    dpi: int,
) -> Tuple[Dict, np.ndarray]:
    model, opts = load_model(run_dir, checkpoint_ext)
    transformed, _raw = canonical_objects(opts)

    with torch.no_grad():
        latent_tensor = run_before_vq(model.encoder1, transformed)
        layer = list(model.encoder1.children())[-1]
        indices_tensor = layer.get_indices(latent_tensor)
        active_k = int(layer.get_num_codes())

        latent_flat = latent_tensor.reshape(latent_tensor.shape[0], -1)
        code_vectors = layer.embedding.weight[:active_k].detach().cpu()
        distances = torch.cdist(latent_flat, code_vectors).pow(2)
        assigned_distances = distances[
            torch.arange(latent_flat.shape[0]),
            indices_tensor,
        ]
        occupancy = ema_occupancy(layer, active_k)

    indices = indices_tensor.cpu().numpy().astype(int)
    latent = latent_flat.cpu().numpy()
    code_vectors_np = code_vectors.numpy()
    assigned_distances_np = assigned_distances.cpu().numpy()

    checkpoint_tag = checkpoint_ext.strip("_") or "checkpoint"
    analysis_dir = run_dir / f"analysis_{checkpoint_tag}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    family_labels = np.repeat(np.arange(5), 10)
    size_labels = np.tile(np.arange(1, 11), 5)
    affordance_labels = AFFORDANCE_GROUP_BY_FAMILY[family_labels]

    assignment_rows = []
    for idx in range(50):
        row = {
            "canonical_index": idx,
            "family_index": int(family_labels[idx]),
            "family": FAMILY_NAMES[family_labels[idx]],
            "size": int(size_labels[idx]),
            "object_code": int(indices[idx]),
            "assigned_squared_distance": float(assigned_distances_np[idx]),
        }
        for dim in range(latent.shape[1]):
            row[f"latent_{dim}"] = float(latent[idx, dim])
        assignment_rows.append(row)
    write_csv(analysis_dir / "canonical_assignments.csv", assignment_rows)

    plot_symbol_map(
        indices,
        analysis_dir / "object_symbol_map",
        dpi,
    )
    plot_latent(
        latent,
        indices,
        code_vectors_np,
        analysis_dir / "latent_space",
        dpi,
    )
    plot_usage(
        indices,
        active_k,
        analysis_dir / "code_usage",
        dpi,
    )
    plot_ema_occupancy(
        occupancy,
        analysis_dir / "ema_training_occupancy",
        dpi,
    )

    canonical_counts = np.bincount(indices, minlength=active_k)
    usage_rows = []
    for code in range(active_k):
        usage_rows.append({
            "object_code": code,
            "canonical_count": int(canonical_counts[code]),
            "canonical_fraction": float(canonical_counts[code] / 50.0),
            "ema_training_occupancy": (
                float(occupancy[code])
                if np.isfinite(occupancy[code]) else float("nan")
            ),
        })
    write_csv(analysis_dir / "code_usage_comparison.csv", usage_rows)

    mse, best_epoch = read_best_mse(run_dir)
    counts = np.bincount(indices, minlength=active_k)
    used_counts = counts[counts > 0]
    used_k = int((counts > 0).sum())

    events = read_structure_events(run_dir)
    event_counts = {
        kind: sum(
            1 for event in events
            if event.get("event_type", "grow") == kind
        )
        for kind in ("grow", "prune", "merge")
    }

    finite_occupancy = occupancy[np.isfinite(occupancy)]
    if finite_occupancy.size:
        occupancy_entropy = -(
            finite_occupancy
            * np.log(np.clip(finite_occupancy, 1e-12, None))
        ).sum()
        occupancy_perplexity = float(np.exp(occupancy_entropy))
        occupancy_min = float(finite_occupancy.min())
        occupancy_max = float(finite_occupancy.max())
    else:
        occupancy_perplexity = float("nan")
        occupancy_min = float("nan")
        occupancy_max = float("nan")

    row = {
        "experiment": experiment,
        "seed": seed,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_level1_mse": mse,
        "active_k": active_k,
        "used_k_canonical": used_k,
        "perplexity_canonical": assignment_perplexity(indices),
        "minimum_code_count_canonical": int(used_counts.min()) if used_counts.size else 0,
        "minimum_code_fraction_canonical": (
            float(used_counts.min() / 50.0) if used_counts.size else 0.0
        ),
        "shape_purity_canonical": cluster_purity(family_labels, indices),
        "affordance_purity_canonical": cluster_purity(affordance_labels, indices),
        "mean_assigned_squared_distance": float(assigned_distances_np.mean()),
        "p95_assigned_squared_distance": float(
            np.quantile(assigned_distances_np, 0.95)
        ),
        "ema_occupancy_min": occupancy_min,
        "ema_occupancy_max": occupancy_max,
        "ema_occupancy_perplexity": occupancy_perplexity,
        "growth_events": event_counts["grow"],
        "prune_events": event_counts["prune"],
        "merge_events": event_counts["merge"],
    }
    return row, indices


def read_metrics_rows(run_dir: Path) -> List[Dict[str, float]]:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row.get("level", -1))) != 1:
                continue
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def plot_experiment_curves(
    experiment_dir: Path,
    seed_dirs: Sequence[Tuple[int, Path]],
    dpi: int,
    checkpoint_ext: str,
) -> None:
    checkpoint_tag = checkpoint_ext.strip("_") or "checkpoint"
    aggregate_dir = experiment_dir / f"aggregate_{checkpoint_tag}"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    fig_k, ax_k = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    fig_mse, ax_mse = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)

    any_rows = False
    for seed, run_dir in seed_dirs:
        rows = read_metrics_rows(run_dir)
        if not rows:
            continue
        any_rows = True
        epochs = [row["epoch"] for row in rows]
        active = [row.get("active_codes", float("nan")) for row in rows]
        eval_mse = [row.get("eval_effect", float("nan")) for row in rows]

        ax_k.step(epochs, active, where="post", label=f"seed {seed}")
        ax_mse.plot(epochs, eval_mse, label=f"seed {seed}", linewidth=1.2)

    if any_rows:
        ax_k.set_xlabel("Epoch")
        ax_k.set_ylabel("Active object codes")
        ax_k.set_title("Dynamic codebook growth")
        ax_k.grid(alpha=0.25)
        ax_k.legend()

        ax_mse.set_xlabel("Epoch")
        ax_mse.set_ylabel("Validation weighted MSE")
        ax_mse.set_title("Level-1 validation error")
        ax_mse.grid(alpha=0.25)
        ax_mse.legend()

        save_figure(fig_k, aggregate_dir / "growth_active_codes", dpi)
        save_figure(fig_mse, aggregate_dir / "validation_mse_curves", dpi)
    else:
        plt.close(fig_k)
        plt.close(fig_mse)


def pairwise_stability(
    experiment_dir: Path,
    assignments: Sequence[Tuple[int, np.ndarray]],
    dpi: int,
    checkpoint_ext: str,
) -> Tuple[List[Dict], Dict]:
    rows: List[Dict] = []
    ari_values: List[float] = []
    nmi_values: List[float] = []

    for i in range(len(assignments)):
        seed_a, labels_a = assignments[i]
        for j in range(i + 1, len(assignments)):
            seed_b, labels_b = assignments[j]
            ari = float(adjusted_rand_score(labels_a, labels_b))
            nmi = float(normalized_mutual_info_score(labels_a, labels_b))
            ari_values.append(ari)
            nmi_values.append(nmi)
            rows.append(
                {
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "adjusted_rand_index": ari,
                    "normalized_mutual_information": nmi,
                }
            )

    checkpoint_tag = checkpoint_ext.strip("_") or "checkpoint"
    aggregate_dir = experiment_dir / f"aggregate_{checkpoint_tag}"
    write_csv(aggregate_dir / "seed_pair_stability.csv", rows)

    n = len(assignments)
    coassignment = np.zeros((50, 50), dtype=float)
    for _seed, labels in assignments:
        coassignment += (labels[:, None] == labels[None, :]).astype(float)
    if n:
        coassignment /= n

    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    image = ax.imshow(
        coassignment,
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
    )
    boundaries = [10, 20, 30, 40]
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.8)
    centers = np.arange(5) * 10 + 4.5
    ax.set_xticks(centers)
    ax.set_xticklabels(FAMILY_NAMES, rotation=35, ha="right")
    ax.set_yticks(centers)
    ax.set_yticklabels(FAMILY_NAMES)
    ax.set_title("Seed consensus: probability that two objects share a code")
    fig.colorbar(image, ax=ax, label="Co-assignment frequency")
    save_figure(fig, aggregate_dir / "coassignment_matrix", dpi)

    summary = {
        "pairwise_ari_mean": (
            float(np.mean(ari_values)) if ari_values else float("nan")
        ),
        "pairwise_ari_min": (
            float(np.min(ari_values)) if ari_values else float("nan")
        ),
        "pairwise_nmi_mean": (
            float(np.mean(nmi_values)) if nmi_values else float("nan")
        ),
        "pairwise_nmi_min": (
            float(np.min(nmi_values)) if nmi_values else float("nan")
        ),
        "mean_offdiagonal_coassignment": float(
            coassignment[~np.eye(50, dtype=bool)].mean()
        ) if n else float("nan"),
    }
    return rows, summary


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    return (
        float(array.mean()),
        float(array.std(ddof=1) if array.size > 1 else 0.0),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if args.experiments:
        experiment_dirs = [root / name for name in args.experiments]
    else:
        # Experiment names are arbitrary (for example grow_only_k16 or
        # prune_1pct_k16).  Discover a directory by its seed_* children
        # instead of requiring the obsolete l1_ prefix.
        experiment_dirs = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and any(child.is_dir() for child in path.glob("seed_*"))
        )

    print(
        "Discovered experiments: "
        + (", ".join(path.name for path in experiment_dirs) or "NONE")
    )

    all_run_rows: List[Dict] = []
    experiment_summaries: List[Dict] = []

    for experiment_dir in experiment_dirs:
        experiment = experiment_dir.name
        seed_dirs: List[Tuple[int, Path]] = []
        for run_dir in sorted(experiment_dir.glob("seed_*")):
            try:
                seed = int(run_dir.name.split("_")[-1])
            except ValueError:
                continue
            encoder_checkpoint = run_dir / f"encoder1{args.checkpoint_ext}.ckpt"
            decoder_checkpoint = run_dir / f"decoder1{args.checkpoint_ext}.ckpt"
            if not encoder_checkpoint.exists() or not decoder_checkpoint.exists():
                print(
                    f"WARNING: missing checkpoint for {run_dir}: "
                    f"{encoder_checkpoint.name}"
                )
                continue
            seed_dirs.append((seed, run_dir))

        if not seed_dirs:
            print(f"WARNING: no completed Level-1 runs in {experiment_dir}")
            continue

        assignments: List[Tuple[int, np.ndarray]] = []
        run_rows: List[Dict] = []
        for seed, run_dir in seed_dirs:
            print(f"Analyzing {experiment} seed {seed}: {run_dir}")
            row, indices = analyze_run(
                run_dir,
                experiment,
                seed,
                args.checkpoint_ext,
                args.dpi,
            )
            run_rows.append(row)
            all_run_rows.append(row)
            assignments.append((seed, indices))

        plot_experiment_curves(
            experiment_dir, seed_dirs, args.dpi, args.checkpoint_ext
        )
        _, stability = pairwise_stability(
            experiment_dir,
            assignments,
            args.dpi,
            args.checkpoint_ext,
        )

        summary: Dict = {
            "experiment": experiment,
            "n_seeds": len(run_rows),
            **stability,
        }
        for key in [
            "best_level1_mse",
            "active_k",
            "used_k_canonical",
            "perplexity_canonical",
            "minimum_code_fraction_canonical",
            "shape_purity_canonical",
            "affordance_purity_canonical",
            "mean_assigned_squared_distance",
            "p95_assigned_squared_distance",
            "ema_occupancy_min",
            "ema_occupancy_max",
            "ema_occupancy_perplexity",
            "growth_events",
            "prune_events",
            "merge_events",
        ]:
            mean, std = mean_std(row[key] for row in run_rows)
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std

        experiment_summaries.append(summary)

    if not all_run_rows:
        raise RuntimeError(
            "No runs were analyzed. Check the root path, experiment folder "
            "layout, and checkpoint suffix."
        )

    checkpoint_tag = args.checkpoint_ext.strip("_") or "checkpoint"
    output_dir = root / f"analysis_{checkpoint_tag}"
    write_csv(output_dir / "level1_run_metrics.csv", all_run_rows)
    write_csv(
        output_dir / "level1_experiment_summary.csv",
        experiment_summaries,
    )

    print()
    print(f"Run metrics: {output_dir / 'level1_run_metrics.csv'}")
    print(
        "Experiment summary: "
        f"{output_dir / 'level1_experiment_summary.csv'}"
    )
    for row in experiment_summaries:
        print(
            f"{row['experiment']}: "
            f"MSE={row['best_level1_mse_mean']:.6f}"
            f"±{row['best_level1_mse_std']:.6f}, "
            f"K={row['active_k_mean']:.2f}"
            f"±{row['active_k_std']:.2f}, "
            f"ARI={row['pairwise_ari_mean']:.3f}, "
            f"affordance purity="
            f"{row['affordance_purity_canonical_mean']:.3f}"
        )


if __name__ == "__main__":
    main()