#!/usr/bin/env python3
"""Inspect semantic stability and Level-1 effect differences across sweep runs.

Outputs under:
    <root>/semantic_analysis_<checkpoint>/

Main questions answered:
1. Do 5-code checkpoints have lower deterministic Level-1 MSE than 4-code ones?
2. Which horizontal-cylinder sizes form satellite categories in each seed?
3. Are those size boundaries consistent across seeds?
4. Do their actual and predicted effect distributions differ by action?
5. Does seed-to-seed ARI improve or deteriorate?

This script evaluates the selected checkpoint directly instead of reading the
historical best MSE from best_level1.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections import Counter, defaultdict
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
    raise SystemExit("Install scikit-learn in the active environment.") from exc


FAMILY_NAMES = [
    "Sphere",
    "Cube",
    "V. cylinder",
    "H. cylinder",
    "Cup",
]
HORIZONTAL_CYLINDER_FAMILY = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="save/dynamic_prune_level1_sweeps",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
    )
    parser.add_argument(
        "--checkpoint-ext",
        default="_last",
        choices=("_best", "_last", "_valpruned"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--action-names",
        nargs="*",
        default=["forward poke", "side poke", "top poke"],
    )
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def model_module_name(opts: Dict) -> str:
    kind = str(opts.get("poster_model", "dynamic"))
    mapping = {
        "dynamic": "models_vq_dynamic",
        "dynamic_prune": "models_vq_dynamic_prune",
        "vq": "models_vq",
    }
    if kind not in mapping:
        raise ValueError(f"Unsupported poster_model={kind!r}")
    return mapping[kind]


def load_model(run_dir: Path, checkpoint_ext: str):
    opts = load_yaml(run_dir / "opts.yaml")
    opts["device"] = "cpu"
    opts["save"] = str(run_dir)
    module = importlib.import_module(model_module_name(opts))
    model = module.EffectRegressorMLP(opts)
    model.load(str(run_dir), checkpoint_ext, 1)
    model.prepare_level(1, training=False)
    return model, opts


def run_before_vq(encoder: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    modules = list(encoder.children())
    if len(modules) < 2 or not hasattr(modules[-1], "get_indices"):
        raise TypeError("encoder1 must end with a VQ layer")
    output = inputs
    for module in modules[:-1]:
        output = module(output)
    return output


def canonical_inputs(opts: Dict) -> torch.Tensor:
    raw = torch.load("data/img/obs_prev_z.pt", map_location="cpu")
    raw = raw.reshape(5, 10, 3, 4, 4, 42, 42)
    raw = raw[:, :, 0, 2, 2].reshape(50, 1, 42, 42).float()

    transform = data.default_transform(
        size=int(opts["size"]),
        affine=False,
        mean=0.279,
        std=0.0094,
    )
    transformed = torch.empty(
        50,
        1,
        int(opts["size"]),
        int(opts["size"]),
    )
    for index in range(50):
        transformed[index] = transform(raw[index])
    return transformed


def canonical_assignments(model, opts: Dict) -> np.ndarray:
    inputs = canonical_inputs(opts)
    with torch.no_grad():
        latent = run_before_vq(model.encoder1, inputs)
        layer = list(model.encoder1.children())[-1]
        return layer.get_indices(latent).cpu().numpy().astype(int)


def deterministic_dataset(opts: Dict):
    transform = data.default_transform(
        size=int(opts["size"]),
        affine=False,
        mean=0.279,
        std=0.0094,
    )
    return data.SingleObjectData(transform=transform)


def sample_metadata(index: int) -> Tuple[int, int, int, int, int]:
    """Decode flat index for [family, size, action, view_x, view_y]."""
    view_y = index % 4
    index //= 4
    view_x = index % 4
    index //= 4
    action = index % 3
    index //= 3
    size = index % 10
    family = index // 10
    return family, size, action, view_x, view_y


def action_label(
    action_index: int,
    action_vector: np.ndarray,
    action_names: Sequence[str],
) -> str:
    if 0 <= action_index < len(action_names):
        return action_names[action_index]
    vector = ",".join(f"{value:g}" for value in action_vector)
    return f"action {action_index} [{vector}]"


def evaluate_full_dataset(
    model,
    opts: Dict,
    canonical_codes: np.ndarray,
    batch_size: int,
    action_names: Sequence[str],
):
    dataset = deterministic_dataset(opts)
    expected = 5 * 10 * 3 * 4 * 4
    if len(dataset) != expected:
        raise ValueError(
            f"Expected {expected} Level-1 samples, found {len(dataset)}. "
            "Update sample_metadata() for this dataset."
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    weights = model.effect_weights1.detach().cpu()
    losses: List[np.ndarray] = []
    rows: List[Dict] = []
    offset = 0

    with torch.no_grad():
        for batch in loader:
            observation = batch["observation"].to(model.device)
            effect = batch["effect"].to(model.device)
            action = batch["action"].to(model.device)

            code = model.encoder1(observation)
            prediction = model.decoder1(torch.cat([code, action], dim=-1))
            per_sample = (
                (prediction - effect).pow(2)
                * model.effect_weights1
            ).mean(dim=1)
            losses.append(per_sample.detach().cpu().numpy())

            prediction_np = prediction.detach().cpu().numpy()
            effect_np = effect.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

            for local in range(len(prediction_np)):
                flat_index = offset + local
                family, size, action_index, view_x, view_y = sample_metadata(
                    flat_index
                )
                canonical_code = int(
                    canonical_codes[family * 10 + size]
                )
                label = action_label(
                    action_index,
                    action_np[local],
                    action_names,
                )

                row = {
                    "flat_index": flat_index,
                    "family_index": family,
                    "family": FAMILY_NAMES[family],
                    "size": size + 1,
                    "action_index": action_index,
                    "action": label,
                    "view_x": view_x,
                    "view_y": view_y,
                    "canonical_code": canonical_code,
                    "sample_weighted_mse": float(
                        per_sample[local].detach().cpu()
                    ),
                }
                for dimension in range(effect_np.shape[1]):
                    row[f"actual_effect_{dimension}"] = float(
                        effect_np[local, dimension]
                    )
                    row[f"predicted_effect_{dimension}"] = float(
                        prediction_np[local, dimension]
                    )
                rows.append(row)

            offset += len(prediction_np)

    loss_vector = np.concatenate(losses)
    return float(loss_vector.mean()), float(loss_vector.std(ddof=1)), rows


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contiguous_ranges(values: Sequence[int]) -> str:
    values = sorted(set(int(value) for value in values))
    if not values:
        return ""
    ranges: List[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(
            str(start) if start == previous else f"{start}-{previous}"
        )
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def horizontal_split_rows(
    experiment: str,
    seed: int,
    assignments: np.ndarray,
) -> Tuple[List[Dict], int, Tuple[int, ...]]:
    horizontal = assignments[
        HORIZONTAL_CYLINDER_FAMILY * 10:
        (HORIZONTAL_CYLINDER_FAMILY + 1) * 10
    ]
    counts = Counter(int(value) for value in horizontal)
    main_code = counts.most_common(1)[0][0]
    satellite_sizes = tuple(
        index + 1
        for index, code in enumerate(horizontal)
        if int(code) != main_code
    )

    rows = []
    for code in sorted(counts):
        sizes = [
            index + 1
            for index, assigned in enumerate(horizontal)
            if int(assigned) == code
        ]
        rows.append({
            "experiment": experiment,
            "seed": seed,
            "active_k": int(np.unique(assignments).size),
            "main_horizontal_code": int(main_code),
            "object_code": int(code),
            "is_main_code": bool(code == main_code),
            "sizes": contiguous_ranges(sizes),
            "size_count": len(sizes),
        })
    return rows, int(main_code), satellite_sizes


def aggregate_effect_rows(
    experiment: str,
    seed: int,
    full_rows: Sequence[Dict],
    main_code: int,
) -> Tuple[List[Dict], List[Dict]]:
    grouped: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    for row in full_rows:
        if int(row["family_index"]) != HORIZONTAL_CYLINDER_FAMILY:
            continue
        grouped[(int(row["canonical_code"]), str(row["action"]))].append(row)

    summaries: List[Dict] = []
    differences: List[Dict] = []
    by_action_code: Dict[Tuple[str, int], Dict] = {}

    for (code, action), members in sorted(grouped.items()):
        effect_dimensions = len([
            key for key in members[0]
            if key.startswith("actual_effect_")
        ])
        summary = {
            "experiment": experiment,
            "seed": seed,
            "object_code": code,
            "is_main_horizontal_code": bool(code == main_code),
            "action": action,
            "sample_count": len(members),
            "size_set": contiguous_ranges(
                [int(member["size"]) for member in members]
            ),
            "weighted_mse_mean": float(np.mean([
                float(member["sample_weighted_mse"])
                for member in members
            ])),
        }
        for dimension in range(effect_dimensions):
            actual = np.asarray([
                float(member[f"actual_effect_{dimension}"])
                for member in members
            ])
            predicted = np.asarray([
                float(member[f"predicted_effect_{dimension}"])
                for member in members
            ])
            summary[f"actual_mean_{dimension}"] = float(actual.mean())
            summary[f"actual_std_{dimension}"] = float(actual.std(ddof=1))
            summary[f"predicted_mean_{dimension}"] = float(predicted.mean())
            summary[f"predicted_std_{dimension}"] = float(
                predicted.std(ddof=1)
            )
        summaries.append(summary)
        by_action_code[(action, code)] = summary

    actions = sorted({action for action, _code in by_action_code})
    satellite_codes = sorted({
        code
        for _action, code in by_action_code
        if code != main_code
    })

    for action in actions:
        main = by_action_code.get((action, main_code))
        if main is None:
            continue
        effect_dimensions = len([
            key for key in main if key.startswith("actual_mean_")
        ])
        for satellite_code in satellite_codes:
            satellite = by_action_code.get((action, satellite_code))
            if satellite is None:
                continue

            actual_delta = np.asarray([
                satellite[f"actual_mean_{dimension}"]
                - main[f"actual_mean_{dimension}"]
                for dimension in range(effect_dimensions)
            ])
            predicted_delta = np.asarray([
                satellite[f"predicted_mean_{dimension}"]
                - main[f"predicted_mean_{dimension}"]
                for dimension in range(effect_dimensions)
            ])
            row = {
                "experiment": experiment,
                "seed": seed,
                "action": action,
                "main_code": main_code,
                "satellite_code": satellite_code,
                "main_sizes": main["size_set"],
                "satellite_sizes": satellite["size_set"],
                "actual_effect_l2_difference": float(
                    np.linalg.norm(actual_delta)
                ),
                "predicted_effect_l2_difference": float(
                    np.linalg.norm(predicted_delta)
                ),
                "main_weighted_mse": main["weighted_mse_mean"],
                "satellite_weighted_mse": satellite["weighted_mse_mean"],
            }
            for dimension in range(effect_dimensions):
                row[f"actual_delta_{dimension}"] = float(
                    actual_delta[dimension]
                )
                row[f"predicted_delta_{dimension}"] = float(
                    predicted_delta[dimension]
                )
            differences.append(row)

    return summaries, differences


def plot_effect_comparison(
    run_dir: Path,
    effect_summaries: Sequence[Dict],
    main_code: int,
    checkpoint_tag: str,
    dpi: int,
) -> None:
    actions = sorted({str(row["action"]) for row in effect_summaries})
    satellite_codes = sorted({
        int(row["object_code"])
        for row in effect_summaries
        if int(row["object_code"]) != main_code
    })
    if not actions or not satellite_codes:
        return

    for satellite_code in satellite_codes:
        figure, axes = plt.subplots(
            len(actions),
            1,
            figsize=(8.5, max(3.4, 3.2 * len(actions))),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, action in enumerate(actions):
            axis = axes[row_index, 0]
            main = next(
                (
                    row for row in effect_summaries
                    if str(row["action"]) == action
                    and int(row["object_code"]) == main_code
                ),
                None,
            )
            satellite = next(
                (
                    row for row in effect_summaries
                    if str(row["action"]) == action
                    and int(row["object_code"]) == satellite_code
                ),
                None,
            )
            if main is None or satellite is None:
                axis.axis("off")
                continue

            dimensions = sorted(
                int(key.split("_")[-1])
                for key in main
                if key.startswith("actual_mean_")
            )
            positions = np.arange(len(dimensions))
            width = 0.2

            axis.bar(
                positions - 1.5 * width,
                [main[f"actual_mean_{dim}"] for dim in dimensions],
                width,
                label=f"main C{main_code} actual",
            )
            axis.bar(
                positions - 0.5 * width,
                [main[f"predicted_mean_{dim}"] for dim in dimensions],
                width,
                label=f"main C{main_code} predicted",
            )
            axis.bar(
                positions + 0.5 * width,
                [
                    satellite[f"actual_mean_{dim}"]
                    for dim in dimensions
                ],
                width,
                label=f"satellite C{satellite_code} actual",
            )
            axis.bar(
                positions + 1.5 * width,
                [
                    satellite[f"predicted_mean_{dim}"]
                    for dim in dimensions
                ],
                width,
                label=f"satellite C{satellite_code} predicted",
            )
            axis.axhline(0.0, linewidth=0.8)
            axis.set_xticks(positions)
            axis.set_xticklabels([
                f"effect {dimension}" for dimension in dimensions
            ])
            axis.set_ylabel("Normalized effect")
            axis.set_title(action)
            axis.legend(fontsize=8)

        figure.suptitle(
            "Horizontal-cylinder main vs satellite category",
            fontweight="bold",
        )
        output_dir = run_dir / f"semantic_analysis_{checkpoint_tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_dir / (
            f"horizontal_effect_mainC{main_code}_satelliteC"
            f"{satellite_code}"
        )
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(
            stem.with_suffix(".png"),
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)


def pairwise_rows(
    experiment: str,
    assignments: Sequence[Tuple[int, np.ndarray]],
    satellite_sets: Dict[int, Tuple[int, ...]],
) -> List[Dict]:
    rows = []
    for left in range(len(assignments)):
        seed_a, labels_a = assignments[left]
        for right in range(left + 1, len(assignments)):
            seed_b, labels_b = assignments[right]
            horizontal_a = labels_a[30:40]
            horizontal_b = labels_b[30:40]
            set_a = set(satellite_sets[seed_a])
            set_b = set(satellite_sets[seed_b])
            union = set_a | set_b
            jaccard = (
                len(set_a & set_b) / len(union)
                if union else 1.0
            )
            rows.append({
                "experiment": experiment,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "active_k_a": int(np.unique(labels_a).size),
                "active_k_b": int(np.unique(labels_b).size),
                "overall_ari": float(
                    adjusted_rand_score(labels_a, labels_b)
                ),
                "overall_nmi": float(
                    normalized_mutual_info_score(labels_a, labels_b)
                ),
                "horizontal_ari": float(
                    adjusted_rand_score(horizontal_a, horizontal_b)
                ),
                "horizontal_nmi": float(
                    normalized_mutual_info_score(
                        horizontal_a,
                        horizontal_b,
                    )
                ),
                "satellite_size_jaccard": float(jaccard),
                "satellite_sizes_a": contiguous_ranges(set_a),
                "satellite_sizes_b": contiguous_ranges(set_b),
            })
    return rows


def group_mse_by_k(run_rows: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for row in run_rows:
        grouped[int(row["active_k"])].append(
            float(row["checkpoint_weighted_mse"])
        )
    rows = []
    for active_k, values in sorted(grouped.items()):
        array = np.asarray(values, dtype=float)
        rows.append({
            "active_k": active_k,
            "run_count": len(values),
            "checkpoint_mse_mean": float(array.mean()),
            "checkpoint_mse_std": float(
                array.std(ddof=1) if len(array) > 1 else 0.0
            ),
            "checkpoint_mse_min": float(array.min()),
            "checkpoint_mse_max": float(array.max()),
        })
    return rows


def discover_experiments(
    root: Path,
    requested: Sequence[str] | None,
) -> List[Path]:
    if requested:
        return [root / name for name in requested]
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and any(child.is_dir() for child in path.glob("seed_*"))
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    checkpoint_tag = args.checkpoint_ext.strip("_")
    output_dir = root / f"semantic_analysis_{checkpoint_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict] = []
    split_rows: List[Dict] = []
    effect_rows: List[Dict] = []
    effect_difference_rows: List[Dict] = []
    stability_rows: List[Dict] = []

    experiments = discover_experiments(root, args.experiments)
    if not experiments:
        raise RuntimeError(f"No experiment directories found under {root}")

    for experiment_dir in experiments:
        experiment = experiment_dir.name
        assignments: List[Tuple[int, np.ndarray]] = []
        satellite_sets: Dict[int, Tuple[int, ...]] = {}

        for run_dir in sorted(experiment_dir.glob("seed_*")):
            try:
                seed = int(run_dir.name.split("_")[-1])
            except ValueError:
                continue
            if not (
                run_dir / f"encoder1{args.checkpoint_ext}.ckpt"
            ).exists():
                continue

            print(f"Analyzing {experiment} seed {seed}")
            model, opts = load_model(run_dir, args.checkpoint_ext)
            assignments_array = canonical_assignments(model, opts)
            checkpoint_mse, checkpoint_mse_std, full_rows = (
                evaluate_full_dataset(
                    model,
                    opts,
                    assignments_array,
                    args.batch_size,
                    args.action_names,
                )
            )
            active_k = int(
                list(model.encoder1.children())[-1].get_num_codes()
            )

            current_split_rows, main_code, satellite_sizes = (
                horizontal_split_rows(
                    experiment,
                    seed,
                    assignments_array,
                )
            )
            current_effect_rows, current_differences = aggregate_effect_rows(
                experiment,
                seed,
                full_rows,
                main_code,
            )
            plot_effect_comparison(
                run_dir,
                current_effect_rows,
                main_code,
                checkpoint_tag,
                args.dpi,
            )

            run_rows.append({
                "experiment": experiment,
                "seed": seed,
                "checkpoint": args.checkpoint_ext,
                "active_k": active_k,
                "checkpoint_weighted_mse": checkpoint_mse,
                "sample_loss_std": checkpoint_mse_std,
                "main_horizontal_code": main_code,
                "horizontal_satellite_sizes": contiguous_ranges(
                    satellite_sizes
                ),
                "horizontal_satellite_size_count": len(satellite_sizes),
            })
            split_rows.extend(current_split_rows)
            effect_rows.extend(current_effect_rows)
            effect_difference_rows.extend(current_differences)
            assignments.append((seed, assignments_array))
            satellite_sets[seed] = satellite_sizes

        stability_rows.extend(
            pairwise_rows(experiment, assignments, satellite_sets)
        )

    if not run_rows:
        raise RuntimeError("No checkpoints were analyzed.")

    write_csv(output_dir / "run_checkpoint_metrics.csv", run_rows)
    write_csv(output_dir / "mse_by_active_k.csv", group_mse_by_k(run_rows))
    write_csv(
        output_dir / "horizontal_cylinder_splits.csv",
        split_rows,
    )
    write_csv(
        output_dir / "horizontal_effect_distributions.csv",
        effect_rows,
    )
    write_csv(
        output_dir / "horizontal_effect_differences.csv",
        effect_difference_rows,
    )
    write_csv(
        output_dir / "seed_pair_stability.csv",
        stability_rows,
    )

    print("\nCheckpoint MSE by active K")
    for row in group_mse_by_k(run_rows):
        print(
            f"  K={row['active_k']}: "
            f"{row['checkpoint_mse_mean']:.6f}"
            f" ± {row['checkpoint_mse_std']:.6f} "
            f"(n={row['run_count']})"
        )

    print("\nHorizontal-cylinder partitions")
    for row in run_rows:
        satellite = row["horizontal_satellite_sizes"] or "none"
        print(
            f"  {row['experiment']} seed {row['seed']}: "
            f"K={row['active_k']}, satellite sizes={satellite}, "
            f"MSE={row['checkpoint_weighted_mse']:.6f}"
        )

    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
