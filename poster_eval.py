"""Evaluate one trained run and create poster-ready metrics and figures.

Outputs are written to <run>/poster/.  PNG files are convenient for quick
inspection; matching PDF files are vector-friendly for LaTeX/Overleaf.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.colors import BoundaryNorm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import data


MODEL_MODULES = {
    "original": "models",
    "vq": "models_vq",
    "dynamic": "models_vq_dynamic",
}
DEFAULT_OBJECT_NAMES = ["Sphere", "Cube", "Vertical cylinder", "Horizontal cylinder", "Cup"]


def save_figure(fig, output_stem: str) -> None:
    fig.savefig(output_stem + ".png", dpi=220, bbox_inches="tight")
    fig.savefig(output_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def load_model(opts: Dict, model_name: str):
    module = importlib.import_module(MODEL_MODULES[model_name])
    model = module.EffectRegressorMLP(opts)
    model.load(opts["save"], "_best", 1)
    model.load(opts["save"], "_best", 2)
    for module_ in (model.encoder1, model.decoder1, model.encoder2, model.decoder2):
        module_.eval()
    return model


def make_eval_loader(opts: Dict, level: int, batch_size: int = 256):
    transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
    if level == 1:
        dataset = data.SingleObjectData(transform=transform)
    else:
        dataset = data.PairedObjectData(transform=transform)
        if hasattr(dataset, "train"):
            dataset.train = False
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset, loader


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weights: Sequence[float]) -> torch.Tensor:
    weight_tensor = torch.as_tensor(weights, dtype=pred.dtype, device=pred.device)
    return (torch.nn.functional.mse_loss(pred, target, reduction="none") * weight_tensor).mean()


def predict_batch(model, sample: Dict[str, torch.Tensor], level: int):
    device = model.device
    if level == 1:
        obs = sample["observation"].to(device)
        target = sample["effect"].to(device)
        action = sample["action"].to(device)
        code = model.encoder1(obs)
        prediction = model.decoder1(torch.cat([code, action], dim=-1))
    else:
        obs = sample["observation"].to(device)
        target = sample["effect"].to(device)
        object_codes = model.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3]))
        object_codes = object_codes.reshape(obs.shape[0], -1)
        relation_code = model.encoder2(obs)
        prediction = model.decoder2(torch.cat([object_codes, relation_code], dim=-1))
    return prediction, target


def evaluate_effect_prediction(model, loader, level: int, weights: Sequence[float]) -> Dict[str, object]:
    squared_sum = None
    count = 0
    weighted_total = 0.0
    with torch.no_grad():
        for sample in loader:
            prediction, target = predict_batch(model, sample, level)
            batch_n = int(target.shape[0])
            squared = (prediction - target).pow(2)
            batch_axis_sum = squared.sum(dim=0).detach().cpu().double()
            squared_sum = batch_axis_sum if squared_sum is None else squared_sum + batch_axis_sum
            weighted_total += float(weighted_mse(prediction, target, weights).cpu()) * batch_n
            count += batch_n
    per_axis = (squared_sum / max(1, count)).numpy()
    return {
        "samples": count,
        "weighted_mse": weighted_total / max(1, count),
        "unweighted_mse": float(per_axis.mean()),
        "per_axis_mse": [float(x) for x in per_axis],
    }


def binary_rows_to_indices(codes: torch.Tensor) -> torch.Tensor:
    bits = (codes > 0).long()
    powers = 2 ** torch.arange(bits.shape[1] - 1, -1, -1, device=bits.device)
    return (bits * powers).sum(dim=1)


def encoder_indices(encoder: torch.nn.Sequential, obs: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "original":
        return binary_rows_to_indices(encoder(obs))
    layer = encoder[-1]
    if not hasattr(layer, "get_indices"):
        raise TypeError("Expected a VQ layer with get_indices()")
    pre_vq = encoder[:-1](obs)
    return layer.get_indices(pre_vq)


def code_capacity(encoder: torch.nn.Sequential, model_name: str, code_dim: int) -> int:
    if model_name == "original":
        return 2 ** int(code_dim)
    layer = encoder[-1]
    if hasattr(layer, "get_num_codes"):
        return int(layer.get_num_codes())
    return 2 ** int(code_dim)


def usage_statistics(indices: np.ndarray, capacity: int, used_fraction: float) -> Dict[str, object]:
    counts = np.bincount(indices.astype(int), minlength=capacity)
    probabilities = counts / max(1, counts.sum())
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())
    threshold = max(1, int(math.ceil(used_fraction * counts.sum())))
    return {
        "capacity": int(capacity),
        "counts": counts.astype(int).tolist(),
        "used_codes_nonzero": int((counts > 0).sum()),
        "used_codes_threshold": int((counts >= threshold).sum()),
        "used_count_threshold": int(threshold),
        "perplexity": float(math.exp(entropy)),
        "entropy": entropy,
    }


def cluster_purity(labels: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for cluster in np.unique(clusters):
        member_labels = labels[clusters == cluster]
        if member_labels.size:
            total += Counter(member_labels.tolist()).most_common(1)[0][1]
    return float(total / max(1, labels.size))


def object_semantic_metrics(indices: np.ndarray, physical_labels: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    return {
        "shape_purity": cluster_purity(physical_labels, indices),
        "shape_nmi": float(normalized_mutual_info_score(physical_labels, indices)),
        "shape_ari": float(adjusted_rand_score(physical_labels, indices)),
        "affordance_purity": cluster_purity(groups, indices),
        "affordance_nmi": float(normalized_mutual_info_score(groups, indices)),
        "affordance_ari": float(adjusted_rand_score(groups, indices)),
    }


def tuple_effect_metrics(
    first_idx: np.ndarray,
    second_idx: np.ndarray,
    relation_idx: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    groups = defaultdict(Counter)
    for a, b, r, label in zip(first_idx, second_idx, relation_idx, labels):
        groups[(int(a), int(b), int(r))][int(label)] += 1
    correct = 0
    conditional_entropy = 0.0
    tuple_rows = []
    n = max(1, len(labels))
    for key, counts in sorted(groups.items()):
        total = sum(counts.values())
        correct += max(counts.values())
        probs = np.asarray(list(counts.values()), dtype=float) / total
        entropy = float(-(probs * np.log(probs)).sum())
        conditional_entropy += (total / n) * entropy
        tuple_rows.append({
            "first_code": key[0],
            "second_code": key[1],
            "relation_code": key[2],
            "samples": total,
            "purity": max(counts.values()) / total,
            "label_counts": dict(counts),
        })
    return {
        "num_observed_tuples": len(groups),
        "weighted_tuple_purity": float(correct / n),
        "conditional_effect_entropy_nats": float(conditional_entropy),
        "tuples": tuple_rows,
    }


def read_training_rows(run_dir: str) -> List[Dict[str, float]]:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    rows = []
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


def plot_loss_curves(rows: List[Dict[str, float]], poster_dir: str) -> None:
    for level in (1, 2):
        selected = [row for row in rows if int(row.get("level", -1)) == level]
        if not selected:
            continue
        epochs = [row["epoch"] for row in selected]
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.plot(epochs, [row["train_effect"] for row in selected], label="Training")
        ax.plot(epochs, [row["eval_effect"] for row in selected], label="Deterministic evaluation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted effect MSE")
        ax.set_title("Level %d effect prediction" % level)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        save_figure(fig, os.path.join(poster_dir, "loss_level%d" % level))


def plot_usage(counts: Sequence[int], title: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(len(counts))
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xlabel("Symbol index")
    ax.set_ylabel("Assigned samples")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_stem)


def plot_object_assignment_frequency(
    physical_labels: np.ndarray,
    indices: np.ndarray,
    object_names: Sequence[str],
    capacity: int,
    output_stem: str,
) -> None:
    matrix = np.zeros((len(object_names), capacity), dtype=float)
    for object_type in range(len(object_names)):
        selected = indices[physical_labels == object_type]
        counts = np.bincount(selected, minlength=capacity)
        matrix[object_type] = 100.0 * counts / max(1, counts.sum())
    fig, ax = plt.subplots(figsize=(max(7.0, capacity * 1.2), 4.8))
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(np.arange(capacity))
    ax.set_xticklabels(["S%d" % i for i in range(capacity)])
    ax.set_yticks(np.arange(len(object_names)))
    ax.set_yticklabels(object_names)
    ax.set_xlabel("Learned object symbol")
    ax.set_title("Object-to-symbol assignment frequency (%%)")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "%.1f" % value, ha="center", va="center", fontsize=8,
                    color="white" if value > 55 else "black")
    fig.colorbar(image, ax=ax, label="Assignment frequency (%)")
    save_figure(fig, output_stem)


def plot_object_symbol_map(
    canonical_indices: np.ndarray,
    object_names: Sequence[str],
    capacity: int,
    output_stem: str,
) -> None:
    matrix = canonical_indices.reshape(len(object_names), 10)
    cmap = plt.get_cmap("tab20", max(1, capacity))
    norm = BoundaryNorm(np.arange(-0.5, capacity + 0.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(object_names)))
    ax.set_yticklabels(object_names)
    ax.set_xticks(np.arange(10))
    ax.set_xticklabels(np.arange(1, 11))
    ax.set_xlabel("Object size index")
    ax.set_title("Learned object symbols across shape and size")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(capacity))
    colorbar.set_label("Object symbol")
    save_figure(fig, output_stem)


def relation_grid_from_canonical(model, canonical: torch.Tensor, model_name: str) -> np.ndarray:
    # grid[slot1_type, slot0_type, slot1_size, slot0_size]
    grids = np.zeros((5, 5, 10, 10), dtype=np.int64)
    device = model.device
    with torch.no_grad():
        for slot1_type in range(5):
            for slot0_type in range(5):
                first = canonical[slot0_type].repeat(10, 1, 1).reshape(-1, 1, canonical.shape[-2], canonical.shape[-1])
                second = canonical[slot1_type].repeat_interleave(10, dim=0).reshape(-1, 1, canonical.shape[-2], canonical.shape[-1])
                pair = torch.cat([first, second], dim=1).to(device)
                idx = encoder_indices(model.encoder2, pair, model_name)
                grids[slot1_type, slot0_type] = idx.reshape(10, 10).detach().cpu().numpy()
    return grids


def plot_relation_symbol_grid(
    grids: np.ndarray,
    object_names: Sequence[str],
    capacity: int,
    output_stem: str,
) -> None:
    cmap = plt.get_cmap("tab20", max(1, capacity))
    norm = BoundaryNorm(np.arange(-0.5, capacity + 0.5, 1), cmap.N)
    fig, axes = plt.subplots(5, 5, figsize=(14.5, 14.5), sharex=True, sharey=True)
    image = None
    for row in range(5):
        for col in range(5):
            ax = axes[row, col]
            image = ax.imshow(grids[row, col], origin="lower", cmap=cmap, norm=norm, aspect="equal")
            ax.set_xticks([0, 9])
            ax.set_yticks([0, 9])
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title("slot0: " + object_names[col], fontsize=8)
            if col == 0:
                ax.set_ylabel("slot1: " + object_names[row], fontsize=8)
    fig.suptitle("Learned relational symbol by ordered object pair and size\n"
                 "x: slot0 size, y: slot1 size", fontsize=15)
    fig.supxlabel("slot0 size index")
    fig.supylabel("slot1 size index")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), ticks=np.arange(capacity), shrink=0.75)
        colorbar.set_label("Relation symbol")
    fig.subplots_adjust(top=0.91, right=0.90, wspace=0.08, hspace=0.14)
    save_figure(fig, output_stem)


def plot_pair_tuple_purity(
    tuple_ids: np.ndarray,
    effect_labels: np.ndarray,
    output_stem: str,
) -> np.ndarray:
    matrix = np.zeros((5, 5), dtype=float)
    # PairedObjectData index: slot0 type = idx//500, slot1 type=(idx//10)%5
    sample_indices = np.arange(len(effect_labels))
    slot0_types = sample_indices // 500
    slot1_types = (sample_indices // 10) % 5
    for slot0 in range(5):
        for slot1 in range(5):
            mask = (slot0_types == slot0) & (slot1_types == slot1)
            matrix[slot0, slot1] = cluster_purity(effect_labels[mask], tuple_ids[mask])
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xlabel("Physical slot1 object type")
    ax.set_ylabel("Physical slot0 object type")
    ax.set_title("Effect-label purity of learned symbolic tuples")
    for row in range(5):
        for col in range(5):
            ax.text(col, row, "%.2f" % matrix[row, col], ha="center", va="center",
                    color="white" if matrix[row, col] < 0.55 else "black")
    fig.colorbar(image, ax=ax, label="Purity")
    save_figure(fig, output_stem)
    return matrix


def plot_growth(rows: List[Dict[str, float]], level: int, output_stem: str) -> None:
    selected = [row for row in rows if int(row.get("level", -1)) == level and "active_codes" in row]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.step([row["epoch"] for row in selected], [row["active_codes"] for row in selected], where="post")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Active codebook size")
    ax.set_title("Dynamic VQ codebook growth — level %d" % level)
    ax.grid(alpha=0.25)
    max_k = max(row["active_codes"] for row in selected)
    ax.set_yticks(np.arange(1, int(max_k) + 1))
    save_figure(fig, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser("Create poster metrics and figures for one DeepSym run.")
    parser.add_argument("-opts", required=True, help="Resolved run opts.yaml")
    parser.add_argument("--model", choices=tuple(MODEL_MODULES), default=None)
    parser.add_argument("--used-fraction", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(args.opts, "r") as handle:
        opts = yaml.safe_load(handle)
    model_name = args.model or opts.get("poster_model")
    if model_name not in MODEL_MODULES:
        raise ValueError("Specify --model original, vq, or dynamic")
    opts["device"] = "cpu"
    run_dir = os.path.abspath(opts["save"])
    poster_dir = os.path.join(run_dir, "poster")
    os.makedirs(poster_dir, exist_ok=True)

    model = load_model(opts, model_name)
    dataset1, loader1 = make_eval_loader(opts, 1, args.batch_size)
    dataset2, loader2 = make_eval_loader(opts, 2, args.batch_size)

    level1_eval = evaluate_effect_prediction(
        model, loader1, 1, opts.get("effect_weights1", [1, 1, 10])
    )
    level2_eval = evaluate_effect_prediction(
        model, loader2, 2, opts.get("effect_weights2", [1, 1, 5, 1, 1, 1])
    )

    all_object_indices = []
    all_physical_types = []
    with torch.no_grad():
        for sample in loader1:
            obs = sample["observation"].to(model.device)
            all_object_indices.append(encoder_indices(model.encoder1, obs, model_name).cpu())
    object_indices = torch.cat(all_object_indices).numpy().astype(int)
    per_type = len(dataset1) // 5
    physical_types = np.repeat(np.arange(5), per_type)
    if len(physical_types) != len(object_indices):
        raise ValueError("SingleObjectData does not follow expected 5-type ordering")
    group_map = np.asarray(opts.get("poster_object_target_groups", [0, 1, 1, 2, 3]), dtype=int)
    affordance_groups = group_map[physical_types]

    first_all, second_all, relation_all = [], [], []
    with torch.no_grad():
        for sample in loader2:
            obs = sample["observation"].to(model.device)
            first_all.append(encoder_indices(model.encoder1, obs[:, 0].unsqueeze(1), model_name).cpu())
            second_all.append(encoder_indices(model.encoder1, obs[:, 1].unsqueeze(1), model_name).cpu())
            relation_all.append(encoder_indices(model.encoder2, obs, model_name).cpu())
    first_idx = torch.cat(first_all).numpy().astype(int)
    second_idx = torch.cat(second_all).numpy().astype(int)
    relation_idx = torch.cat(relation_all).numpy().astype(int)

    object_capacity = code_capacity(model.encoder1, model_name, int(opts["code1_dim"]))
    relation_capacity = code_capacity(model.encoder2, model_name, int(opts["code2_dim"]))
    object_usage = usage_statistics(object_indices, object_capacity, args.used_fraction)
    relation_usage = usage_statistics(relation_idx, relation_capacity, args.used_fraction)
    semantic_metrics = object_semantic_metrics(object_indices, physical_types, affordance_groups)

    labels_path = os.path.join(run_dir, "label.pt")
    tuple_metrics = None
    pair_purity = None
    tuple_ids = None
    if os.path.exists(labels_path):
        effect_labels = torch.load(labels_path, map_location="cpu").detach().cpu().numpy().astype(int)
        if len(effect_labels) != len(relation_idx):
            raise ValueError("label.pt length does not match PairedObjectData")
        tuple_metrics = tuple_effect_metrics(first_idx, second_idx, relation_idx, effect_labels)
        tuple_keys = list(zip(first_idx.tolist(), second_idx.tolist(), relation_idx.tolist()))
        key_to_id = {key: i for i, key in enumerate(sorted(set(tuple_keys)))}
        tuple_ids = np.asarray([key_to_id[key] for key in tuple_keys], dtype=int)
        pair_purity = plot_pair_tuple_purity(
            tuple_ids,
            effect_labels,
            os.path.join(poster_dir, "tuple_effect_purity_by_pair"),
        )

    # Canonical central-view object codes: 5 shapes x 10 sizes.
    full_batch = next(iter(torch.utils.data.DataLoader(dataset1, batch_size=len(dataset1), shuffle=False)))["observation"]
    canonical_images = full_batch.reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])[:, :, 0, 2, 2]
    canonical_flat = canonical_images.reshape(-1, 1, opts["size"], opts["size"]).to(model.device)
    with torch.no_grad():
        canonical_indices = encoder_indices(model.encoder1, canonical_flat, model_name).cpu().numpy().astype(int)
    relation_grids = relation_grid_from_canonical(model, canonical_images, model_name)

    object_names = opts.get("poster_object_names", DEFAULT_OBJECT_NAMES)
    if len(object_names) != 5:
        object_names = DEFAULT_OBJECT_NAMES

    rows = read_training_rows(run_dir)
    plot_loss_curves(rows, poster_dir)
    plot_usage(object_usage["counts"], "Object-symbol usage", os.path.join(poster_dir, "object_code_usage"))
    plot_usage(relation_usage["counts"], "Relation-symbol usage", os.path.join(poster_dir, "relation_code_usage"))
    plot_object_assignment_frequency(
        physical_types,
        object_indices,
        object_names,
        object_capacity,
        os.path.join(poster_dir, "object_assignment_frequency"),
    )
    plot_object_symbol_map(
        canonical_indices,
        object_names,
        object_capacity,
        os.path.join(poster_dir, "object_symbol_map"),
    )
    plot_relation_symbol_grid(
        relation_grids,
        object_names,
        relation_capacity,
        os.path.join(poster_dir, "relation_symbol_grid"),
    )
    if model_name == "dynamic":
        plot_growth(rows, 1, os.path.join(poster_dir, "codebook_growth_level1"))
        plot_growth(rows, 2, os.path.join(poster_dir, "codebook_growth_level2"))

    active1 = code_capacity(model.encoder1, model_name, int(opts["code1_dim"]))
    active2 = code_capacity(model.encoder2, model_name, int(opts["code2_dim"]))
    def load_growth_events(level):
        path = os.path.join(run_dir, "growth_events_level%d.json" % level)
        if os.path.exists(path):
            with open(path, "r") as handle:
                return json.load(handle)
        return []

    metrics = {
        "model": model_name,
        "seed": int(opts.get("seed", -1)),
        "run_dir": run_dir,
        "level1": level1_eval,
        "level2": level2_eval,
        "object_codes": object_usage,
        "relation_codes": relation_usage,
        "object_semantics": semantic_metrics,
        "active_object_codes": active1,
        "active_relation_codes": active2,
        "tuple_effect": tuple_metrics,
        "growth_events_level1": load_growth_events(1),
        "growth_events_level2": load_growth_events(2),
    }
    with open(os.path.join(run_dir, "poster_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    np.savez_compressed(
        os.path.join(run_dir, "poster_assignments.npz"),
        object_indices=object_indices,
        physical_object_types=physical_types,
        affordance_groups=affordance_groups,
        canonical_object_indices=canonical_indices,
        first_object_indices=first_idx,
        second_object_indices=second_idx,
        relation_indices=relation_idx,
        relation_grids=relation_grids,
        tuple_ids=np.asarray([]) if tuple_ids is None else tuple_ids,
        pair_tuple_purity=np.asarray([]) if pair_purity is None else pair_purity,
    )

    print(json.dumps({
        "model": model_name,
        "seed": metrics["seed"],
        "level1_weighted_mse": level1_eval["weighted_mse"],
        "level2_weighted_mse": level2_eval["weighted_mse"],
        "object_used_codes": object_usage["used_codes_threshold"],
        "relation_used_codes": relation_usage["used_codes_threshold"],
        "object_affordance_purity": semantic_metrics["affordance_purity"],
        "tuple_effect_purity": None if tuple_metrics is None else tuple_metrics["weighted_tuple_purity"],
        "poster_dir": poster_dir,
    }, indent=2))


if __name__ == "__main__":
    main()