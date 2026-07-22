#!/usr/bin/env python3
"""
Extract poster-ready evidence from a DeepSym VQ rule-learning run.

Outputs
-------
1) Exact decision-tree paths and leaf outcome distributions for target symbolic
   tuples such as:
       below=objtype0, above=objtype2, relation0/1

2) Canonical object-pair examples that actually receive those same symbols in
   the exact 50 x 50 pair order used by save_cat_vq.py.

The script works with the files produced by:
    save_cat_vq.py
    learn_rules_vq.py

Typical use
-----------
python extract_poster_rule_evidence.py \
  --method-root save/poster_5seed/dynamic \
  --tuples 0,2,0 0,2,1 \
  --output-dir poster_assets/rule_evidence \
  --examples-per-tuple 3 \
  --prefer-below-types 4 \
  --prefer-above-types 1 2

Object-type order:
    0 Sphere
    1 Cube
    2 Vertical cylinder
    3 Horizontal cylinder
    4 Cup
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.tree import _tree


DEFAULT_OBJECT_NAMES = [
    "Sphere",
    "Cube",
    "Vertical cylinder",
    "Horizontal cylinder",
    "Cup",
]


def choose_best_run(method_root: Path) -> Path:
    candidates: List[Tuple[float, Path]] = []
    for metrics_path in sorted(method_root.glob("seed_*/poster_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        value = None
        level2 = metrics.get("level2")
        if isinstance(level2, dict):
            value = level2.get("weighted_mse")
        if value is None:
            value = metrics.get("level2_weighted_mse")
        if value is None:
            continue

        candidates.append((float(value), metrics_path.parent))

    if not candidates:
        raise FileNotFoundError(
            f"No usable seed_*/poster_metrics.json files found under {method_root}"
        )

    candidates.sort(key=lambda item: item[0])
    print("Level-2 weighted MSE ranking:")
    for mse, run in candidates:
        print(f"  {run.name}: {mse:.8f}")
    print(f"\nSelected best run: {candidates[0][1]} (MSE={candidates[0][0]:.8f})")
    return candidates[0][1]


def parse_tuple(text: str) -> Tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Tuple must be below,above,relation, received {text!r}"
        )
    try:
        values = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return values  # type: ignore[return-value]


def load_meta(run_dir: Path) -> Dict:
    path = run_dir / "category_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run save_cat_vq.py for the selected run first."
        )
    with path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("encoding") != "vq_onehot":
        raise ValueError(
            "This script expects category_meta.json with encoding='vq_onehot'."
        )
    return meta


def number_from_meta(meta: Dict, *keys: str) -> int:
    for key in keys:
        if key in meta:
            return int(meta[key])
    raise KeyError(f"None of the metadata keys exist: {keys}")


def one_hot_tuple(
    below: int,
    above: int,
    relation: int,
    n_obj: int,
    n_rel: int,
) -> np.ndarray:
    row = np.zeros(2 * n_obj + n_rel, dtype=np.float32)
    row[below] = 1.0
    row[n_obj + above] = 1.0
    row[2 * n_obj + relation] = 1.0
    return row


def feature_names(n_obj: int, n_rel: int) -> List[str]:
    names = [f"below_is_objtype{i}" for i in range(n_obj)]
    names += [f"above_is_objtype{i}" for i in range(n_obj)]
    names += [f"relation_is_relation{i}" for i in range(n_rel)]
    return names


def leaf_path(tree, leaf_id: int) -> List[Tuple[int, str, float]]:
    """Return the exact root-to-leaf split path."""
    t = tree.tree_
    found: List[Tuple[int, str, float]] = []

    def recurse(node: int, path: List[Tuple[int, str, float]]) -> bool:
        if node == leaf_id:
            found.extend(path)
            return True

        left = int(t.children_left[node])
        right = int(t.children_right[node])
        if left == _tree.TREE_LEAF and right == _tree.TREE_LEAF:
            return False

        feature = int(t.feature[node])
        threshold = float(t.threshold[node])
        if recurse(left, path + [(feature, "<=", threshold)]):
            return True
        if recurse(right, path + [(feature, ">", threshold)]):
            return True
        return False

    if not recurse(0, []):
        raise ValueError(f"Leaf {leaf_id} is not present in the tree.")
    return found


def leaf_distribution(tree, leaf_id: int, effect_names: Sequence[str]) -> List[Dict]:
    values = np.asarray(tree.tree_.value[leaf_id][0], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return []

    result = []
    for class_position, count in enumerate(values):
        if count <= 0:
            continue
        label_index = int(tree.classes_[class_position])
        name = (
            str(effect_names[label_index])
            if 0 <= label_index < len(effect_names)
            else f"class_{label_index}"
        )
        result.append(
            {
                "effect": name,
                "count": int(round(float(count))),
                "probability": float(count / total),
            }
        )
    result.sort(key=lambda item: item["probability"], reverse=True)
    return result


def decode_category_indices(
    category: np.ndarray,
    n_obj: int,
    n_rel: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = 2 * n_obj + n_rel
    if category.ndim != 2 or category.shape[1] != expected:
        raise ValueError(
            f"category.pt shape {category.shape} does not match "
            f"2*n_obj+n_rel={expected}"
        )

    below = np.argmax(category[:, :n_obj], axis=1).astype(int)
    above = np.argmax(category[:, n_obj : 2 * n_obj], axis=1).astype(int)
    relation = np.argmax(
        category[:, 2 * n_obj : 2 * n_obj + n_rel], axis=1
    ).astype(int)
    return below, above, relation


def extract_action_blocks(domain_text: str) -> List[Tuple[str, str]]:
    starts = list(re.finditer(r"\(:action\s+([^\s()]+)", domain_text))
    blocks = []
    for index, match in enumerate(starts):
        start = match.start()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(domain_text)
        blocks.append((match.group(1), domain_text[start:end]))
    return blocks


def matching_domain_actions(
    domain_path: Path,
    below: int,
    above: int,
    relation: int,
) -> List[str]:
    if not domain_path.exists():
        return []

    text = domain_path.read_text(encoding="utf-8")
    required = [
        f"(objtype{below} ?below)",
        f"(objtype{above} ?above)",
        f"(relation{relation} ?below ?above)",
    ]
    matches = []
    for name, block in extract_action_blocks(text):
        if name.startswith("stack") and all(token in block for token in required):
            matches.append(name)
    return matches


def load_canonical_images(path: Path) -> np.ndarray:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, torch.Tensor):
        raw = torch.as_tensor(raw)

    expected = 5 * 10 * 3 * 4 * 4 * 42 * 42
    if raw.numel() != expected:
        raise ValueError(
            f"{path} has {raw.numel()} values; expected {expected} for "
            "shape (5,10,3,4,4,42,42)."
        )

    # Exact canonical crop used by the original-order VQ exporter.
    images = raw.reshape(5, 10, 3, 4, 4, 42, 42)
    images = images[:, :, 0, 2, 2].float().numpy()
    return images


def pair_metadata(pair_index: int) -> Dict[str, int]:
    below_canonical = pair_index // 50
    above_canonical = pair_index % 50
    return {
        "pair_index": pair_index,
        "below_canonical": below_canonical,
        "above_canonical": above_canonical,
        "below_type": below_canonical // 10,
        "below_size": below_canonical % 10 + 1,
        "above_type": above_canonical // 10,
        "above_size": above_canonical % 10 + 1,
    }


def candidate_score(
    item: Dict[str, int],
    prefer_below: Sequence[int],
    prefer_above: Sequence[int],
) -> Tuple[float, float, int]:
    type_penalty = 0.0
    if prefer_below and item["below_type"] not in prefer_below:
        type_penalty += 100.0
    if prefer_above and item["above_type"] not in prefer_above:
        type_penalty += 100.0

    # Prefer medium-sized examples once the requested physical types match.
    centre_penalty = abs(item["below_size"] - 5.5) + abs(item["above_size"] - 5.5)
    return type_penalty, centre_penalty, item["pair_index"]


def save_candidates_csv(
    path: Path,
    rows: Sequence[Dict],
) -> None:
    fields = [
        "tuple_below_symbol",
        "tuple_above_symbol",
        "tuple_relation_symbol",
        "pair_index",
        "below_type",
        "below_type_name",
        "below_size",
        "above_type",
        "above_type_name",
        "above_size",
        "effect_label",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compact_distribution(distribution: Sequence[Dict]) -> str:
    return ", ".join(
        f"{item['effect']} {100.0 * item['probability']:.1f}%"
        for item in distribution
    )


def save_pair_montage(
    output_stem: Path,
    tuple_results: Sequence[Dict],
    canonical_images: np.ndarray,
    object_names: Sequence[str],
    examples_per_tuple: int,
) -> None:
    rows = len(tuple_results)
    columns = max(1, examples_per_tuple)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 4.0 * rows),
        squeeze=False,
        facecolor="white",
    )

    global_min = float(np.min(canonical_images))
    global_max = float(np.max(canonical_images))

    for row_idx, result in enumerate(tuple_results):
        selected = result["selected_examples"]
        for col_idx in range(columns):
            ax = axes[row_idx, col_idx]
            ax.axis("off")

            if col_idx >= len(selected):
                continue

            item = selected[col_idx]
            below_img = canonical_images[
                item["below_type"], item["below_size"] - 1
            ]
            above_img = canonical_images[
                item["above_type"], item["above_size"] - 1
            ]

            spacer = np.full((below_img.shape[0], 5), global_max, dtype=float)
            combined = np.concatenate([below_img, spacer, above_img], axis=1)

            ax.imshow(
                combined,
                cmap="plasma",
                vmin=global_min,
                vmax=global_max,
                interpolation="nearest",
            )
            ax.axvline(below_img.shape[1] + 2, color="white", linewidth=2)
            ax.text(
                0.24,
                0.97,
                "BELOW",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", alpha=0.50, pad=2, edgecolor="none"),
            )
            ax.text(
                0.76,
                0.97,
                "ABOVE",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox=dict(facecolor="black", alpha=0.50, pad=2, edgecolor="none"),
            )
            ax.set_title(
                f"{object_names[item['below_type']]} s{item['below_size']}"
                f"  →  {object_names[item['above_type']]} s{item['above_size']}",
                fontsize=11,
                fontweight="bold",
                pad=7,
            )

        tuple_label = (
            f"objtype{result['below_symbol']} below, "
            f"objtype{result['above_symbol']} above, "
            f"relation{result['relation_symbol']}"
        )
        action_text = (
            ", ".join(result["domain_actions"])
            if result["domain_actions"]
            else "action name not resolved"
        )
        fig.text(
            0.01,
            1.0 - (row_idx + 0.52) / rows,
            f"{tuple_label}\n{action_text}\n"
            f"{compact_distribution(result['distribution'])}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )

    fig.suptitle(
        "Canonical object-pair examples for selected learned rules",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.08, 0.02, 1.0, 0.96))
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def wrap_path_line(name: str, op: str, threshold: float) -> str:
    if abs(threshold - 0.5) < 1e-6:
        if op == ">":
            return f"{name} = YES"
        return f"{name} = NO"
    return f"{name} {op} {threshold:.4f}"


def save_path_figure(output_stem: Path, tuple_results: Sequence[Dict]) -> None:
    rows = len(tuple_results)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(13.0, 4.1 * rows),
        squeeze=False,
        facecolor="white",
    )

    for row_idx, result in enumerate(tuple_results):
        ax = axes[row_idx, 0]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        path_lines = [
            wrap_path_line(item["feature_name"], item["operator"], item["threshold"])
            for item in result["path"]
        ]

        # Keep the exact path readable without drawing a huge full tree.
        path_text = "\n".join(f"{i+1}. {line}" for i, line in enumerate(path_lines))
        rule_text = (
            f"WHEN\n"
            f"below = object symbol {result['below_symbol']}\n"
            f"above = object symbol {result['above_symbol']}\n"
            f"relation = symbol {result['relation_symbol']}\n"
            f"action = stack"
        )
        outcome_text = "PREDICT\n" + "\n".join(
            f"{100.0 * item['probability']:.1f}%  {item['effect']}"
            for item in result["distribution"]
        )

        boxes = [
            (0.01, 0.12, 0.31, 0.76, "Exact decision path", path_text),
            (0.345, 0.12, 0.27, 0.76, "Readable rule", rule_text),
            (0.64, 0.12, 0.35, 0.76, "Leaf outcome distribution", outcome_text),
        ]

        for x, y, width, height, title, body in boxes:
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    width,
                    height,
                    fill=True,
                    facecolor="#F4F6F8",
                    edgecolor="#17365D",
                    linewidth=1.4,
                )
            )
            ax.text(
                x + 0.015,
                y + height - 0.06,
                title,
                fontsize=12,
                fontweight="bold",
                va="top",
                color="#17365D",
            )
            ax.text(
                x + 0.015,
                y + height - 0.14,
                body,
                fontsize=9.5,
                va="top",
                family="monospace" if title == "Exact decision path" else "sans-serif",
                linespacing=1.25,
            )

        ax.annotate(
            "",
            xy=(0.635, 0.5),
            xytext=(0.62, 0.5),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )
        ax.annotate(
            "",
            xy=(0.34, 0.5),
            xytext=(0.325, 0.5),
            arrowprops=dict(arrowstyle="->", linewidth=1.5),
        )

        actions = ", ".join(result["domain_actions"]) or "unresolved action name"
        ax.text(
            0.01,
            0.96,
            f"{actions}: objtype{result['below_symbol']} below, "
            f"objtype{result['above_symbol']} above, "
            f"relation{result['relation_symbol']}  |  leaf {result['leaf_id']}",
            fontsize=13,
            fontweight="bold",
            va="top",
        )

    fig.suptitle(
        "From a decision-tree path to a probabilistic planning rule",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.97))
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        "Extract exact decision paths and canonical pair examples for poster use."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument(
        "--method-root",
        type=Path,
        help="Directory such as save/poster_5seed/dynamic; best Level-2-MSE seed is selected.",
    )
    parser.add_argument(
        "--tuples",
        type=parse_tuple,
        nargs="+",
        required=True,
        metavar="BELOW,ABOVE,REL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("poster_assets/rule_evidence"),
    )
    parser.add_argument(
        "--canonical-data",
        type=Path,
        default=Path("data/img/obs_prev_z.pt"),
    )
    parser.add_argument("--examples-per-tuple", type=int, default=3)
    parser.add_argument(
        "--prefer-below-types",
        type=int,
        nargs="*",
        default=[],
        help="Physical type indices preferred for examples; Cup is 4.",
    )
    parser.add_argument(
        "--prefer-above-types",
        type=int,
        nargs="*",
        default=[],
        help="Physical type indices preferred for examples; Cube=1, V-cylinder=2.",
    )
    args = parser.parse_args()

    run_dir = (
        choose_best_run(args.method_root.resolve())
        if args.method_root is not None
        else args.run_dir.resolve()
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [
        run_dir / "opts.yaml",
        run_dir / "category.pt",
        run_dir / "category_meta.json",
        run_dir / "label.pt",
        run_dir / "effect_names.npy",
        run_dir / "tree_vq_onehot.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        lines = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            "The selected run is missing rule-extraction files:\n"
            f"{lines}\n\n"
            "Run save_cat_vq.py and learn_rules_vq.py for this run first."
        )

    with (run_dir / "opts.yaml").open("r", encoding="utf-8") as handle:
        opts = yaml.safe_load(handle)
    object_names = opts.get("poster_object_names", DEFAULT_OBJECT_NAMES)
    if len(object_names) != 5:
        object_names = DEFAULT_OBJECT_NAMES

    meta = load_meta(run_dir)
    n_obj = number_from_meta(meta, "num_object_codes", "num_obj_codes")
    n_rel = number_from_meta(meta, "num_relation_codes", "num_rel_codes")

    category_tensor = torch.load(run_dir / "category.pt", map_location="cpu")
    category = (
        category_tensor.detach().cpu().numpy()
        if isinstance(category_tensor, torch.Tensor)
        else np.asarray(category_tensor)
    )
    labels_tensor = torch.load(run_dir / "label.pt", map_location="cpu")
    labels = (
        labels_tensor.detach().cpu().numpy().reshape(-1).astype(int)
        if isinstance(labels_tensor, torch.Tensor)
        else np.asarray(labels_tensor).reshape(-1).astype(int)
    )
    effect_names = [
        str(item)
        for item in np.load(run_dir / "effect_names.npy", allow_pickle=True)
    ]

    with (run_dir / "tree_vq_onehot.pkl").open("rb") as handle:
        tree = pickle.load(handle)

    below_idx, above_idx, relation_idx = decode_category_indices(
        category, n_obj, n_rel
    )
    leaves = tree.apply(category)
    names = feature_names(n_obj, n_rel)
    canonical_images = load_canonical_images(args.canonical_data.resolve())

    domain_path = run_dir / "domain.pddl"
    tuple_results = []
    all_candidate_rows = []

    for below_symbol, above_symbol, relation_symbol in args.tuples:
        if not (0 <= below_symbol < n_obj):
            raise ValueError(f"below symbol {below_symbol} outside 0..{n_obj-1}")
        if not (0 <= above_symbol < n_obj):
            raise ValueError(f"above symbol {above_symbol} outside 0..{n_obj-1}")
        if not (0 <= relation_symbol < n_rel):
            raise ValueError(f"relation symbol {relation_symbol} outside 0..{n_rel-1}")

        tuple_row = one_hot_tuple(
            below_symbol, above_symbol, relation_symbol, n_obj, n_rel
        )
        leaf_id = int(tree.apply(tuple_row.reshape(1, -1))[0])
        exact_path = leaf_path(tree, leaf_id)
        distribution = leaf_distribution(tree, leaf_id, effect_names)
        actions = matching_domain_actions(
            domain_path, below_symbol, above_symbol, relation_symbol
        )

        mask = (
            (below_idx == below_symbol)
            & (above_idx == above_symbol)
            & (relation_idx == relation_symbol)
        )
        pair_indices = np.flatnonzero(mask).tolist()
        candidates = []
        for pair_index in pair_indices:
            item = pair_metadata(int(pair_index))
            item["effect_label"] = (
                effect_names[int(labels[pair_index])]
                if pair_index < len(labels)
                else ""
            )
            candidates.append(item)

            all_candidate_rows.append(
                {
                    "tuple_below_symbol": below_symbol,
                    "tuple_above_symbol": above_symbol,
                    "tuple_relation_symbol": relation_symbol,
                    "pair_index": item["pair_index"],
                    "below_type": item["below_type"],
                    "below_type_name": object_names[item["below_type"]],
                    "below_size": item["below_size"],
                    "above_type": item["above_type"],
                    "above_type_name": object_names[item["above_type"]],
                    "above_size": item["above_size"],
                    "effect_label": item["effect_label"],
                }
            )

        candidates.sort(
            key=lambda item: candidate_score(
                item,
                args.prefer_below_types,
                args.prefer_above_types,
            )
        )
        selected = candidates[: max(1, args.examples_per_tuple)]

        path_records = [
            {
                "feature_index": feature,
                "feature_name": names[feature],
                "operator": operator,
                "threshold": threshold,
            }
            for feature, operator, threshold in exact_path
        ]

        tuple_results.append(
            {
                "run_dir": str(run_dir),
                "below_symbol": below_symbol,
                "above_symbol": above_symbol,
                "relation_symbol": relation_symbol,
                "leaf_id": leaf_id,
                "domain_actions": actions,
                "path": path_records,
                "distribution": distribution,
                "num_matching_canonical_pairs": len(candidates),
                "selected_examples": selected,
            }
        )

    save_candidates_csv(
        output_dir / "symbolic_pair_candidates.csv",
        all_candidate_rows,
    )

    summary = {
        "run_dir": str(run_dir),
        "model": opts.get("poster_model"),
        "seed": opts.get("seed"),
        "n_object_codes": n_obj,
        "n_relation_codes": n_rel,
        "role_convention": "first/below, second/above",
        "tuple_results": tuple_results,
    }
    with (output_dir / "rule_evidence.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    text_lines = [
        f"Run: {run_dir}",
        f"Model: {opts.get('poster_model')}",
        f"Seed: {opts.get('seed')}",
        "",
    ]
    for result in tuple_results:
        actions = ", ".join(result["domain_actions"]) or "<action name unresolved>"
        text_lines += [
            "=" * 78,
            f"{actions}",
            (
                f"below=objtype{result['below_symbol']} "
                f"above=objtype{result['above_symbol']} "
                f"relation=relation{result['relation_symbol']}"
            ),
            f"leaf={result['leaf_id']}",
            f"matching canonical pairs={result['num_matching_canonical_pairs']}",
            "",
            "Exact decision path:",
        ]
        for item in result["path"]:
            text_lines.append(
                f"  {item['feature_name']} {item['operator']} "
                f"{item['threshold']:.6f}"
            )
        text_lines += ["", "Leaf outcome distribution:"]
        for item in result["distribution"]:
            text_lines.append(
                f"  {item['effect']:12s} count={item['count']:4d} "
                f"p={item['probability']:.6f}"
            )
        text_lines += ["", "Selected canonical examples:"]
        for item in result["selected_examples"]:
            text_lines.append(
                f"  pair={item['pair_index']:4d}: "
                f"below={object_names[item['below_type']]} size {item['below_size']}, "
                f"above={object_names[item['above_type']]} size {item['above_size']}, "
                f"label={item['effect_label']}"
            )
        text_lines.append("")

    (output_dir / "rule_evidence.txt").write_text(
        "\n".join(text_lines), encoding="utf-8"
    )

    save_pair_montage(
        output_dir / "symbolic_pair_examples",
        tuple_results,
        canonical_images,
        object_names,
        max(1, args.examples_per_tuple),
    )
    save_path_figure(
        output_dir / "decision_path_to_rule",
        tuple_results,
    )

    print("\nCreated:")
    for path in [
        output_dir / "rule_evidence.txt",
        output_dir / "rule_evidence.json",
        output_dir / "symbolic_pair_candidates.csv",
        output_dir / "symbolic_pair_examples.pdf",
        output_dir / "symbolic_pair_examples.png",
        output_dir / "decision_path_to_rule.pdf",
        output_dir / "decision_path_to_rule.png",
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()