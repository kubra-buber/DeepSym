#!/usr/bin/env python3
"""
Rank learned DeepSym/VQ rule pairs for a poster example.

The script searches all learned object-symbol pairs and compares the two
relation-symbol branches. It prioritizes candidates that:

1. contain requested physical object combinations among their canonical pairs;
2. assign substantial probability to insertion/stacking-like outcomes;
3. show a clear difference between relation-symbol outcome distributions.

It does not modify the run.

Example
-------
python rank_poster_rule_candidates.py \
  --method-root save/poster_5seed/dynamic \
  --prefer-below-types 4 \
  --prefer-above-types 1 2 \
  --top 15 \
  --output poster_assets/rule_candidate_ranking.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


OBJECT_NAMES = [
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
        if isinstance(metrics.get("level2"), dict):
            value = metrics["level2"].get("weighted_mse")
        if value is None:
            value = metrics.get("level2_weighted_mse")
        if value is not None:
            candidates.append((float(value), metrics_path.parent))

    if not candidates:
        raise FileNotFoundError(
            f"No seed_*/poster_metrics.json with Level-2 MSE under {method_root}"
        )

    candidates.sort(key=lambda item: item[0])
    print("Level-2 weighted-MSE ranking:")
    for mse, run_dir in candidates:
        print(f"  {run_dir.name}: {mse:.8f}")
    print(f"\nSelected run: {candidates[0][1]}\n")
    return candidates[0][1]


def meta_integer(meta: Dict, *keys: str) -> int:
    for key in keys:
        if key in meta:
            return int(meta[key])
    raise KeyError(f"Missing metadata keys: {keys}")


def one_hot_tuple(
    below_symbol: int,
    above_symbol: int,
    relation_symbol: int,
    n_obj: int,
    n_rel: int,
) -> np.ndarray:
    row = np.zeros(2 * n_obj + n_rel, dtype=np.float32)
    row[below_symbol] = 1.0
    row[n_obj + above_symbol] = 1.0
    row[2 * n_obj + relation_symbol] = 1.0
    return row


def leaf_distribution(tree, row: np.ndarray, effect_names: Sequence[str]) -> Dict[str, float]:
    leaf_id = int(tree.apply(row.reshape(1, -1))[0])
    values = np.asarray(tree.tree_.value[leaf_id][0], dtype=float)
    total = float(values.sum())

    distribution: Dict[str, float] = {}
    if total <= 0:
        return distribution

    for class_position, count in enumerate(values):
        if count <= 0:
            continue
        class_index = int(tree.classes_[class_position])
        name = (
            str(effect_names[class_index])
            if 0 <= class_index < len(effect_names)
            else f"class_{class_index}"
        )
        distribution[name] = float(count / total)

    return distribution


def normalized_vector(
    distribution: Dict[str, float],
    keys: Sequence[str],
) -> np.ndarray:
    vector = np.array([distribution.get(key, 0.0) for key in keys], dtype=float)
    total = float(vector.sum())
    return vector / total if total > 0 else vector


def js_divergence(first: Dict[str, float], second: Dict[str, float]) -> float:
    keys = sorted(set(first) | set(second))
    p = normalized_vector(first, keys)
    q = normalized_vector(second, keys)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def target_mass(distribution: Dict[str, float]) -> float:
    """Probability assigned to insertion/stacking-like outcomes."""
    total = 0.0
    for name, probability in distribution.items():
        lower = name.lower()
        if "insert" in lower or "stack" in lower:
            total += probability
    return float(total)


def compact_distribution(distribution: Dict[str, float], limit: int = 4) -> str:
    ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    shown = ordered[:limit]
    return "; ".join(f"{name} {100.0 * probability:.1f}%" for name, probability in shown)


def decode_assignments(
    category: np.ndarray,
    n_obj: int,
    n_rel: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = 2 * n_obj + n_rel
    if category.ndim != 2 or category.shape[1] != expected:
        raise ValueError(
            f"category.pt shape {category.shape}; expected second dimension {expected}"
        )

    below = np.argmax(category[:, :n_obj], axis=1).astype(int)
    above = np.argmax(category[:, n_obj : 2 * n_obj], axis=1).astype(int)
    relation = np.argmax(category[:, 2 * n_obj :], axis=1).astype(int)
    return below, above, relation


def pair_physical_types(pair_index: int) -> Tuple[int, int]:
    """Original canonical pair order: first/below, second/above."""
    below_canonical = pair_index // 50
    above_canonical = pair_index % 50
    return below_canonical // 10, above_canonical // 10


def action_blocks(domain_text: str) -> List[Tuple[str, str]]:
    starts = list(re.finditer(r"\(:action\s+([^\s()]+)", domain_text))
    result = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(domain_text)
        result.append((match.group(1), domain_text[match.start():end]))
    return result


def find_action_name(
    blocks: Sequence[Tuple[str, str]],
    below: int,
    above: int,
    relation: int,
) -> str:
    required = [
        f"(objtype{below} ?below)",
        f"(objtype{above} ?above)",
        f"(relation{relation} ?below ?above)",
    ]
    for name, block in blocks:
        if name.startswith("stack") and all(token in block for token in required):
            return name
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank poster-worthy pairs of learned relational rules."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--method-root", type=Path)

    parser.add_argument(
        "--prefer-below-types",
        type=int,
        nargs="*",
        default=[4],
        help="Physical type indices preferred below. Default: Cup (4).",
    )
    parser.add_argument(
        "--prefer-above-types",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Physical type indices preferred above. Default: Cube/V-cylinder.",
    )
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poster_assets/rule_candidate_ranking.csv"),
    )
    args = parser.parse_args()

    run_dir = (
        choose_best_run(args.method_root.resolve())
        if args.method_root is not None
        else args.run_dir.resolve()
    )

    required = [
        run_dir / "category_meta.json",
        run_dir / "category.pt",
        run_dir / "effect_names.npy",
        run_dir / "tree_vq_onehot.pkl",
        run_dir / "domain.pddl",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files:\n" + "\n".join(f"  {path}" for path in missing)
        )

    with (run_dir / "category_meta.json").open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    n_obj = meta_integer(meta, "num_object_codes", "num_obj_codes")
    n_rel = meta_integer(meta, "num_relation_codes", "num_rel_codes")
    if n_rel < 2:
        raise ValueError("At least two relation symbols are needed for comparison.")

    raw_category = torch.load(run_dir / "category.pt", map_location="cpu")
    category = (
        raw_category.detach().cpu().numpy()
        if isinstance(raw_category, torch.Tensor)
        else np.asarray(raw_category)
    )
    below_assign, above_assign, relation_assign = decode_assignments(
        category, n_obj, n_rel
    )

    effect_names = [
        str(item)
        for item in np.load(run_dir / "effect_names.npy", allow_pickle=True)
    ]
    with (run_dir / "tree_vq_onehot.pkl").open("rb") as handle:
        tree = pickle.load(handle)

    domain_text = (run_dir / "domain.pddl").read_text(encoding="utf-8")
    blocks = action_blocks(domain_text)

    rows: List[Dict] = []

    # Compare relation 0 and relation 1 because this experiment has two
    # relation symbols. If there are more, the user can extend the pair list.
    relation_a, relation_b = 0, 1

    for below_symbol in range(n_obj):
        for above_symbol in range(n_obj):
            row_a = one_hot_tuple(
                below_symbol, above_symbol, relation_a, n_obj, n_rel
            )
            row_b = one_hot_tuple(
                below_symbol, above_symbol, relation_b, n_obj, n_rel
            )
            dist_a = leaf_distribution(tree, row_a, effect_names)
            dist_b = leaf_distribution(tree, row_b, effect_names)

            mask_pair = (
                (below_assign == below_symbol)
                & (above_assign == above_symbol)
            )
            pair_indices = np.flatnonzero(mask_pair)

            preferred_count = 0
            physical_counts: Dict[Tuple[int, int], int] = {}
            for pair_index in pair_indices:
                below_type, above_type = pair_physical_types(int(pair_index))
                key = (below_type, above_type)
                physical_counts[key] = physical_counts.get(key, 0) + 1
                if (
                    below_type in args.prefer_below_types
                    and above_type in args.prefer_above_types
                ):
                    preferred_count += 1

            physical_summary = "; ".join(
                f"{OBJECT_NAMES[b]}→{OBJECT_NAMES[a]}:{count}"
                for (b, a), count in sorted(
                    physical_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
            )

            js = js_divergence(dist_a, dist_b)
            target_a = target_mass(dist_a)
            target_b = target_mass(dist_b)
            max_target = max(target_a, target_b)
            target_difference = abs(target_a - target_b)

            # Ranking priority:
            #   preferred physical examples > useful stack/insert effects >
            #   relation-dependent contrast.
            has_preferred = 1 if preferred_count > 0 else 0
            score = (
                100.0 * has_preferred
                + 10.0 * max_target
                + 5.0 * target_difference
                + 3.0 * js
                + min(preferred_count, 100) / 1000.0
            )

            rows.append(
                {
                    "score": score,
                    "below_symbol": below_symbol,
                    "above_symbol": above_symbol,
                    "relation0_action": find_action_name(
                        blocks, below_symbol, above_symbol, relation_a
                    ),
                    "relation1_action": find_action_name(
                        blocks, below_symbol, above_symbol, relation_b
                    ),
                    "matching_pair_count": int(len(pair_indices)),
                    "preferred_physical_pair_count": preferred_count,
                    "physical_pair_summary": physical_summary,
                    "relation0_target_mass": target_a,
                    "relation1_target_mass": target_b,
                    "target_mass_difference": target_difference,
                    "js_divergence": js,
                    "relation0_distribution": compact_distribution(dist_a),
                    "relation1_distribution": compact_distribution(dist_b),
                }
            )

    rows.sort(key=lambda row: row["score"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Run: {run_dir}")
    print(f"Saved full ranking: {args.output}\n")
    print(f"Top {min(args.top, len(rows))} candidates:\n")

    for rank, row in enumerate(rows[: args.top], start=1):
        print(
            f"{rank:2d}. below=objtype{row['below_symbol']} "
            f"above=objtype{row['above_symbol']} | "
            f"{row['relation0_action']}/{row['relation1_action']}"
        )
        print(
            f"    preferred physical pairs={row['preferred_physical_pair_count']}, "
            f"JS={row['js_divergence']:.3f}, "
            f"stack/insert mass={row['relation0_target_mass']:.3f}/"
            f"{row['relation1_target_mass']:.3f}"
        )
        print(f"    physical: {row['physical_pair_summary']}")
        print(f"    relation0: {row['relation0_distribution']}")
        print(f"    relation1: {row['relation1_distribution']}")
        print(
            "    extract command tuple arguments: "
            f"--tuples {row['below_symbol']},{row['above_symbol']},0 "
            f"{row['below_symbol']},{row['above_symbol']},1\n"
        )


if __name__ == "__main__":
    main()