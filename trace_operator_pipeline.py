#!/usr/bin/env python3
"""Trace a DeepSym/Railroad operator all the way back to training samples.

What it prints / saves:
  railroad operator -> source_leaf -> samples in leaf -> physical slot types/sizes
  -> label.pt labels -> raw delta_pix_3 effects -> object-pair image grid -> CSV.

Usage examples:
  python trace_operator_pipeline.py -opts opts.yaml --operator stack29
  python trace_operator_pipeline.py -opts opts.yaml --below objtype1 --above objtype2 --relation relation1
  python trace_operator_pipeline.py -opts opts.yaml --leaf 57

Notes:
  In the paired dataset, an index is decoded as:
    slot0_type = idx // 500
    slot0_size = (idx // 50) % 10
    slot1_type = (idx // 10) % 5
    slot1_size = idx % 10

  This script deliberately says slot0/slot1, not below/above.  Use the printed
  role-convention section to decide whether slot0 is actually below or above in
  your current pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont


def load_torch(path: str):
    return torch.load(path, map_location="cpu")


def objtype_to_int(name: str) -> int:
    if not name.startswith("objtype"):
        raise ValueError(f"Expected objtypeN, got {name!r}")
    return int(name.replace("objtype", ""))


def relation_to_int(name: str) -> int:
    if not name.startswith("relation"):
        raise ValueError(f"Expected relationN, got {name!r}")
    return int(name.replace("relation", ""))


def decode_dataset_index(idx: int) -> Dict[str, int]:
    return {
        "slot0_type": idx // 500,
        "slot0_size": (idx // 50) % 10,
        "slot1_type": (idx // 10) % 5,
        "slot1_size": idx % 10,
    }


def load_tree(save_dir: str, explicit_tree: Optional[str]) -> Tuple[Any, str]:
    candidates = []
    if explicit_tree:
        candidates.append(explicit_tree)
    candidates += ["tree_vq_onehot.pkl", "tree.pkl"]
    for name in candidates:
        path = name if os.path.isabs(name) else os.path.join(save_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), path
    raise FileNotFoundError(f"Could not find tree file in {save_dir}; tried {candidates}")


def load_operators(save_dir: str) -> Dict[str, Any]:
    path = os.path.join(save_dir, "railroad_operators.json")
    with open(path, "r") as f:
        return json.load(f)


def find_matching_operators(
    ops: Sequence[Dict[str, Any]],
    operator_name: Optional[str],
    below: Optional[str],
    above: Optional[str],
    relation: Optional[str],
    leaf: Optional[int],
) -> List[Dict[str, Any]]:
    matches = []
    for op in ops:
        if operator_name and op.get("name") != operator_name:
            continue
        if below and op.get("below_type", op.get("obj1_type")) != below:
            continue
        if above and op.get("above_type", op.get("obj2_type")) != above:
            continue
        if relation and op.get("relation") != relation:
            continue
        if leaf is not None and int(op.get("source_leaf", -999999)) != int(leaf):
            continue
        matches.append(op)
    return matches


def leaf_path(tree: Any, leaf_id: int) -> List[Tuple[int, str, float]]:
    """Return split constraints to reach a leaf."""
    t = tree.tree_
    out: List[Tuple[int, str, float]] = []

    def rec(node: int, path: List[Tuple[int, str, float]]) -> bool:
        if node == leaf_id:
            out.extend(path)
            return True
        left = int(t.children_left[node])
        right = int(t.children_right[node])
        if left == -1 and right == -1:
            return False
        feat = int(t.feature[node])
        thr = float(t.threshold[node])
        if rec(left, path + [(feat, "<=", thr)]):
            return True
        if rec(right, path + [(feat, ">", thr)]):
            return True
        return False

    rec(0, [])
    return out


def detect_onehot_slices(save_dir: str, category: np.ndarray) -> Optional[Dict[str, Tuple[int, int]]]:
    meta_path = os.path.join(save_dir, "category_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        fs = meta.get("feature_slices", {})
        # Support either current names or neutral names.
        s0 = fs.get("slot0_object") or fs.get("first_object") or fs.get("below_object")
        s1 = fs.get("slot1_object") or fs.get("second_object") or fs.get("above_object")
        sr = fs.get("relation")
        if s0 and s1 and sr:
            return {
                "slot0": (int(s0[0]), int(s0[1])),
                "slot1": (int(s1[0]), int(s1[1])),
                "relation": (int(sr[0]), int(sr[1])),
            }
    # Fallback for 4 object codes + 2 relation codes: 4+4+2 = 10 features.
    if category.shape[1] == 10:
        return {"slot0": (0, 4), "slot1": (4, 8), "relation": (8, 10)}
    return None


def decode_category_row(row: np.ndarray, slices: Optional[Dict[str, Tuple[int, int]]]) -> Dict[str, Optional[int]]:
    if not slices:
        return {"slot0_symbol": None, "slot1_symbol": None, "relation_symbol": None}
    ans = {}
    for key, (a, b) in slices.items():
        sub = row[a:b]
        ans[key + "_symbol"] = int(np.argmax(sub))
    return ans


def label_distribution(labels: np.ndarray, effect_names: Sequence[str]) -> str:
    if len(labels) == 0:
        return "  <empty>"
    c = Counter(labels.tolist())
    lines = []
    for lab, n in sorted(c.items()):
        lines.append(f"  {effect_names[int(lab)]:10s} count={n:4d} p={n/len(labels):.6f}")
    return "\n".join(lines)


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu().float()
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-8:
        return torch.zeros_like(x)
    return (x - mn) / (mx - mn)


def make_leaf_image_grid(
    idxs: Sequence[int],
    labels: np.ndarray,
    effect_names: Sequence[str],
    save_dir: str,
    leaf_id: int,
    max_images: int,
) -> Optional[str]:
    obs_path = "data/img/obs_prev_z.pt"
    if not os.path.exists(obs_path):
        print(f"WARNING: {obs_path} not found; skipping image grid")
        return None

    obs = load_torch(obs_path)
    obs = obs.reshape(5, 10, 3, 4, 4, 42, 42)
    obs = obs[:, :, 0]  # same subset used by PairedObjectData

    chosen = list(idxs[:max_images])
    if not chosen:
        return None

    tile_w = 42 * 2
    tile_h = 42
    text_h = 34
    pad = 8
    ncol = 5
    nrow = int(np.ceil(len(chosen) / ncol))
    canvas = Image.new("RGB", (ncol * (tile_w + pad) + pad, nrow * (tile_h + text_h + pad) + pad), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for k, idx in enumerate(chosen):
        d = decode_dataset_index(int(idx))
        img0 = normalize_img(obs[d["slot0_type"], d["slot0_size"], 2, 2])
        img1 = normalize_img(obs[d["slot1_type"], d["slot1_size"], 2, 2])
        pair = torch.cat([img0, img1], dim=1)
        arr = (pair.numpy() * 255).astype(np.uint8)
        tile = Image.fromarray(arr, mode="L").convert("RGB")

        col = k % ncol
        row = k // ncol
        x0 = pad + col * (tile_w + pad)
        y0 = pad + row * (tile_h + text_h + pad)
        canvas.paste(tile, (x0, y0))
        txt = f"{idx} {effect_names[int(labels[idx])]}\n"
        txt += f"s0 {d['slot0_type']}/{d['slot0_size']} s1 {d['slot1_type']}/{d['slot1_size']}"
        draw.text((x0, y0 + tile_h + 2), txt, fill="black", font=font)

    out = os.path.join(save_dir, f"leaf_{leaf_id}_pairs_slot0_left_slot1_right.png")
    canvas.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Trace a Railroad operator to its training samples and raw effects.")
    ap.add_argument("-opts", required=True)
    ap.add_argument("--operator", type=str, default=None, help="operator name, e.g. stack29")
    ap.add_argument("--below", type=str, default=None, help="below type, e.g. objtype1")
    ap.add_argument("--above", type=str, default=None, help="above type, e.g. objtype2")
    ap.add_argument("--relation", type=str, default=None, help="relation name, e.g. relation1")
    ap.add_argument("--leaf", type=int, default=None, help="source leaf id")
    ap.add_argument("--tree", type=str, default=None, help="optional explicit tree pickle path/name")
    ap.add_argument("--max-images", type=int, default=40)
    ap.add_argument("--max-print", type=int, default=120)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    category_t = load_torch(os.path.join(save_dir, "category.pt"))
    labels_t = load_torch(os.path.join(save_dir, "label.pt"))
    category = category_t.detach().cpu().numpy() if isinstance(category_t, torch.Tensor) else np.asarray(category_t)
    labels = labels_t.detach().cpu().numpy().reshape(-1).astype(int) if isinstance(labels_t, torch.Tensor) else np.asarray(labels_t).reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"))
    effect_names = [str(x) for x in effect_names]
    tree, tree_path = load_tree(save_dir, args.tree)
    leaves = tree.apply(category)

    raw_effect_path = "data/img/delta_pix_3.pt"
    raw_effect = load_torch(raw_effect_path) if os.path.exists(raw_effect_path) else None

    ops_json = load_operators(save_dir)
    ops = ops_json.get("stack_operators", [])
    matches = find_matching_operators(
        ops,
        operator_name=args.operator,
        below=args.below,
        above=args.above,
        relation=args.relation,
        leaf=args.leaf,
    )

    target_leaf = args.leaf
    print("=" * 90)
    print("BASIC INFO")
    print("=" * 90)
    print(f"save_dir: {save_dir}")
    print(f"tree: {tree_path}")
    print(f"category shape: {category.shape}")
    print(f"num samples: {len(labels)}")
    print(f"effect names: {effect_names}")
    print("\nOverall label distribution:")
    print(label_distribution(labels, effect_names))

    if matches:
        print("\n" + "=" * 90)
        print("MATCHING RAILROAD OPERATORS")
        print("=" * 90)
        for op in matches:
            print(json.dumps(op, indent=2))
        if target_leaf is None:
            leaves_from_ops = sorted({int(op.get("source_leaf")) for op in matches if "source_leaf" in op})
            if len(leaves_from_ops) == 1:
                target_leaf = leaves_from_ops[0]
            elif len(leaves_from_ops) > 1:
                print(f"\nMultiple leaves matched: {leaves_from_ops}. Re-run with --leaf N for one leaf.")
    elif args.operator or args.below or args.above or args.relation:
        print("\nNo matching railroad operator found for the requested filter.")

    if target_leaf is None:
        raise SystemExit("No target leaf selected. Use --operator, --below/--above/--relation, or --leaf.")

    idxs = np.where(leaves == int(target_leaf))[0]
    print("\n" + "=" * 90)
    print(f"LEAF {target_leaf} SUMMARY")
    print("=" * 90)
    print(f"num samples in leaf: {len(idxs)}")
    print("\nLeaf label distribution:")
    print(label_distribution(labels[idxs], effect_names))

    path = leaf_path(tree, int(target_leaf))
    print("\nDecision path constraints:")
    if path:
        for feat, op, thr in path:
            print(f"  feature[{feat}] {op} {thr:.6f}")
    else:
        print("  <could not reconstruct path or root is target leaf>")

    slices = detect_onehot_slices(save_dir, category)
    if slices:
        print("\nDetected one-hot/category slices:")
        print(f"  slot0:    {slices['slot0']}")
        print(f"  slot1:    {slices['slot1']}")
        print(f"  relation: {slices['relation']}")
    else:
        print("\nCould not detect one-hot slices; symbolic indices will be omitted.")

    print("\nRole-convention check:")
    print("  Interpretation A: slot0 = BELOW, slot1 = ABOVE")
    print("  Interpretation B: slot0 = ABOVE, slot1 = BELOW")
    print("  Use the rendered images + raw effects to decide which convention matches the simulator.")

    print("\n" + "=" * 90)
    print("SAMPLES IN LEAF")
    print("=" * 90)
    header = [
        "idx", "label", "slot0_type", "slot0_size", "slot1_type", "slot1_size",
        "slot0_symbol", "slot1_symbol", "relation_symbol",
        "raw_dx0", "raw_dy0", "raw_dz0", "raw_dx1", "raw_dy1", "raw_dz1",
    ]

    rows: List[Dict[str, Any]] = []
    for idx in idxs:
        idx = int(idx)
        d = decode_dataset_index(idx)
        dec = decode_category_row(category[idx], slices)
        row: Dict[str, Any] = {
            "idx": idx,
            "label": effect_names[int(labels[idx])],
            **d,
            "slot0_symbol": dec.get("slot0_symbol"),
            "slot1_symbol": dec.get("slot1_symbol"),
            "relation_symbol": dec.get("relation_symbol"),
        }
        if raw_effect is not None:
            eff = raw_effect[idx].detach().cpu().numpy().astype(float).tolist()
            for name, val in zip(["raw_dx0", "raw_dy0", "raw_dz0", "raw_dx1", "raw_dy1", "raw_dz1"], eff):
                row[name] = val
        else:
            for name in ["raw_dx0", "raw_dy0", "raw_dz0", "raw_dx1", "raw_dy1", "raw_dz1"]:
                row[name] = None
        rows.append(row)

    for row in rows[: args.max_print]:
        print(
            f"idx={row['idx']:4d} label={row['label']:10s} | "
            f"slot0 phys={row['slot0_type']}/{row['slot0_size']} sym={row['slot0_symbol']} | "
            f"slot1 phys={row['slot1_type']}/{row['slot1_size']} sym={row['slot1_symbol']} | "
            f"rel_sym={row['relation_symbol']} | "
            f"raw=({row['raw_dx0']}, {row['raw_dy0']}, {row['raw_dz0']}, "
            f"{row['raw_dx1']}, {row['raw_dy1']}, {row['raw_dz1']})"
        )
    if len(rows) > args.max_print:
        print(f"... {len(rows) - args.max_print} more rows omitted from print")

    print("\nDistribution by physical slot pair:")
    pair_cnt: Dict[Tuple[int, int], Counter] = defaultdict(Counter)
    for row in rows:
        pair_cnt[(row["slot0_type"], row["slot1_type"])][row["label"]] += 1
    for pair, c in sorted(pair_cnt.items()):
        total = sum(c.values())
        print(f"  slot0_type={pair[0]} slot1_type={pair[1]} total={total}: {dict(c)}")

    if raw_effect is not None:
        print("\nMean raw effect by label in this leaf:")
        for lab in sorted(set(labels[idxs].tolist())):
            lab_idxs = idxs[labels[idxs] == lab]
            mean_eff = raw_effect[lab_idxs].float().mean(dim=0).tolist()
            print(f"  {effect_names[int(lab)]:10s} n={len(lab_idxs):3d} mean={mean_eff}")

    csv_out = os.path.join(save_dir, f"leaf_{target_leaf}_trace.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV trace: {csv_out}")

    img_out = make_leaf_image_grid(
        idxs=idxs,
        labels=labels,
        effect_names=effect_names,
        save_dir=save_dir,
        leaf_id=int(target_leaf),
        max_images=args.max_images,
    )
    if img_out:
        print(f"Saved rendered pair image grid: {img_out}")
        print("  In the grid, slot0 image is LEFT and slot1 image is RIGHT.")

    print("\n" + "=" * 90)
    print("NEXT CHECK")
    print("=" * 90)
    print("If this leaf contains both inserted and stacked for the same symbolic tuple,")
    print("then the exported probability is an average over those samples.")
    print("If that average is physically wrong for a specific scene, the symbolic relation")
    print("is not separating that scene finely enough, or the below/above convention is swapped.")


if __name__ == "__main__":
    main()