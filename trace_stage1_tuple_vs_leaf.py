import os
import json
import pickle
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml


def load_tree(save_dir, railroad):
    meta = railroad.get("metadata", {})
    tree_name = meta.get("tree_file")
    candidates = []
    if tree_name:
        candidates.append(os.path.join(save_dir, tree_name))
    candidates += [
        os.path.join(save_dir, "tree_vq_onehot.pkl"),
        os.path.join(save_dir, "tree.pkl"),
    ]

    for p in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return p, pickle.load(f)

    raise FileNotFoundError(f"No tree file found. Tried: {candidates}")


def label_dist(labels, effect_names, idxs):
    c = Counter(labels[idxs].tolist())
    total = len(idxs)
    lines = []
    for k, v in sorted(c.items()):
        lines.append(f"  {effect_names[k]:10s} count={v:4d} p={v/total:.6f}")
    return "\n".join(lines)


def decode_onehot_category(category):
    n = category.shape[1]
    # Your VQ one-hot setup: 4 below + 4 above + 2 relation = 10
    if n != 10:
        raise ValueError(f"Expected one-hot category shape [N,10], got {category.shape}")

    below = category[:, 0:4].argmax(axis=1)
    above = category[:, 4:8].argmax(axis=1)
    rel = category[:, 8:10].argmax(axis=1)
    return below, above, rel


def physical_indices(idx):
    # Original DeepSym ordered-pair index convention:
    # left/slot0: type = idx // 500, size = (idx // 50) % 10
    # right/slot1: type = (idx // 10) % 5, size = idx % 10
    slot0_type = idx // 500
    slot0_size = (idx // 50) % 10
    slot1_type = (idx // 10) % 5
    slot1_size = idx % 10
    return slot0_type, slot0_size, slot1_type, slot1_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--below", required=True, help="example: objtype1")
    ap.add_argument("--above", required=True, help="example: objtype2")
    ap.add_argument("--relation", required=True, help="example: relation1")
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    save = opts["save"]

    railroad_path = os.path.join(save, "railroad_operators.json")
    railroad = json.load(open(railroad_path))

    category = torch.load(os.path.join(save, "category.pt"), map_location="cpu").detach().cpu().numpy()
    labels = torch.load(os.path.join(save, "label.pt"), map_location="cpu").detach().cpu().numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))

    raw_effect_path = "data/img/delta_pix_3.pt"
    raw_effect = torch.load(raw_effect_path, map_location="cpu").detach().cpu().numpy()

    tree_path, tree = load_tree(save, railroad)
    leaves = tree.apply(category)

    target_ops = [
        op for op in railroad["stack_operators"]
        if op.get("below_type") == args.below
        and op.get("above_type") == args.above
        and op.get("relation") == args.relation
    ]

    print("=" * 90)
    print("TARGET")
    print("=" * 90)
    print(f"save: {save}")
    print(f"tree: {tree_path}")
    print(f"target tuple: below={args.below}, above={args.above}, relation={args.relation}")

    if not target_ops:
        print("NO MATCHING OPERATOR FOUND IN railroad_operators.json")
        return

    for op in target_ops:
        print("\nMatching railroad operator:")
        print(json.dumps(op, indent=2))

    op = target_ops[0]
    source_leaf = int(op["source_leaf"])

    below_idx = int(args.below.replace("objtype", ""))
    above_idx = int(args.above.replace("objtype", ""))
    rel_idx = int(args.relation.replace("relation", ""))

    cat_below, cat_above, cat_rel = decode_onehot_category(category)

    exact_tuple_idxs = np.where(
        (cat_below == below_idx) &
        (cat_above == above_idx) &
        (cat_rel == rel_idx)
    )[0]

    leaf_idxs = np.where(leaves == source_leaf)[0]

    exact_in_leaf = np.intersect1d(exact_tuple_idxs, leaf_idxs)

    print("\n" + "=" * 90)
    print("A) EXACT SYMBOLIC TUPLE DISTRIBUTION, BEFORE TREE GROUPING")
    print("=" * 90)
    print(f"exact tuple rows: {len(exact_tuple_idxs)}")
    if len(exact_tuple_idxs):
        print(label_dist(labels, effect_names, exact_tuple_idxs))

    print("\n" + "=" * 90)
    print("B) SOURCE LEAF DISTRIBUTION")
    print("=" * 90)
    print(f"source_leaf: {source_leaf}")
    print(f"leaf rows: {len(leaf_idxs)}")
    print(label_dist(labels, effect_names, leaf_idxs))

    print("\n" + "=" * 90)
    print("C) EXACT TUPLE INSIDE SOURCE LEAF")
    print("=" * 90)
    print(f"exact tuple rows inside source leaf: {len(exact_in_leaf)}")
    if len(exact_in_leaf):
        print(label_dist(labels, effect_names, exact_in_leaf))

    print("\n" + "=" * 90)
    print("D) LEAF BREAKDOWN BY SYMBOLIC TUPLE")
    print("=" * 90)
    bucket = defaultdict(list)
    for idx in leaf_idxs:
        key = (cat_below[idx], cat_above[idx], cat_rel[idx])
        bucket[key].append(idx)

    for key, idxs in sorted(bucket.items()):
        b, a, r = key
        idxs = np.array(idxs)
        print(f"\nleaf tuple below=objtype{b}, above=objtype{a}, relation=relation{r}, n={len(idxs)}")
        print(label_dist(labels, effect_names, idxs))

    print("\n" + "=" * 90)
    print("E) FIRST 80 ROWS IN SOURCE LEAF")
    print("=" * 90)
    print("idx | label | cat_below cat_above rel | slot0_type/size slot1_type/size | raw dx0 dy0 dz0 dx1 dy1 dz1")
    for idx in leaf_idxs[:80]:
        s0t, s0s, s1t, s1s = physical_indices(int(idx))
        eff = raw_effect[idx]
        print(
            f"{idx:4d} | {effect_names[labels[idx]]:10s} | "
            f"objtype{cat_below[idx]} objtype{cat_above[idx]} relation{cat_rel[idx]} | "
            f"s0 {s0t}/{s0s} s1 {s1t}/{s1s} | "
            f"{eff[0]: .4f} {eff[1]: .4f} {eff[2]: .5f} "
            f"{eff[3]: .4f} {eff[4]: .4f} {eff[5]: .5f}"
        )


if __name__ == "__main__":
    main()
