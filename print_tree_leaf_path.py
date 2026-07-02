#!/usr/bin/env python3
"""Print the decision-tree path and matched symbolic tuples for a DeepSym/Railroad leaf."""
import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import torch
import yaml


def load_meta(save_dir, category_shape, opts):
    meta_path = os.path.join(save_dir, "category_meta.json")
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))
    n_obj = 2 ** code1_dim
    n_rel = 2 ** code2_dim
    if category_shape[1] == 2 * n_obj + n_rel:
        return {
            "encoding": "vq_onehot",
            "num_obj_codes": n_obj,
            "num_rel_codes": n_rel,
            "feature_slices": {
                "below_object": [0, n_obj],
                "above_object": [n_obj, 2*n_obj],
                "relation": [2*n_obj, 2*n_obj+n_rel],
            }
        }
    raise ValueError("Only vq_onehot category.pt is supported by this diagnostic.")


def feature_name(i, meta):
    sl = meta["feature_slices"]
    b0, b1 = sl["below_object"]
    a0, a1 = sl["above_object"]
    r0, r1 = sl["relation"]
    if b0 <= i < b1:
        return f"below_is_objtype{i-b0}"
    if a0 <= i < a1:
        return f"above_is_objtype{i-a0}"
    if r0 <= i < r1:
        return f"relation_is_relation{i-r0}"
    return f"feature_{i}"


def tuple_feature(below, above, rel, meta):
    n_obj = int(meta["num_obj_codes"])
    n_rel = int(meta["num_rel_codes"])
    x = np.zeros(2*n_obj+n_rel, dtype=np.float32)
    x[below] = 1.0
    x[n_obj + above] = 1.0
    x[2*n_obj + rel] = 1.0
    return x


def path_for_leaf(tree, leaf_id):
    tr = tree.tree_
    path = []
    found = []
    def dfs(node, cur):
        if node == leaf_id:
            found.extend(cur)
            return True
        f = tr.feature[node]
        if f < 0:
            return False
        th = float(tr.threshold[node])
        if dfs(tr.children_left[node], cur + [(int(f), "<=", th)]):
            return True
        if dfs(tr.children_right[node], cur + [(int(f), ">", th)]):
            return True
        return False
    if not dfs(0, []):
        raise ValueError(f"Leaf {leaf_id} not found in tree")
    return found


def satisfies(x, path, eps=1e-12):
    for f, op, th in path:
        v = float(x[f])
        if op == "<=" and v > th + eps:
            return False
        if op == ">" and v <= th + eps:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--leaf", type=int, required=True)
    ap.add_argument("--tree", default=None, help="optional tree filename in save dir")
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    save = opts["save"]
    cat = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
    labels = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))
    meta = load_meta(save, cat.shape, opts)

    if args.tree:
        tree_path = os.path.join(save, args.tree)
    else:
        candidates = [
            p for p in os.listdir(save)
            if p.startswith("tree_vq_onehot") and p.endswith(".pkl")
        ]
        if not candidates:
            candidates = ["tree_vq_onehot.pkl", "tree.pkl"]
        # prefer newest if multiple
        existing = [os.path.join(save, p) for p in candidates if os.path.exists(os.path.join(save, p))]
        if not existing:
            raise FileNotFoundError("No tree pickle found")
        tree_path = max(existing, key=os.path.getmtime)

    tree = pickle.load(open(tree_path, "rb"))
    leaves = tree.apply(cat)
    idxs = np.where(leaves == args.leaf)[0]
    path = path_for_leaf(tree, args.leaf)

    print(f"save={save}")
    print(f"tree={tree_path}")
    print(f"leaf={args.leaf}")
    print(f"samples in leaf={len(idxs)}")
    print("\nPath constraints:")
    for f, op, th in path:
        print(f"  {feature_name(f, meta):24s} {op} {th:.6f}")

    print("\nLeaf label distribution:")
    c = Counter(labels[idxs])
    for lab, n in sorted(c.items()):
        print(f"  {effect_names[lab]:10s} {n:4d} p={n/max(1,len(idxs)):.6f}")

    print("\nAll symbolic tuples satisfying this leaf:")
    n_obj = int(meta["num_obj_codes"])
    n_rel = int(meta["num_rel_codes"])
    for below in range(n_obj):
        for above in range(n_obj):
            for rel in range(n_rel):
                x = tuple_feature(below, above, rel, meta)
                if satisfies(x, path):
                    print(f"  below=objtype{below} above=objtype{above} relation=relation{rel}")

    print("\nPhysical sample distribution in this leaf using dataset index formula:")
    pair_c = Counter()
    pair_label_c = {}
    for idx in idxs:
        slot0_type = idx // 500
        slot0_size = (idx // 50) % 10
        slot1_type = (idx // 10) % 5
        slot1_size = idx % 10
        key = (slot0_type, slot1_type)
        pair_c[key] += 1
        pair_label_c.setdefault(key, Counter())[effect_names[labels[idx]]] += 1
    for key, n in sorted(pair_c.items()):
        print(f"  slot0_type={key[0]} slot1_type={key[1]} total={n}: {dict(pair_label_c[key])}")

if __name__ == "__main__":
    main()