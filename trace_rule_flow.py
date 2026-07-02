#!/usr/bin/env python3
"""
Trace DeepSym / VQ / Railroad rule probabilities end-to-end.

Use this to answer: where did a high inserted/stacked probability in
railroad_operators.json come from?

It prints, for a target symbolic tuple:
  - the matching exported Railroad operator(s)
  - their source decision-tree leaf
  - the class counts inside that leaf
  - the raw training samples that reached that leaf
  - direct empirical distributions for both possible role conventions:
        slot0=below, slot1=above
        slot0=above, slot1=below

Examples:
  python trace_rule_flow.py -opts opts.yaml --below objtype2 --above objtype1 --relation relation1
  python trace_rule_flow.py -opts opts.yaml --below objtype1 --above objtype2 --relation relation1
  python trace_rule_flow.py -opts opts.yaml --leaf 57
"""

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import torch
import yaml


def decimal_to_binary_signed(number, length):
    bits = []
    n = int(number)
    if n == 0:
        bits = [0]
    else:
        while n > 0:
            bits.append(n % 2)
            n //= 2
        bits = list(reversed(bits))
    if len(bits) < length:
        bits = [0] * (length - len(bits)) + bits
    return tuple(1 if b == 1 else -1 for b in bits[-length:])


def binary_signed_to_decimal(bits):
    out = 0
    for b in bits:
        out *= 2
        if int(b) == 1:
            out += 1
    return out


def detect_encoding(category, code1_dim, code2_dim):
    nfeat = category.shape[1]
    old_n = 2 * code1_dim + code2_dim
    onehot_n = 2 * (2 ** code1_dim) + (2 ** code2_dim)
    vals = set(np.unique(category.cpu().numpy()).tolist())

    if nfeat == onehot_n and vals.issubset({0, 1, 0.0, 1.0}):
        return "onehot"
    if nfeat == old_n:
        return "signed_binary"
    # fallback: one-hot-like if 0/1 and block sums look okay
    if vals.issubset({0, 1, 0.0, 1.0}):
        return "onehot_unknown"
    return "unknown"


def decode_category_row(row, code1_dim, code2_dim, encoding):
    arr = row.detach().cpu().numpy()
    n_obj = 2 ** code1_dim
    n_rel = 2 ** code2_dim

    if encoding.startswith("onehot"):
        block0 = arr[:n_obj]
        block1 = arr[n_obj:2*n_obj]
        blockr = arr[2*n_obj:2*n_obj+n_rel]
        slot0 = int(np.argmax(block0))
        slot1 = int(np.argmax(block1))
        rel = int(np.argmax(blockr))
        return slot0, slot1, rel

    if encoding == "signed_binary":
        b0 = arr[:code1_dim]
        b1 = arr[code1_dim:2*code1_dim]
        br = arr[2*code1_dim:2*code1_dim+code2_dim]
        slot0 = binary_signed_to_decimal(b0)
        slot1 = binary_signed_to_decimal(b1)
        rel = binary_signed_to_decimal(br)
        return slot0, slot1, rel

    raise ValueError(f"Cannot decode category encoding: {encoding}, shape={arr.shape}")


def label_dist(labels, effect_names):
    c = Counter(int(x) for x in labels)
    total = sum(c.values())
    parts = []
    for k, v in sorted(c.items(), key=lambda kv: kv[0]):
        name = str(effect_names[k]) if k < len(effect_names) else f"label{k}"
        parts.append((name, v, v / total if total else 0.0))
    return parts


def print_dist(title, labels, effect_names, indent="  "):
    print(title)
    if len(labels) == 0:
        print(indent + "EMPTY")
        return
    for name, cnt, p in label_dist(labels, effect_names):
        print(f"{indent}{name:10s} count={cnt:4d} p={p:.6f}")


def idx_to_physical(idx):
    # This follows data.PairedObjectData indexing:
    # obj_i = idx // 500; size_i = (idx // 50) % 10
    # obj_j = (idx // 10) % 5; size_j = idx % 10
    idx = int(idx)
    slot0_phys = idx // 500
    slot0_size = (idx // 50) % 10
    slot1_phys = (idx // 10) % 5
    slot1_size = idx % 10
    return slot0_phys, slot0_size, slot1_phys, slot1_size


def load_tree(save_dir):
    candidates = [
        "tree_vq_onehot.pkl",
        "tree.pkl",
    ]
    for name in candidates:
        path = os.path.join(save_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f), path
    raise FileNotFoundError(f"No tree_vq_onehot.pkl or tree.pkl found in {save_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--below", default=None, help="e.g. objtype2")
    ap.add_argument("--above", default=None, help="e.g. objtype1")
    ap.add_argument("--relation", default=None, help="e.g. relation1")
    ap.add_argument("--leaf", type=int, default=None, help="analyze a specific tree leaf/source_leaf")
    ap.add_argument("--max-samples", type=int, default=30)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]
    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))
    n_obj = 2 ** code1_dim
    n_rel = 2 ** code2_dim

    category = torch.load(os.path.join(save_dir, "category.pt"), map_location="cpu")
    labels = torch.load(os.path.join(save_dir, "label.pt"), map_location="cpu").detach().cpu().numpy().astype(int)
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"), allow_pickle=True)
    tree, tree_path = load_tree(save_dir)

    encoding = detect_encoding(category, code1_dim, code2_dim)
    decoded = np.array([decode_category_row(category[i], code1_dim, code2_dim, encoding)
                        for i in range(category.shape[0])])
    slot0_sym = decoded[:, 0]
    slot1_sym = decoded[:, 1]
    rel_sym = decoded[:, 2]

    leaves = tree.apply(category.detach().cpu().numpy())

    print("=" * 80)
    print("BASIC INFO")
    print("=" * 80)
    print(f"save_dir: {save_dir}")
    print(f"tree_path: {tree_path}")
    print(f"category shape: {tuple(category.shape)}")
    print(f"category encoding detected: {encoding}")
    print(f"num object symbols: {n_obj}; num relation symbols: {n_rel}")
    print(f"num labels: {len(effect_names)} -> {[str(x) for x in effect_names]}")
    print(f"num samples: {len(labels)}")

    print("\nOverall label distribution:")
    print_dist("", labels, effect_names, indent="  ")

    # Operator lookup
    ops_path = os.path.join(save_dir, "railroad_operators.json")
    if os.path.exists(ops_path):
        with open(ops_path) as f:
            ops = json.load(f).get("stack_operators", [])
        print("\nMatching exported Railroad operators:")
        found = False
        for op in ops:
            if args.below and op.get("below_type", op.get("obj1_type")) != args.below:
                continue
            if args.above and op.get("above_type", op.get("obj2_type")) != args.above:
                continue
            if args.relation and op.get("relation") != args.relation:
                continue
            found = True
            print(json.dumps(op, indent=2))
        if not found:
            print("  No exported operator matched the requested below/above/relation.")
    else:
        print(f"\nNo railroad_operators.json found at {ops_path}")

    def summarize_mask(title, mask):
        idxs = np.where(mask)[0]
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"matched samples: {len(idxs)}")
        if len(idxs) == 0:
            return
        print_dist("Effect labels in this set:", labels[idxs], effect_names)

        leaf_counts = Counter(int(x) for x in leaves[idxs])
        print("\nDecision-tree leaves hit by this set:")
        for leaf, cnt in leaf_counts.most_common(20):
            leaf_mask = leaves == leaf
            leaf_total = int(leaf_mask.sum())
            print(f"  leaf={leaf:4d} count_in_set={cnt:4d} leaf_total={leaf_total:4d}")
            print_dist("    leaf distribution:", labels[leaf_mask], effect_names, indent="      ")

        print("\nFirst matched samples:")
        for idx in idxs[:args.max_samples]:
            p0, s0, p1, s1 = idx_to_physical(idx)
            print(
                f"  idx={idx:4d} leaf={int(leaves[idx]):4d} "
                f"label={str(effect_names[labels[idx]]):10s} "
                f"slot0_sym=objtype{slot0_sym[idx]} slot1_sym=objtype{slot1_sym[idx]} rel=relation{rel_sym[idx]} "
                f"physical_slot0_type={p0} size={s0} physical_slot1_type={p1} size={s1}"
            )

    if args.below is not None and args.above is not None and args.relation is not None:
        below_id = int(args.below.replace("objtype", ""))
        above_id = int(args.above.replace("objtype", ""))
        rel_id = int(args.relation.replace("relation", ""))

        # Test both interpretations. This is the key role-order diagnostic.
        mask_below_above = (slot0_sym == below_id) & (slot1_sym == above_id) & (rel_sym == rel_id)
        mask_above_below = (slot0_sym == above_id) & (slot1_sym == below_id) & (rel_sym == rel_id)

        summarize_mask(
            f"INTERPRETATION A: category slot0=BELOW, slot1=ABOVE for target {args.below} <- {args.above}, {args.relation}",
            mask_below_above,
        )
        summarize_mask(
            f"INTERPRETATION B: category slot0=ABOVE, slot1=BELOW for target {args.below} <- {args.above}, {args.relation}",
            mask_above_below,
        )

    if args.leaf is not None:
        leaf_mask = leaves == int(args.leaf)
        summarize_mask(f"EXPLICIT LEAF ANALYSIS: leaf {args.leaf}", leaf_mask)

    print("\n" + "=" * 80)
    print("HOW TO READ THIS")
    print("=" * 80)
    print("1. The exported operator probability is exactly the effect-label distribution of its source_leaf.")
    print("2. If Interpretation B matches your physical case but the exporter uses Interpretation A, your object roles are swapped.")
    print("3. If the target set itself has high inserted labels, the problem is label/effect clustering, not rule export.")
    print("4. If the target set labels are mostly stacked but its leaf is mixed/inserted, the problem is tree generalization or symbolic categories.")


if __name__ == "__main__":
    main()