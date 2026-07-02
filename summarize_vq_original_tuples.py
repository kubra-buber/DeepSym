import os
import argparse
from collections import defaultdict, Counter

import torch
import yaml
import numpy as np

def idx_of(slot0_type, slot0_size, slot1_type, slot1_size):
    return slot0_type * 500 + slot0_size * 50 + slot1_type * 10 + slot1_size

def decode_vq_onehot(cat):
    below = cat[:, 0:4].argmax(axis=1)
    above = cat[:, 4:8].argmax(axis=1)
    rel = cat[:, 8:10].argmax(axis=1)
    return below, above, rel

def bits_to_code(bits):
    bits = np.asarray(bits)
    b0 = 1 if bits[0] > 0 else 0
    b1 = 1 if bits[1] > 0 else 0
    return b0 * 2 + b1

def decode_original_binary(cat):
    below = np.array([bits_to_code(row[0:2]) for row in cat])
    above = np.array([bits_to_code(row[2:4]) for row in cat])
    rel = np.array([1 if row[4] > 0 else 0 for row in cat])
    return below, above, rel

def print_dist(title, d, effect_names):
    print("\n" + title)
    print("-" * len(title))
    for tup, c in sorted(d.items()):
        total = sum(c.values())
        parts = []
        for k, v in sorted(c.items()):
            parts.append(f"{effect_names[k]}={v}/{total}={v/total:.3f}")
        print(f"{tup:6s} n={total:3d}  " + "  ".join(parts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq-opts", default="opts.yaml")
    ap.add_argument("--orig-save", required=True)
    ap.add_argument("--slot0-type", type=int, required=True)
    ap.add_argument("--slot1-type", type=int, required=True)
    args = ap.parse_args()

    vq_opts = yaml.safe_load(open(args.vq_opts))
    vq_save = vq_opts["save"]

    vq_cat = torch.load(os.path.join(vq_save, "category.pt"), map_location="cpu").numpy()
    orig_cat = torch.load(os.path.join(args.orig_save, "category.pt"), map_location="cpu").numpy()

    labels = torch.load(os.path.join(vq_save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(vq_save, "effect_names.npy"))

    vb, va, vr = decode_vq_onehot(vq_cat)
    ob, oa, orr = decode_original_binary(orig_cat)

    vq_dist = defaultdict(Counter)
    orig_dist = defaultdict(Counter)
    joint_dist = defaultdict(Counter)

    for s0 in range(10):
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            lab = labels[idx]

            vt = f"{vb[idx]}{va[idx]}r{vr[idx]}"
            ot = f"{ob[idx]}{oa[idx]}r{orr[idx]}"

            vq_dist[vt][lab] += 1
            orig_dist[ot][lab] += 1
            joint_dist[f"VQ={vt} ORIG={ot}"][lab] += 1

    print("=" * 90)
    print(f"slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print(f"VQ save: {vq_save}")
    print(f"Original save: {args.orig_save}")
    print("=" * 90)

    print_dist("VQ tuple label distributions", vq_dist, effect_names)
    print_dist("Original tuple label distributions", orig_dist, effect_names)
    print_dist("Joint VQ-vs-Original tuple distributions", joint_dist, effect_names)

if __name__ == "__main__":
    main()
