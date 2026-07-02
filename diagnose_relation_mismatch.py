import os
import argparse
from collections import Counter, defaultdict

import torch
import yaml
import numpy as np

def idx_of(t0, s0, t1, s1):
    return t0 * 500 + s0 * 50 + t1 * 10 + s1

def decode_vq_rel(cat):
    return cat[:, 8:10].argmax(axis=1)

def decode_orig_rel(cat):
    return (cat[:, 4] > 0).astype(int)

def short_label(name):
    return str(name)

def print_counter(title, c, effect_names):
    total = sum(c.values())
    print(title, f"n={total}")
    for k, v in sorted(c.items()):
        print(f"  {effect_names[k]:10s} {v:4d} p={v/total:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vq-opts", default="opts.yaml")
    ap.add_argument("--orig-save", required=True)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.vq_opts))
    vq_save = opts["save"]

    vq_cat = torch.load(os.path.join(vq_save, "category.pt"), map_location="cpu").numpy()
    orig_cat = torch.load(os.path.join(args.orig_save, "category.pt"), map_location="cpu").numpy()

    labels = torch.load(os.path.join(vq_save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(vq_save, "effect_names.npy"))

    vq_rel = decode_vq_rel(vq_cat)
    orig_rel = decode_orig_rel(orig_cat)

    print("=" * 90)
    print("GLOBAL RELATION CONFUSION: rows=VQ relation, cols=Original GS relation")
    print("=" * 90)
    for vr in [0, 1]:
        row = []
        for orr in [0, 1]:
            n = int(((vq_rel == vr) & (orig_rel == orr)).sum())
            row.append(n)
        print(f"VQ r{vr}: orig r0={row[0]:4d}, orig r1={row[1]:4d}")

    print("\n" + "=" * 90)
    print("LABEL DISTRIBUTION BY (VQ relation, Original relation)")
    print("=" * 90)
    for vr in [0, 1]:
        for orr in [0, 1]:
            idxs = np.where((vq_rel == vr) & (orig_rel == orr))[0]
            if len(idxs) == 0:
                continue
            c = Counter(labels[idxs])
            print_counter(f"VQ r{vr}, ORIG r{orr}", c, effect_names)

    print("\n" + "=" * 90)
    print("PER PHYSICAL PAIR MISMATCH SUMMARY")
    print("=" * 90)

    rows = []
    for t0 in range(5):
        for t1 in range(5):
            idxs = []
            for s0 in range(10):
                for s1 in range(10):
                    idxs.append(idx_of(t0, s0, t1, s1))
            idxs = np.array(idxs)

            mismatch = int((vq_rel[idxs] != orig_rel[idxs]).sum())
            total = len(idxs)

            vq_r1 = int((vq_rel[idxs] == 1).sum())
            orig_r1 = int((orig_rel[idxs] == 1).sum())

            inserted = int((labels[idxs] == list(effect_names).index("inserted")).sum()) if "inserted" in effect_names else -1
            stacked = int((labels[idxs] == list(effect_names).index("stacked")).sum()) if "stacked" in effect_names else -1

            rows.append((mismatch / total, mismatch, total, t0, t1, vq_r1, orig_r1, inserted, stacked))

    rows.sort(reverse=True)

    print("mismatch_rate mismatch/total slot0 slot1  VQ_r1 ORIG_r1 inserted stacked")
    for rate, mismatch, total, t0, t1, vq_r1, orig_r1, inserted, stacked in rows:
        print(
            f"{rate:6.3f}        {mismatch:3d}/{total:3d}      "
            f"{t0:2d}    {t1:2d}    {vq_r1:3d}    {orig_r1:3d}     "
            f"{inserted:3d}     {stacked:3d}"
        )

if __name__ == "__main__":
    main()
