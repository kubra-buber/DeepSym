import os
import argparse
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
    # original DeepSym: tree paths treat <=0.5 as -1, >0.5 as +1
    bits = np.asarray(bits)
    b0 = 1 if bits[0] > 0 else 0
    b1 = 1 if bits[1] > 0 else 0
    return b0 * 2 + b1

def decode_original_binary(cat):
    # original category shape should be [2500, 5]:
    # first 2 bits object1, next 2 bits object2, last bit relation
    below = np.array([bits_to_code(row[0:2]) for row in cat])
    above = np.array([bits_to_code(row[2:4]) for row in cat])
    rel = np.array([1 if row[4] > 0 else 0 for row in cat])
    return below, above, rel

def short_label(name):
    return {
        "inserted": "I",
        "stacked": "S",
        "roll1": "R1",
        "roll2": "R2",
        "tumble1": "T1",
        "tumble2": "T2",
    }.get(str(name), str(name)[:2])

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

    vq_b, vq_a, vq_r = decode_vq_onehot(vq_cat)
    og_b, og_a, og_r = decode_original_binary(orig_cat)

    print("=" * 120)
    print(f"VQ save:      {vq_save}")
    print(f"Original save:{args.orig_save}")
    print(f"Pair: slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print("Rows = slot0_size 9..0, Cols = slot1_size 0..9")
    print("cell = label | VQtuple | ORIGtuple")
    print("=" * 120)

    for s0 in reversed(range(10)):
        cells = []
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            lab = short_label(effect_names[labels[idx]])
            vq = f"{vq_b[idx]}{vq_a[idx]}r{vq_r[idx]}"
            og = f"{og_b[idx]}{og_a[idx]}r{og_r[idx]}"
            cells.append(f"{lab}:{vq}:{og}")
        print(f"s0={s0}: " + " | ".join(cells))

if __name__ == "__main__":
    main()
