import os
import argparse
import torch
import yaml
import numpy as np
from collections import Counter

def idx_of(slot0_type, slot0_size, slot1_type, slot1_size):
    return slot0_type * 500 + slot0_size * 50 + slot1_type * 10 + slot1_size

def decode_onehot_category(category):
    below = category[:, 0:4].argmax(axis=1)
    above = category[:, 4:8].argmax(axis=1)
    rel = category[:, 8:10].argmax(axis=1)
    return below, above, rel

def short(name):
    return {
        "inserted": "I",
        "stacked": "S",
        "roll1": "R1",
        "roll2": "R2",
        "tumble1": "T1",
        "tumble2": "T2",
    }.get(str(name), str(name)[:2])

def summarize_pair(category, labels, effect_names, raw, t0, t1):
    cb, ca, cr = decode_onehot_category(category)
    print("=" * 100)
    print(f"slot0_type={t0}, slot1_type={t1}")
    print("rows=slot0_size 9..0, cols=slot1_size 0..9")
    print("cell = label:tuple:dz0")
    print("=" * 100)

    tuple_counter = Counter()
    label_counter = Counter()

    for s0 in reversed(range(10)):
        cells = []
        for s1 in range(10):
            idx = idx_of(t0, s0, t1, s1)
            lab = short(effect_names[labels[idx]])
            tup = f"{cb[idx]}{ca[idx]}r{cr[idx]}"
            dz0 = raw[idx, 2]
            cells.append(f"{lab}:{tup}:{dz0:.3f}")
            tuple_counter[tup] += 1
            label_counter[effect_names[labels[idx]]] += 1
        print(f"s0={s0}: " + " | ".join(cells))

    print("\nTuple counts:")
    for k, v in sorted(tuple_counter.items()):
        print(f"  {k}: {v}")

    print("\nLabel counts:")
    for k, v in sorted(label_counter.items()):
        print(f"  {k}: {v}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    save = opts["save"]

    category = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
    labels = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))
    raw = torch.load("data/img/delta_pix_3.pt", map_location="cpu").numpy()

    summarize_pair(category, labels, effect_names, raw, args.a, args.b)
    summarize_pair(category, labels, effect_names, raw, args.b, args.a)

if __name__ == "__main__":
    main()
