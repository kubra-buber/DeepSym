import os
import argparse
import torch
import yaml
import numpy as np

def idx_of(slot0_type, slot0_size, slot1_type, slot1_size):
    # PairedObjectData index formula:
    # obj_i = idx // 500
    # size_i = (idx // 50) % 10
    # obj_j = (idx // 10) % 5
    # size_j = idx % 10
    return slot0_type * 500 + slot0_size * 50 + slot1_type * 10 + slot1_size

def decode_onehot_category(category):
    if category.shape[1] != 10:
        raise ValueError(f"Expected one-hot category.pt with shape [N,10], got {category.shape}")
    below = category[:, 0:4].argmax(axis=1)
    above = category[:, 4:8].argmax(axis=1)
    rel = category[:, 8:10].argmax(axis=1)
    return below, above, rel

def short_label(name):
    mapping = {
        "inserted": "I",
        "stacked": "S",
        "roll1": "R1",
        "roll2": "R2",
        "tumble1": "T1",
        "tumble2": "T2",
    }
    return mapping.get(str(name), str(name)[:2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--slot0-type", type=int, required=True)
    ap.add_argument("--slot1-type", type=int, required=True)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    save = opts["save"]

    category = torch.load(os.path.join(save, "category.pt"), map_location="cpu").detach().cpu().numpy()
    label = torch.load(os.path.join(save, "label.pt"), map_location="cpu").detach().cpu().numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))
    raw_effect = torch.load("data/img/delta_pix_3.pt", map_location="cpu").detach().cpu().numpy()

    cat_below, cat_above, cat_rel = decode_onehot_category(category)

    print("=" * 100)
    print(f"PAIR GRID: slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print("Rows = slot0_size 9..0, columns = slot1_size 0..9")
    print("Each cell: label / tuple below-above-rel / dz0")
    print("=" * 100)

    for s0 in reversed(range(10)):
        cells = []
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            lab = short_label(effect_names[label[idx]])
            tup = f"{cat_below[idx]}{cat_above[idx]}r{cat_rel[idx]}"
            dz0 = raw_effect[idx, 2]
            cells.append(f"{lab}:{tup}:{dz0:.3f}")
        print(f"s0={s0}: " + " | ".join(cells))

    print("\nLegend:")
    print("  I=inserted, S=stacked, R1/R2=roll, T1/T2=tumble")
    print("  tuple format: belowCode aboveCode rRelation")
    print("  Example 12r1 means below=objtype1, above=objtype2, relation1")
    print("\nTarget tuple we are tracking: 12r1")

if __name__ == "__main__":
    main()
