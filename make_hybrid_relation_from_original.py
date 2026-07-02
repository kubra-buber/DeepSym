import os
import argparse
import torch
import yaml
import numpy as np

def decode_original_relation(orig_cat):
    # original category shape: [N, 5]
    # first 2 bits object1, next 2 bits object2, last bit relation
    return (orig_cat[:, 4] > 0).long()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--orig-save", required=True)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    vq_save = opts["save"]

    vq_cat = torch.load(os.path.join(vq_save, "category.pt"), map_location="cpu").float()
    orig_cat = torch.load(os.path.join(args.orig_save, "category.pt"), map_location="cpu")

    if vq_cat.shape[1] != 10:
        raise ValueError(f"Expected VQ one-hot category [N,10], got {vq_cat.shape}")
    if orig_cat.shape[1] != 5:
        raise ValueError(f"Expected original binary category [N,5], got {orig_cat.shape}")
    if vq_cat.shape[0] != orig_cat.shape[0]:
        raise ValueError(f"Row mismatch: VQ {vq_cat.shape}, original {orig_cat.shape}")

    orig_rel = decode_original_relation(orig_cat)
    orig_rel_oh = torch.nn.functional.one_hot(orig_rel, num_classes=2).float()

    hybrid = vq_cat.clone()
    hybrid[:, 8:10] = orig_rel_oh

    out = os.path.join(vq_save, "category_hybrid_vq_objects_original_relation.pt")
    torch.save(hybrid, out)

    print(f"Saved: {out}")
    print("Relation counts:")
    print("  VQ original relation counts:", vq_cat[:, 8:10].argmax(dim=1).bincount(minlength=2).tolist())
    print("  Original GS relation counts:", orig_rel.bincount(minlength=2).tolist())
    print("  Hybrid relation counts:", hybrid[:, 8:10].argmax(dim=1).bincount(minlength=2).tolist())

if __name__ == "__main__":
    main()
