"""Save DeepSym/VQ categories using the ORIGINAL DeepSym save_cat.py pair order.

Why this file exists
--------------------
The original DeepSym save_cat.py does NOT iterate through PairedObjectData.
It builds the 50 canonical object crops, forms all 50 x 50 ordered pairs as:

    left_img  = Y.repeat_interleave(B, 0)
    right_img = Y.repeat(B, 1, 1, 1)
    concat    = [left_img, right_img]

and saves:

    category.pt = [left_object_code, right_object_code, relation_code]

The original learn_rules.py / utils.tree_to_code then interprets the first
object-code segment as ?below and the second segment as ?above.  Whether those
names are physically intuitive is less important than preserving the exact
convention used by the original PDDL pipeline.

For VQ models, codebook indices are arbitrary categorical labels, so this file
stores each VQ index as a one-hot vector rather than converting the index to
signed-binary bits.

Usage:
    python save_cat.py -opts opts.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import torch
import yaml

import data
from models import EffectRegressorMLP


def _one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    indices = indices.detach().cpu().long().view(-1)
    if indices.numel() == 0:
        return torch.empty((0, num_classes), dtype=torch.float32)
    if int(indices.min()) < 0 or int(indices.max()) >= num_classes:
        raise ValueError(
            f"VQ index out of range: min={int(indices.min())}, "
            f"max={int(indices.max())}, classes={num_classes}"
        )
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).to(torch.float32)


def _vq_indices_from_encoder(encoder: torch.nn.Sequential, obs: torch.Tensor) -> torch.Tensor:
    """Return VQ code indices for obs.

    Prefer computing the input to the final VQ layer explicitly.  This avoids
    calling get_indices() on an already-quantized output and makes the intended
    computation clear.
    """
    vq_layer = encoder[-1]
    if not hasattr(vq_layer, "get_indices"):
        raise TypeError(
            "Expected encoder[-1] to be a VQ layer with get_indices(). "
            "This save_cat.py is for the VQ implementation."
        )
    pre_vq = encoder[:-1](obs)
    return vq_layer.get_indices(pre_vq)


def main() -> None:
    parser = argparse.ArgumentParser("Save VQ one-hot categories in original DeepSym order.")
    parser.add_argument("-opts", help="option file", type=str, required=True)
    args = parser.parse_args()

    opts: Dict = yaml.safe_load(open(args.opts, "r"))
    opts["device"] = "cpu"
    device = torch.device("cpu")
    save_dir = opts["save"]

    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))
    num_obj_codes = 2 ** code1_dim
    num_rel_codes = 2 ** code2_dim

    print("Loading models...")
    model = EffectRegressorMLP(opts)
    model.load(save_dir, "_best", 1)
    model.load(save_dir, "_best", 2)
    model.encoder1.eval()
    model.encoder2.eval()

    transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)

    # This block intentionally mirrors the original save_cat.py exactly.
    X = torch.load("data/img/obs_prev_z.pt", map_location="cpu")
    X = X.reshape(5, 10, 3, 4, 4, 42, 42)
    X = X[:, :, 0, 2, 2]
    X = X.reshape(-1, 1, 42, 42)
    B, _, _, _ = X.shape

    Y = torch.empty(B, 1, opts["size"], opts["size"])
    for i in range(B):
        Y[i] = transform(X[i])

    with torch.no_grad():
        obj_idx = _vq_indices_from_encoder(model.encoder1, Y.to(device))

        left_img = Y.repeat_interleave(B, 0)
        right_img = Y.repeat(B, 1, 1, 1)
        concat = torch.cat([left_img, right_img], dim=1).to(device)
        rel_idx = _vq_indices_from_encoder(model.encoder2, concat)

    left_idx = obj_idx.repeat_interleave(B, 0)
    right_idx = obj_idx.repeat(B)

    left_oh = _one_hot(left_idx, num_obj_codes)
    right_oh = _one_hot(right_idx, num_obj_codes)
    rel_oh = _one_hot(rel_idx, num_rel_codes)

    category_all = torch.cat([left_oh, right_oh, rel_oh], dim=-1).cpu()

    os.makedirs(save_dir, exist_ok=True)
    torch.save(category_all, os.path.join(save_dir, "category.pt"))
    torch.save(
        {
            "left_object_original_order": left_idx.cpu(),
            "right_object_original_order": right_idx.cpu(),
            "relation_original_order": rel_idx.cpu(),
        },
        os.path.join(save_dir, "category_indices.pt"),
    )

    meta = {
        "encoding": "vq_onehot",
        "order": "original_deepsym_save_cat",
        "code1_dim": code1_dim,
        "code2_dim": code2_dim,
        "num_obj_codes": num_obj_codes,
        "num_rel_codes": num_rel_codes,
        "category_shape": list(category_all.shape),
        "feature_slices": {
            "first_object_original_left_segment": [0, num_obj_codes],
            "second_object_original_right_segment": [num_obj_codes, 2 * num_obj_codes],
            "relation_segment": [2 * num_obj_codes, 2 * num_obj_codes + num_rel_codes],
        },
        "role_convention": (
            "This file preserves original DeepSym save_cat.py order: "
            "[left_object, right_object, relation]. learn_rules_railroad.py binds "
            "left_object to ?below and right_object to ?above to match original "
            "utils.tree_to_code / PDDL export."
        ),
        "note": (
            "VQ indices are one-hot categorical labels. This is original-order, "
            "not role-swapped. Use this when comparing against original PDDL."
        ),
    }
    with open(os.path.join(save_dir, "category_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved category.pt to {os.path.join(save_dir, 'category.pt')}")
    print(f"  category shape: {tuple(category_all.shape)}")
    print(f"  object code counts:  {torch.bincount(obj_idx.cpu(), minlength=num_obj_codes).tolist()}")
    print(f"  relation code counts:{torch.bincount(rel_idx.cpu(), minlength=num_rel_codes).tolist()}")
    print(f"Saved category_meta.json to {os.path.join(save_dir, 'category_meta.json')}")


if __name__ == "__main__":
    main()