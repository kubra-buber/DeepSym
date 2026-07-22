#!/usr/bin/env python3
"""Export one-hot VQ symbolic tuples for learn_rules_vq.py."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import yaml

import data


def load_model_class(kind: str):
    module_name = {
        "dynamic": "models_vq_dynamic",
        "vq": "models_vq",
    }[kind]
    return importlib.import_module(module_name).EffectRegressorMLP


def vq_indices(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    modules = list(encoder.children())
    if len(modules) < 2 or not hasattr(modules[-1], "get_indices"):
        raise TypeError("Final encoder layer is not a compatible VQ layer")
    h = x
    for layer in modules[:-1]:
        h = layer(h)
    return modules[-1].get_indices(h)


def main() -> None:
    parser = argparse.ArgumentParser("Save VQ categories for rule learning.")
    parser.add_argument("-opts", required=True)
    parser.add_argument(
        "--model-kind",
        choices=["dynamic", "vq"],
        default="dynamic",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Checkpoint directory override; defaults to opts['save']",
    )
    args = parser.parse_args()

    with open(args.opts, "r") as handle:
        opts = yaml.safe_load(handle)
    save_dir = Path(os.path.abspath(args.ckpt or opts["save"]))
    opts["save"] = str(save_dir)
    opts["device"] = "cpu"

    EffectRegressorMLP = load_model_class(args.model_kind)
    model = EffectRegressorMLP(opts)
    model.load(str(save_dir), "_best", 1)
    model.load(str(save_dir), "_best", 2)
    model.encoder1.eval()
    model.encoder2.eval()

    transform = data.default_transform(
        size=opts["size"], affine=False, mean=0.279, std=0.0094
    )
    X = torch.load("data/img/obs_prev_z.pt", map_location="cpu")
    X = X.reshape(5, 10, 3, 4, 4, 42, 42)
    X = X[:, :, 0, 2, 2].reshape(-1, 1, 42, 42)

    Y = torch.empty(X.shape[0], 1, opts["size"], opts["size"])
    for i in range(X.shape[0]):
        Y[i] = transform(X[i])

    with torch.no_grad():
        object_idx = vq_indices(model.encoder1, Y).long()

        left_img = Y.repeat_interleave(Y.shape[0], dim=0)
        right_img = Y.repeat(Y.shape[0], 1, 1, 1)
        pairs = torch.cat([left_img, right_img], dim=1)
        relation_idx = vq_indices(model.encoder2, pairs).long()

    object_layer = list(model.encoder1.children())[-1]
    relation_layer = list(model.encoder2.children())[-1]
    n_obj = int(object_layer.get_num_codes())
    n_rel = int(relation_layer.get_num_codes())

    left_idx = object_idx.repeat_interleave(Y.shape[0])
    right_idx = object_idx.repeat(Y.shape[0])

    left_hot = torch.nn.functional.one_hot(left_idx, n_obj)
    right_hot = torch.nn.functional.one_hot(right_idx, n_obj)
    relation_hot = torch.nn.functional.one_hot(relation_idx, n_rel)
    category = torch.cat(
        [left_hot, right_hot, relation_hot], dim=1
    ).float()

    torch.save(category, save_dir / "category.pt")
    torch.save(object_idx.cpu(), save_dir / "object_category_indices.pt")
    torch.save(
        relation_idx.reshape(Y.shape[0], Y.shape[0]).cpu(),
        save_dir / "relation_category_indices.pt",
    )

    meta = {
        "encoding": "vq_onehot",
        "model_kind": args.model_kind,
        "num_object_codes": n_obj,
        "num_relation_codes": n_rel,
        "num_objects": int(Y.shape[0]),
        "num_pairs": int(Y.shape[0] ** 2),
        "feature_order": [
            "below_object_onehot",
            "above_object_onehot",
            "relation_onehot",
        ],
    }
    with (save_dir / "category_meta.json").open("w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved {category.shape} category matrix to {save_dir}")
    print(f"Object codes: active={n_obj}, used={sorted(object_idx.unique().tolist())}")
    print(f"Relation codes: active={n_rel}, used={sorted(relation_idx.unique().tolist())}")


if __name__ == "__main__":
    main()