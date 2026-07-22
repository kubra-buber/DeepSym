#!/usr/bin/env python3
"""Recognize a randomly generated CoppeliaSim scene with EMA-VQ symbols.

This is the VQ counterpart of the original recognize.py. It writes:
  <save>/problem.pddl
  <save>/objects.txt

Important: VQ categories are nominal integer code indices. The index must be
computed from the encoder latent *before* the final VQ layer.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import rospy
import torch
import yaml

import data
import utils
from simtools.rosutils import RosNode


def load_model_class(kind: str):
    module_name = {
        "dynamic": "models_vq_dynamic",
        "vq": "models_vq",
    }[kind]
    return importlib.import_module(module_name).EffectRegressorMLP


def vq_indices(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    modules = list(encoder.children())
    if len(modules) < 2 or not hasattr(modules[-1], "get_indices"):
        raise TypeError(
            "Expected the encoder's final layer to be a VQ layer with "
            "get_indices()."
        )
    h = x
    for layer in modules[:-1]:
        h = layer(h)
    return modules[-1].get_indices(h)


def main() -> None:
    parser = argparse.ArgumentParser("Recognize a scene using VQ DeepSym.")
    parser.add_argument("-opts", required=True, help="Options YAML")
    parser.add_argument(
        "-goal", default="(H3) (S4)", help="Goal predicates, e.g. '(H3) (S4)'"
    )
    parser.add_argument(
        "-uri", default="http://localhost:11311", help="ROS master URI"
    )
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
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--num-objects", type=int, default=5)
    args = parser.parse_args()

    with open(args.opts, "r") as handle:
        opts = yaml.safe_load(handle)

    ckpt = os.path.abspath(args.ckpt or opts["save"])
    opts["save"] = ckpt
    device = torch.device(opts["device"])

    if not 1 <= args.num_objects <= 5:
        raise ValueError("--num-objects must be between 1 and 5")

    EffectRegressorMLP = load_model_class(args.model_kind)
    model = EffectRegressorMLP(opts)
    model.load(ckpt, "_best", 1)
    model.load(ckpt, "_best", 2)
    model.encoder1.eval()
    model.encoder2.eval()

    node = RosNode("recognize_scene_vq", args.uri)
    # The opened .ttt scene is reset here; random objects are then generated.
    node.stopSimulation()
    rospy.sleep(2.0)
    node.startSimulation()
    rospy.sleep(2.0)

    H = torch.load("H.pt", map_location="cpu")

    rng = np.random.default_rng(args.seed)
    num_objects = args.num_objects
    obj_types = rng.integers(1, 6, size=num_objects)
    obj_sizes = rng.uniform(1.0, 2.0, size=num_objects).tolist()

    candidate_locations = np.array(
        [
            [-0.69, -0.09],
            [-0.90, -0.35],
            [-0.45, 0.175],
            [-0.45, -0.35],
            [-0.90, 0.175],
        ],
        dtype=np.float32,
    )
    candidate_locations = candidate_locations[rng.permutation(5)]
    locations_np = candidate_locations[:num_objects]
    locations = locations_np.tolist()

    for i in range(num_objects):
        node.generateObject(
            int(obj_types[i]),
            float(obj_sizes[i]),
            locations[i] + [float(obj_sizes[i]) * 0.05 + 0.7],
        )
    rospy.sleep(1.0)

    locations_tensor = torch.tensor(locations_np, dtype=torch.float32)
    depth = torch.tensor(node.getDepthImage(8), dtype=torch.float32)
    objs, locs, _ = utils.find_objects(depth, opts["size"])
    if len(objs) != num_objects:
        print(
            f"WARNING: generated {num_objects} objects but detected {len(objs)}."
        )

    transform = data.default_transform(
        size=opts["size"], affine=False, mean=0.279, std=0.0094
    )
    transformed = [transform(obj)[0] for obj in objs]
    objs = torch.stack(transformed).to(device)

    locs_h = torch.cat(
        [
            locs.float(),
            torch.ones(locs.shape[0], 1, device=locs.device),
        ],
        dim=1,
    )
    locs_world = torch.matmul(locs_h, H.T)
    locs_world = locs_world / locs_world[:, 2].reshape(-1, 1)

    _, nearest = torch.cdist(
        locs_world[:, :2].cpu(), locations_tensor
    ).min(dim=1)

    obj_infos = []
    comparisons = []

    with torch.no_grad():
        object_indices = vq_indices(
            model.encoder1, objs.unsqueeze(1)
        ).cpu().tolist()

        for i, obj in enumerate(objs):
            source_idx = int(nearest[i].item())
            cat = int(object_indices[i])
            x, y = locations_tensor[source_idx].tolist()

            print(
                f"Object O{i+1}: category={cat}, "
                f"location=({x:.5f}, {y:.5f})"
            )
            obj_infos.append(
                {
                    "name": f"O{i+1}",
                    "loc": (x, y),
                    "size": float(obj_sizes[source_idx]) * 0.1,
                    "type": f"objtype{cat}",
                }
            )

            for j in range(len(objs)):
                if i == j:
                    continue
                pair = torch.stack([obj, objs[j]], dim=0).unsqueeze(0)
                rel = int(vq_indices(model.encoder2, pair).item())
                comparisons.append(f"(relation{rel} O{i+1} O{j+1})")

    print("Objects:", obj_infos)
    print("Relations:", comparisons)

    save_dir = Path(ckpt)
    save_dir.mkdir(parents=True, exist_ok=True)
    problem_path = save_dir / "problem.pddl"
    objects_path = save_dir / "objects.txt"

    object_str = "\t(:objects " + " ".join(x["name"] for x in obj_infos) + ")"
    init_lines = ["\t(:init"]
    for obj in obj_infos:
        init_lines.append(
            f"\t\t(pickloc {obj['name']}) "
            f"({obj['type']} {obj['name']})"
        )
    init_lines.extend(f"\t\t{x}" for x in comparisons)
    init_lines.extend(["\t\t(H0)", "\t\t(S0)", "\t)"])

    goal_str = (
        f"\t(:goal (and {args.goal} "
        f"(not (stacked)) (not (inserted))))"
    )

    with problem_path.open("w") as handle:
        print("(define (problem dom1) (:domain stack)", file=handle)
        print(object_str, file=handle)
        print("\n".join(init_lines), file=handle)
        print(goal_str, file=handle)
        print(")", file=handle)

    with objects_path.open("w") as handle:
        print(len(obj_infos), file=handle)
        for obj in obj_infos:
            print(
                f"{obj['name']} {obj['loc'][0]:.5f} "
                f"{obj['loc'][1]:.5f} {obj['size']:.5f}",
                file=handle,
            )

    print(f"Wrote: {problem_path}")
    print(f"Wrote: {objects_path}")


if __name__ == "__main__":
    main()