#!/usr/bin/env python3
"""Create canonical object-category montages for dynamic_prune checkpoints."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

import data


FAMILY_NAMES = [
    "Sphere",
    "Cube",
    "Vertical cylinder",
    "Horizontal cylinder",
    "Cup",
]


def run_before_vq(encoder: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    modules = list(encoder.children())
    if len(modules) < 2 or not hasattr(modules[-1], "get_indices"):
        raise TypeError("Encoder must end with a VQ layer exposing get_indices().")
    h = x
    for layer in modules[:-1]:
        h = layer(h)
    return h


def choose_representatives(
    members: List[int],
    distances: np.ndarray,
    maximum: int,
) -> List[int]:
    ranked = sorted(members, key=lambda index: float(distances[index]))
    selected: List[int] = []
    families = set()

    for index in ranked:
        family = index // 10
        if family not in families:
            selected.append(index)
            families.add(family)
            if len(selected) >= maximum:
                return selected

    for index in ranked:
        if index not in selected:
            selected.append(index)
            if len(selected) >= maximum:
                break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--png", default="")
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    run_dir = Path(args.ckpt).expanduser().resolve()
    opts_path = run_dir / "opts.yaml"
    if not opts_path.exists():
        raise FileNotFoundError(opts_path)

    with opts_path.open("r") as handle:
        opts = yaml.safe_load(handle)
    opts["device"] = "cpu"
    opts["save"] = str(run_dir)

    model_name = str(opts.get("poster_model", "dynamic_prune"))
    module_name = {
        "dynamic_prune": "models_vq_dynamic_prune",
        "dynamic": "models_vq_dynamic",
        "vq": "models_vq",
    }.get(model_name)
    if module_name is None:
        raise ValueError(f"Unsupported poster_model: {model_name}")

    model_class = importlib.import_module(module_name).EffectRegressorMLP
    model = model_class(opts)
    model.load(str(run_dir), "_best", 1)
    model.encoder1.eval()

    raw_path = Path("data/img/obs_prev_z.pt")
    raw_all = torch.load(raw_path, map_location="cpu")
    raw_all = raw_all.reshape(5, 10, 3, 4, 4, 42, 42)
    raw = raw_all[:, :, 0, 2, 2].reshape(50, 1, 42, 42).float()

    size = int(opts["size"])
    transform = data.default_transform(
        size=size,
        affine=False,
        mean=0.279,
        std=0.0094,
    )
    encoded = torch.empty(50, 1, size, size)
    for index in range(50):
        encoded[index] = transform(raw[index])

    with torch.no_grad():
        latent = run_before_vq(model.encoder1, encoded)
        layer = list(model.encoder1.children())[-1]
        indices = layer.get_indices(latent).cpu().numpy().astype(int)
        active = int(layer.get_num_codes())
        flat = latent.reshape(latent.shape[0], -1)
        vectors = layer.embedding.weight[:active]
        distances = torch.cdist(flat, vectors).pow(2)
        assigned = distances[
            torch.arange(flat.shape[0]),
            torch.as_tensor(indices, dtype=torch.long),
        ].cpu().numpy()

    used_codes = sorted(np.unique(indices).tolist())
    representatives = {
        code: choose_representatives(
            np.flatnonzero(indices == code).tolist(),
            assigned,
            max(1, args.max_examples),
        )
        for code in used_codes
    }

    columns = max(len(items) for items in representatives.values())
    rows = len(used_codes)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.0 * columns, 2.05 * rows),
        squeeze=False,
        constrained_layout=True,
    )

    raw_np = raw[:, 0].numpy()
    valid = raw_np[np.isfinite(raw_np)]
    vmin = float(np.percentile(valid, 1.0))
    vmax = float(np.percentile(valid, 99.0))

    for row, code in enumerate(used_codes):
        chosen = representatives[code]
        for column in range(columns):
            ax = axes[row, column]
            ax.set_xticks([])
            ax.set_yticks([])
            if column >= len(chosen):
                ax.axis("off")
                continue

            index = chosen[column]
            family = index // 10
            size_index = index % 10 + 1
            ax.imshow(
                raw_np[index],
                cmap="gray_r",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(
                f"{FAMILY_NAMES[family]}\nsize {size_index}",
                fontsize=10,
            )

        axes[row, 0].set_ylabel(
            f"Object code {code}",
            fontsize=12,
            fontweight="bold",
            labelpad=12,
        )

    fig.suptitle(
        "Canonical observations grouped by learned object code",
        fontsize=15,
        fontweight="bold",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if args.png:
        png = Path(args.png)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Active codes: {active}")
    print(f"Used codes on canonical set: {used_codes}")
    for code in used_codes:
        count = int((indices == code).sum())
        print(f"  code {code}: {count}/50 canonical objects")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()