import argparse
import os
import math
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt

import data
import utils


def load_model_class(model_kind: str):
    if model_kind == "original":
        from models import EffectRegressorMLP
    elif model_kind == "vq":
        from models_vq import EffectRegressorMLP
    elif model_kind == "dynamic":
        from models_vq_dynamic import EffectRegressorMLP
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")
    return EffectRegressorMLP


def robust_model_load(model, ckpt_path):
    """
    Tries common load signatures used across your DeepSym variants.
    """
    tried = []

    for args in [
        (ckpt_path, "_best"),
        (ckpt_path, "_best", 1),
        (ckpt_path,),
    ]:
        try:
            model.load(*args)
            return
        except Exception as exc:
            tried.append((args, repr(exc)))

    msg = ["Could not load checkpoint. Tried these signatures:"]
    for args, exc in tried:
        msg.append(f"  model.load{args} -> {exc}")
    raise RuntimeError("\n".join(msg))


def infer_category(model, code, model_kind):
    """
    Extract integer symbol index from encoder output.
    """
    # VQ / Dynamic VQ path
    if model_kind in ["vq", "dynamic"]:
        try:
            idx = model.encoder1[-1].get_indices(code)
            return int(idx[0].item())
        except Exception:
            pass

    # Original DeepSym path (binary code -> decimal)
    try:
        return int(utils.binary_to_decimal(code[0]))
    except Exception as exc:
        raise RuntimeError(f"Failed to infer category index: {exc}")


def build_object_tensor(sample_observation, size):
    """
    Replicates the reshaping logic from test_first.py.
    """
    B = sample_observation.shape[0]
    dim1 = B // 480
    if dim1 == 0:
        raise RuntimeError(
            f"Unexpected batch size {B}; cannot reshape like test_first.py"
        )

    objects = sample_observation.reshape(dim1, 10, 3, 4, 4, size, size)
    objects = objects[:, :, 0].reshape(-1, 1, size, size)
    return objects


def collect_examples(model, objects, num_symbols, max_examples, model_kind):
    """
    Collect up to max_examples canonical observations for each learned object symbol.
    """
    grouped = [[] for _ in range(num_symbols)]

    model.encoder1.eval()
    with torch.no_grad():
        max_elements = objects.shape[0]
        it = 0

        while it < max_elements:
            code = model.encoder1(objects[it].reshape(1, 1, objects.shape[-2], objects.shape[-1]))
            cat = infer_category(model, code, model_kind)

            if 0 <= cat < num_symbols:
                if len(grouped[cat]) < max_examples:
                    grouped[cat].append(objects[it].clone())

            it += 1

            done = all(len(grouped[k]) >= max_examples for k in range(num_symbols))
            if done:
                break

    return grouped


def plasma_colorize(stacked_gray_images):
    """
    Same spirit as test_first.py:
    normalize globally and apply matplotlib plasma colormap.
    Input: [N, H, W]
    Output: [N, H, W, 3]
    """
    arr = stacked_gray_images.cpu().numpy().astype(np.float32)

    vmin = arr.min()
    vmax = arr.max()
    if vmax - vmin < 1e-8:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - vmin) / (vmax - vmin)

    cmap = plt.cm.plasma
    colored = cmap(arr)[..., :3]  # drop alpha
    return colored


def make_pdf(grouped, output_pdf, title):
    """
    Render grouped examples as a clean poster-ready PDF.
    """
    counts = [len(g) for g in grouped]
    min_len = min(counts)

    if min_len == 0:
        raise RuntimeError(
            "At least one learned object symbol has zero collected examples. "
            "Training may have collapsed or num_symbols may be incorrect."
        )

    # Trim all rows equally for a clean grid
    grouped = [torch.stack(g[:min_len]) for g in grouped]

    # Stack into [K, C, 1, H, W] -> [K*min_len, H, W]
    all_gray = torch.stack(grouped)[:, :, 0].reshape(-1, grouped[0].shape[-2], grouped[0].shape[-1])
    all_rgb = plasma_colorize(all_gray)

    num_symbols = len(grouped)
    H, W = all_rgb.shape[1], all_rgb.shape[2]
    all_rgb = all_rgb.reshape(num_symbols, min_len, H, W, 3)

    # Figure sizing chosen to stay compact and poster-friendly
    fig_w = max(10, min_len * 1.6)
    fig_h = max(6, num_symbols * 1.8 + 1.0)

    fig, axes = plt.subplots(num_symbols, min_len, figsize=(fig_w, fig_h))
    if num_symbols == 1:
        axes = np.expand_dims(axes, axis=0)
    if min_len == 1:
        axes = np.expand_dims(axes, axis=1)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    for r in range(num_symbols):
        for c in range(min_len):
            ax = axes[r, c]
            ax.imshow(all_rgb[r, c], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if c == 0:
                ax.set_ylabel(
                    f"Object symbol {r}",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    labelpad=18,
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved poster-style object-symbol figure to: {output_pdf}")


def main():
    parser = argparse.ArgumentParser("Poster-friendly object-symbol figure based on test_first.py")
    parser.add_argument("-opts", required=True, help="Path to opts.yaml")
    parser.add_argument("--ckpt", required=True, help="Checkpoint directory")
    parser.add_argument(
        "--model-kind",
        choices=["original", "vq", "dynamic"],
        default="dynamic",
        help="Which model file to import"
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=4,
        help="Number of learned object symbols to visualize"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum number of examples per symbol"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="poster_assets/object_symbol_grid",
        help="Output path prefix or .pdf path"
    )
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    ckpt_opts_path = os.path.join(args.ckpt, "opts.yaml")
    if os.path.exists(ckpt_opts_path):
        ckpt_opts = yaml.safe_load(open(ckpt_opts_path, "r"))
        opts.update(ckpt_opts)

    opts["device"] = "cpu"

    ModelClass = load_model_class(args.model_kind)
    model = ModelClass(opts)
    robust_model_load(model, args.ckpt)

    transform = data.default_transform(
        size=opts["size"], affine=False, mean=0.279, std=0.0094
    )
    trainset = data.SingleObjectData(transform=transform)

    loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=len(trainset),
        shuffle=True
    )

    sample = next(iter(loader))
    objects = build_object_tensor(sample["observation"], opts["size"])

    grouped = collect_examples(
        model=model,
        objects=objects,
        num_symbols=args.num_symbols,
        max_examples=args.max_examples,
        model_kind=args.model_kind,
    )

    output_pdf = args.output if args.output.endswith(".pdf") else args.output + ".pdf"

    make_pdf(
        grouped=grouped,
        output_pdf=output_pdf,
        title="Representative canonical observations grouped by learned object symbol",
    )


if __name__ == "__main__":
    main()