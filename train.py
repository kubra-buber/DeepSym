import os
import argparse
import time
import yaml
import torch

from models import EffectRegressorMLP
import data


def make_level1_loader(opts):
    transform = data.default_transform(
        size=opts["size"],
        affine=True,
        mean=0.279,
        std=0.0094,
    )
    trainset = data.SingleObjectData(transform=transform)
    return torch.utils.data.DataLoader(
        trainset,
        batch_size=opts["batch_size1"],
        shuffle=True,
    )


def make_level2_loader(opts):
    transform = data.default_transform(
        size=opts["size"],
        affine=True,
        mean=0.279,
        std=0.0094,
    )
    trainset = data.PairedObjectData(transform=transform)
    return torch.utils.data.DataLoader(
        trainset,
        batch_size=opts["batch_size2"],
        shuffle=True,
    )


def freeze_level1_for_level2(model):
    """
    Level 2 uses encoder1 outputs, but encoder1 must stay fixed.

    This is especially important for EMA-VQ: torch.no_grad() alone does not stop
    EMA codebook updates while the module is in training mode.
    """
    model.encoder1.eval()
    model.decoder1.eval()

    for parameter in model.encoder1.parameters():
        parameter.requires_grad_(False)
    for parameter in model.decoder1.parameters():
        parameter.requires_grad_(False)


def require_level1_checkpoint(path, ext):
    encoder_path = os.path.join(path, f"encoder1{ext}.ckpt")
    decoder_path = os.path.join(path, f"decoder1{ext}.ckpt")

    missing = [p for p in (encoder_path, decoder_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Level-1 checkpoint is required before level-2 training. Missing:\n  "
            + "\n  ".join(missing)
        )


def train_level1(model, opts):
    print("\n=== TRAINING LEVEL 1 ===")
    loader = make_level1_loader(opts)
    model.print_model(1)
    model.train(opts["epoch1"], loader, 1)
    print(
        "\nLevel 1 complete.\n"
        f"Best checkpoint: {os.path.join(opts['save'], 'encoder1_best.ckpt')}"
    )


def train_level2(model, opts, stage1_dir, stage1_ext):
    print("\n=== LOADING FIXED LEVEL-1 ENCODER ===")
    require_level1_checkpoint(stage1_dir, stage1_ext)
    model.load(stage1_dir, stage1_ext, 1)
    freeze_level1_for_level2(model)

    print(f"Loaded level 1 from: {stage1_dir}")
    print(f"Checkpoint suffix: {stage1_ext}")
    print("encoder1 is now in eval mode and frozen.")

    print("\n=== TRAINING LEVEL 2 ===")
    loader = make_level2_loader(opts)
    model.print_model(2)
    model.train(opts["epoch2"], loader, 2)
    print(
        "\nLevel 2 complete.\n"
        f"Best checkpoint: {os.path.join(opts['save'], 'encoder2_best.ckpt')}"
    )


def main():
    parser = argparse.ArgumentParser("Train DeepSym effect prediction models by level.")
    parser.add_argument("-opts", required=True, help="YAML option file.")
    parser.add_argument(
        "--level",
        choices=("1", "2", "both"),
        default="both",
        help=(
            "Train only level 1, only level 2, or both sequentially. "
            "Default: both, preserving the old behavior."
        ),
    )
    parser.add_argument(
        "--stage1-dir",
        default=None,
        help=(
            "Directory containing the trained level-1 checkpoint for --level 2. "
            "Default: opts['save']."
        ),
    )
    parser.add_argument(
        "--stage1-ext",
        default="_best",
        help="Level-1 checkpoint suffix. Default: _best.",
    )
    args = parser.parse_args()

    with open(args.opts, "r") as file:
        opts = yaml.safe_load(file)

    os.makedirs(opts["save"], exist_ok=True)
    opts["time"] = time.asctime(time.localtime())

    with open(os.path.join(opts["save"], "opts.yaml"), "w") as file:
        yaml.safe_dump(opts, file)

    print(yaml.safe_dump(opts))

    model = EffectRegressorMLP(opts)

    if args.level == "1":
        train_level1(model, opts)
        return

    if args.level == "2":
        stage1_dir = args.stage1_dir or opts["save"]
        train_level2(model, opts, stage1_dir, args.stage1_ext)
        return

    train_level1(model, opts)
    train_level2(model, opts, opts["save"], "_best")


if __name__ == "__main__":
    main()