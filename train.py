"""Reproducible staged training for the DeepSym poster experiments.

The same script trains three controlled bottleneck variants:

    original : original DeepSym straight-through binary bottleneck
    vq       : fixed-size EMA vector quantization
    dynamic  : dynamically growing EMA vector quantization

Each run writes checkpoints, epoch metrics, dynamic-growth logs, resolved opts,
and (unless disabled) invokes poster_eval.py to create poster-ready figures.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml

import data


MODEL_MODULES = {
    "original": "models",
    "vq": "models_vq",
    "dynamic": "models_vq_dynamic",
}


def resolve_device(requested: str) -> str:
    requested = str(requested or "cpu")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        try:
            if not torch.cuda.is_available():
                print("WARNING: CUDA unavailable; falling back to CPU.")
                return "cpu"
            torch.empty(1, device=requested)
        except Exception as exc:
            print("WARNING: CUDA initialization failed (%s); falling back to CPU." % exc)
            return "cpu"
    return requested


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except (AttributeError, TypeError):
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_loader(
    opts: Dict,
    level: int,
    training: bool,
    seed: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    transform = data.default_transform(
        size=opts["size"],
        affine=bool(training),
        mean=0.279,
        std=0.0094,
    )
    if level == 1:
        dataset = data.SingleObjectData(transform=transform)
        batch_size = int(opts["batch_size1"])
    elif level == 2:
        dataset = data.PairedObjectData(transform=transform)
        if hasattr(dataset, "train"):
            dataset.train = bool(training)
        batch_size = int(opts["batch_size2"])
    else:
        raise ValueError("level must be 1 or 2")

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(training),
        num_workers=int(num_workers),
        pin_memory=str(opts["device"]).startswith("cuda"),
        generator=make_generator(seed),
        worker_init_fn=seed_worker if int(num_workers) > 0 else None,
    )


def batch_size_of(sample: Dict[str, torch.Tensor]) -> int:
    for value in sample.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from sample")


def average_dict(total: Dict[str, float], denominator: int) -> Dict[str, float]:
    denominator = max(1, int(denominator))
    return {key: value / denominator for key, value in total.items()}


def run_training_epoch(model, loader, level: int) -> Dict[str, float]:
    model.prepare_level(level, training=True)
    totals = {"total": 0.0, "effect": 0.0, "vq": 0.0}
    samples = 0
    for sample in loader:
        batch_n = batch_size_of(sample)
        losses = model.optimize_batch(sample, level)
        for key in totals:
            totals[key] += float(losses.get(key, 0.0)) * batch_n
        samples += batch_n
    return average_dict(totals, samples)


def run_evaluation_epoch(model, loader, level: int) -> Dict[str, float]:
    model.prepare_level(level, training=False)
    totals = {"total": 0.0, "effect": 0.0, "vq": 0.0}
    samples = 0
    with torch.no_grad():
        for sample in loader:
            batch_n = batch_size_of(sample)
            losses = model.loss_components(sample, level)
            for key in totals:
                totals[key] += float(losses[key].detach().cpu()) * batch_n
            samples += batch_n
    return average_dict(totals, samples)


def write_csv(rows: List[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    all_fields = []
    seen = set()
    preferred = [
        "model", "seed", "level", "epoch", "train_total", "train_effect",
        "train_vq", "eval_total", "eval_effect", "eval_vq",
    ]
    for field in preferred:
        if any(field in row for row in rows):
            all_fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                all_fields.append(field)
                seen.add(field)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)


def save_growth_events(model, level: int, save_dir: str) -> None:
    events = model.growth_events(level)
    path = os.path.join(save_dir, "growth_events_level%d.json" % level)
    with open(path, "w") as handle:
        json.dump(events, handle, indent=2)


def checkpoint_exists(path: str, level: int, ext: str = "_best") -> bool:
    return all(
        os.path.exists(os.path.join(path, "%s%d%s.ckpt" % (kind, level, ext)))
        for kind in ("encoder", "decoder")
    )


def train_level(
    model,
    opts: Dict,
    model_name: str,
    seed: int,
    level: int,
    rows: List[Dict[str, object]],
    num_workers: int,
) -> None:
    train_loader = make_loader(opts, level, True, seed + 1000 * level, num_workers)
    eval_loader = make_loader(opts, level, False, seed + 2000 * level, num_workers)
    epochs = int(opts["epoch%d" % level])
    best_effect = float("inf")

    print("\n=== %s | seed=%d | level=%d | epochs=%d ===" % (model_name, seed, level, epochs))
    model.print_model(level)

    for epoch in range(1, epochs + 1):
        # Reset augmentation/view RNG independently of model initialization so
        # all bottleneck variants see the same stochastic training stream.
        epoch_seed = int(seed) * 100000 + int(level) * 10000 + int(epoch)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed % (2 ** 32))
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)

        train_metrics = run_training_epoch(model, train_loader, level)
        eval_metrics = run_evaluation_epoch(model, eval_loader, level)
        extra = model.additional_metrics(level)
        row = {
            "model": model_name,
            "seed": seed,
            "level": level,
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_effect": train_metrics["effect"],
            "train_vq": train_metrics["vq"],
            "eval_total": eval_metrics["total"],
            "eval_effect": eval_metrics["effect"],
            "eval_vq": eval_metrics["vq"],
        }
        row.update(extra)
        rows.append(row)

        if eval_metrics["effect"] < best_effect:
            best_effect = eval_metrics["effect"]
            model.save(opts["save"], "_best", level)
            with open(os.path.join(opts["save"], "best_level%d.json" % level), "w") as handle:
                json.dump({
                    "level": level,
                    "epoch": epoch,
                    "eval_effect": best_effect,
                    "additional_metrics": extra,
                }, handle, indent=2)
        model.save(opts["save"], "_last", level)
        write_csv(rows, os.path.join(opts["save"], "metrics.csv"))
        save_growth_events(model, level, opts["save"])

        extras_text = " ".join(
            "%s=%.4g" % (key, value)
            for key, value in extra.items()
            if isinstance(value, (int, float))
        )
        print(
            "level=%d epoch=%d/%d train_effect=%.6f eval_effect=%.6f "
            "train_vq=%.6f %s"
            % (
                level,
                epoch,
                epochs,
                train_metrics["effect"],
                eval_metrics["effect"],
                train_metrics["vq"],
                extras_text,
            )
        )

    print("Best level-%d deterministic weighted MSE: %.8f" % (level, best_effect))


def copy_shared_symbol_labels(source_save: str, destination: str) -> None:
    """Use one shared physical-effect clustering for every bottleneck run."""
    if not source_save:
        return
    for filename in ("label.pt", "effect_names.npy"):
        source = os.path.join(source_save, filename)
        target = os.path.join(destination, filename)
        if os.path.exists(source) and os.path.abspath(source) != os.path.abspath(target):
            shutil.copy2(source, target)
            print("Copied shared %s from %s" % (filename, source))


def invoke_poster_eval(opts_path: str, model_name: str) -> None:
    evaluator = os.path.join(os.path.dirname(os.path.abspath(__file__)), "poster_eval.py")
    command = [sys.executable, evaluator, "-opts", opts_path, "--model", model_name]
    print("\nRunning poster evaluation:\n  " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser("Train controlled DeepSym poster baselines.")
    parser.add_argument("-opts", required=True, help="Base opts.yaml")
    parser.add_argument("--model", choices=tuple(MODEL_MODULES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-dir", required=True, help="Output directory for this exact run")
    parser.add_argument("--level", choices=("1", "2", "both"), default="both")
    parser.add_argument("--stage1-dir", default=None, help="Level-1 checkpoint directory for --level 2")
    parser.add_argument("--stage1-ext", default="_best")
    parser.add_argument("--device", default=None, help="Override opts device; supports auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--skip-poster-eval", action="store_true")
    args = parser.parse_args()

    with open(args.opts, "r") as handle:
        opts = yaml.safe_load(handle)
    if not isinstance(opts, dict):
        raise ValueError("opts file must contain a YAML mapping")

    source_save = str(opts.get("save", ""))
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    opts["source_opts"] = os.path.abspath(args.opts)
    opts["source_save"] = source_save
    opts["save"] = save_dir
    opts["poster_model"] = args.model
    opts["seed"] = int(args.seed)
    opts["time"] = time.asctime(time.localtime())
    opts["device"] = resolve_device(args.device or opts.get("device", "cpu"))

    # Fair defaults. Values in opts.yaml override these.
    opts.setdefault("effect_weights1", [1.0, 1.0, 10.0])
    opts.setdefault("effect_weights2", [1.0, 1.0, 5.0, 1.0, 1.0, 1.0])
    opts.setdefault("vq_commitment_cost", 0.25)
    opts.setdefault("vq_decay", 0.99)
    opts.setdefault("vq_epsilon", 1.0e-5)
    opts.setdefault("vq_num_embeddings1", 2 ** int(opts["code1_dim"]))
    opts.setdefault("vq_num_embeddings2", 2 ** int(opts["code2_dim"]))
    opts.setdefault("dynamic_initial_embeddings", 1)
    opts.setdefault("dynamic_warmup_steps", 200)
    opts.setdefault("dynamic_growth_interval", 100)
    opts.setdefault("dynamic_min_support", 8)
    opts.setdefault("dynamic_min_support_fraction", 0.05)
    opts.setdefault("dynamic_required_checks", 2)
    opts.setdefault("surprise_threshold_1", 1.0)
    opts.setdefault("surprise_threshold_2", 1.0)

    set_global_seed(args.seed, deterministic=not args.non_deterministic)
    copy_shared_symbol_labels(source_save, save_dir)

    resolved_opts_path = os.path.join(save_dir, "opts.yaml")
    with open(resolved_opts_path, "w") as handle:
        yaml.safe_dump(opts, handle, sort_keys=False)
    print(yaml.safe_dump(opts, sort_keys=False))

    module = importlib.import_module(MODEL_MODULES[args.model])
    model = module.EffectRegressorMLP(opts)
    rows: List[Dict[str, object]] = []

    if args.level in ("1", "both"):
        train_level(model, opts, args.model, args.seed, 1, rows, args.num_workers)

    if args.level in ("2", "both"):
        if args.level == "both":
            stage1_dir = save_dir
            stage1_ext = "_best"
        else:
            stage1_dir = args.stage1_dir or source_save
            stage1_ext = args.stage1_ext
        if not checkpoint_exists(stage1_dir, 1, stage1_ext):
            raise FileNotFoundError(
                "Level-1 checkpoint missing in %s with suffix %s" % (stage1_dir, stage1_ext)
            )
        model.load(stage1_dir, stage1_ext, 1)
        model.freeze_level1()
        train_level(model, opts, args.model, args.seed, 2, rows, args.num_workers)

    write_csv(rows, os.path.join(save_dir, "metrics.csv"))
    with open(os.path.join(save_dir, "run_manifest.json"), "w") as handle:
        json.dump(
            {
                "model": args.model,
                "seed": args.seed,
                "save_dir": save_dir,
                "levels": args.level,
                "device": opts["device"],
                "deterministic": not args.non_deterministic,
            },
            handle,
            indent=2,
        )

    if not args.skip_poster_eval and checkpoint_exists(save_dir, 1) and checkpoint_exists(save_dir, 2):
        invoke_poster_eval(resolved_opts_path, args.model)
    elif not args.skip_poster_eval:
        print("Poster evaluation skipped because both level checkpoints are not present yet.")


if __name__ == "__main__":
    main()