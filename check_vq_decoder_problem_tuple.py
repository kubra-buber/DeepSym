import os
import argparse
import torch
import yaml
import numpy as np
from collections import defaultdict

import data
from models import EffectRegressorMLP

def idx_of(t0, s0, t1, s1):
    return t0 * 500 + s0 * 50 + t1 * 10 + s1

def decode_vq(cat):
    b = cat[:, 0:4].argmax(axis=1)
    a = cat[:, 4:8].argmax(axis=1)
    r = cat[:, 8:10].argmax(axis=1)
    return b, a, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-opts", required=True)
    ap.add_argument("--slot0-type", type=int, required=True)
    ap.add_argument("--slot1-type", type=int, required=True)
    args = ap.parse_args()

    opts = yaml.safe_load(open(args.opts))
    opts["device"] = "cpu"
    save = opts["save"]

    transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)

    # Rebuild the same ordered 50x50 pair tensor as original save_cat.py
    X = torch.load("data/img/obs_prev_z.pt", map_location="cpu")
    X = X.reshape(5, 10, 3, 4, 4, 42, 42)
    X = X[:, :, 0, 2, 2]
    X = X.reshape(-1, 1, 42, 42)

    Y = torch.empty(X.shape[0], 1, opts["size"], opts["size"])
    for i in range(X.shape[0]):
        Y[i] = transform(X[i])

    left_img = Y.repeat_interleave(Y.shape[0], 0)
    right_img = Y.repeat(Y.shape[0], 1, 1, 1)
    pair_obs = torch.cat([left_img, right_img], dim=1)

    raw = torch.load("data/img/delta_pix_3.pt", map_location="cpu").float()
    labels = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))
    cat = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
    cb, ca, cr = decode_vq(cat)

    model = EffectRegressorMLP(opts)
    model.load(save, "_best", 1)
    model.load(save, "_best", 2)
    model.eval()

    with torch.no_grad():
        # Try the model's normal forward path for paired observations.
        # If your model API differs, this may error; send the error if it does.
        pred = model(pair_obs)

    pred = pred.detach().cpu().float()

    # Some implementations return tuple/list. Keep first tensor if needed.
    if isinstance(pred, (tuple, list)):
        pred = pred[0].detach().cpu().float()

    print("=" * 100)
    print(f"Pair slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print("Rows = slot0_size 9..0, Cols = slot1_size 0..9")
    print("cell = label:tuple: raw_dz0 -> pred_dz0")
    print("=" * 100)

    mse_by_tuple = defaultdict(list)
    dzerr_by_tuple = defaultdict(list)

    for s0 in reversed(range(10)):
        cells = []
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            tup = f"{cb[idx]}{ca[idx]}r{cr[idx]}"
            lab = str(effect_names[labels[idx]])

            raw_i = raw[idx]
            pred_i = pred[idx]

            mse = torch.mean((pred_i - raw_i) ** 2).item()
            dzerr = abs(float(pred_i[2] - raw_i[2]))

            mse_by_tuple[tup].append(mse)
            dzerr_by_tuple[tup].append(dzerr)

            cells.append(f"{lab[0].upper()}:{tup}:{raw_i[2]:.3f}->{pred_i[2]:.3f}")
        print(f"s0={s0}: " + " | ".join(cells))

    print("\nMean prediction error by tuple:")
    for tup in sorted(mse_by_tuple):
        print(
            f"{tup}: n={len(mse_by_tuple[tup])} "
            f"mse={np.mean(mse_by_tuple[tup]):.6f} "
            f"abs_dz0_err={np.mean(dzerr_by_tuple[tup]):.6f}"
        )

if __name__ == "__main__":
    main()
