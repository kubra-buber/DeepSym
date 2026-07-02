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
    dataset = data.PairedObjectData(transform=transform)

    obs_all, eff_all = [], []
    for i in range(len(dataset)):
        s = dataset[i]
        obs_all.append(s["observation"])
        eff_all.append(s["effect"])

    obs = torch.stack(obs_all).float()
    target = torch.stack(eff_all).float()

    labels = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().reshape(-1).astype(int)
    effect_names = np.load(os.path.join(save, "effect_names.npy"))

    cat = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
    cb, ca, cr = decode_vq(cat)

    model = EffectRegressorMLP(opts)
    model.load(save, "_best", 1)
    model.load(save, "_best", 2)

    model.encoder1.eval()
    model.encoder2.eval()
    model.decoder1.eval()
    model.decoder2.eval()

    with torch.no_grad():
        h1 = model.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3]))
        h1 = h1.reshape(obs.shape[0], -1)
        h2 = model.encoder2(obs)
        pred = model.decoder2(torch.cat([h1, h2], dim=-1))

    print("=" * 100)
    print(f"Pair slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print("Rows = slot0_size 9..0, Cols = slot1_size 0..9")
    print("cell = label:tuple target_z0->pred_z0")
    print("=" * 100)

    mse_by_tuple = defaultdict(list)
    dzerr_by_tuple = defaultdict(list)

    for s0 in reversed(range(10)):
        cells = []
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            tup = f"{cb[idx]}{ca[idx]}r{cr[idx]}"
            lab = str(effect_names[labels[idx]])

            mse = torch.mean((pred[idx] - target[idx]) ** 2).item()
            dzerr = abs(float(pred[idx, 2] - target[idx, 2]))

            mse_by_tuple[tup].append(mse)
            dzerr_by_tuple[tup].append(dzerr)

            cells.append(f"{lab[0].upper()}:{tup}:{target[idx,2]:.3f}->{pred[idx,2]:.3f}")

        print(f"s0={s0}: " + " | ".join(cells))

    print("\nMean normalized prediction error by tuple:")
    for tup in sorted(mse_by_tuple):
        print(
            f"{tup}: n={len(mse_by_tuple[tup])} "
            f"mse={np.mean(mse_by_tuple[tup]):.6f} "
            f"abs_z0_err={np.mean(dzerr_by_tuple[tup]):.6f}"
        )

if __name__ == "__main__":
    main()
