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

    buckets = defaultdict(list)

    for s0 in range(10):
        for s1 in range(10):
            idx = idx_of(args.slot0_type, s0, args.slot1_type, s1)
            tup = f"{cb[idx]}{ca[idx]}r{cr[idx]}"
            lab = str(effect_names[labels[idx]])
            buckets[(tup, lab)].append((idx, s0, s1, float(target[idx,2]), float(pred[idx,2])))

    print("=" * 100)
    print(f"slot0_type={args.slot0_type}, slot1_type={args.slot1_type}")
    print("Stats grouped by tuple and label. z0 is normalized.")
    print("=" * 100)

    for key in sorted(buckets):
        vals = buckets[key]
        target_z = np.array([v[3] for v in vals])
        pred_z = np.array([v[4] for v in vals])

        print(
            f"{key[0]:5s} {key[1]:10s} n={len(vals):3d} "
            f"target_z mean={target_z.mean(): .4f} std={target_z.std(): .4f} "
            f"pred_z mean={pred_z.mean(): .4f} std={pred_z.std(): .4f} "
            f"abs_err={np.mean(np.abs(pred_z-target_z)): .4f}"
        )

    print("\nProblem tuple 12r1 samples:")
    for key in sorted(buckets):
        if key[0] != "12r1":
            continue
        for idx, s0, s1, tz, pz in buckets[key]:
            print(f"  {key[1]:10s} idx={idx:4d} s0={s0} s1={s1} target_z={tz: .4f} pred_z={pz: .4f}")

if __name__ == "__main__":
    main()
