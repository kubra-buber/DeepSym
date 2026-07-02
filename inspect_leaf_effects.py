import os
import pickle
import torch
import yaml
import numpy as np
from collections import Counter

opts = yaml.safe_load(open("opts.yaml"))
save = opts["save"]

category = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
label = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().astype(int)
effect_names = np.load(os.path.join(save, "effect_names.npy"))

raw_effect = torch.load("data/img/delta_pix_3.pt", map_location="cpu")

tree_path = os.path.join(save, "tree_vq_onehot.pkl")
if not os.path.exists(tree_path):
    tree_path = os.path.join(save, "tree.pkl")

with open(tree_path, "rb") as f:
    tree = pickle.load(f)

leaves = tree.apply(category)
leaf_id = 57
idxs = np.where(leaves == leaf_id)[0]

print("idx | label | raw delta_pix_3")
for idx in idxs:
    print(f"{idx:4d} | {effect_names[label[idx]]:10s} | {raw_effect[idx].tolist()}")

print("\nMean raw effect by label:")
for lab in sorted(set(label[idxs])):
    lab_idxs = [i for i in idxs if label[i] == lab]
    mean_eff = raw_effect[lab_idxs].float().mean(dim=0)
    print(f"{effect_names[lab]:10s} n={len(lab_idxs):3d} mean={mean_eff.tolist()}")
