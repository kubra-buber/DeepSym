import os
import pickle
import torch
import yaml
import numpy as np
from collections import Counter, defaultdict

opts = yaml.safe_load(open("opts.yaml"))
save = opts["save"]

category = torch.load(os.path.join(save, "category.pt"), map_location="cpu").numpy()
label = torch.load(os.path.join(save, "label.pt"), map_location="cpu").numpy().astype(int)
effect_names = np.load(os.path.join(save, "effect_names.npy"))

tree_path = os.path.join(save, "tree_vq_onehot.pkl")
if not os.path.exists(tree_path):
    tree_path = os.path.join(save, "tree.pkl")

with open(tree_path, "rb") as f:
    tree = pickle.load(f)

leaves = tree.apply(category)

leaf_id = 57
idxs = np.where(leaves == leaf_id)[0]

print(f"save={save}")
print(f"tree={tree_path}")
print(f"leaf={leaf_id}")
print(f"num samples={len(idxs)}")

print("\nLabel distribution:")
cnt = Counter(label[idxs])
for k, v in sorted(cnt.items()):
    print(f"  {effect_names[k]:10s} {v:4d} p={v/len(idxs):.6f}")

print("\nSamples:")
print("idx | label | slot0_type slot0_size | slot1_type slot1_size")
for idx in idxs:
    # This is the PairedObjectData indexing convention.
    slot0_type = idx // 500
    slot0_size = (idx // 50) % 10
    slot1_type = (idx // 10) % 5
    slot1_size = idx % 10
    print(
        f"{idx:4d} | {effect_names[label[idx]]:10s} | "
        f"{slot0_type:2d} {slot0_size:2d} | "
        f"{slot1_type:2d} {slot1_size:2d}"
    )

print("\nDistribution by physical slot pair:")
pair_cnt = defaultdict(Counter)
for idx in idxs:
    slot0_type = idx // 500
    slot1_type = (idx // 10) % 5
    pair_cnt[(slot0_type, slot1_type)][effect_names[label[idx]]] += 1

for pair, c in sorted(pair_cnt.items()):
    total = sum(c.values())
    print(f"  slot0_type={pair[0]} slot1_type={pair[1]} total={total}: {dict(c)}")
