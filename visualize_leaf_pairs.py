import os
import pickle
import torch
import yaml
import numpy as np
import torchvision
from PIL import Image, ImageDraw

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

# Load the object image bank used by PairedObjectData.
obs = torch.load("data/img/obs_prev_z.pt", map_location="cpu")
obs = obs.reshape(5, 10, 3, 4, 4, 42, 42)
obs = obs[:, :, 0]  # same camera/view subset used in data.py

tiles = []
texts = []

for idx in idxs[:40]:
    slot0_type = idx // 500
    slot0_size = (idx // 50) % 10
    slot1_type = (idx // 10) % 5
    slot1_size = idx % 10

    img0 = obs[slot0_type, slot0_size, 2, 2]
    img1 = obs[slot1_type, slot1_size, 2, 2]

    # Put slot0 left, slot1 right.
    pair = torch.cat([img0, img1], dim=1).unsqueeze(0)
    pair = (pair - pair.min()) / (pair.max() - pair.min() + 1e-6)
    tiles.append(pair)

    texts.append(
        f"idx={idx} {effect_names[label[idx]]}\\n"
        f"s0=t{slot0_type}/z{slot0_size} s1=t{slot1_type}/z{slot1_size}"
    )

grid = torchvision.utils.make_grid(torch.stack(tiles), nrow=5, padding=4, pad_value=1.0)
grid = (grid.squeeze(0).numpy() * 255).astype(np.uint8)
img = Image.fromarray(grid).convert("RGB")

# Optional: add text below would require bigger canvas; keep filenames printed.
out = os.path.join(save, f"leaf_{leaf_id}_pairs_slot0_left_slot1_right.png")
img.save(out)

txt_out = os.path.join(save, f"leaf_{leaf_id}_pairs_labels.txt")
with open(txt_out, "w") as f:
    for t in texts:
        f.write(t + "\\n")

print(f"Saved image: {out}")
print(f"Saved labels: {txt_out}")
