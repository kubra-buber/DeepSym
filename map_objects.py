import os
import torch
import yaml
import torchvision
import data
from models import EffectRegressorMLP

opts = yaml.safe_load(open("save/myrun_vq_weighted/opts.yaml", "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.encoder1.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
# Extract 50 base objects
X = torch.load("data/img/obs_prev_z.pt").reshape(5, 10, 3, 4, 4, 42, 42)[:, :, 0, 2, 2].reshape(-1, 1, 42, 42)
Y = torch.empty(50, 1, opts["size"], opts["size"])
for i in range(50):
    Y[i] = transform(X[i])

with torch.no_grad():
    raw_cat = model.encoder1(Y)
    indices = model.encoder1[-1].get_indices(raw_cat)

grouped_imgs = {0: [], 1: [], 2: [], 3: []}
for i in range(50):
    idx = indices[i].item()
    if len(grouped_imgs[idx]) < 10:
        grouped_imgs[idx].append(Y[i])

print("Generating visual mappings...")
for obj_idx, imgs in grouped_imgs.items():
    if imgs:
        grid = torchvision.utils.make_grid(torch.stack(imgs), nrow=10, normalize=True)
        out_name = f"objtype{obj_idx}_visual.png"
        torchvision.utils.save_image(grid, out_name)
        print(f"Saved {out_name}! Open this file to see what objtype{obj_idx} is.")