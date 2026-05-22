import argparse
import os
import torch
import torchvision
import yaml
import matplotlib.pyplot as plt
import data
import utils
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)

# Robust Loading Block
try:
    model.load(args.ckpt, "_best")
except TypeError:
    model.load(args.ckpt, "_best", 1)

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
# Changed shuffle to False to cleanly extract the 50 base objects
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)

sample = next(iter(loader))
# Extract exactly 50 base images
objects = sample["observation"].reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])
objects = objects[:, :, 0].reshape(-1, 1, 42, 42)

# Use a dynamic dictionary instead of a hardcoded list of 4
colored = {}

model.encoder1.eval()
with torch.no_grad():
    for i in range(len(objects)):
        img = objects[i].reshape(1, 1, 42, 42)
        c = model.encoder1(img)
        
        # --- THE VQ FIX: Extract the integer class index ---
        try:
            # If using VQLayer, grab the codebook index
            cat = int(model.encoder1[-1].get_indices(c)[0].item())
        except AttributeError:
            # If using original Gumbel STLayer, calculate decimal from binary
            cat = int(utils.binary_to_decimal(c[0]))
        # ---------------------------------------------------
        
        # Dynamically add new clusters as they are discovered
        if cat not in colored:
            colored[cat] = []
        
        colored[cat].append(objects[i].clone())

# Find the max number of objects in any single cluster to pad our image grid
keys = sorted(list(colored.keys()))
max_len = max([len(colored[k]) for k in keys]) if keys else 0

print(f"Dynamic VQ discovered {len(keys)} active clusters!")
for k in keys:
    print(f" -> Cluster {k} contains {len(colored[k])} objects.")

grid_rows = []
for k in keys:
    imgs = colored[k]
    # Pad the row with black/blank images so every row is the same width
    while len(imgs) < max_len:
        imgs.append(torch.zeros_like(imgs[0]))
    grid_rows.append(torch.stack(imgs))

if grid_rows:
    colored_tensor = torch.stack(grid_rows)
    colored_tensor = colored_tensor.reshape(-1, 42, 42)
    
    # Normalize for colormap
    t_min = colored_tensor.min()
    t_max = colored_tensor.max()
    if t_max > t_min:
        colored_tensor = (colored_tensor - t_min) / (t_max - t_min)
        
    cm = plt.cm.plasma
    colored_tensor = torch.tensor(cm(colored_tensor.numpy()), dtype=torch.float).permute(0, 3, 1, 2)[:, :3]

    # Save image where each row represents one VQ cluster
    out_name = "colored-objects_vq.png"
    torchvision.utils.save_image(colored_tensor, out_name, nrow=max_len)
    print(f"Saved visual mapping to {out_name}")
else:
    print("No objects found to visualize.")