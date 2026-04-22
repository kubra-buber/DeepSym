import argparse
import os
import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import data
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

model = EffectRegressorMLP(opts)

# --- THE FIX: Robust Loading Block ---
try:
    # Try the end-to-end loading signature
    model.load(args.ckpt, "_best")
except TypeError:
    # Fallback to the progressive loading signature
    model.load(args.ckpt, "_best", 1)
    model.load(args.ckpt, "_best", 2)
# ------------------------------------

model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
trainset.train = False
loader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True)

sample = next(iter(loader))["observation"]

with torch.no_grad():
    codes = model.encoder2(sample)

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
for i in range(6):
    for j in range(6):
        idx = i * 6 + j
        ax[i, j].imshow(sample[idx].permute(1, 0, 2).reshape(sample.shape[3], sample.shape[3]*2)*0.0094+0.279)
        ax[i, j].axis("off")
        ax[i, j].set_title(str(codes[idx].numpy()))

# --- THE FIX: Remove plt.show() and save to checkpoint folder ---
# output_pdf = os.path.join(args.ckpt, "paired.pdf")
pp = PdfPages("paired.pdf")
pp.savefig(fig) # Pass the figure explicitly
pp.close()