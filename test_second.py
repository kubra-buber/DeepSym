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

# Robust Loading Block
try:
    model.load(args.ckpt, "_best")
except TypeError:
    model.load(args.ckpt, "_best", 1)
    model.load(args.ckpt, "_best", 2)

model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
trainset.train = False
loader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True)

sample = next(iter(loader))["observation"]

with torch.no_grad():
    codes = model.encoder2(sample)
    
    # --- THE VQ FIX: Extract the integer class index ---
    try:
        # If using VQLayer, grab the codebook index
        indices = model.encoder2[-1].get_indices(codes)
        display_codes = indices.numpy()
    except AttributeError:
        # If using original Gumbel STLayer, display the binary array
        display_codes = codes.numpy()
    # ---------------------------------------------------

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
for i in range(6):
    for j in range(6):
        idx = i * 6 + j
        ax[i, j].imshow(sample[idx].permute(1, 0, 2).reshape(sample.shape[3], sample.shape[3]*2)*0.0094+0.279)
        ax[i, j].axis("off")
        ax[i, j].set_title(str(display_codes[idx]))

# output_pdf = os.path.join(args.ckpt, "paired.pdf")
pp = PdfPages("save/run3/paired.pdf")
pp.savefig(fig)
pp.close()