import os
import argparse
import torch
import yaml
from models import EffectRegressorMLP
import data
import utils

parser = argparse.ArgumentParser("Save categories.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
X = torch.load("data/img/obs_prev_z.pt")
X = X.reshape(5, 10, 3, 4, 4, 42, 42)
X = X[:, :, 0, 2, 2]
X = X.reshape(-1, 1, 42, 42)
B, _, H, W = X.shape
Y = torch.empty(B, 1, opts["size"], opts["size"])

for i in range(B):
    Y[i] = transform(X[i])

# --- EXTRACT CATEGORY 1 (SHAPES) ---
with torch.no_grad():
    raw_cat1 = model.encoder1(Y.to(device))
    try:
        # VQ Logic: Get index and convert to binary array
        idx1 = model.encoder1[-1].get_indices(raw_cat1)
        # Convert each index (e.g., 2) to a binary list (e.g., [1, 0])
        bin_list = [utils.decimal_to_binary(i.item(), length=opts["code1_dim"]) for i in idx1]
        category1 = torch.tensor(bin_list, dtype=torch.int32)
    except AttributeError:
        # Gumbel Fallback
        category1 = raw_cat1.int()

left_img = Y.repeat_interleave(B, 0)
right_img = Y.repeat(B, 1, 1, 1)
concat = torch.cat([left_img, right_img], dim=1)

# --- EXTRACT CATEGORY 2 (RELATIONS) ---
with torch.no_grad():
    raw_cat2 = model.encoder2(concat.to(device))
    try:
        # VQ Logic: Get index and convert to binary array
        idx2 = model.encoder2[-1].get_indices(raw_cat2)
        bin_list2 = [utils.decimal_to_binary(i.item(), length=opts["code2_dim"]) for i in idx2]
        category2 = torch.tensor(bin_list2, dtype=torch.int32)
    except AttributeError:
         # Gumbel Fallback
        category2 = raw_cat2.int()

left_cat = category1.repeat_interleave(B, 0)
right_cat = category1.repeat(B, 1)
category_all = torch.cat([left_cat, right_cat, category2], dim=-1)
torch.save(category_all.cpu(), os.path.join(opts["save"], "category.pt"))
print(f"Saved binary categories to {os.path.join(opts['save'], 'category.pt')}")