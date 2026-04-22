import argparse
import os
import torch
import yaml
import data
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("Evaluate Mean Effect Prediction Error (MSE).")
parser.add_argument("-ckpt", help="checkpoint folder path (e.g., save/myrun_end)", type=str, required=True)
args = parser.parse_args()

# Load options
file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"  # Evaluate on CPU for simplicity

# Initialize model
model = EffectRegressorMLP(opts)

# Smart loading: Handles both the End-to-End model and the Original Progressive model
try:
    # Try the end-to-end loading signature (takes 2 arguments)
    model.load(args.ckpt, "_best")
    print(f"Loaded End-to-End model from: {args.ckpt}")
except TypeError:
    # Fallback to the progressive loading signature (takes 3 arguments)
    model.load(args.ckpt, "_best", 1)
    model.load(args.ckpt, "_best", 2)
    print(f"Loaded Progressive Baseline model from: {args.ckpt}")

# Set all sub-networks to evaluation mode (disables dropout, batchnorm updates, etc.)
model.encoder1.eval()
model.decoder1.eval()
model.encoder2.eval()
model.decoder2.eval()

# --- Evaluate Level 1 (Single Object) ---
transform1 = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
dataset1 = data.SingleObjectData(transform=transform1)
loader1 = torch.utils.data.DataLoader(dataset1, batch_size=256, shuffle=False)

total_loss1 = 0.0
total_samples1 = 0

with torch.no_grad():
    for sample in loader1:
        # We can use the model's built-in loss function to calculate MSE
        batch_loss = model.loss1(sample).item()
        batch_size = sample["observation"].size(0)
        
        total_loss1 += batch_loss * batch_size
        total_samples1 += batch_size

mse1 = total_loss1 / total_samples1
print(f"\n---> Level 1 (Single Object) MSE: {mse1:.6f}")

# --- Evaluate Level 2 (Paired Objects) ---
transform2 = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
dataset2 = data.PairedObjectData(transform=transform2)
loader2 = torch.utils.data.DataLoader(dataset2, batch_size=256, shuffle=False)

total_loss2 = 0.0
total_samples2 = 0

with torch.no_grad():
    for sample in loader2:
        batch_loss = model.loss2(sample).item()
        batch_size = sample["observation"].size(0)
        
        total_loss2 += batch_loss * batch_size
        total_samples2 += batch_size

mse2 = total_loss2 / total_samples2
print(f"---> Level 2 (Paired Object) MSE: {mse2:.6f}\n")