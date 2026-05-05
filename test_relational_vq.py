import argparse
import os
import torch
import yaml
import random
import matplotlib.pyplot as plt
import data
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("Test Relational VQ Logic")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str, required=True)
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

model.encoder2.eval()

# Load Data
transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2000, shuffle=True)

# Extract objects and flatten them into a single pool of images
sample = next(iter(loader))
# Shape: (Categories, Instances, Sizes, View1, View2, H, W)
objects = sample["observation"].reshape(-1, 1, 42, 42) 

print(f"Loaded a pool of {len(objects)} objects. Testing relational pairs...")

predicted_cat_0 = []
predicted_cat_1 = []

with torch.no_grad():
    # Test 500 random pairs to ensure we have enough for both categories
    for _ in range(500):
        # Pick two random objects
        idx_top = random.randint(0, len(objects)-1)
        idx_bottom = random.randint(0, len(objects)-1)
        
        top_obj = objects[idx_top].unsqueeze(0)    # Shape: (1, 1, 42, 42)
        bottom_obj = objects[idx_bottom].unsqueeze(0) # Shape: (1, 1, 42, 42)
        
        # Concatenate on channel dimension for the encoder (Shape: 1, 2, 42, 42)
        xy = torch.cat([top_obj, bottom_obj], dim=1)
        
        # Run through Level 2 Relational Encoder
        raw_rel = model.encoder2(xy)
        
        # Extract VQ Index
        try:
            rel_idx = int(model.encoder2[-1].get_indices(raw_rel)[0].item())
        except AttributeError:
            # Fallback for Gumbel Baseline
            rel_val = raw_rel[0, 0]
            rel_idx = 0 if rel_val == -1 else 1

        # For plotting, concatenate side-by-side horizontally (Shape: 42, 84)
        visual_pair = torch.cat([top_obj[0, 0], bottom_obj[0, 0]], dim=1)
        
        if rel_idx == 0:
            predicted_cat_0.append(visual_pair)
        else:
            predicted_cat_1.append(visual_pair)

print(f"Found {len(predicted_cat_0)} pairs for Cat 0, and {len(predicted_cat_1)} pairs for Cat 1.")

# --- MATPLOTLIB VISUALIZATION ---
# We will plot up to 10 examples for each category
num_examples = 10
fig, axes = plt.subplots(2, num_examples, figsize=(20, 5))
fig.suptitle("Relational VQ (Level 2) Consistency Test\n[Left side of each cell = Top Object | Right side = Bottom Object]", fontsize=16, fontweight='bold', color='white')
fig.patch.set_facecolor('black')
plt.subplots_adjust(wspace=0.1, hspace=0.3)

for row, cat_list in enumerate([predicted_cat_0, predicted_cat_1]):
    # Shuffle to get a random sample of the results
    random.shuffle(cat_list) 
    
    for col in range(num_examples):
        ax = axes[row, col]
        ax.axis('off')
        
        if col == 0:
            ax.set_ylabel(f"VQ Index {row}", fontsize=14, fontweight='bold', color='white', rotation=0, labelpad=40, va='center')
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        
        if col < len(cat_list):
            img = cat_list[col].numpy()
            ax.imshow(img, cmap='plasma')
        else:
            # If the network collapsed and couldn't find 10 examples for a category
            ax.text(0.5, 0.5, "NO DATA", color='red', ha='center', va='center', transform=ax.transAxes)

output_filename = os.path.join(args.ckpt, "relational_vq_test.png")
plt.savefig(output_filename, bbox_inches='tight', dpi=150, facecolor='black')
print(f"Saved visual test to: {output_filename}")