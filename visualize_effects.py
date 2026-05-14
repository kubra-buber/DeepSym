import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import data
from models import EffectRegressorMLP
from blocks import build_encoder

parser = argparse.ArgumentParser("Visualize Predicted vs Actual Effects per Dimension.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)

# --- ARCHITECTURE SNIFFER ---
ckpt_path = os.path.join(opts["save"], "encoder1_best.ckpt")
state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
is_vq = any("cluster_size" in k for k in state_dict.keys())

if not is_vq:
    print(f"Detected original architecture in {opts['save']}. Reverting VQ layers...")
    orig_enc1 = build_encoder(opts, 1).to(device)
    orig_enc2 = build_encoder(opts, 2).to(device)
    model.encoder1[-1] = orig_enc1[-1]
    model.encoder2[-1] = orig_enc2[-1]
# ----------------------------

model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.decoder1.eval()
model.encoder2.eval()
model.decoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
dataset1 = data.SingleObjectData(transform=transform)
loader1 = torch.utils.data.DataLoader(dataset1, batch_size=200, shuffle=False)
dataset2 = data.PairedObjectData(transform=transform)
loader2 = torch.utils.data.DataLoader(dataset2, batch_size=200, shuffle=False)

# --- LEVEL 1 PREDICTIONS ---
print("Running Level 1 Predictions...")
true_1, pred_1 = [], []
with torch.no_grad():
    for sample in loader1:
        obs, action, effect = sample["observation"].to(device), sample["action"].to(device), sample["effect"].to(device)
        h = model.encoder1(obs)
        eff_pred = model.decoder1(torch.cat([h, action], dim=-1))
        true_1.append(effect)
        pred_1.append(eff_pred)

# Keep as (N, D) arrays instead of flattening
true_1 = torch.cat(true_1, dim=0).numpy()
pred_1 = torch.cat(pred_1, dim=0).numpy()

# --- LEVEL 2 PREDICTIONS ---
print("Running Level 2 Predictions...")
true_2, pred_2 = [], []
with torch.no_grad():
    for sample in loader2:
        obs, effect = sample["observation"].to(device), sample["effect"].to(device)
        h1 = model.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3])).reshape(obs.shape[0], -1)
        h2 = model.encoder2(obs)
        eff_pred = model.decoder2(torch.cat([h1, h2], dim=-1))
        true_2.append(effect)
        pred_2.append(eff_pred)

# Keep as (N, D) arrays
true_2 = torch.cat(true_2, dim=0).numpy()
pred_2 = torch.cat(pred_2, dim=0).numpy()

# --- PLOTTING FUNCTION ---
def plot_dimensions(true_arr, pred_arr, title_prefix, color, save_name):
    num_dims = true_arr.shape[1]
    cols = 3 if num_dims > 4 else 2
    rows = int(np.ceil(num_dims / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.patch.set_facecolor('black')
    fig.suptitle(f"{title_prefix}: Predicted vs Actual per Dimension", color='white', fontsize=16, y=1.02)
    
    axes = axes.flatten() if num_dims > 1 else [axes]
    
    for i in range(num_dims):
        ax = axes[i]
        ax.set_facecolor('#111111')
        ax.scatter(true_arr[:, i], pred_arr[:, i], alpha=0.3, c=color, s=15, edgecolors='none')
        
        # Perfect prediction line
        min_val = min(true_arr[:, i].min(), pred_arr[:, i].min())
        max_val = max(true_arr[:, i].max(), pred_arr[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax.set_title(f"Dimension {i}", color='white', fontsize=12)
        ax.set_xlabel("Actual", color='white', fontsize=10)
        ax.set_ylabel("Predicted", color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.grid(color='gray', linestyle=':', alpha=0.5)

    # Hide extra empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_filename = os.path.join(opts["save"], save_name)
    plt.savefig(output_filename, bbox_inches='tight', dpi=150, facecolor='black')
    print(f"Saved: {output_filename}")

# Generate the plots
plot_dimensions(true_1, pred_1, "Level 1", 'cyan', "effects_dim_level1.png")
plot_dimensions(true_2, pred_2, "Level 2", 'magenta', "effects_dim_level2.png")