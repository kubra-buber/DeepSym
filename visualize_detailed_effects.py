import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import data
from models import EffectRegressorMLP
from blocks import build_encoder

parser = argparse.ArgumentParser("Visualize Input Images vs Effects")
parser.add_argument("-opt1", help="option file for VQ3", type=str, required=True)
parser.add_argument("-opt2", help="option file for Stable1", type=str, required=True)
parser.add_argument("-start", help="Starting index for the 10 data points", type=int, default=0)
args = parser.parse_args()

def load_model_safely(opt_path):
    opts = yaml.safe_load(open(opt_path, "r"))
    opts["device"] = "cpu"
    device = torch.device("cpu")
    model = EffectRegressorMLP(opts)
    
    ckpt_path = os.path.join(opts["save"], "encoder1_best.ckpt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    is_vq = any("cluster_size" in k for k in state_dict.keys())
    
    if not is_vq:
        orig_enc1 = build_encoder(opts, 1).to(device)
        orig_enc2 = build_encoder(opts, 2).to(device)
        model.encoder1[-1] = orig_enc1[-1]
        model.encoder2[-1] = orig_enc2[-1]
        
    model.load(opts["save"], "_best", 1)
    model.load(opts["save"], "_best", 2)
    model.encoder1.eval()
    model.decoder1.eval()
    model.encoder2.eval()
    model.decoder2.eval()
    return model, opts

print("Loading models...")
model1, opts1 = load_model_safely(args.opt1)
model2, opts2 = load_model_safely(args.opt2)

print("Loading specific dataset indices...")
transform = data.default_transform(size=opts1["size"], affine=False, mean=0.279, std=0.0094)
dataset2 = data.PairedObjectData(transform=transform)

# Extract exactly 10 samples starting from args.start
start_idx = max(0, args.start)
end_idx = min(start_idx + 10, len(dataset2))

obs_list = []
actuals_list = []
for i in range(start_idx, end_idx):
    sample = dataset2[i]
    obs_list.append(sample["observation"])
    actuals_list.append(sample["effect"])

obs = torch.stack(obs_list)
actuals = torch.stack(actuals_list).numpy()

try:
    relations = torch.load("data/relations.pt", map_location='cpu')
    obj_names = np.load("data/obj_names.npy")
    has_names = True
except Exception:
    has_names = False

print("Generating predictions...")
with torch.no_grad():
    h1_1 = model1.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3])).reshape(obs.shape[0], -1)
    h2_1 = model1.encoder2(obs)
    preds1 = model1.decoder2(torch.cat([h1_1, h2_1], dim=-1)).numpy()
    
    h1_2 = model2.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3])).reshape(obs.shape[0], -1)
    h2_2 = model2.encoder2(obs)
    preds2 = model2.decoder2(torch.cat([h1_2, h2_2], dim=-1)).numpy()

print("Generating images...")
import matplotlib as mpl
mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'

num_samples = end_idx - start_idx
fig, axes = plt.subplots(num_samples, 3, figsize=(18, 2.5 * num_samples))
fig.patch.set_facecolor('#111111')
fig.suptitle(f"Data Points {start_idx} to {end_idx - 1}: Input Depth vs 6D Effect", color='white', fontsize=20, y=0.99)

# Handle edge case where there's only 1 sample left at the end of the dataset
if num_samples == 1:
    axes = np.expand_dims(axes, axis=0)

for row in range(num_samples):
    true_idx = start_idx + row
    
    if has_names:
        top_obj = obj_names[relations[true_idx, 0]]
        bot_obj = obj_names[relations[true_idx, 1]]
    else:
        top_obj, bot_obj = "Unknown", "Unknown"

    ax_top = axes[row, 0]
    ax_top.imshow(obs[row, 0], cmap='plasma')
    ax_top.set_title(f"[{true_idx}] Top: {top_obj}", color='cyan', fontsize=14)
    ax_top.axis('off')

    ax_bot = axes[row, 1]
    ax_bot.imshow(obs[row, 1], cmap='plasma')
    ax_bot.set_title(f"[{true_idx}] Bottom: {bot_obj}", color='magenta', fontsize=14)
    ax_bot.axis('off')

    ax_bar = axes[row, 2]
    ax_bar.set_facecolor('#222222')
    x = np.arange(6)
    width = 0.25

    ax_bar.bar(x - width, actuals[row], width, label='Actual', color='white', alpha=0.9)
    ax_bar.bar(x, preds1[row], width, label='VQ3', color='cyan', alpha=0.8)
    ax_bar.bar(x + width, preds2[row], width, label='Stable1', color='magenta', alpha=0.8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['X(Top)', 'Y(Top)', 'Z(Top)', 'X(Bot)', 'Y(Bot)', 'Z(Bot)'])
    ax_bar.grid(color='gray', linestyle=':', alpha=0.3)
    
    if row == 0:
        ax_bar.legend(loc='upper right')

plt.tight_layout()
out_name = f"detailed_view_{start_idx}_to_{end_idx - 1}.png"
plt.savefig(out_name, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=150)
print(f"Saved {out_name}")