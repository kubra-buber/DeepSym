import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import data
from models import EffectRegressorMLP
from blocks import build_encoder

parser = argparse.ArgumentParser("Compare Predictions: VQ vs Stable vs Actual")
parser.add_argument("-opt1", help="option file for Model 1 (e.g. myrun_vq3)", type=str, required=True)
parser.add_argument("-opt2", help="option file for Model 2 (e.g. stable1)", type=str, required=True)
parser.add_argument("-start", help="Start index to plot", type=int, default=0)
parser.add_argument("-end", help="End index to plot", type=int, default=150)
args = parser.parse_args()

def load_model_safely(opt_path):
    opts = yaml.safe_load(open(opt_path, "r"))
    opts["device"] = "cpu"
    device = torch.device(opts["device"])
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
    return model, opts, device

print("Loading models...")
model1, opts1, device = load_model_safely(args.opt1)
model2, opts2, _ = load_model_safely(args.opt2)

print("Loading data...")
transform = data.default_transform(size=opts1["size"], affine=False, mean=0.279, std=0.0094)
dataset1 = data.SingleObjectData(transform=transform)
loader1 = torch.utils.data.DataLoader(dataset1, batch_size=200, shuffle=False)
dataset2 = data.PairedObjectData(transform=transform)
loader2 = torch.utils.data.DataLoader(dataset2, batch_size=200, shuffle=False)

def get_predictions(model, loader, level):
    preds, actuals = [], []
    with torch.no_grad():
        for sample in loader:
            obs = sample["observation"].to(device)
            effect = sample["effect"].to(device)
            
            if level == 1:
                action = sample["action"].to(device)
                h = model.encoder1(obs)
                eff_pred = model.decoder1(torch.cat([h, action], dim=-1))
            else:
                h1 = model.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3])).reshape(obs.shape[0], -1)
                h2 = model.encoder2(obs)
                eff_pred = model.decoder2(torch.cat([h1, h2], dim=-1))
                
            actuals.append(effect)
            preds.append(eff_pred)
    return torch.cat(actuals, dim=0).numpy(), torch.cat(preds, dim=0).numpy()

print("Generating predictions...")
actual_1, pred1_1 = get_predictions(model1, loader1, level=1)
_, pred2_1 = get_predictions(model2, loader1, level=1)
actual_2, pred1_2 = get_predictions(model1, loader2, level=2)
_, pred2_2 = get_predictions(model2, loader2, level=2)

def plot_comparison(actual, pred1, pred2, title_prefix, save_name):
    num_dims = actual.shape[1]
    
    # Restrict to user-defined range
    start_idx = max(0, args.start)
    end_idx = min(args.end, actual.shape[0])
    x = np.arange(start_idx, end_idx)
    
    fig, axes = plt.subplots(num_dims, 1, figsize=(15, 4 * num_dims))
    fig.patch.set_facecolor('black')
    fig.suptitle(f"{title_prefix}: Actual vs Predictions (Data Points {start_idx} to {end_idx-1})", color='white', fontsize=18, y=0.99)
    
    axes = axes.flatten() if num_dims > 1 else [axes]
    
    for i in range(num_dims):
        ax = axes[i]
        ax.set_facecolor('#111111')
        
        ax.plot(x, actual[start_idx:end_idx, i], color='white', alpha=0.3, label='Actual (Line)', zorder=1)
        ax.scatter(x, actual[start_idx:end_idx, i], color='white', s=25, label='Actual Ground Truth', zorder=2)
        ax.scatter(x, pred1[start_idx:end_idx, i], color='cyan', s=15, alpha=0.8, label=f'{opts1["save"]} (VQ3)', zorder=3)
        ax.scatter(x, pred2[start_idx:end_idx, i], color='magenta', s=15, alpha=0.8, label=f'{opts2["save"]} (Stable1)', zorder=4)
        
        ax.set_title(f"Dimension {i}", color='white', fontsize=14)
        ax.set_xlabel("Data Point Index", color='white', fontsize=10)
        ax.set_ylabel("Effect Value", color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.grid(color='gray', linestyle=':', alpha=0.3)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', dpi=150, facecolor='black')
    print(f"Saved: {save_name}")

plot_comparison(actual_1, pred1_1, pred2_1, "Level 1", "compare_level1.png")
plot_comparison(actual_2, pred1_2, pred2_2, "Level 2", "compare_level2.png")