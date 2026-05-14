import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import data

parser = argparse.ArgumentParser("Visualize K-Means Physics Clusters.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
save_dir = opts["save"]

print("Loading dataset and cluster labels...")
# Load the raw 6D physics data
transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
effects = trainset.effect.numpy() # Shape: [2500, 6]

# Load the labels you assigned via cluster.py
try:
    labels = torch.load(os.path.join(save_dir, "label.pt")).numpy()
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"))
except FileNotFoundError:
    print(f"Error: label.pt or effect_names.npy not found in {save_dir}. Run cluster.py first!")
    exit()

print("Running PCA projection...")
pca = PCA(n_components=2)
effects_pca = pca.fit_transform(effects)

print("Running t-SNE projection (this might take 10-20 seconds)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
effects_tsne = tsne.fit_transform(effects)

print("Generating plots...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('black')

# Create a distinct color map for the clusters
colors = plt.cm.get_cmap('tab10', len(effect_names))

# --- Plot 1: PCA ---
ax = axes[0]
ax.set_facecolor('#111111')
for i in range(len(effect_names)):
    idx = labels == i
    ax.scatter(effects_pca[idx, 0], effects_pca[idx, 1], label=effect_names[i], color=colors(i), s=15, alpha=0.7)
ax.set_title("PCA Projection (Linear Variance)", color='white', fontsize=14)
ax.tick_params(colors='white')
ax.grid(color='gray', linestyle=':', alpha=0.3)

# --- Plot 2: t-SNE ---
ax = axes[1]
ax.set_facecolor('#111111')
for i in range(len(effect_names)):
    idx = labels == i
    ax.scatter(effects_tsne[idx, 0], effects_tsne[idx, 1], label=effect_names[i], color=colors(i), s=15, alpha=0.7)
ax.set_title("t-SNE Projection (Cluster Separability)", color='white', fontsize=14)
ax.tick_params(colors='white')
ax.grid(color='gray', linestyle=':', alpha=0.3)

# Add a shared legend at the bottom
handles = [mpatches.Patch(color=colors(i), label=effect_names[i]) for i in range(len(effect_names))]
fig.legend(handles=handles, loc='lower center', ncol=len(effect_names), facecolor='black', labelcolor='white', fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out_file = os.path.join(save_dir, "cluster_visualization.png")
plt.savefig(out_file, bbox_inches='tight', facecolor='black', dpi=150)
print(f"Success! Saved cluster visualization to: {out_file}")