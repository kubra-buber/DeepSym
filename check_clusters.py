import torch
import numpy as np
import yaml
import data
import os

# We want to inspect the stable1 folder
folder = "save/stable1"

# Load the saved names and cluster assignments
names = np.load(os.path.join(folder, "effect_names.npy"))
labels = torch.load(os.path.join(folder, "label.pt"))

# Load the dataset to get the actual ground-truth physics
opts = yaml.safe_load(open(os.path.join(folder, "opts.yaml"), "r"))
transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)

effects = trainset.effect
mu = trainset.eff_mu
std = trainset.eff_std

print("=== STABLE1 CLUSTER NAMES & EFFECTS ===")
for i, name in enumerate(names):
    mask = (labels == i)
    if mask.sum() > 0:
        cluster_eff = effects[mask]
        # Un-normalize back to real physics values
        cluster_eff = cluster_eff * (std + 1e-6) + mu
        centroid = cluster_eff.mean(dim=0)
        
        print(f"Cluster [{i}] Name: '{name}' | Datapoints: {mask.sum()}")
        # We usually care most about the Z-axis (height) changes, which are index 2 and 5
        print(f"    Avg Physics (6D): {centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}, {centroid[3]:.3f}, {centroid[4]:.3f}, {centroid[5]:.3f}\n")
    else:
        print(f"Cluster [{i}] Name: '{name}' | (NO DATA)\n")