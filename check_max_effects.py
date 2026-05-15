import torch
import data
import yaml

# We just need any opts file to initialize the dataset properly
opts = yaml.safe_load(open("save/myrun_vq3/opts.yaml", "r"))

print("==================================================")
print("1. INSPECTING RAW FILE (data/effects_2.pt)")
print("==================================================")
try:
    raw_effects = torch.load("data/effects_2.pt", map_location='cpu', weights_only=False)
    for i in range(6):
        dim_data = raw_effects[:, i]
        print(f"Dimension {i}: Min = {dim_data.min().item():>8.4f} | Max = {dim_data.max().item():>8.4f} | Mean = {dim_data.mean().item():>8.4f}")
except Exception as e:
    print(f"Error loading raw file: {e}")

print("\n==================================================")
print("2. INSPECTING DATALOADER (What the NN actually trains on)")
print("==================================================")
try:
    transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
    dataset2 = data.PairedObjectData(transform=transform)
    processed_effects = dataset2.effect
    
    for i in range(6):
        dim_data = processed_effects[:, i]
        print(f"Dimension {i}: Min = {dim_data.min().item():>8.4f} | Max = {dim_data.max().item():>8.4f} | Mean = {dim_data.mean().item():>8.4f}")
except Exception as e:
    print(f"Error loading dataloader: {e}")