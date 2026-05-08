import os
import torch
import numpy as np

data_dir = "data"

pt_files = [
    "actions.pt", 
    "effects_1.pt", 
    "effects_2.pt", 
    "objectsZ.pt", 
    "relations.pt", 
    "targets.pt"
]

npy_files = [
    "action_names.npy", 
    "obj_names.npy"
]

print("=== PyTorch Datasets (.pt) ===")
for file in pt_files:
    filepath = os.path.join(data_dir, file)
    if os.path.exists(filepath):
        try:
            tensor_data = torch.load(filepath, map_location='cpu')
            print(f"{file:<15}: shape {tensor_data.shape}")
        except Exception as e:
             print(f"{file:<15}: Error loading - {e}")
    else:
        print(f"{file:<15}: File not found")

print("\n=== NumPy Labels (.npy) ===")
for file in npy_files:
    filepath = os.path.join(data_dir, file)
    if os.path.exists(filepath):
        try:
            array_data = np.load(filepath, allow_pickle=True)
            print(f"{file:<15}: shape {array_data.shape} -> {array_data}")
        except Exception as e:
            print(f"{file:<15}: Error loading - {e}")
    else:
        print(f"{file:<15}: File not found")