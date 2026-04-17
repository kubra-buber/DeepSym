import torch

# Load the file
# Use map_location='cpu' to avoid errors if the file was saved on a GPU
data = torch.load('data/relations.pt', map_location='cpu')

# 1. If it's a simple tensor
if torch.is_tensor(data):
    print(data.shape)
    print(data)

# 2. If it's a model's state_dict (very common)
elif isinstance(data, dict):
    for key, value in data.items():
        if torch.is_tensor(value):
            print(f"Layer: {key} | Shape: {value.shape}")
        else:
            print(f"Key: {key} | Value: {value}")