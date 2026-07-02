import argparse
import os
import torch
import torchvision
import yaml
import matplotlib.pyplot as plt
import data
import utils
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
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

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)

# Set batch size automatically to the total size of your dataset to ensure max availability
loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)

sample = next(iter(loader))
B = sample["observation"].shape[0]

# Dynamically calculate the division factor based on the size of the incoming batch
dim1 = B // 480  
objects = sample["observation"].reshape(dim1, 10, 3, 4, 4, opts["size"], opts["size"])
objects = objects[:, :, 0].reshape(-1, 1, 42, 42)
colored = [[], [], [], []]

model.encoder1.eval()
with torch.no_grad():
    done = False
    it = 0
    max_elements = objects.shape[0]  # Total safe available elements after reshaping
    
    while not done:
        # SAFETY CHECK: If we run out of objects before finding 20 items per cluster, break safely
        if it >= max_elements:
            print(f"Warning: All {max_elements} items scanned, but some clusters could not reach 20 samples.")
            print("Proceeding to save the plot with the available items.")
            break
            
        c = model.encoder1(objects[it].reshape(1, 1, 42, 42))
        
        # --- THE VQ FIX: Extract the integer class index ---
        try:
            # If using VQLayer, grab the codebook index
            cat = int(model.encoder1[-1].get_indices(c)[0].item())
        except AttributeError:
            # If using original Gumbel STLayer, calculate decimal from binary
            cat = int(utils.binary_to_decimal(c[0]))
        # ---------------------------------------------------
        
        # Guard against index errors if an unexpected cluster category is predicted
        if 0 <= cat < 4:
            if len(colored[cat]) < 20:
                colored[cat].append(objects[it].clone())
        
        it += 1

        done = True
        for i in range(4):
            if len(colored[i]) < 20:
                done = False
                break

# Find the lowest count among all clusters to prevent stacking dimension mismatches
min_len = min(len(colored[i]) for i in range(4))

if min_len == 0:
    print("Error: One or more clusters contain 0 elements. The codebook might have collapsed or training failed.")
    exit()

# Trim all clusters to match the smallest length so they can be securely stacked into a grid
for i in range(4):
    colored[i] = torch.stack(colored[i][:min_len])

colored = torch.stack(colored)
colored = colored.reshape(-1, 42, 42)
colored = (colored - colored.min()) / (colored.max() - colored.min())
cm = plt.cm.plasma
colored = torch.tensor(cm(colored.numpy()), dtype=torch.float).permute(0, 3, 1, 2)[:, :3]

# Ensure save directory directory exists
os.makedirs("save/run4", exist_ok=True)

output_path = "save/run4/colored_objects_vq.png"
torchvision.utils.save_image(colored, output_path, nrow=min_len)
print(f"Success! Cluster grid visualization saved to: {output_path} (Images per row: {min_len})")