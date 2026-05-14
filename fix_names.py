import numpy as np
import os
import argparse

parser = argparse.ArgumentParser("Fix swapped cluster names.")
parser.add_argument("-opts", help="option file (to find the save folder)", type=str, required=True)
args = parser.parse_args()

# Extract the save folder from opts
import yaml
opts = yaml.safe_load(open(args.opts, "r"))
save_folder = opts["save"]

file_path = os.path.join(save_folder, "effect_names.npy")

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}")
    exit()

# Load the current names
names = np.load(file_path)
print(f"Old names: {names}")

# Swap 'stacked' and 'inserted'
for i in range(len(names)):
    if names[i] == 'TEMP_INS':
        print("yes")
        names[i] = 'inserted'

print(f"New names: {names}")

# Save the fixed array back to the file
np.save(file_path, names)
print(f"Successfully saved fixed names to {file_path}!")