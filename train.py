import os
import argparse
import time
import yaml
import torch
from models import EffectRegressorMLP
import data

parser = argparse.ArgumentParser("Train effect prediction models end-to-end.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))

# Standard opts.yaml file generation
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the first level data
transform1 = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset1 = data.SingleObjectData(transform=transform1)
loader1 = torch.utils.data.DataLoader(trainset1, batch_size=opts["batch_size1"], shuffle=True)

# load the second level data
transform2 = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset2 = data.PairedObjectData(transform=transform2)
loader2 = torch.utils.data.DataLoader(trainset2, batch_size=opts.get("batch_size2", opts["batch_size1"]), shuffle=True)

# Initialize Model
model = EffectRegressorMLP(opts)
if opts.get("load") is not None:
    model.load(opts["load"], ext="")
    
model.print_model()

# Define total epochs 
epochs = opts.get("epoch", opts.get("epoch1", 100))

# Pass loader1 and loader2 separately
print(f"Starting end-to-end training for {epochs} epochs...")
model.train(epochs, loader1, loader2)