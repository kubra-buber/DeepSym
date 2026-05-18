import os
import argparse
import torch
import yaml
from models import EffectRegressorMLP
import data
import utils

parser = argparse.ArgumentParser("Save categories.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = torch.device("cpu")

print("Loading models...")
model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
# Using PairedObjectData GUARANTEES perfect alignment with label.pt
dataset2 = data.PairedObjectData(transform=transform)
loader2 = torch.utils.data.DataLoader(dataset2, batch_size=200, shuffle=False)

all_cats = []

with torch.no_grad():
    for sample in loader2:
        obs = sample["observation"].to(device)
        # DeepSym dataset: obs[:, 0] is Top Object, obs[:, 1] is Bottom Object
        # obs_top = obs[:, 0].unsqueeze(1)
        # obs_bot = obs[:, 1].unsqueeze(1)
        obs_bot = obs[:, 0].unsqueeze(1)
        obs_top = obs[:, 1].unsqueeze(1)

        h_top = model.encoder1(obs_top)
        h_bot = model.encoder1(obs_bot)
        h_rel = model.encoder2(obs)

        # Get VQ Integer Indices
        idx_top = model.encoder1[-1].get_indices(h_top)
        idx_bot = model.encoder1[-1].get_indices(h_bot)
        idx_rel = model.encoder2[-1].get_indices(h_rel)

        # Safely convert to binary lists using the paper's native util
        bits_top = [utils.decimal_to_binary(i.item(), length=opts["code1_dim"]) for i in idx_top]
        bits_bot = [utils.decimal_to_binary(i.item(), length=opts["code1_dim"]) for i in idx_bot]
        bits_rel = [utils.decimal_to_binary(i.item(), length=opts["code2_dim"]) for i in idx_rel]

        t_top = torch.tensor(bits_top, dtype=torch.int32)
        t_bot = torch.tensor(bits_bot, dtype=torch.int32)
        t_rel = torch.tensor(bits_rel, dtype=torch.int32)

        # Concatenate [Top, Bottom, Relation]
        cat = torch.cat([t_top, t_bot, t_rel], dim=-1)
        all_cats.append(cat)

final_categories = torch.cat(all_cats, dim=0)
torch.save(final_categories.cpu(), os.path.join(opts["save"], "category.pt"))
print(f"Saved robust VQ binary categories to {os.path.join(opts['save'], 'category.pt')}")