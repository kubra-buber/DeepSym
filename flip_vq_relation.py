import os
import torch
import yaml

opts = yaml.safe_load(open("opts.yaml"))
save = opts["save"]

cat = torch.load(os.path.join(save, "category.pt"), map_location="cpu").float()

if cat.shape[1] != 10:
    raise ValueError(f"Expected one-hot VQ category [N,10], got {cat.shape}")

flipped = cat.clone()
flipped[:, 8:10] = cat[:, 8:10].flip(dims=[1])

torch.save(flipped, os.path.join(save, "category_vq_relation_flipped.pt"))

print("Before counts:", cat[:, 8:10].argmax(dim=1).bincount(minlength=2).tolist())
print("After counts: ", flipped[:, 8:10].argmax(dim=1).bincount(minlength=2).tolist())
print("Saved:", os.path.join(save, "category_vq_relation_flipped.pt"))
