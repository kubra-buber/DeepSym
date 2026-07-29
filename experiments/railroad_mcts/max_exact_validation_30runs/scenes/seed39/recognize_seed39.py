import os
import argparse
import json
import rospy
import yaml
import numpy as np
import torch
from models import EffectRegressorMLP
import data
import utils
from simtools.rosutils import RosNode


def resolve_device(requested):
    """Return a safe torch.device string.

    DeepSym opts files often contain `device: cuda`. On some ROS/desktop
    sessions PyTorch can see the NVIDIA driver but fail CUDA initialization.
    For scene recognition, CPU is fast enough, so fall back cleanly instead of
    crashing before planning.
    """
    requested = str(requested or "cpu").strip()

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda"):
        try:
            if not torch.cuda.is_available():
                print("WARNING: CUDA requested but torch.cuda.is_available() is False; using CPU for recognize.py.")
                return torch.device("cpu")
            # Force a tiny allocation so CUDA init errors are caught here.
            torch.empty(1, device=requested)
            return torch.device(requested)
        except Exception as exc:
            print(f"WARNING: CUDA requested but initialization failed ({exc}); using CPU for recognize.py.")
            return torch.device("cpu")

    return torch.device(requested)


parser = argparse.ArgumentParser("Recognize scene and write DeepSym Railroad scene JSON/objects.txt.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-goal", help="goal state", type=str, default="(H3) (S0)")
parser.add_argument("-uri", help="master uri", type=str, default="http://localhost:11311")
parser.add_argument(
    "-device",
    "--device",
    help="recognition device override: auto, cpu, cuda, cuda:0. Default: opts.yaml device with safe CUDA fallback.",
    type=str,
    default=None,
)
parser.add_argument(
    "--write-pddl",
    action="store_true",
    help="also write legacy problem.pddl for comparison/debugging; Railroad planners do not use it",
)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
requested_device = args.device if args.device is not None else opts.get("device", "cpu")
device = resolve_device(requested_device)
opts["device"] = str(device)
print(f"recognize.py using device: {device}")

node = RosNode("recognize_scene", args.uri)
node.stopSimulation()
rospy.sleep(2.0)
node.startSimulation()
rospy.sleep(2.0)

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()
model.encoder1.to(device)
model.encoder2.to(device)

# Homogeneous transformation matrix. Keep it on CPU because locs are CPU.
H = torch.load("H.pt", map_location="cpu")

np.random.seed(39)

# GENERATE A RANDOM SCENE
NUM_OBJECTS = 5
objTypes = np.random.randint(1, 6, (NUM_OBJECTS,))
objSizes = np.random.uniform(1.0, 2, (5,)).tolist()
locations = np.array([
    [-0.69, -0.09],
    [-0.9, -0.35],
    [-0.45, 0.175],
    [-0.45, -0.35],
    [-0.9, 0.175],
])
locations = locations[np.random.permutation(5)]
locations = locations[:NUM_OBJECTS].tolist()

for i in range(NUM_OBJECTS):
    node.generateObject(objTypes[i], objSizes[i], locations[i] + [objSizes[i] * 0.05 + 0.7])
rospy.sleep(1.0)
locations = torch.tensor(locations, dtype=torch.float)

x = torch.tensor(node.getDepthImage(8), dtype=torch.float)
objs, locs, _ = utils.find_objects(x, opts["size"])

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
for i, o in enumerate(objs):
    objs[i] = transform(o)[0]
objs = objs.to(device)

locs = torch.cat([locs.float(), torch.ones(locs.shape[0], 1, device=locs.device)], dim=1)
locs = torch.matmul(locs, H.T)
locs = locs / locs[:, 2].reshape(-1, 1)

_, indices = torch.cdist(locs[:, :2], locations).min(dim=1)
obj_infos = []
comparisons = []
with torch.no_grad():
    for i, obj in enumerate(objs):
        cat = model.encoder1(obj.unsqueeze(0).unsqueeze(0))
        cat_cpu = cat.detach().cpu()
        # TODO: this uses true location and size.
        print(
            "Category: (%d %d), Location: (%.5f %.5f)"
            % (
                cat_cpu[0, 0],
                cat_cpu[0, 1],
                locations[indices[i], 0],
                locations[indices[i], 1],
            )
        )
        info = {}
        info["name"] = "O{}".format(i + 1)
        info["loc"] = (locations[indices[i], 0].item(), locations[indices[i], 1].item())
        info["size"] = objSizes[indices[i]] * 0.1
        info["type"] = "objtype{}".format(utils.binary_to_decimal([int(cat_cpu[0, 0]), int(cat_cpu[0, 1])]))

        obj_infos.append(info)
        for j in range(len(objs)):
            if i != j:
                rel = model.encoder2(torch.stack([obj, objs[j]]).unsqueeze(0))[0, 0]
                rel_value = int(rel.detach().cpu().item())
                if rel_value == -1:
                    comparisons.append("(relation0 O%d O%d)" % (i + 1, j + 1))
                else:
                    comparisons.append("(relation1 O%d O%d)" % (i + 1, j + 1))
print(obj_infos)
print(comparisons)

scene_file = os.path.join(opts["save"], "railroad_problem.json")
file_obj = os.path.join(opts["save"], "objects.txt")
legacy_pddl = os.path.join(opts["save"], "problem.pddl")
os.makedirs(opts["save"], exist_ok=True)

# Write Railroad-native scene description. This is the symbolic problem input
# for make_plan_railroad.py, make_plan_railroad_expected.py and
# closed_loop_railroad.py. It replaces problem.pddl in the Railroad path.
scene = {
    "format": "deepsym_railroad_problem_v1",
    "goal": args.goal,
    "objects": [
        {
            "name": obj_i["name"],
            "type": obj_i["type"],
            "loc": [float(obj_i["loc"][0]), float(obj_i["loc"][1])],
            "size": float(obj_i["size"]),
        }
        for obj_i in obj_infos
    ],
    "relations": [],
    "counters": {"H": "H0", "S": "S0"},
}

for c_i in comparisons:
    # comparison strings are of the form "(relation0 O1 O2)".
    parts = c_i.strip("()").split()
    if len(parts) == 3:
        scene["relations"].append({"name": parts[0], "below": parts[1], "above": parts[2]})

with open(scene_file, "w") as f_scene:
    json.dump(scene, f_scene, indent=2, sort_keys=True)

# Keep objects.txt because execute_plan.py uses the compact legacy plan header.
# This file is execution geometry, not the symbolic planning problem.
with open(file_obj, "w") as f_objects:
    print(str(len(obj_infos)), file=f_objects)
    for obj_i in obj_infos:
        print(
            "%s %.5f %.5f %.5f" % (obj_i["name"], obj_i["loc"][0], obj_i["loc"][1], obj_i["size"]),
            file=f_objects,
        )

if args.write_pddl:
    with open(legacy_pddl, "w") as f_problem:
        print("(define (problem dom1) (:domain stack)", file=f_problem)
        object_str = "\t(:objects"
        init_str = "\t(:init\n"
        for obj_i in obj_infos:
            object_str += " " + obj_i["name"]
            init_str += "\t\t(pickloc " + obj_i["name"] + ") (" + obj_i["type"] + " " + obj_i["name"] + ")\n"
        object_str += ")"
        for c_i in comparisons:
            init_str += "\t\t" + c_i + "\n"
        init_str += "\t\t(H0)\n"
        init_str += "\t\t(S0)\n"
        init_str += "\t)"
        goal_str = "\t(:goal (and %s (not (stacked)) (not (inserted))))\n)" % args.goal
        print(object_str, file=f_problem)
        print(init_str, file=f_problem)
        print(goal_str, file=f_problem)
    print(f"Wrote legacy {legacy_pddl}")

print(f"Wrote {scene_file}")
print(f"Wrote {file_obj}")