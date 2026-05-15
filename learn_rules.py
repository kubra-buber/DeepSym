import os
import argparse
import yaml
from sklearn.tree import DecisionTreeClassifier
import torch
import pickle
import numpy as np
import utils

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

save_name = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

category = torch.load(os.path.join(opts["save"], "category.pt"))
label = torch.load(os.path.join(opts["save"], "label.pt"))
effect_names = np.load(os.path.join(opts["save"], "effect_names.npy"))
K = len(effect_names)

tree = DecisionTreeClassifier()
tree.fit(category, label)
with open(os.path.join(opts["save"], "tree.pkl"), "wb") as file:
    pickle.dump(tree, file)

CODE1_DIM = opts.get("code1_dim", 2)
CODE2_DIM = opts.get("code2_dim", 1)

obj_names = {}
for i in range(2**CODE1_DIM):
    cat_bin = utils.decimal_to_binary(i, length=CODE1_DIM)
    obj_names[cat_bin] = "objtype{}".format(i)

file_loc = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(file_loc):
    os.remove(file_loc)

# CRITICAL FIX: Base feature names strictly on category array shape, not K
num_features = category.shape[1]
feature_names = ["f%d" % i for i in range(num_features)]

pddl_code = utils.tree_to_code(tree, feature_names, effect_names, obj_names)

pretext = "(define (domain stack)\n"
pretext += "\t(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects :disjunctive-preconditions)\n"
pretext += "\t(:predicates"

for i in range(K):
    pretext += "\n\t\t(%s) " % effect_names[i]
pretext += "(base) \n\t\t(pickloc ?x)\n\t\t(instack ?x)\n\t\t(stackloc ?x)"

# Dynamically generate relation predicates based on code2_dim
for i in range(2**CODE2_DIM):
    pretext += f"\n\t\t(relation{i} ?x ?y)"

for i in range(2**CODE1_DIM):
    pretext += "\n\t\t(" + obj_names[utils.decimal_to_binary(i, length=CODE1_DIM)] + " ?x)"
for i in range(7):
    pretext += "\n\t\t(H%d)" % i
for i in range(7):
    pretext += "\n\t\t(S%d)" % i
pretext += "\n\t)"

with open(file_loc, "a") as f:
    print(pretext, file=f)

action_template = "\t(:action stack%d\n\t\t:parameters (?below ?above)"
with open(file_loc, "a") as f:
    for i, (precond, effect) in enumerate(pddl_code):
        print(action_template % i, file=f)
        print("\t\t"+precond, file=f)
        print("\t\t"+effect, file=f)
        print("\t)", file=f)
    for i in range(6):
        print("\t(:action increase-height%d" % (i+1), file=f)
        print("\t\t:precondition (and (stacked) (H%d))" % i, file=f)
        print("\t\t:effect (and (not (H%d)) (H%d) (not (stacked)))\n\t)" % (i, i+1), file=f)
    for i in range(6):
        print("\t(:action increase-stack%d" % (i+1), file=f)
        print("\t\t:precondition (and (inserted) (S%d))" % i, file=f)
        print("\t\t:effect (and (not (S%d)) (S%d) (not (inserted)))\n\t)" % (i, i+1), file=f)
    print("\t(:action makebase", file=f)
    print("\t\t:parameters (?obj)", file=f)
    print("\t\t:precondition (not (base))", file=f)
    print("\t\t:effect (and (base) (stacked) (inserted) (not (pickloc ?obj)) (stackloc ?obj))", file=f)
    print("\t)", file=f)
    print(")", file=f)
print("Successfully generated domain.pddl")