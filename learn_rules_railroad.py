"""Export DeepSym learned decision-tree rules as Railroad-compatible JSON.

This file replaces the PDDL domain generation part of DeepSym's original
learn_rules.py. It does NOT change the neuro-symbolic learning pipeline:

    category.pt + label.pt + effect_names.npy
        -> DecisionTreeClassifier
        -> symbolic stack operator specs with learned probabilistic effects

The output, railroad_operators.json, is consumed by make_plan_railroad.py.
Unlike the old domain.pddl, this JSON keeps the learned probabilistic effect
branches in a format that can be reconstructed as Railroad Operators.

Usage:
    python learn_rules_railroad.py -opts opts.yaml
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.tree import DecisionTreeClassifier, _tree

import utils


def _normalize_prob_effects(prob_effects: List[Dict], *, tol: float = 1e-9) -> List[Dict]:
    """Merge duplicate effect names and normalize probabilities to sum to 1.

    The original DeepSym code rounds leaf distributions to 3 decimals. That can
    cause tiny residual errors. For true probabilistic planning we want a valid
    distribution, so this function merges duplicates and normalizes.
    """
    merged: Dict[str, float] = {}
    for pe in prob_effects:
        name = str(pe["effect_name"])
        p = float(pe["probability"])
        if p <= 0.0:
            continue
        merged[name] = merged.get(name, 0.0) + p

    total = sum(merged.values())
    if total <= tol:
        raise ValueError(f"Probabilistic effect distribution is empty: {prob_effects}")

    normalized = [
        {"probability": float(p / total), "effect_name": name}
        for name, p in sorted(merged.items())
    ]

    # Round only for readability, then fix residual on largest branch.
    rounded = [
        {"probability": round(pe["probability"], 6), "effect_name": pe["effect_name"]}
        for pe in normalized
    ]
    residual = round(1.0 - sum(pe["probability"] for pe in rounded), 6)
    if rounded and abs(residual) > 0.0:
        max_i = max(range(len(rounded)), key=lambda i: rounded[i]["probability"])
        rounded[max_i]["probability"] = round(rounded[max_i]["probability"] + residual, 6)

    return [pe for pe in rounded if pe["probability"] > 0.0]


def _rule_to_types(rule: Sequence[int], obj_names: Dict[Tuple[int, ...], str], code1_dim: int, code2_dim: int):
    """Convert a decision-tree path into possible object types and relations.

    Role convention matches the original DeepSym PDDL code:
        first object code  -> ?below / obj1_type
        second object code -> ?above / obj2_type
        relation code      -> relation ?below ?above
    """
    num_features = code1_dim * 2 + code2_dim
    absrules = np.abs(rule).tolist()

    indices: List[int] = []
    for feature_id in range(1, num_features + 1):
        if feature_id in absrules:
            indices.append(absrules.index(feature_id))
        else:
            indices.append(-1)

    possible_obj_1 = list(obj_names.keys())
    for bit_i, idx in enumerate(indices[:code1_dim]):
        if idx == -1:
            continue
        sign = int(np.sign(rule[idx]))
        possible_obj_1 = [code for code in possible_obj_1 if code[bit_i] == sign]

    possible_obj_2 = list(obj_names.keys())
    for bit_i, idx in enumerate(indices[code1_dim:2 * code1_dim]):
        if idx == -1:
            continue
        sign = int(np.sign(rule[idx]))
        possible_obj_2 = [code for code in possible_obj_2 if code[bit_i] == sign]

    obj1_types = [obj_names[code] for code in possible_obj_1]
    obj2_types = [obj_names[code] for code in possible_obj_2]

    relation_indices = indices[2 * code1_dim:]
    possible_relations = list(range(2 ** code2_dim))
    for bit_i, idx in enumerate(relation_indices):
        if idx == -1:
            continue
        sign = int(np.sign(rule[idx]))
        filtered = []
        for rel_id in possible_relations:
            rel_binary = utils.decimal_to_binary(rel_id, length=code2_dim)
            if rel_binary[bit_i] == sign:
                filtered.append(rel_id)
        possible_relations = filtered

    relation_names = [f"relation{rel_id}" for rel_id in possible_relations]
    return obj1_types, obj2_types, relation_names


def tree_to_operator_specs(tree: DecisionTreeClassifier,
                           effect_names: Sequence[str],
                           obj_names: Dict[Tuple[int, ...], str],
                           code1_dim: int,
                           code2_dim: int) -> List[Dict]:
    """Walk the decision tree and convert leaves to operator specs."""
    tree_ = tree.tree_
    specs: List[Dict] = []
    counter = [0]

    def recurse(node: int, rules: List[int]) -> None:
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature_index = int(tree_.feature[node]) + 1
            left_rules = list(rules) + [-feature_index]
            right_rules = list(rules) + [feature_index]
            recurse(tree_.children_left[node], left_rules)
            recurse(tree_.children_right[node], right_rules)
            return

        obj1_types, obj2_types, relation_names = _rule_to_types(
            rules, obj_names, code1_dim, code2_dim
        )

        leaf_counts = tree_.value[node][0].astype(float)
        total = float(leaf_counts.sum())
        if total <= 0.0:
            return

        raw_effects: List[Dict] = []
        for i, count in enumerate(leaf_counts):
            p = float(count / total)
            if p <= 0.0:
                continue
            raw_effects.append({
                "probability": p,
                "effect_name": str(effect_names[i]),
            })
        prob_effects = _normalize_prob_effects(raw_effects)

        for obj1_type in obj1_types:
            for obj2_type in obj2_types:
                for rel_name in relation_names:
                    specs.append({
                        "name": f"stack{counter[0]}",
                        "obj1_type": obj1_type,
                        "obj2_type": obj2_type,
                        "below_type": obj1_type,
                        "above_type": obj2_type,
                        "relation": rel_name,
                        "prob_effects": prob_effects,
                    })
                    counter[0] += 1

    recurse(0, [])
    return specs


def create_auxiliary_specs(num_heights: int = 7, num_stacks: int = 7) -> List[Dict]:
    """Create DeepSym bookkeeping actions."""
    aux: List[Dict] = []

    for i in range(num_heights - 1):
        aux.append({
            "type": "increase_height",
            "name": f"increase_height{i + 1}",
            "from_counter": f"H{i}",
            "to_counter": f"H{i + 1}",
        })

    for i in range(num_stacks - 1):
        aux.append({
            "type": "increase_stack",
            "name": f"increase_stack{i + 1}",
            "from_counter": f"S{i}",
            "to_counter": f"S{i + 1}",
        })

    aux.append({"type": "makebase", "name": "makebase"})
    return aux


def main() -> None:
    parser = argparse.ArgumentParser("Convert DeepSym decision-tree rules to Railroad JSON.")
    parser.add_argument("-opts", type=str, required=True, help="option file")
    parser.add_argument("--retrain-tree", action="store_true",
                        help="ignore existing tree.pkl and retrain the decision tree")
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))

    category = torch.load(os.path.join(save_dir, "category.pt"))
    label = torch.load(os.path.join(save_dir, "label.pt"))
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"))

    tree_path = os.path.join(save_dir, "tree.pkl")
    if os.path.exists(tree_path) and not args.retrain_tree:
        with open(tree_path, "rb") as f:
            tree = pickle.load(f)
        print(f"Loaded existing decision tree from {tree_path}")
    else:
        tree = DecisionTreeClassifier()
        tree.fit(category, label)
        with open(tree_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"Trained and saved decision tree to {tree_path}")

    obj_names: Dict[Tuple[int, ...], str] = {}
    for i in range(2 ** code1_dim):
        code = tuple(utils.decimal_to_binary(i, length=code1_dim))
        obj_names[code] = f"objtype{i}"

    stack_specs = tree_to_operator_specs(tree, effect_names, obj_names, code1_dim, code2_dim)
    aux_specs = create_auxiliary_specs()

    all_specs = {
        "stack_operators": stack_specs,
        "auxiliary_operators": aux_specs,
        "metadata": {
            "planner_semantics": "probabilistic_expected_reachability",
            "code1_dim": code1_dim,
            "code2_dim": code2_dim,
            "effect_names": [str(e) for e in effect_names],
            "obj_names": {str(k): v for k, v in obj_names.items()},
            "num_obj_types": 2 ** code1_dim,
            "num_relations": 2 ** code2_dim,
            "role_convention": "obj1_type/below_type is ?below; obj2_type/above_type is ?above",
        },
    }

    out_path = os.path.join(save_dir, "railroad_operators.json")
    with open(out_path, "w") as f:
        json.dump(all_specs, f, indent=2)

    print(f"Created {len(stack_specs)} stack operator specs")
    print(f"Created {len(aux_specs)} auxiliary operator specs")
    print(f"Saved {len(stack_specs) + len(aux_specs)} operator specs to {out_path}")


if __name__ == "__main__":
    main()