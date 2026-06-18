"""Convert DeepSym decision-tree rules into Railroad operator specs.

This is the Railroad-side replacement for the original PDDL domain generation in
learn_rules.py. It intentionally keeps the DeepSym neuro-symbolic structure:

    learned object/relation codes  ->  decision-tree rule preconditions
    learned effect clusters        ->  probabilistic symbolic effects

The output is JSON rather than Python objects because Railroad Operator/Effect
objects are not safely JSON-serializable. make_plan_railroad.py reconstructs the
actual Railroad operators from this file.
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


def _normalize_leaf_distribution(
    values: np.ndarray,
    effect_names: Sequence[str],
    granularity: int = 1000,
) -> List[Dict[str, float]]:
    """Convert a sklearn leaf class-count vector into probabilities.

    The original DeepSym PDDL writer quantized probabilities to 3 decimals for
    PPDDL output. We keep the same 1/1000 granularity here so comparisons with
    the old PPDDL pipeline remain close, but we also fix both positive and
    negative rounding residuals so the distribution sums exactly to 1.0.

    Args:
        values: class counts at a decision-tree leaf.
        effect_names: symbolic names for the effect clusters.
        granularity: probability denominator; 1000 -> 3 decimal places.

    Returns:
        A list of {"probability": p, "effect_name": name} entries.
    """
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Decision-tree leaf has zero total class count.")

    probs = values / total
    scaled = np.rint(probs * granularity).astype(int)

    # Ensure exact sum after rounding. Assign residual to the most common class.
    residual = int(granularity - scaled.sum())
    if residual != 0:
        scaled[int(np.argmax(values))] += residual

    if np.any(scaled < 0):
        raise ValueError(
            "Probability rounding produced a negative entry. "
            "Increase granularity or inspect the decision-tree leaf."
        )

    out = []
    for count, effect_name in zip(scaled, effect_names):
        if count <= 0:
            continue
        out.append({
            "probability": float(count) / float(granularity),
            "effect_name": str(effect_name),
        })

    # Final safety check.
    prob_sum = sum(x["probability"] for x in out)
    if abs(prob_sum - 1.0) > 1e-9:
        raise AssertionError("Normalized probabilities do not sum to 1.0.")

    return out


def _rule_to_types(
    rule: Sequence[int],
    obj_names: Dict[Tuple[int, ...], str],
    code1_dim: int,
    code2_dim: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Convert a decision-tree path into symbolic type/relation predicates.

    This mirrors the role convention used by the original DeepSym
    utils.tree_to_code:

        first object code  -> ?below
        second object code -> ?above
        relation code      -> relation(?below, ?above)

    If you later discover that your dataset/category writer uses the opposite
    order, fix the role convention here and in make_plan_railroad.py together.
    """
    num_features = code1_dim * 2 + code2_dim
    absrules = np.abs(rule).tolist()

    # indices[i] stores the position in `rule` that constrains feature i+1.
    # -1 means unconstrained by this decision-tree path.
    indices = []
    for feat_idx in range(1, num_features + 1):
        if feat_idx in absrules:
            indices.append(absrules.index(feat_idx))
        else:
            indices.append(-1)

    possible_below_codes = list(obj_names.keys())
    for bit_idx, rule_idx in enumerate(indices[:code1_dim]):
        if rule_idx == -1:
            continue
        sign = int(np.sign(rule[rule_idx]))
        possible_below_codes = [
            code for code in possible_below_codes if code[bit_idx] == sign
        ]

    possible_above_codes = list(obj_names.keys())
    for bit_idx, rule_idx in enumerate(indices[code1_dim:2 * code1_dim]):
        if rule_idx == -1:
            continue
        sign = int(np.sign(rule[rule_idx]))
        possible_above_codes = [
            code for code in possible_above_codes if code[bit_idx] == sign
        ]

    possible_relation_ids = list(range(2 ** code2_dim))
    for bit_idx, rule_idx in enumerate(indices[2 * code1_dim:]):
        if rule_idx == -1:
            continue
        sign = int(np.sign(rule[rule_idx]))
        filtered = []
        for rel_id in possible_relation_ids:
            rel_bits = utils.decimal_to_binary(rel_id, length=code2_dim)
            if rel_bits[bit_idx] == sign:
                filtered.append(rel_id)
        possible_relation_ids = filtered

    below_types = [obj_names[code] for code in possible_below_codes]
    above_types = [obj_names[code] for code in possible_above_codes]
    relation_names = [f"relation{rel_id}" for rel_id in possible_relation_ids]

    if not below_types or not above_types or not relation_names:
        raise ValueError(
            "A decision-tree rule produced an empty type/relation set. "
            f"rule={rule}, below={below_types}, above={above_types}, "
            f"relations={relation_names}"
        )

    return below_types, above_types, relation_names


def tree_to_operator_specs(
    tree: DecisionTreeClassifier,
    effect_names: Sequence[str],
    obj_names: Dict[Tuple[int, ...], str],
    code1_dim: int,
    code2_dim: int,
) -> List[Dict]:
    """Convert each decision-tree leaf into Railroad stack operator specs.

    A single leaf may correspond to multiple symbolic operators if the rule does
    not constrain every object/relation bit. This mirrors the original PDDL
    expansion of disjunctive preconditions into allowed type/relation cases.
    """
    tree_ = tree.tree_
    op_specs = []
    action_counter = 0

    def recurse(node: int, rules: List[int]) -> None:
        nonlocal action_counter

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            # sklearn features are 0-indexed. DeepSym rule encoding uses
            # +/- (feature_index + 1), preserving the original utility format.
            feat = int(tree_.feature[node]) + 1

            left_rules = list(rules)
            left_rules.append(-feat)
            recurse(int(tree_.children_left[node]), left_rules)

            right_rules = list(rules)
            right_rules.append(feat)
            recurse(int(tree_.children_right[node]), right_rules)
            return

        below_types, above_types, relation_names = _rule_to_types(
            rules, obj_names, code1_dim, code2_dim
        )

        leaf_values = tree_.value[node][0]
        prob_effects = _normalize_leaf_distribution(leaf_values, effect_names)

        for below_type in below_types:
            for above_type in above_types:
                for relation_name in relation_names:
                    op_specs.append({
                        "name": f"stack{action_counter}",
                        "below_type": below_type,
                        "above_type": above_type,
                        # Backward-compatible aliases for older loaders.
                        "obj1_type": below_type,
                        "obj2_type": above_type,
                        "relation": relation_name,
                        "prob_effects": prob_effects,
                    })
                    action_counter += 1

    recurse(0, [])
    return op_specs


def create_auxiliary_specs(num_heights: int = 7, num_stacks: int = 7) -> List[Dict]:
    """Create deterministic auxiliary operators used by DeepSym.

    These match the original generated PPDDL actions:
        - increase-heightN consumes `stacked` and increments H counter
        - increase-stackN consumes `inserted` and increments S counter
        - makebase chooses the first base object
    """
    aux_specs = []

    for i in range(num_heights - 1):
        aux_specs.append({
            "type": "increase_height",
            "name": f"increase_height{i + 1}",
            "from_counter": f"H{i}",
            "to_counter": f"H{i + 1}",
        })

    for i in range(num_stacks - 1):
        aux_specs.append({
            "type": "increase_stack",
            "name": f"increase_stack{i + 1}",
            "from_counter": f"S{i}",
            "to_counter": f"S{i + 1}",
        })

    aux_specs.append({
        "type": "makebase",
        "name": "makebase",
    })

    return aux_specs


def main() -> None:
    parser = argparse.ArgumentParser(
        "Convert DeepSym decision-tree rules to Railroad operator specs."
    )
    parser.add_argument("-opts", help="option file", type=str, required=True)
    parser.add_argument(
        "--force-retrain-tree",
        action="store_true",
        help="Ignore an existing tree.pkl and retrain the decision tree.",
    )
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))

    save_dir = opts["save"]
    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))
    num_heights = int(opts.get("num_heights", 7))
    num_stacks = int(opts.get("num_stacks", 7))

    category = torch.load(os.path.join(save_dir, "category.pt"))
    label = torch.load(os.path.join(save_dir, "label.pt"))
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"))

    # Torch tensors are accepted by sklearn in many environments, but explicit
    # NumPy conversion makes the script more predictable.
    category_np = category.detach().cpu().numpy() if hasattr(category, "detach") else np.asarray(category)
    label_np = label.detach().cpu().numpy() if hasattr(label, "detach") else np.asarray(label)

    expected_features = code1_dim * 2 + code2_dim
    if category_np.shape[1] != expected_features:
        raise ValueError(
            f"category.pt has {category_np.shape[1]} features, but "
            f"code1_dim*2 + code2_dim = {expected_features}. "
            "Check save_cat.py and opts.yaml."
        )

    tree_path = os.path.join(save_dir, "tree.pkl")
    if os.path.exists(tree_path) and not args.force_retrain_tree:
        with open(tree_path, "rb") as f:
            tree = pickle.load(f)
        print(f"Loaded existing decision tree from {tree_path}")
    else:
        tree = DecisionTreeClassifier(random_state=0)
        tree.fit(category_np, label_np)
        with open(tree_path, "wb") as f:
            pickle.dump(tree, f)
        print(f"Trained and saved new decision tree to {tree_path}")

    obj_names = {}
    for i in range(2 ** code1_dim):
        code = utils.decimal_to_binary(i, length=code1_dim)
        obj_names[code] = f"objtype{i}"

    stack_specs = tree_to_operator_specs(
        tree=tree,
        effect_names=[str(x) for x in effect_names],
        obj_names=obj_names,
        code1_dim=code1_dim,
        code2_dim=code2_dim,
    )
    aux_specs = create_auxiliary_specs(
        num_heights=num_heights,
        num_stacks=num_stacks,
    )

    all_specs = {
        "stack_operators": stack_specs,
        "auxiliary_operators": aux_specs,
        "metadata": {
            "code1_dim": code1_dim,
            "code2_dim": code2_dim,
            "num_heights": num_heights,
            "num_stacks": num_stacks,
            "effect_names": [str(e) for e in effect_names],
            "obj_names": {str(k): v for k, v in obj_names.items()},
            "num_obj_types": 2 ** code1_dim,
            "num_relations": 2 ** code2_dim,
            "role_convention": {
                "first_object_code": "?below",
                "second_object_code": "?above",
                "relation": "relation(?below, ?above)",
            },
            "probability_granularity": 1000,
        },
    }

    save_path = os.path.join(save_dir, "railroad_operators.json")
    with open(save_path, "w") as f:
        json.dump(all_specs, f, indent=2, sort_keys=True)

    print(f"Created {len(stack_specs)} stack operator specs")
    print(f"Created {len(aux_specs)} auxiliary operator specs")
    print(f"Saved {len(stack_specs) + len(aux_specs)} operator specs to {save_path}")


if __name__ == "__main__":
    main()