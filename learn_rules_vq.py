"""Learn PPDDL rules from one-hot VQ categories.

The original DeepSym learn_rules.py assumes signed binary bottleneck bits.
VQ indices are nominal categories, so this variant trains the same decision-tree
rule model on one-hot features and exports leaf preconditions by enumerating the
finite symbolic tuple space.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def one_hot_tuple(first: int, second: int, relation: int, n_obj: int, n_rel: int) -> np.ndarray:
    row = np.zeros(2 * n_obj + n_rel, dtype=np.float32)
    row[first] = 1.0
    row[n_obj + second] = 1.0
    row[2 * n_obj + relation] = 1.0
    return row


def normalized_leaf_effects(
    tree: DecisionTreeClassifier,
    leaf_node: int,
    effect_names: Sequence[str],
    granularity: int = 1000,
) -> List[Tuple[float, str]]:
    values = np.asarray(tree.tree_.value[leaf_node][0], dtype=float)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Empty decision-tree leaf")
    raw = values / total
    scaled = np.rint(raw * granularity).astype(int)
    residual = int(granularity - scaled.sum())
    if residual:
        scaled[int(np.argmax(values))] += residual

    effects = []
    for class_position, count in enumerate(scaled):
        if count <= 0:
            continue
        label_index = int(tree.classes_[class_position])
        if label_index < 0 or label_index >= len(effect_names):
            raise IndexError("Tree class %d outside effect_names" % label_index)
        effects.append((count / float(granularity), str(effect_names[label_index])))
    return effects


def effect_expression(effect_name: str) -> str:
    if effect_name == "stacked":
        return "(and (stacked) (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
    if effect_name == "inserted":
        return "(and (inserted) (instack ?above) (stackloc ?above) (not (stackloc ?below)))"
    return "(%s)" % effect_name


def tuple_condition(symbol_tuple: Tuple[int, int, int]) -> str:
    first, second, relation = symbol_tuple
    return "(and (objtype%d ?below) (objtype%d ?above) (relation%d ?below ?above))" % (
        first, second, relation
    )


def leaf_precondition(tuples: List[Tuple[int, int, int]]) -> str:
    conditions = [tuple_condition(item) for item in tuples]
    if len(conditions) == 1:
        symbolic = conditions[0]
    else:
        symbolic = "(or %s)" % " ".join(conditions)
    return (
        "(and (not (stacked)) (not (inserted)) "
        "(pickloc ?above) (stackloc ?below) %s)" % symbolic
    )


def leaf_effect(effects: List[Tuple[float, str]]) -> str:
    branches = []
    for probability, name in effects:
        branches.append("%.3f %s" % (probability, effect_expression(name)))
    return "(and (probabilistic %s) (not (pickloc ?above)))" % " ".join(branches)


def write_domain(
    path: str,
    tree: DecisionTreeClassifier,
    leaf_tuples: Dict[int, List[Tuple[int, int, int]]],
    effect_names: Sequence[str],
    n_obj: int,
    n_rel: int,
    num_heights: int,
    num_stacks: int,
) -> None:
    predicates = []
    for name in effect_names:
        if str(name) not in predicates:
            predicates.append(str(name))

    with open(path, "w") as handle:
        print("(define (domain stack)", file=handle)
        print("\t(:requirements :typing :negative-preconditions :probabilistic-effects :conditional-effects :disjunctive-preconditions)", file=handle)
        print("\t(:predicates", file=handle)
        for name in predicates:
            print("\t\t(%s)" % name, file=handle)
        print("\t\t(base)", file=handle)
        print("\t\t(pickloc ?x)", file=handle)
        print("\t\t(instack ?x)", file=handle)
        print("\t\t(stackloc ?x)", file=handle)
        for relation in range(n_rel):
            print("\t\t(relation%d ?x ?y)" % relation, file=handle)
        for obj in range(n_obj):
            print("\t\t(objtype%d ?x)" % obj, file=handle)
        for index in range(num_heights):
            print("\t\t(H%d)" % index, file=handle)
        for index in range(num_stacks):
            print("\t\t(S%d)" % index, file=handle)
        print("\t)", file=handle)

        for action_index, leaf_node in enumerate(sorted(leaf_tuples)):
            effects = normalized_leaf_effects(tree, leaf_node, effect_names)
            print("\t(:action stack%d" % action_index, file=handle)
            print("\t\t:parameters (?below ?above)", file=handle)
            print("\t\t:precondition %s" % leaf_precondition(leaf_tuples[leaf_node]), file=handle)
            print("\t\t:effect %s" % leaf_effect(effects), file=handle)
            print("\t)", file=handle)

        for index in range(num_heights - 1):
            print("\t(:action increase-height%d" % (index + 1), file=handle)
            print("\t\t:precondition (and (stacked) (H%d))" % index, file=handle)
            print("\t\t:effect (and (not (H%d)) (H%d) (not (stacked)))" % (index, index + 1), file=handle)
            print("\t)", file=handle)
        for index in range(num_stacks - 1):
            print("\t(:action increase-stack%d" % (index + 1), file=handle)
            print("\t\t:precondition (and (inserted) (S%d))" % index, file=handle)
            print("\t\t:effect (and (not (S%d)) (S%d) (not (inserted)))" % (index, index + 1), file=handle)
            print("\t)", file=handle)
        print("\t(:action makebase", file=handle)
        print("\t\t:parameters (?obj)", file=handle)
        print("\t\t:precondition (not (base))", file=handle)
        print("\t\t:effect (and (base) (stacked) (inserted) (not (pickloc ?obj)) (stackloc ?obj))", file=handle)
        print("\t)", file=handle)
        print(")", file=handle)


def main() -> None:
    parser = argparse.ArgumentParser("Learn PPDDL rules from VQ one-hot categories.")
    parser.add_argument("-opts", required=True)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    with open(args.opts, "r") as handle:
        opts = yaml.safe_load(handle)
    save_dir = os.path.abspath(opts["save"])
    category = torch.load(os.path.join(save_dir, "category.pt"), map_location="cpu").numpy()
    labels = torch.load(os.path.join(save_dir, "label.pt"), map_location="cpu").numpy().astype(int)
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"), allow_pickle=True)
    with open(os.path.join(save_dir, "category_meta.json"), "r") as handle:
        meta = json.load(handle)

    if meta.get("encoding") != "vq_onehot":
        raise ValueError("learn_rules_vq.py expects category_meta.json encoding=vq_onehot")
    n_obj = int(meta["num_object_codes"])
    n_rel = int(meta["num_relation_codes"])
    expected_features = 2 * n_obj + n_rel
    if category.shape[1] != expected_features:
        raise ValueError("category feature count %d != expected %d" % (category.shape[1], expected_features))

    tree = DecisionTreeClassifier(
        random_state=args.random_state,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    tree.fit(category, labels)
    predictions = tree.predict(category)
    leaves = tree.apply(category)

    tuple_rows = []
    symbolic_tuples = []
    for first, second, relation in itertools.product(range(n_obj), range(n_obj), range(n_rel)):
        symbolic_tuples.append((first, second, relation))
        tuple_rows.append(one_hot_tuple(first, second, relation, n_obj, n_rel))
    tuple_leaves = tree.apply(np.asarray(tuple_rows, dtype=np.float32))
    leaf_tuples: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for symbolic_tuple, leaf in zip(symbolic_tuples, tuple_leaves):
        leaf_tuples[int(leaf)].append(symbolic_tuple)

    tree_path = os.path.join(save_dir, "tree_vq_onehot.pkl")
    with open(tree_path, "wb") as handle:
        pickle.dump(tree, handle)
    domain_path = os.path.join(save_dir, "domain.pddl")
    write_domain(
        domain_path,
        tree,
        leaf_tuples,
        effect_names,
        n_obj,
        n_rel,
        int(opts.get("num_heights", 7)),
        int(opts.get("num_stacks", 7)),
    )

    leaf_stats = []
    for leaf in sorted(set(leaves.tolist())):
        selected = labels[leaves == leaf]
        counts = Counter(selected.tolist())
        leaf_stats.append({
            "leaf": int(leaf),
            "samples": int(len(selected)),
            "purity": float(max(counts.values()) / len(selected)),
            "label_counts": {str(effect_names[int(k)]): int(v) for k, v in sorted(counts.items())},
            "symbolic_tuples": [list(x) for x in leaf_tuples.get(int(leaf), [])],
        })
    metrics = {
        "training_accuracy": float(accuracy_score(labels, predictions)),
        "weighted_leaf_purity": float(sum(item["samples"] * item["purity"] for item in leaf_stats) / len(labels)),
        "num_training_samples": int(len(labels)),
        "num_tree_leaves": int(tree.get_n_leaves()),
        "tree_depth": int(tree.get_depth()),
        "num_object_codes": n_obj,
        "num_relation_codes": n_rel,
        "num_possible_symbolic_tuples": len(symbolic_tuples),
        "num_exported_stack_actions": len(leaf_tuples),
        "leaf_stats": leaf_stats,
    }
    with open(os.path.join(save_dir, "rule_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("Saved tree:", tree_path)
    print("Saved domain:", domain_path)
    print("Training accuracy: %.6f" % metrics["training_accuracy"])
    print("Weighted leaf purity: %.6f" % metrics["weighted_leaf_purity"])
    print("Tree depth/leaves: %d/%d" % (metrics["tree_depth"], metrics["num_tree_leaves"]))
    print("Exported stack actions:", metrics["num_exported_stack_actions"])


if __name__ == "__main__":
    main()