"""Export DeepSym learned decision-tree rules as Railroad-compatible JSON.

This file is intentionally ORIGINAL-COMPATIBLE.

It mirrors original learn_rules.py / utils.tree_to_code semantics as closely as
possible while supporting both:

  1. original signed-binary category.pt, shape [N, 2*code1_dim + code2_dim]
  2. VQ one-hot category.pt from save_cat.py, shape [N, 2*num_obj_codes + num_rel_codes]

For VQ one-hot, leaves are converted to operator specs by enumerating every
valid categorical tuple that satisfies the tree path.  This avoids the invalid
VQ-index -> signed-binary conversion.

Important convention
--------------------
The original DeepSym save_cat.py stores categories as:

    [first_object, second_object, relation]

and utils.tree_to_code binds:

    first_object  -> ?below / obj1_type
    second_object -> ?above / obj2_type

This file preserves that convention exactly.  Do not role-swap here if your goal
is to compare Railroad against the original PDDL pipeline.

Usage:
    python learn_rules_railroad.py -opts opts.yaml --retrain-tree
    python learn_rules_railroad.py -opts opts.yaml --retrain-tree --min-samples-leaf 1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import yaml
from sklearn.tree import DecisionTreeClassifier, _tree

import utils


@dataclass(frozen=True)
class CategoryEncoding:
    encoding: str
    code1_dim: int
    code2_dim: int
    num_obj_codes: int
    num_rel_codes: int
    num_features: int


def _load_category_encoding(save_dir: str, opts: Dict, category: torch.Tensor) -> CategoryEncoding:
    code1_dim = int(opts.get("code1_dim", 2))
    code2_dim = int(opts.get("code2_dim", 1))
    default_obj = 2 ** code1_dim
    default_rel = 2 ** code2_dim
    num_features = int(category.shape[1])

    meta_path = os.path.join(save_dir, "category_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if str(meta.get("encoding", "")).strip() == "vq_onehot":
            num_obj = int(meta.get("num_obj_codes", default_obj))
            num_rel = int(meta.get("num_rel_codes", default_rel))
            expected = 2 * num_obj + num_rel
            if num_features != expected:
                raise ValueError(
                    f"category.pt has {num_features} features but category_meta.json "
                    f"expects {expected}."
                )
            return CategoryEncoding("vq_onehot", code1_dim, code2_dim, num_obj, num_rel, num_features)

    signed_expected = 2 * code1_dim + code2_dim
    onehot_expected = 2 * default_obj + default_rel
    if num_features == onehot_expected:
        return CategoryEncoding("vq_onehot", code1_dim, code2_dim, default_obj, default_rel, num_features)
    if num_features == signed_expected:
        return CategoryEncoding("signed_binary", code1_dim, code2_dim, default_obj, default_rel, num_features)

    raise ValueError(
        f"Cannot infer category encoding for category.pt shape {tuple(category.shape)}. "
        f"Expected {signed_expected} signed-binary or {onehot_expected} VQ-one-hot features."
    )


def _feature_vector(first_idx: int, second_idx: int, rel_idx: int, enc: CategoryEncoding) -> np.ndarray:
    if enc.encoding == "vq_onehot":
        x = np.zeros(enc.num_features, dtype=np.float32)
        x[first_idx] = 1.0
        x[enc.num_obj_codes + second_idx] = 1.0
        x[2 * enc.num_obj_codes + rel_idx] = 1.0
        return x
    if enc.encoding == "signed_binary":
        first_bits = utils.decimal_to_binary(first_idx, length=enc.code1_dim)
        second_bits = utils.decimal_to_binary(second_idx, length=enc.code1_dim)
        rel_bits = utils.decimal_to_binary(rel_idx, length=enc.code2_dim)
        return np.asarray(tuple(first_bits) + tuple(second_bits) + tuple(rel_bits), dtype=np.float32)
    raise ValueError(enc.encoding)


def _all_symbolic_tuples(enc: CategoryEncoding) -> Iterable[Tuple[int, int, int, np.ndarray]]:
    for first_idx in range(enc.num_obj_codes):
        for second_idx in range(enc.num_obj_codes):
            for rel_idx in range(enc.num_rel_codes):
                yield first_idx, second_idx, rel_idx, _feature_vector(first_idx, second_idx, rel_idx, enc)


def _satisfies_path(x: np.ndarray, path: Sequence[Tuple[int, str, float]], eps: float = 1e-12) -> bool:
    for feature, op, threshold in path:
        value = float(x[feature])
        if op == "<=":
            if value > threshold + eps:
                return False
        elif op == ">":
            if value <= threshold + eps:
                return False
        else:
            raise ValueError(op)
    return True


def _tuples_for_leaf(path: Sequence[Tuple[int, str, float]], enc: CategoryEncoding) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for first_idx, second_idx, rel_idx, x in _all_symbolic_tuples(enc):
        if _satisfies_path(x, path):
            out.append((first_idx, second_idx, rel_idx))
    return out


def _normalize_prob_effects(raw: List[Dict], effect_names: Sequence[str]) -> List[Dict]:
    merged = {str(name): 0.0 for name in effect_names}
    for pe in raw:
        p = float(pe["probability"])
        if p <= 0.0:
            continue
        name = str(pe["effect_name"])
        merged[name] = merged.get(name, 0.0) + p

    total = sum(p for p in merged.values() if p > 0.0)
    if total <= 0.0:
        return []

    out = []
    for name in [str(e) for e in effect_names]:
        p = merged.get(name, 0.0)
        if p > 0.0:
            out.append({"probability": round(float(p / total), 6), "effect_name": name})

    residual = round(1.0 - sum(pe["probability"] for pe in out), 6)
    if out and abs(residual) > 0.0:
        max_i = max(range(len(out)), key=lambda i: out[i]["probability"])
        out[max_i]["probability"] = round(out[max_i]["probability"] + residual, 6)
    return [pe for pe in out if pe["probability"] > 0.0]


def _leaf_prob_effects(tree_: _tree.Tree, node: int, effect_names: Sequence[str]) -> List[Dict]:
    counts = tree_.value[node][0].astype(float)
    total = float(counts.sum())
    if total <= 0.0:
        return []
    raw = []
    for i, count in enumerate(counts):
        if count > 0.0:
            raw.append({"probability": float(count / total), "effect_name": str(effect_names[i])})
    return _normalize_prob_effects(raw, effect_names)


def tree_to_operator_specs(tree: DecisionTreeClassifier, effect_names: Sequence[str], enc: CategoryEncoding) -> List[Dict]:
    tree_ = tree.tree_
    specs: List[Dict] = []
    counter = [0]

    def recurse(node: int, path: List[Tuple[int, str, float]]) -> None:
        feature = int(tree_.feature[node])
        if feature != _tree.TREE_UNDEFINED:
            threshold = float(tree_.threshold[node])
            recurse(tree_.children_left[node], path + [(feature, "<=", threshold)])
            recurse(tree_.children_right[node], path + [(feature, ">", threshold)])
            return

        tuples = _tuples_for_leaf(path, enc)
        if not tuples:
            return
        prob_effects = _leaf_prob_effects(tree_, node, effect_names)
        if not prob_effects:
            return

        for first_idx, second_idx, rel_idx in tuples:
            # Original DeepSym convention: first category segment -> ?below;
            # second category segment -> ?above.  Keep this exact mapping.
            below_type = f"objtype{first_idx}"
            above_type = f"objtype{second_idx}"
            relation = f"relation{rel_idx}"
            specs.append({
                "name": f"stack{counter[0]}",
                "obj1_type": below_type,
                "obj2_type": above_type,
                "below_type": below_type,
                "above_type": above_type,
                "relation": relation,
                "prob_effects": prob_effects,
                "source_leaf": int(node),
            })
            counter[0] += 1

    recurse(0, [])
    return specs


def create_auxiliary_specs() -> List[Dict]:
    """Create only the remaining non-learned decision operator.

    H and S are native integer state variables. Their updates are attached
    directly to makebase and learned stack outcome branches, so numbered
    increase_height*/increase_stack* bookkeeping actions are no longer
    generated.
    """
    return [
        {
            "type": "makebase",
            "name": "makebase",
        }
    ]


def _train_or_load_tree(
    path: str,
    X: np.ndarray,
    y: np.ndarray,
    args,
) -> DecisionTreeClassifier:
    """Load the existing decision tree when compatible, otherwise train it."""
    if os.path.exists(path) and not args.retrain_tree:
        try:
            with open(path, "rb") as f:
                tree = pickle.load(f)

            if getattr(tree, "n_features_in_", X.shape[1]) == X.shape[1]:
                print(f"Loaded existing decision tree from {path}")
                return tree

            print("Existing tree feature count mismatch; retraining.")
        except Exception as exc:
            print(f"Could not load existing tree ({exc}); retraining.")

    tree = DecisionTreeClassifier(
        min_samples_leaf=args.min_samples_leaf,
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )

    print(
        "Training DecisionTreeClassifier("
        f"min_samples_leaf={args.min_samples_leaf}, "
        f"min_samples_split={args.min_samples_split}, "
        f"max_depth={args.max_depth}, "
        f"random_state={args.random_state})"
    )

    tree.fit(X, y)

    with open(path, "wb") as f:
        pickle.dump(tree, f)

    print(f"Saved decision tree to {path}")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser("Convert DeepSym tree rules to Railroad JSON.")
    parser.add_argument("-opts", type=str, required=True)
    parser.add_argument("--retrain-tree", action="store_true")
    parser.add_argument("--tree-name", type=str, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    opts = yaml.safe_load(open(args.opts, "r"))
    save_dir = opts["save"]

    category = torch.load(os.path.join(save_dir, "category.pt"))
    label = torch.load(os.path.join(save_dir, "label.pt"))
    effect_names = np.load(os.path.join(save_dir, "effect_names.npy"))

    enc = _load_category_encoding(save_dir, opts, category)
    print(f"Category encoding: {enc.encoding}")
    print(f"Category shape: {tuple(category.shape)}")
    print(f"Object codes: {enc.num_obj_codes}, relation codes: {enc.num_rel_codes}")

    X = category.detach().cpu().numpy() if isinstance(category, torch.Tensor) else np.asarray(category)
    y = label.detach().cpu().numpy().reshape(-1) if isinstance(label, torch.Tensor) else np.asarray(label).reshape(-1)

    if args.tree_name:
        tree_name = args.tree_name
    elif enc.encoding == "vq_onehot":
        depth_tag = "None" if args.max_depth is None else str(args.max_depth)
        tree_name = f"tree_vq_onehot_original_order_leaf{args.min_samples_leaf}_split{args.min_samples_split}_depth{depth_tag}.pkl"
    else:
        tree_name = "tree.pkl"
    tree_path = os.path.join(save_dir, tree_name)
    tree = _train_or_load_tree(tree_path, X, y, args)

    stack_specs = tree_to_operator_specs(tree, effect_names, enc)
    aux_specs = create_auxiliary_specs()

    out = {
        "stack_operators": stack_specs,
        "auxiliary_operators": aux_specs,
        "metadata": {
            "planner_semantics": "probabilistic_numeric_counters",
            "category_encoding": enc.encoding,
            "order": "original_deepsym_compatible",
            "code1_dim": enc.code1_dim,
            "code2_dim": enc.code2_dim,
            "num_obj_types": enc.num_obj_codes,
            "num_relations": enc.num_rel_codes,
            "num_category_features": enc.num_features,
            "effect_names": [str(e) for e in effect_names],
            "obj_names": {str(i): f"objtype{i}" for i in range(enc.num_obj_codes)},
            "relation_names": {str(i): f"relation{i}" for i in range(enc.num_rel_codes)},
            "role_convention": (
                "Preserves original DeepSym mapping: first category segment -> ?below; "
                "second category segment -> ?above."
            ),
            "tree_file": tree_name,
            "tree_min_samples_leaf": args.min_samples_leaf,
            "tree_min_samples_split": args.min_samples_split,
            "tree_max_depth": args.max_depth,
            "tree_random_state": args.random_state,
            "note": "VQ one-hot leaves are expanded by valid categorical tuple enumeration.",
        },
    }

    out_path = os.path.join(save_dir, "railroad_operators.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Created {len(stack_specs)} stack operator specs")
    print(f"Created {len(aux_specs)} auxiliary operator specs")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()