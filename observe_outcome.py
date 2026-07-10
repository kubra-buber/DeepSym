"""Automatic symbolic outcome observer for DeepSym closed-loop Railroad.

Robust/flexible version.

It reads the one-step execution record written by execute_plan.py and classifies
only the last executed stack action as one of:

    stacked, inserted, roll1, tumble1, roll2, tumble2

This version is intentionally conservative and more tolerant of simulator
/getObjectPosition formats.  It automatically searches for the coordinate layout
that best matches the plan-header object locations, instead of assuming that the
raw vector is exactly [x,y,z, x,y,z, ...].
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


Vec3 = Tuple[float, float, Optional[float]]


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _as_floats(raw: Sequence[float]) -> List[float]:
    return [float(x) for x in raw]


def _strip_optional_count(vals: List[float]) -> List[float]:
    if not vals:
        return vals
    n0 = int(round(vals[0]))
    if abs(vals[0] - n0) < 1e-6 and n0 > 0:
        rest = vals[1:]
        # Common flat layouts: N + N*stride.
        for stride in (2, 3, 4, 5, 6, 7):
            if len(rest) == n0 * stride:
                return rest
    return vals


def candidate_raw_records(raw: Sequence[float], expected_count: int) -> List[Tuple[int, List[List[float]]]]:
    """Return possible row-stride interpretations of the raw position vector.

    The simulator side is not documented here.  We therefore try several flat
    row strides and later select the coordinate columns that best match the
    known pre-action DeepSym object locations.
    """
    vals = _strip_optional_count(_as_floats(raw))
    out: List[Tuple[int, List[List[float]]]] = []
    if not vals:
        return out

    # Prefer interpretations that yield the expected object count.
    for stride in (3, 2, 4, 5, 6, 7):
        if len(vals) % stride == 0:
            rows = [vals[i:i + stride] for i in range(0, len(vals), stride)]
            if len(rows) == expected_count:
                out.append((stride, rows))

    # Fall back to any valid row split if expected count matching failed.
    if not out:
        for stride in (3, 2, 4, 5, 6, 7):
            if len(vals) % stride == 0:
                rows = [vals[i:i + stride] for i in range(0, len(vals), stride)]
                if rows:
                    out.append((stride, rows))
    return out


def _plan_header_positions(data: Dict) -> Dict[str, Tuple[float, float]]:
    names = [str(x).upper() for x in data.get("object_names", [])]
    locs = data.get("object_locs", [])
    if len(names) != len(locs):
        raise ValueError("executed action JSON has inconsistent object_names/object_locs")
    return {name: (float(loc[0]), float(loc[1])) for name, loc in zip(names, locs)}


def _assignment_for_points(expected_xy: Dict[str, Tuple[float, float]],
                           raw_xy: Sequence[Tuple[float, float]],
                           max_match_distance: float) -> Optional[Tuple[float, Dict[str, int]]]:
    """Find minimum-distance one-to-one assignment.

    N is tiny in DeepSym scenes, so brute force permutations are simpler and more
    reliable than a greedy matcher.
    """
    names = list(expected_xy.keys())
    n = len(names)
    if len(raw_xy) < n:
        return None

    best: Optional[Tuple[float, Dict[str, int]]] = None
    for perm in itertools.permutations(range(len(raw_xy)), n):
        total = 0.0
        ok = True
        for name, idx in zip(names, perm):
            d = _dist2(expected_xy[name], raw_xy[idx])
            if d > max_match_distance:
                ok = False
                break
            total += d
        if not ok:
            continue
        mapping = {name: idx for name, idx in zip(names, perm)}
        if best is None or total < best[0]:
            best = (total, mapping)
    return best


def _best_layout(raw_before: Sequence[float],
                 raw_after: Sequence[float],
                 expected_xy: Dict[str, Tuple[float, float]],
                 max_match_distance: float) -> Tuple[Dict[str, int], List[Vec3], List[Vec3], Dict]:
    expected_count = len(expected_xy)
    before_candidates = candidate_raw_records(raw_before, expected_count)
    after_candidates = candidate_raw_records(raw_after, expected_count)

    if not before_candidates:
        raise ValueError("Could not parse before_object_positions_raw into rows")
    if not after_candidates:
        raise ValueError("Could not parse after_object_positions_raw into rows")

    best = None

    for stride_b, rows_b in before_candidates:
        for stride_a, rows_a in after_candidates:
            if len(rows_b) != len(rows_a):
                continue
            stride = min(stride_b, stride_a)
            if stride < 2:
                continue

            # Try every ordered pair of columns as the horizontal plane.
            for x_col, y_col in itertools.permutations(range(stride), 2):
                raw_xy = [(row[x_col], row[y_col]) for row in rows_b]
                assignment = _assignment_for_points(expected_xy, raw_xy, max_match_distance)
                if assignment is None:
                    continue
                total, mapping = assignment

                # Choose a vertical column if one exists.  For the intended
                # /getAllObjectPoses layout [x, y, z, qx, qy, qz, qw], z is
                # column 2.  Do not choose quaternion w as z merely because its
                # absolute value is large.  Fall back to a value-based guess only
                # for unknown layouts.
                remaining_cols = [c for c in range(stride) if c not in (x_col, y_col)]
                z_col: Optional[int] = None
                if 2 in remaining_cols:
                    z_col = 2
                elif remaining_cols:
                    def col_score(c: int) -> float:
                        vals = [abs(row[c]) for row in rows_b]
                        return sum(vals) / max(len(vals), 1)
                    z_col = max(remaining_cols, key=col_score)

                before_vecs: List[Vec3] = []
                after_vecs: List[Vec3] = []
                for rb, ra in zip(rows_b, rows_a):
                    zb = None if z_col is None or z_col >= len(rb) else rb[z_col]
                    za = None if z_col is None or z_col >= len(ra) else ra[z_col]
                    # Use after vertical value for after vector; before vertical for before.
                    before_vecs.append((rb[x_col], rb[y_col], zb))
                    after_vecs.append((ra[x_col], ra[y_col], za))

                candidate = {
                    "score": total,
                    "mapping": mapping,
                    "before_vecs": before_vecs,
                    "after_vecs": after_vecs,
                    "layout": {
                        "before_stride": stride_b,
                        "after_stride": stride_a,
                        "x_col": x_col,
                        "y_col": y_col,
                        "z_col": z_col,
                        "num_rows": len(rows_b),
                    },
                }
                if best is None or total < best["score"]:
                    best = candidate

    if best is None:
        # Give a useful error, not just "missing names".
        details = {
            "expected_xy": expected_xy,
            "before_raw_len": len(raw_before) if hasattr(raw_before, "__len__") else None,
            "after_raw_len": len(raw_after) if hasattr(raw_after, "__len__") else None,
            "before_candidate_shapes": [(s, len(r)) for s, r in before_candidates],
            "after_candidate_shapes": [(s, len(r)) for s, r in after_candidates],
            "hint": "Try larger --max-match-distance or inspect raw object positions.",
        }
        raise RuntimeError("Could not match simulator object positions to plan objects: " + json.dumps(details))

    return best["mapping"], best["before_vecs"], best["after_vecs"], best["layout"]


def positions_by_name(raw_positions: Sequence[Vec3], mapping: Dict[str, int]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for name, idx in mapping.items():
        x, y, z = raw_positions[idx]
        out[name] = {"x": float(x), "y": float(y), "z": None if z is None else float(z)}
    return out


def _object_size(data: Dict, name: str) -> Optional[float]:
    names = [str(x).upper() for x in data.get("object_names", [])]
    sizes = data.get("object_sizes", [])
    for n, s in zip(names, sizes):
        if n == name:
            return float(s)
    return None


def classify_outcome(data: Dict,
                     *,
                     max_match_distance: float,
                     near_xy: float,
                     base_move_threshold: float,
                     stack_z_threshold: float,
                     no_move_threshold: float,
                     inserted_z_threshold: Optional[float]) -> Dict:
    executed = data.get("executed", [])
    if not executed:
        raise ValueError("executed action JSON does not contain an executed stack action")
    action = executed[0]
    below = str(action["below"]).upper()
    above = str(action["above"]).upper()

    before_raw = data.get("before_object_positions_raw")
    after_raw = data.get("after_object_positions_raw")
    if before_raw is None or after_raw is None:
        raise ValueError(
            "executed action JSON is missing before_object_positions_raw or "
            "after_object_positions_raw. Use the edited execute_plan.py."
        )

    expected_xy = _plan_header_positions(data)
    mapping, before_positions, after_positions, layout = _best_layout(
        before_raw,
        after_raw,
        expected_xy,
        max_match_distance=max_match_distance,
    )

    before_named = positions_by_name(before_positions, mapping)
    after_named = positions_by_name(after_positions, mapping)

    if below not in after_named or above not in after_named:
        raise RuntimeError(f"Missing below/above object in matched positions: {below}, {above}")

    b0 = before_named[below]
    a0 = before_named[above]
    b1 = after_named[below]
    a1 = after_named[above]

    below_move = _dist2((b0["x"], b0["y"]), (b1["x"], b1["y"]))
    above_move = _dist2((a0["x"], a0["y"]), (a1["x"], a1["y"]))
    above_to_below_xy = _dist2((a1["x"], a1["y"]), (b1["x"], b1["y"]))

    dz = None
    if a1.get("z") is not None and b1.get("z") is not None:
        dz = float(a1["z"]) - float(b1["z"])

    if inserted_z_threshold is None:
        # If object size is available, use a conservative fraction of the moved
        # object's size to distinguish inserted vs stacked.
        above_size = _object_size(data, above)
        inserted_z_threshold = max(0.015, 0.35 * above_size) if above_size is not None else stack_z_threshold

    if below_move > base_move_threshold:
        outcome = "tumble1"
        reason = f"below/base object moved {below_move:.4f} m > threshold {base_move_threshold:.4f}"
    elif above_to_below_xy <= near_xy:
        if dz is not None:
            if dz > stack_z_threshold:
                outcome = "stacked"
                reason = (
                    f"above object is near below in XY ({above_to_below_xy:.4f} m) "
                    f"and higher by dz={dz:.4f} m > {stack_z_threshold:.4f}"
                )
            elif abs(dz) <= inserted_z_threshold:
                outcome = "inserted"
                reason = (
                    f"above object is near below in XY ({above_to_below_xy:.4f} m) "
                    f"with small height difference dz={dz:.4f} m"
                )
            else:
                # Near in XY but height relation is odd; conservative failure of moved object.
                outcome = "tumble2"
                reason = (
                    f"above object is near below, but dz={dz:.4f} m is not consistent "
                    "with clean stacked/inserted thresholds"
                )
        else:
            outcome = "inserted"
            reason = (
                f"above object is near below in XY ({above_to_below_xy:.4f} m); "
                "z unavailable, conservatively treating as inserted"
            )
    else:
        if above_move < no_move_threshold:
            reason = (
                f"above object did not move enough ({above_move:.4f} m); "
                "treating as moved-object failure"
            )
        else:
            reason = (
                f"above object ended far from below ({above_to_below_xy:.4f} m > {near_xy:.4f}); "
                "treating as moved-object failure"
            )
        outcome = "tumble2"

    return {
        "outcome": outcome,
        "below": below,
        "above": above,
        "confidence": 1.0,
        "reason": reason,
        "metrics": {
            "below_move_xy": below_move,
            "above_move_xy": above_move,
            "above_to_below_xy": above_to_below_xy,
            "dz_above_minus_below": dz,
            "near_xy_threshold": near_xy,
            "base_move_threshold": base_move_threshold,
            "stack_z_threshold": stack_z_threshold,
            "inserted_z_threshold": inserted_z_threshold,
            "max_match_distance": max_match_distance,
        },
        "raw_mapping": mapping,
        "raw_layout": layout,
        "before_positions_by_name": before_named,
        "after_positions_by_name": after_named,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Observe/classify the last DeepSym one-step stack outcome.")
    parser.add_argument("--executed-action-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--max-match-distance", type=float, default=0.35)
    parser.add_argument("--near-xy", type=float, default=0.14)
    parser.add_argument("--base-move-threshold", type=float, default=0.10)
    parser.add_argument("--stack-z-threshold", type=float, default=0.030)
    parser.add_argument("--inserted-z-threshold", type=float, default=None)
    parser.add_argument("--no-move-threshold", type=float, default=0.035)
    args = parser.parse_args()

    path = Path(args.executed_action_file)
    with path.open("r") as f:
        data = json.load(f)

    result = classify_outcome(
        data,
        max_match_distance=args.max_match_distance,
        near_xy=args.near_xy,
        base_move_threshold=args.base_move_threshold,
        stack_z_threshold=args.stack_z_threshold,
        inserted_z_threshold=args.inserted_z_threshold,
        no_move_threshold=args.no_move_threshold,
    )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"observe_outcome.py ERROR: {exc}", file=sys.stderr)
        raise