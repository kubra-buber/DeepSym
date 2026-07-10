"""Execute DeepSym plan.txt commands.

Backward-compatible with the original execute_plan.py, with additions for
closed-loop planning and automatic observation:

  --one-step
      Execute only the first physical stack command in the file.

  --support-z Z
      Override the placement support height for one-step execution.

  --executed-action-file PATH
      Write a JSON record containing the executed command and raw simulator
      object positions before/after execution.  observe_outcome.py uses this to
      classify the actual symbolic outcome automatically.

The plan file format remains:

    N
    O1 x y size
    ...
    plan probability: 0.123
    stack O1 O2
    ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from simtools.rosutils import RosNode


def read_plan(path: str):
    lines = [line.strip() for line in open(path, "r") if line.strip()]
    if not lines:
        raise ValueError(f"Empty plan file: {path}")

    try:
        n = int(lines[0])
    except Exception as exc:
        raise ValueError(
            f"Plan file must start with object count. First line was: {lines[0]!r}"
        ) from exc

    if len(lines) < n + 2:
        raise ValueError(f"Plan file {path} is too short for {n} objects")

    obj_names: List[str] = []
    obj_locs: List[List[float]] = []
    obj_sizes: List[float] = []
    for i in range(n):
        name, x, y, size = lines[i + 1].split()[:4]
        obj_names.append(name.upper())
        obj_locs.append([float(x), float(y)])
        obj_sizes.append(float(size))

    status_line = lines[n + 1]
    if status_line == "not found.":
        return obj_names, obj_locs, obj_sizes, 0.0, []
    if not status_line.startswith("plan probability:"):
        raise ValueError(f"Expected 'plan probability:' line, got: {status_line!r}")
    probability = float(status_line.split(":", 1)[1].strip())

    stack_lines = []
    for line in lines[n + 2:]:
        parts = line.split()
        if len(parts) == 3 and parts[0].lower() == "stack":
            stack_lines.append((parts[1].upper(), parts[2].upper()))
    return obj_names, obj_locs, obj_sizes, probability, stack_lines


def safe_get_object_positions(node: RosNode, label: str):
    """Return raw all-object poses for observation.

    Prefer /getAllObjectPoses, which should publish all generated objects.
    Fall back to legacy /getObjectPosition only so old scenes still run, but
    that legacy topic is usually just one 3D point and is not enough for true
    automatic observation.
    """
    try:
        return node.getAllObjectPoses(timeout=1.0)
    except Exception as exc_all:
        print(f"WARNING: could not read /getAllObjectPoses {label}: {exc_all}")
        try:
            return node.getObjectPosition()
        except Exception as exc_one:
            print(f"WARNING: could not read /getObjectPosition {label}: {exc_one}")
            return None


def execute_stack(node: RosNode,
                  obj_names: List[str],
                  obj_locs: List[List[float]],
                  obj_sizes: List[float],
                  below: str,
                  above: str,
                  *,
                  support_z: float | None,
                  base_level_state: Dict[str, float]) -> None:
    below = below.upper()
    above = above.upper()
    below_idx = obj_names.index(below)
    above_idx = obj_names.index(above)

    below_loc = obj_locs[below_idx]
    above_loc = obj_locs[above_idx]
    above_size = obj_sizes[above_idx]

    node.initArmPose()

    # Pick the object from its current 2D location.
    node.move(above_loc + [1.0, 0.0, 0.0, 0.0, 1.0])
    node.move(above_loc + [0.7 + 0.85 * above_size, 0.0, 0.0, 0.0, 1.0])
    node.handGraspPose()

    if support_z is None:
        # Backward-compatible whole-plan behavior.
        base_level_state["base_level"] += obj_sizes[below_idx]
        place_z = base_level_state["base_level"] + above_size + 0.05
    else:
        # Closed-loop behavior: support_z is the current support/top height.
        place_z = support_z + above_size + 0.05

    node.move(above_loc + [place_z, 0.0, 0.0, 0.0, 1.0])
    node.move(below_loc + [place_z, 0.0, 0.0, 0.0, 1.0])
    node.handOpenPose()
    node.wait(0.5)

    # Local bookkeeping for whole-plan execution.  In closed-loop mode the
    # controller maintains the authoritative object/tower state.
    obj_locs[above_idx] = list(below_loc)


def main() -> None:
    parser = argparse.ArgumentParser("Execute DeepSym plan.txt")
    parser.add_argument("-p", help="plan file", type=str, required=True)
    parser.add_argument("-uri", help="ROS master URI", type=str, default="http://localhost:11311")
    parser.add_argument("--one-step", action="store_true", help="execute only the first stack command")
    parser.add_argument(
        "--support-z",
        type=float,
        default=None,
        help="closed-loop placement support height for the first stack command",
    )
    parser.add_argument(
        "--executed-action-file",
        type=str,
        default=None,
        help="optional JSON file to record the executed stack action and before/after object positions",
    )
    args = parser.parse_args()

    obj_names, obj_locs, obj_sizes, probability, stack_lines = read_plan(args.p)

    print(f"Plan success probability: {probability:.6f}")
    if not stack_lines:
        print("No physical stack command to execute.")
        return

    # Keep the plan-header locations exactly as read.  execute_stack mutates
    # obj_locs for local bookkeeping after a successful command; the observer
    # needs the pre-action plan locations, not that mutated version.
    object_locs_before_plan = [list(xy) for xy in obj_locs]

    node = RosNode("execute_plan", args.uri, wait_time=2.5)
    base_level_state = {"base_level": 0.7}

    commands = stack_lines[:1] if args.one_step else stack_lines
    before_positions_raw = safe_get_object_positions(node, "before execution")

    for idx, (below, above) in enumerate(commands):
        support_z = args.support_z if args.one_step and idx == 0 else None
        print(f"Executing: stack {below} {above}")
        if support_z is not None:
            print(f"  support_z={support_z:.5f}")
        execute_stack(
            node,
            obj_names,
            obj_locs,
            obj_sizes,
            below,
            above,
            support_z=support_z,
            base_level_state=base_level_state,
        )

    node.initArmPose()
    node.wait(0.5)
    after_positions_raw = safe_get_object_positions(node, "after execution")

    if args.executed_action_file:
        out = {
            "plan_file": str(Path(args.p).resolve()),
            "probability": probability,
            "object_names": obj_names,
            "object_locs": object_locs_before_plan,
            "object_locs_after_bookkeeping": obj_locs,
            "object_sizes": obj_sizes,
            "support_z": args.support_z,
            "one_step": bool(args.one_step),
            "executed": [
                {"below": below.upper(), "above": above.upper()}
                for below, above in commands
            ],
            "before_object_positions_raw": before_positions_raw,
            "after_object_positions_raw": after_positions_raw,
        }
        with open(args.executed_action_file, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        print(f"Wrote executed-action observation data to {args.executed_action_file}")


if __name__ == "__main__":
    main()