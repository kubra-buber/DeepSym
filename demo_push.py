#!/usr/bin/env python3
"""Run one controlled push interaction for a poster screenshot.

Open simtools/rosscene_first.ttt in CoppeliaSim and start roscore first.
The script resets the scene, generates one object, moves the robot to the
pre-contact pose, waits for Enter, performs the selected interaction, then
waits again so screenshots can be captured.
"""

from __future__ import annotations

import argparse

import rospy
from scipy.spatial.transform import Rotation

from simtools.rosutils import RosNode


def main() -> None:
    parser = argparse.ArgumentParser("Single DeepSym push demo")
    parser.add_argument(
        "--action",
        choices=["top", "front", "side"],
        default="front",
    )
    parser.add_argument(
        "--object-type",
        type=int,
        choices=range(1, 6),
        default=2,
        help="1..5 using the simulator's original object-family indexing",
    )
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--x", type=float, default=-0.69)
    parser.add_argument("--y", type=float, default=-0.09)
    parser.add_argument(
        "-uri",
        default="http://localhost:11311",
        help="ROS master URI",
    )
    args = parser.parse_args()

    node = RosNode("poster_demo_push", args.uri, wait_time=2.5)
    node.stopSimulation()
    rospy.sleep(1.0)
    node.startSimulation()
    rospy.sleep(1.0)

    size = args.scale * 0.1
    x, y = args.x, args.y
    node.generateObject(
        args.object_type,
        args.scale,
        [x, y, 0.7 + size / 2.0],
    )
    rospy.sleep(1.0)

    node.initArmPose()
    node.handOpenPose()

    if args.action == "top":
        node.handPokePose()
        node.move([x - 0.05, y, 1.0, 0.0, 0.0, 0.0, 1.0])
        input(
            "Robot is above the object. Capture an approach image, "
            "then press Enter to perform the top poke..."
        )
        node.move([x - 0.05, y, 0.77 + size, 0.0, 0.0, 0.0, 1.0])

    elif args.action == "front":
        node.handFistPose()
        quat = [0.0, 0.0, 0.0, 1.0]
        node.move([x + 0.17, y, 1.0] + quat)
        node.move([x + 0.17, y, 0.70 + size / 2.0] + quat)
        input(
            "Robot is at the front pre-contact pose. Capture an image, "
            "then press Enter to push..."
        )
        node.move([x - 0.05, y, 0.70 + size / 2.0] + quat)

    else:
        node.handFistPose()
        quat = Rotation.from_euler(
            "z", 270, degrees=True
        ).as_quat().tolist()
        node.move([x, y - 0.17, 1.0] + quat)
        node.move([x, y - 0.17, 0.70 + size / 2.0] + quat)
        input(
            "Robot is at the side pre-contact pose. Capture an image, "
            "then press Enter to push..."
        )
        node.move([x, y + 0.05, 0.70 + size / 2.0] + quat)

    input(
        "Interaction finished. Capture the result image, then press Enter "
        "to return the arm to its initial pose..."
    )
    node.initArmPose()
    print("Demo finished; the simulation remains open.")


if __name__ == "__main__":
    main()