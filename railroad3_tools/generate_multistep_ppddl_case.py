#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAIN = """(define (domain controlled-multistep)
  (:requirements :strips :typing :probabilistic-effects)

  (:predicates
    (start ?x - object)
    (ready ?x - object)
    (goal ?x - object)
    (dead ?x - object)
  )

  (:action risky_direct
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.55 (and
        (goal ?x)
        (not (start ?x))
      )
      0.45 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )

  (:action prepare_safe
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (and
      (ready ?x)
      (not (start ?x))
    )
  )

  (:action finish_safe
    :parameters (?x - object)
    :precondition (ready ?x)
    :effect (probabilistic
      0.90 (and
        (goal ?x)
        (not (ready ?x))
      )
      0.10 (and
        (dead ?x)
        (not (ready ?x))
      )
    )
  )
)
"""

PROBLEM = """(define (problem controlled-multistep-problem)
  (:domain controlled-multistep)

  (:objects
    obj0 - object
  )

  (:init
    (start obj0)
  )

  (:goal
    (goal obj0)
  )
)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="railroad3_multistep",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_path = output_dir / "multistep_domain_prob.pddl"
    problem_path = output_dir / "multistep_problem.pddl"
    manifest_path = output_dir / "multistep_manifest.json"

    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)

    manifest = {
        "domain": domain_path.name,
        "problem": problem_path.name,
        "horizon": 2,
        "expected_action": "prepare_safe obj0",
        "expected_value": 0.90,
        "root_action_values": {
            "risky_direct obj0": 0.55,
            "prepare_safe obj0": 0.90,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote: {domain_path}")
    print(f"Wrote: {problem_path}")
    print(f"Wrote: {manifest_path}")
    print("Expected exact action: prepare_safe obj0")
    print("Expected reachability: 0.90 within horizon 2")


if __name__ == "__main__":
    main()