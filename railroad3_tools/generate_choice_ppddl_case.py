#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAIN = """(define (domain controlled-choice)
  (:requirements :strips :typing :negative-preconditions :probabilistic-effects)

  (:predicates
    (start ?x - object)
    (goal ?x - object)
    (dead ?x - object)
  )

  (:action low_success
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.20 (and
        (goal ?x)
        (not (start ?x))
      )
      0.80 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )

  (:action high_success
    :parameters (?x - object)
    :precondition (start ?x)
    :effect (probabilistic
      0.80 (and
        (goal ?x)
        (not (start ?x))
      )
      0.20 (and
        (dead ?x)
        (not (start ?x))
      )
    )
  )
)
"""

PROBLEM = """(define (problem controlled-choice-problem)
  (:domain controlled-choice)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        "Create a two-action PPDDL case for testing probabilistic action choice."
    )
    parser.add_argument(
        "--output-dir",
        default="railroad3_choice",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_path = output_dir / "choice_domain_prob.pddl"
    problem_path = output_dir / "choice_problem.pddl"
    manifest_path = output_dir / "choice_manifest.json"

    domain_path.write_text(DOMAIN)
    problem_path.write_text(PROBLEM)

    manifest = {
        "domain": domain_path.name,
        "problem": problem_path.name,
        "expected_action": "high_success obj0",
        "goal_probabilities": {
            "low_success obj0": 0.20,
            "high_success obj0": 0.80,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Wrote: {domain_path}")
    print(f"Wrote: {problem_path}")
    print(f"Wrote: {manifest_path}")
    print("Expected MCTS action: high_success obj0")


if __name__ == "__main__":
    main()