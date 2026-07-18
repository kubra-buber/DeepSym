#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators

from railroad.planner import MCTSPlanner


def load_case(case_dir: Path):
    manifest = json.loads((case_dir / "controlled_manifest.json").read_text())
    problem_path = case_dir / manifest["files"]["problem"]
    probabilistic_domain_path = case_dir / manifest["files"]["probabilistic_domain"]
    nominal_domain_path = case_dir / manifest["files"]["nominal_domain"]
    return manifest, problem_path, probabilistic_domain_path, nominal_domain_path


def build_mcts_inputs(domain_path: Path, problem_path: Path):
    domain = parse_domain(domain_path)
    problem = parse_problem(problem_path)

    if problem.domain_name != domain.name:
        raise ValueError(
            f"Problem expects domain {problem.domain_name!r}, "
            f"but domain file defines {domain.name!r}"
        )

    rr_problem = build_problem(problem)
    built_operators = build_operators(domain)
    grounded_actions = ground_operators(
        built_operators,
        rr_problem.objects_by_type,
    )

    return rr_problem, grounded_actions


def expected_action_name(manifest: dict) -> str:
    explicit = manifest.get("expected_ground_action")
    if explicit:
        return str(explicit)

    binding = manifest["binding"]
    # The controlled generator currently uses ?o0, ?o1 ordering.
    return (
        f"{manifest['selected_action']} "
        f"{binding['?o0']} {binding['?o1']}"
    )


def run_planner(
    label: str,
    domain_path: Path,
    problem_path: Path,
    *,
    runs: int,
    iterations: int,
    max_depth: int,
    c: float,
    heuristic_multiplier: float,
    show_trace: bool,
):
    rr_problem, grounded_actions = build_mcts_inputs(domain_path, problem_path)

    print(f"\n=== {label} ===")
    print(f"Grounded actions: {len(grounded_actions)}")
    for action in grounded_actions:
        print(f"  {action.name}")

    selections = Counter()
    last_planner = None

    for _ in range(runs):
        # Use a fresh planner for every run so previous search trees do not
        # influence the next result.
        planner = MCTSPlanner(grounded_actions)
        selected = planner(
            rr_problem.initial_state,
            rr_problem.goal,
            max_iterations=iterations,
            max_depth=max_depth,
            c=c,
            heuristic_multiplier=heuristic_multiplier,
        )
        selections[str(selected)] += 1
        last_planner = planner

    print("MCTS selections:")
    for action_name, count in selections.most_common():
        print(f"  {action_name!r}: {count}/{runs}")

    if show_trace and last_planner is not None:
        print("\nLast MCTS trace:")
        try:
            trace = last_planner.get_trace_from_last_mcts_tree()
            print(trace)
        except Exception as exc:
            print(f"Trace unavailable: {exc}")

    return selections


def main():
    parser = argparse.ArgumentParser(
        "Run Railroad MCTSPlanner on the controlled PPDDL case."
    )
    parser.add_argument(
        "--case-dir",
        default="railroad3_controlled",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--c", type=float, default=1.414)
    parser.add_argument("--heuristic-multiplier", type=float, default=5.0)
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument(
        "--include-nominal",
        action="store_true",
        help=(
            "Also run MCTS on the nominal domain. In the current controlled "
            "case the nominal effect cannot reach the goal, so this output is "
            "diagnostic rather than a success assertion."
        ),
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    (
        manifest,
        problem_path,
        probabilistic_domain_path,
        nominal_domain_path,
    ) = load_case(case_dir)

    expected = expected_action_name(manifest)

    probabilistic_selections = run_planner(
        "PROBABILISTIC MCTS",
        probabilistic_domain_path,
        problem_path,
        runs=args.runs,
        iterations=args.iterations,
        max_depth=args.max_depth,
        c=args.c,
        heuristic_multiplier=args.heuristic_multiplier,
        show_trace=args.show_trace,
    )

    selected_expected = probabilistic_selections.get(expected, 0)
    print(f"\nExpected controlled action: {expected!r}")

    if selected_expected != args.runs:
        raise AssertionError(
            f"Expected MCTS to select {expected!r} in all {args.runs} runs, "
            f"but it selected it {selected_expected} times. "
            f"Selections: {dict(probabilistic_selections)}"
        )

    print("PASS: Railroad MCTS selected the controlled probabilistic action.")

    if args.include_nominal:
        run_planner(
            "NOMINAL MCTS (diagnostic)",
            nominal_domain_path,
            problem_path,
            runs=args.runs,
            iterations=args.iterations,
            max_depth=args.max_depth,
            c=args.c,
            heuristic_multiplier=args.heuristic_multiplier,
            show_trace=args.show_trace,
        )
        print(
            "\nNote: the current nominal branch has zero goal probability. "
            "Its returned action should not be interpreted as a valid plan."
        )


if __name__ == "__main__":
    main()