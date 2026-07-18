#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import signal
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ppddl_parser import parse_domain, parse_problem
from railroad_builder import build_operators, build_problem, ground_operators
from generic_expected_planner import ExactExpectedReachabilityPlanner
from railroad.core import transition


# ---------------------------------------------------------------------------
# TIM second-prototype determinization logic
# Reproduces src/bilevel/parse.py and evaluate.py:
#   - sample 0 is domain.pddl
#   - samples 1..N-1 come from domain_prob.pddl
#   - seed for sample i is 12 * (i - 1)
# ---------------------------------------------------------------------------

def extract_actions(domain_str: str):
    actions = []
    lines = domain_str.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("(:action"):
            action_block = []
            paren_count = 0
            in_action = False

            while i < len(lines):
                current_line = lines[i]
                action_block.append(current_line)
                paren_count += current_line.count("(")
                paren_count -= current_line.count(")")

                if not in_action:
                    tokens = current_line.replace("(", "").split()
                    if len(tokens) >= 2:
                        action_name = tokens[1]
                        in_action = True

                if in_action and paren_count == 0:
                    break
                i += 1

            actions.append((action_name, "\n".join(action_block)))
        else:
            i += 1

    return actions


def extract_probabilistic_block(full_action_block: str):
    start_idx = full_action_block.find(":effect")
    if start_idx == -1:
        return None, -1, -1

    effect_start = full_action_block[start_idx:]
    prob_start_idx = effect_start.find("(probabilistic")
    if prob_start_idx == -1:
        return None, -1, -1

    abs_prob_start = start_idx + prob_start_idx
    paren_count = 0

    for offset, char in enumerate(full_action_block[abs_prob_start:]):
        if char == "(":
            paren_count += 1
        elif char == ")":
            paren_count -= 1

        if paren_count == 0:
            end_idx = abs_prob_start + offset
            return (
                full_action_block[abs_prob_start:end_idx + 1],
                abs_prob_start,
                end_idx + 1,
            )

    return None, -1, -1


def sample_effect_from_prob_block(prob_block: str, seed: Optional[int] = None):
    # This intentionally matches TIM: the RNG is reset for every action block.
    if seed is not None:
        random.seed(seed)

    prob_block = prob_block.strip()
    if not prob_block.startswith("(probabilistic"):
        raise ValueError("Block must start with '(probabilistic'")

    content = prob_block[len("(probabilistic"):].strip()
    i = 0
    effects = []

    while i < len(content):
        while i < len(content) and content[i].isspace():
            i += 1
        if i >= len(content):
            break

        start = i
        while i < len(content) and (
            content[i].isdigit() or content[i] in ".eE-+"
        ):
            i += 1

        prob_str = content[start:i].strip()
        if not prob_str:
            break
        probability = float(prob_str)

        while i < len(content) and content[i].isspace():
            i += 1
        if i >= len(content) or content[i] != "(":
            raise ValueError(
                f"Expected '(' after probability {probability}"
            )

        effect_start = i
        paren_count = 0
        while i < len(content):
            if content[i] == "(":
                paren_count += 1
            elif content[i] == ")":
                paren_count -= 1
                if paren_count == 0:
                    i += 1
                    break
            i += 1

        effects.append(
            (probability, content[effect_start:i].strip())
        )

    if not effects:
        raise ValueError("No effects found in probabilistic block")

    probabilities, effect_bodies = zip(*effects)
    selected = random.choices(
        range(len(probabilities)),
        weights=probabilities,
    )[0]
    return effect_bodies[selected]


def replace_probabilistic_with_sampled_effect(
    full_action_block: str,
    seed: Optional[int] = None,
):
    prob_block, start_idx, end_idx = extract_probabilistic_block(
        full_action_block
    )
    if prob_block is None:
        return full_action_block

    sampled_effect = sample_effect_from_prob_block(
        prob_block,
        seed=seed,
    )
    return (
        full_action_block[:start_idx]
        + sampled_effect
        + full_action_block[end_idx:]
    )


def determinize_domain(action_blocks, domain_str: str, seed: int):
    new_domain = domain_str

    for _, full_block in action_blocks:
        if ":effect" in full_block and "probabilistic" in full_block:
            new_block = replace_probabilistic_with_sampled_effect(
                full_block,
                seed=seed,
            )
            new_domain = new_domain.replace(full_block, new_block)

    return new_domain


def sample_domains_like_tim(
    nominal_domain: Path,
    probabilistic_domain: Path,
    number: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    prob_template = probabilistic_domain.read_text().replace(
        ":probabilistic-effects",
        "",
    )
    action_blocks = extract_actions(prob_template)

    paths = []

    first = output_dir / "domain_determinized_0.pddl"
    shutil.copyfile(nominal_domain, first)
    paths.append(first.resolve())

    for index in range(1, number):
        t = index - 1
        seed = t + 11 * t  # exact TIM expression: 12*t
        content = determinize_domain(
            action_blocks,
            prob_template,
            seed=seed,
        )
        path = output_dir / f"domain_determinized_{index}.pddl"
        path.write_text(content)
        paths.append(path.resolve())

    return paths


# ---------------------------------------------------------------------------
# Fast Downward
# ---------------------------------------------------------------------------

def locate_fast_downward(explicit: Optional[str]) -> Path:
    candidates: List[Path] = []

    if explicit:
        candidates.append(Path(explicit).expanduser())

    for executable in ("fast-downward.py", "downward"):
        found = shutil.which(executable)
        if found:
            candidates.append(Path(found))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "downward" / "fast-downward.py",
            cwd.parent / "downward" / "fast-downward.py",
            Path.home() / "downward" / "fast-downward.py",
            Path.home() / "DeepSym" / "downward" / "fast-downward.py",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Fast Downward bulunamadı. --downward ile fast-downward.py "
        "dosyasının yolunu verin."
    )


def run_fast_downward(
    downward: Path,
    domain: Path,
    problem: Path,
    plan_file: Path,
    timeout: int,
):
    plan_file.parent.mkdir(parents=True, exist_ok=True)
    if plan_file.exists():
        plan_file.unlink()

    command = [
        str(downward),
        "--plan-file",
        str(plan_file.resolve()),
        str(domain.resolve()),
        str(problem.resolve()),
        "--search",
        "astar(blind())",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()
        return False, output, "timeout"

    solved = process.returncode == 0 and plan_file.exists()
    return solved, output, str(process.returncode)


def parse_raw_plan(plan_file: Path) -> Tuple[str, ...]:
    if not plan_file.exists():
        return tuple()

    actions = []
    for raw_line in plan_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1].strip()
        if line:
            actions.append(" ".join(line.lower().split()))

    return tuple(actions)


# ---------------------------------------------------------------------------
# PPDDL fixed-plan evaluation through Railroad transitions
# ---------------------------------------------------------------------------

def state_key(state) -> Tuple[str, ...]:
    return tuple(sorted(str(fluent) for fluent in state.fluents))


def normalize_action_name(name: str) -> str:
    return " ".join(name.lower().strip().strip("()").split())


def logical_action_name(name: str) -> str:
    """Ignore the final learned sample-count suffix, e.g. _c364/_c599."""
    normalized = normalize_action_name(name)
    if not normalized:
        return normalized
    parts = normalized.split()
    parts[0] = re.sub(r"_c\d+$", "", parts[0])
    return " ".join(parts)


def evaluate_fixed_plan(
    initial_state,
    goal,
    plan: Sequence[str],
    action_map: Dict[str, object],
):
    distribution = {state_key(initial_state): [initial_state, 1.0]}

    for step in plan:
        action = action_map.get(logical_action_name(step))
        if action is None:
            return 0.0, f"unknown logical action: {logical_action_name(step)}"

        next_distribution = defaultdict(float)
        next_states = {}

        for _, (state, state_probability) in distribution.items():
            try:
                outcomes = transition(state, action)
            except RuntimeError as exc:
                if "precondition not satisfied" in str(exc).lower():
                    continue
                raise

            for next_state, outcome_probability in outcomes:
                key = state_key(next_state)
                next_states[key] = next_state
                next_distribution[key] += (
                    state_probability * float(outcome_probability)
                )

        distribution = {
            key: [next_states[key], probability]
            for key, probability in next_distribution.items()
            if probability > 0.0
        }

        if not distribution:
            return 0.0, "all branches became inapplicable"

    probability = sum(
        state_probability
        for state, state_probability in distribution.values()
        if goal.evaluate(state.fluents)
    )
    return probability, None


def load_railroad_problem(prob_domain: Path, problem: Path):
    parsed_domain = parse_domain(prob_domain)
    parsed_problem = parse_problem(problem)
    rr_problem = build_problem(parsed_problem)

    actions = ground_operators(
        build_operators(parsed_domain),
        rr_problem.objects_by_type,
    )
    action_map = {}
    for action in actions:
        key = logical_action_name(action.name)
        if key in action_map and action_map[key].name != action.name:
            raise RuntimeError(
                f"Logical alias collision for {key!r}: "
                f"{action_map[key].name!r} vs {action.name!r}"
            )
        action_map[key] = action

    return parsed_domain, parsed_problem, rr_problem, actions, action_map


def format_plan(plan: Sequence[str]) -> str:
    return " -> ".join(plan) if plan else "<NO PLAN>"


def main():
    parser = argparse.ArgumentParser(
        "Compare Railroad with TIM's nominal and sampled-determinization "
        "Fast Downward planning."
    )
    parser.add_argument("--domain", default="domain.pddl")
    parser.add_argument("--prob-domain", default="domain_prob.pddl")
    parser.add_argument(
        "--problem",
        default="railroad3_real_choice/real_choice_problem.pddl",
    )
    parser.add_argument("--downward")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--railroad-result",
        default="railroad3_real_choice/planner_result.json",
    )
    parser.add_argument(
        "--output-dir",
        default="railroad3_real_choice/tim_comparison",
    )
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    nominal_domain = Path(args.domain)
    probabilistic_domain = Path(args.prob_domain)
    problem = Path(args.problem)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downward = locate_fast_downward(args.downward)
    print(f"Fast Downward: {downward}")

    (
        parsed_domain,
        parsed_problem,
        rr_problem,
        actions,
        action_map,
    ) = load_railroad_problem(probabilistic_domain, problem)

    exact = ExactExpectedReachabilityPlanner(actions)
    exact_action, exact_value, exact_root_values = exact.solve(
        rr_problem.initial_state,
        rr_problem.goal,
        horizon=args.horizon,
    )

    railroad_mcts_action = None
    railroad_result_path = Path(args.railroad_result)
    if railroad_result_path.exists():
        railroad_result = json.loads(railroad_result_path.read_text())
        railroad_mcts_action = (
            railroad_result.get("mcts", {}).get("selected_action")
        )

    print("\n=== RAILROAD REFERENCE ===")
    print(f"Exact action: {exact_action}")
    print(f"Exact reachability: {exact_value:.8f}")
    print(f"MCTS modal action: {railroad_mcts_action}")

    # 1) Original deterministic path.
    nominal_plan_path = output_dir / "sas_plan_nominal"
    nominal_solved, nominal_log, nominal_status = run_fast_downward(
        downward,
        nominal_domain,
        problem,
        nominal_plan_path,
        args.timeout,
    )
    (output_dir / "fast_downward_nominal.log").write_text(nominal_log)
    nominal_plan = parse_raw_plan(nominal_plan_path)
    nominal_probability, nominal_error = evaluate_fixed_plan(
        rr_problem.initial_state,
        rr_problem.goal,
        nominal_plan,
        action_map,
    )

    print("\n=== FAST DOWNWARD ON domain.pddl ===")
    print(f"Solved: {nominal_solved} (status={nominal_status})")
    print(f"Plan: {format_plan(nominal_plan)}")
    print(f"PPDDL fixed-plan success: {nominal_probability:.8f}")
    if nominal_error:
        print(f"Evaluation note: {nominal_error}")

    # 2) TIM sampled determinization.
    sampled_domain_dir = output_dir / "sampled_domains"
    sampled_plan_dir = output_dir / "sampled_plans"
    sampled_plan_dir.mkdir(parents=True, exist_ok=True)

    sampled_domains = sample_domains_like_tim(
        nominal_domain,
        probabilistic_domain,
        args.samples,
        sampled_domain_dir,
    )

    plan_counter = Counter()
    sample_records = []
    solved_count = 0

    for index, sampled_domain in enumerate(sampled_domains):
        plan_path = sampled_plan_dir / f"sas_plan_{index}"
        solved, log, status = run_fast_downward(
            downward,
            sampled_domain,
            problem,
            plan_path,
            args.timeout,
        )
        (sampled_plan_dir / f"fast_downward_{index}.log").write_text(log)

        plan = parse_raw_plan(plan_path)
        plan_counter[plan] += 1
        solved_count += int(solved)

        sample_records.append(
            {
                "index": index,
                "domain": str(sampled_domain),
                "solved": solved,
                "status": status,
                "plan": list(plan),
            }
        )

    selected_sampled_plan = (
        plan_counter.most_common(1)[0][0]
        if plan_counter
        else tuple()
    )
    sampled_probability, sampled_error = evaluate_fixed_plan(
        rr_problem.initial_state,
        rr_problem.goal,
        selected_sampled_plan,
        action_map,
    )

    print("\n=== TIM SAMPLED DETERMINIZATION ===")
    print(f"Samples: {args.samples}")
    print(f"Fast Downward solved: {solved_count}/{args.samples}")
    print(
        "Selected most-frequent plan: "
        f"{format_plan(selected_sampled_plan)}"
    )
    print(
        f"Frequency: "
        f"{plan_counter[selected_sampled_plan]}/{args.samples}"
    )
    print(f"PPDDL fixed-plan success: {sampled_probability:.8f}")
    if sampled_error:
        print(f"Evaluation note: {sampled_error}")

    print("\nTop sampled plans:")
    unique_plan_results = []
    for plan, count in plan_counter.most_common(args.top):
        probability, error = evaluate_fixed_plan(
            rr_problem.initial_state,
            rr_problem.goal,
            plan,
            action_map,
        )
        unique_plan_results.append(
            {
                "plan": list(plan),
                "count": count,
                "frequency": count / args.samples,
                "ppddl_success_probability": probability,
                "evaluation_error": error,
            }
        )
        print(
            f"  {count:>3}/{args.samples}: "
            f"P={probability:.8f} | {format_plan(plan)}"
        )

    nominal_first = nominal_plan[0] if nominal_plan else None
    sampled_first = (
        selected_sampled_plan[0]
        if selected_sampled_plan
        else None
    )

    print("\n=== FIRST-ACTION COMPARISON ===")
    print(f"Railroad exact: {exact_action}")
    print(f"Railroad MCTS:  {railroad_mcts_action}")
    print(f"FD nominal:     {nominal_first}")
    print(f"TIM sampled:    {sampled_first}")

    result = {
        "domain": parsed_domain.name,
        "problem": parsed_problem.name,
        "fast_downward": str(downward),
        "railroad": {
            "exact_action": exact_action,
            "exact_reachability": exact_value,
            "mcts_modal_action": railroad_mcts_action,
            "root_action_values": exact_root_values,
        },
        "fast_downward_nominal": {
            "solved": nominal_solved,
            "status": nominal_status,
            "plan": list(nominal_plan),
            "first_action": nominal_first,
            "ppddl_fixed_plan_success_probability": nominal_probability,
            "evaluation_error": nominal_error,
        },
        "tim_sampled_determinization": {
            "samples": args.samples,
            "solved_count": solved_count,
            "selected_plan": list(selected_sampled_plan),
            "selected_first_action": sampled_first,
            "selected_frequency": (
                plan_counter[selected_sampled_plan]
                if plan_counter
                else 0
            ),
            "ppddl_fixed_plan_success_probability": sampled_probability,
            "evaluation_error": sampled_error,
            "top_plans": unique_plan_results,
            "sample_records": sample_records,
        },
        "agreement": {
            "exact_vs_nominal_first": (
                logical_action_name(exact_action)
                == logical_action_name(nominal_first or "")
            ),
            "exact_vs_sampled_first": (
                logical_action_name(exact_action)
                == logical_action_name(sampled_first or "")
            ),
            "mcts_vs_nominal_first": (
                railroad_mcts_action is not None
                and logical_action_name(railroad_mcts_action)
                == logical_action_name(nominal_first or "")
            ),
            "mcts_vs_sampled_first": (
                railroad_mcts_action is not None
                and logical_action_name(railroad_mcts_action)
                == logical_action_name(sampled_first or "")
            ),
        },
    }

    result_path = output_dir / "comparison_result.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote: {result_path}")


if __name__ == "__main__":
    main()