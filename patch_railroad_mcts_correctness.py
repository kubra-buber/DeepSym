#!/usr/bin/env python3
"""Patch Railroad 0.2.0 MCTS correctness issues, tolerant of formatting changes."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def backup_once(path: Path) -> Path:
    backup = path.with_name(path.name + ".before_mcts_fix_v3")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def patch_penalty(path: Path) -> None:
    text = path.read_text()
    pattern = re.compile(
        r"(HEURISTIC_CANNOT_FIND_GOAL_PENALTY\s*=\s*)"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))(\s*;)"
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Penalty constant not found in {path}")

    if float(match.group(2)) == 100.0:
        print("[ok] dead-end fallback penalty is already 100.0")
        return

    backup_once(path)
    path.write_text(pattern.sub(r"\g<1>100.0\g<3>", text, count=1))
    print("[changed] dead-end fallback penalty -> 100.0")


def patch_planner(path: Path) -> None:
    text = path.read_text()
    original = text

    # 1. Keep the full grounded action set.
    full_action = "const auto &all_actions = all_actions_base;"
    if full_action in text:
        print("[ok] full grounded action set is already enabled")
    else:
        root_filter = re.compile(
            r"auto\s+all_actions\s*=\s*"
            r"get_usable_actions\s*\(\s*root_state\s*,\s*all_actions_base\s*\)\s*;"
        )
        text, count = root_filter.subn(full_action, text, count=1)
        if count != 1:
            raise RuntimeError("Could not locate root-state action filtering")
        print("[changed] enabled full grounded action set")

    # 2. Reorder the beginning of the Selection loop:
    #    goal -> untried actions -> no children.
    goal_first_pattern = re.compile(
        r"while\s*\(\s*depth\s*<\s*max_depth\s*\)\s*\{\s*"
        r"(?://[^\n]*\n\s*)?"
        r"if\s*\(\s*goal\s*->\s*evaluate\s*\(\s*node\s*->\s*state\.fluents\s*\(\s*\)\s*\)\s*\)\s*\{",
        re.DOTALL,
    )

    if goal_first_pattern.search(text):
        print("[ok] goal is already checked first in Selection")
    else:
        old_prefix = re.compile(
            r"(?P<indent>[ \t]*)while\s*\(\s*depth\s*<\s*max_depth\s*\)\s*\{\s*"
            r"if\s*\(\s*!node\s*->\s*untried_actions\.empty\s*\(\s*\)\s*\)\s*break\s*;\s*"
            r"if\s*\(\s*node\s*->\s*children\.empty\s*\(\s*\)\s*\)\s*break\s*;\s*"
            r"(?:(?://[^\n]*\n)\s*)?"
            r"if\s*\(\s*goal\s*->\s*evaluate\s*\(\s*node\s*->\s*state\.fluents\s*\(\s*\)\s*\)\s*\)\s*\{\s*"
            r"is_node_goal\s*=\s*true\s*;\s*"
            r"break\s*;\s*"
            r"\}",
            re.DOTALL,
        )

        def repl(match: re.Match[str]) -> str:
            i = match.group("indent")
            return (
                f"{i}while (depth < max_depth) {{\n"
                f"{i}  // Goal states are terminal even when more actions are applicable.\n"
                f"{i}  if (goal->evaluate(node->state.fluents())) {{\n"
                f"{i}    is_node_goal = true;\n"
                f"{i}    break;\n"
                f"{i}  }}\n"
                f"{i}  if (!node->untried_actions.empty()) break;\n"
                f"{i}  if (node->children.empty()) break;"
            )

        text, count = old_prefix.subn(repl, text, count=1)
        if count != 1:
            # Emit the actual local selection block to make any remaining mismatch obvious.
            marker = text.find("Selection")
            excerpt = text[max(0, marker - 100): marker + 1200] if marker >= 0 else text[:1200]
            raise RuntimeError(
                "Could not patch the Selection prefix. Local excerpt follows:\n\n" + excerpt
            )
        print("[changed] goal states are now terminal before action checks")

    # 3. Prevent one expansion beyond max_depth.
    fixed_expansion = re.compile(
        r"if\s*\(\s*depth\s*<\s*max_depth\s*&&\s*"
        r"!node\s*->\s*untried_actions\.empty\s*\(\s*\)\s*&&\s*"
        r"!is_node_goal\s*\)\s*\{",
        re.DOTALL,
    )
    if fixed_expansion.search(text):
        print("[ok] expansion already respects max_depth")
    else:
        old_expansion = re.compile(
            r"if\s*\(\s*!node\s*->\s*untried_actions\.empty\s*\(\s*\)\s*&&\s*"
            r"!is_node_goal\s*\)\s*\{",
            re.DOTALL,
        )
        replacement = (
            "if (depth < max_depth &&\n"
            "        !node->untried_actions.empty() &&\n"
            "        !is_node_goal) {"
        )
        text, count = old_expansion.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError("Could not locate MCTS expansion condition")
        print("[changed] expansion can no longer exceed max_depth")

    if text != original:
        backup = backup_once(path)
        path.write_text(text)
        print(f"[backup] {backup}")
        print(f"[written] {path}")


def verify(planner: Path, constants: Path) -> None:
    p = planner.read_text()
    c = constants.read_text()

    checks = [
        bool(re.search(
            r"HEURISTIC_CANNOT_FIND_GOAL_PENALTY\s*=\s*100(?:\.0+)?\s*;", c
        )),
        "const auto &all_actions = all_actions_base;" in p,
        bool(re.search(
            r"while\s*\(\s*depth\s*<\s*max_depth\s*\)\s*\{\s*"
            r"(?://[^\n]*\n\s*)?"
            r"if\s*\(\s*goal\s*->\s*evaluate",
            p,
            re.DOTALL,
        )),
        bool(re.search(
            r"if\s*\(\s*depth\s*<\s*max_depth\s*&&\s*"
            r"!node\s*->\s*untried_actions\.empty\s*\(\s*\)\s*&&\s*"
            r"!is_node_goal\s*\)\s*\{",
            p,
            re.DOTALL,
        )),
    ]

    if not all(checks):
        raise RuntimeError(f"Verification failed: {checks}")

    print("[verified] all four MCTS corrections are present")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "DeepSym" / "railroad_source",
    )
    args = parser.parse_args()

    include = args.source_root / "packages" / "railroad" / "include" / "railroad"
    planner = include / "planner.hpp"
    constants = include / "constants.hpp"

    if not planner.exists():
        raise FileNotFoundError(planner)
    if not constants.exists():
        raise FileNotFoundError(constants)

    patch_penalty(constants)
    patch_planner(planner)
    verify(planner, constants)


if __name__ == "__main__":
    main()
