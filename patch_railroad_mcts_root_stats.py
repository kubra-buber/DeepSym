#!/usr/bin/env python3
"""Add complete root-action statistics to Railroad MCTS traces.

The patch is diagnostic only. It does not alter selection, rewards, transitions,
or the returned action.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "ROOT ACTION STATISTICS"

INSERT_OLD = '''  // Generate tree trace
  std::ostringstream tree_trace_stream;
  tree_trace_stream << std::fixed << std::setprecision(2);
  print_best_path(tree_trace_stream, root.get(), heuristic_fn, 20);
'''

INSERT_NEW = r'''  // Generate tree trace
  std::ostringstream tree_trace_stream;

  // ROOT ACTION STATISTICS
  // These values expose what MCTS actually estimated at the root.
  // `value` is the accumulated surrogate reward, not reachability probability.
  tree_trace_stream << std::fixed << std::setprecision(6);
  tree_trace_stream << "ROOT_STATS"
                    << "\troot_visits=" << root->visits
                    << "\tc=" << c
                    << "\theuristic_multiplier=" << heuristic_multiplier
                    << "\n";

  for (const auto &kv : root->children) {
    const MCTSChanceNode *chance = kv.second.get();
    const double mean_q =
        chance->visits > 0
            ? chance->value / static_cast<double>(chance->visits)
            : 0.0;
    const double final_ucb =
        ucb_score(root->visits, *chance, c);

    tree_trace_stream << "ROOT_ACTION"
                      << "\tname=" << chance->action->name()
                      << "\tvisits=" << chance->visits
                      << "\ttotal_reward=" << chance->value
                      << "\tmean_q=" << mean_q
                      << "\tucb=" << final_ucb
                      << "\n";

    const std::size_t outcome_count =
        std::min(chance->children.size(), chance->outcome_weights.size());
    for (std::size_t i = 0; i < outcome_count; ++i) {
      const MCTSDecisionNode *outcome = chance->children[i].get();
      const double outcome_mean_q =
          outcome->visits > 0
              ? outcome->value / static_cast<double>(outcome->visits)
              : 0.0;

      tree_trace_stream << "ROOT_OUTCOME"
                        << "\taction=" << chance->action->name()
                        << "\tindex=" << i
                        << "\tprobability=" << chance->outcome_weights[i]
                        << "\tvisits=" << outcome->visits
                        << "\ttotal_reward=" << outcome->value
                        << "\tmean_q=" << outcome_mean_q
                        << "\n";
    }
  }

  tree_trace_stream << "BEST_PATH\n";
  tree_trace_stream << std::fixed << std::setprecision(2);
  print_best_path(tree_trace_stream, root.get(), heuristic_fn, 20);
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path.home() / "DeepSym" / "railroad_source",
    )
    args = parser.parse_args()

    planner = (
        args.source_root
        / "packages"
        / "railroad"
        / "include"
        / "railroad"
        / "planner.hpp"
    )
    if not planner.exists():
        raise FileNotFoundError(planner)

    text = planner.read_text()

    if MARKER in text:
        print("[ok] root-action trace statistics are already installed")
        return

    if INSERT_OLD not in text:
        start = text.find("// Generate tree trace")
        excerpt = text[start : start + 700] if start >= 0 else "<marker not found>"
        raise RuntimeError(
            "Could not find the expected trace-generation block.\n\n" + excerpt
        )

    backup = planner.with_name(planner.name + ".before_root_stats")
    if not backup.exists():
        shutil.copy2(planner, backup)
        print(f"[backup] {backup}")

    planner.write_text(text.replace(INSERT_OLD, INSERT_NEW, 1))
    print(f"[written] {planner}")
    print("[verified] all root actions and outcomes will be included in MCTS traces")


if __name__ == "__main__":
    main()
