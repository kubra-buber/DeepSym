#!/bin/bash
# Railroad-based planning pipeline
# Replaces the old mdpsim + mini-gpt pipeline with Railroad's MCTSPlanner
#
# Usage: ./make_plan.sh opts.yaml "(S4)"

# Railroad conda environment (Python 3.12 required)
RAILROAD_PYTHON="${RAILROAD_PYTHON:-/home/kubra/miniconda3/envs/deepsym_railroad/bin/python}"

# get save location
loc="$(grep save: $1 | sed 's/^.*: //')"

# transform image to pddl problem (scene recognition)
python recognize.py -opts "$1" -goal "$2"

# Run Railroad planner (replaces mdpsim server + mini-gpt planner + parse_plan.py)
$RAILROAD_PYTHON make_plan_railroad.py -opts "$1" -goal "$2"

# The plan is now in {savepath}/plan.txt, ready for execute_plan.py
echo "Plan saved to $loc/plan.txt"