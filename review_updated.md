# Railroad MCTS Review Notes

## Scope and test domain

- Package: `railroad 0.2.0`
- Upstream commit: `6bc74416138eed331cde6615421a0b75dc04a8ed`
- Domain: DeepSym probabilistic object stacking
- Original DeepSym repository: <https://github.com/alper111/DeepSym>

DeepSym learns symbolic object types, relations and probabilistic action operators from interaction data. In the stacking domain, the planner must construct action sequences that satisfy symbolic goals such as a target height `Hk` and/or stack count `Sk`. A typical plan alternates between selecting a base, updating symbolic counters and applying learned probabilistic stacking operators:

```text
makebase
→ increase_height / increase_stack
→ stack
→ increase_height / increase_stack
→ stack
```

The learned stacking actions may lead to outcomes such as successful stacking, insertion, rolling or tumbling. Railroad MCTS was tested as a replacement for the original DeepSym `mdpsim + mini-gpt` planning pipeline. An exact finite-horizon expected-reachability planner was used as a reference and to evaluate the exact linear success probability of generated physical plans.

`representative_branch_probability` refers only to the selected diagnostic progress branch and should not be interpreted as closed-loop policy success.

## Issues encountered and local handling

### Unreachable outcomes could be valued like successful outcomes

The original configuration used zero reward for both successful states and states for which the heuristic could not find a path to the goal:

```cpp
HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
SUCCESS_REWARD = 0.0;
```

This could make actions with different success probabilities receive nearly identical Q values when their accumulated action costs were equal. In controlled tests, action selection could therefore become dependent on action ordering rather than reachability probability.

For validation, the unreachable-state penalty was changed to:

```cpp
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 100.0;
```

After this change, higher-reachability actions were selected consistently or near-consistently in controlled tests.

### Deeper search was restricted to actions usable at the root

The tested implementation filtered the grounded action set in the root state and reused that filtered set throughout the tree. Actions that became applicable only after earlier actions could therefore not appear later in the same MCTS search.

This is particularly restrictive for DeepSym because required counter-update and stacking actions become applicable sequentially. During validation, the complete grounded action set was retained:

```cpp
const auto &all_actions = all_actions_base;
```

Applicability was then evaluated independently at each visited state.

### Goal states and depth limits were not always terminal

Goal states could still be expanded when other actions remained applicable, adding unnecessary action costs after success. The expansion stage could also create a node one level beyond `max_depth`.

The local validation version checked the goal before further selection or expansion and prevented expansion when the depth limit had been reached.

### Applicability checks could disagree

At least one action returned by `get_usable_actions()` was rejected by `transition()` with a precondition error. During validation, `transition()` was treated as the authoritative applicability check and rejected actions were discarded.

### `NONE` could be treated as an action name

During sequential replanning, Railroad could return the sentinel value `NONE` when no action was selected. The Python wrapper originally attempted to resolve it as a grounded action name. The local wrapper instead recorded it as a planner failure:

```python
if text.upper() == "NONE":
    return None
```

Root action and outcome statistics were also added to the trace to inspect visit counts, Q values, UCB values and probabilistic outcomes during diagnosis.

## Heuristic configurations

The original configuration combined additive and FF heuristics:

```text
c = 1.41421356237
heuristic_multiplier = 5
lambda_add = 0.5
lambda_max = 0
lambda_ff = 0.5
```

In the tested stacking domain, this configuration often produced unstable or low-quality plans. A likely reason is that several symbolic subgoals are achieved jointly by the same physical stacking sequence. The additive heuristic treats subgoal costs as independent and can therefore count shared supporting actions more than once.

The main retained configuration used only the max heuristic:

```text
c = 3
heuristic_multiplier = 1
lambda_add = 0
lambda_max = 1
lambda_ff = 0
```

The strongest alternative combined max and FF:

```text
c = 5
heuristic_multiplier = 1
lambda_add = 0
lambda_max = 0.5
lambda_ff = 0.5
```

These should be interpreted as complete MCTS configurations rather than a pure heuristic-only ablation because the exploration constant and heuristic multiplier were also changed.

## Validation summary

Initial tests showed that the original and FF-only configurations produced more varied plans and more low-probability solutions. Configurations containing `h_max` produced more concentrated plan distributions and plans that agreed more closely with the exact and original DeepSym planners.

The final comparison used 10 scene-goal combinations and 30 independent runs per configuration, giving 300 runs per method. Every unique MCTS plan was evaluated using its exact linear success probability.

Max-only and Max+FF performed almost identically. Max-only achieved a mean exact plan probability of `0.711756`, compared with `0.711354` for Max+FF, and a mean regret of `0.027904`, compared with `0.028307`. Both selected an exact optimal plan in `49%` of runs and produced no planner failures. Max+FF was within `95%` of the optimum in `66.0%` of runs, compared with `65.3%` for Max-only.

The FF component did not provide a consistent practical improvement. Max-only was therefore retained as the main configuration because it achieved equivalent plan quality with a simpler and more interpretable heuristic, while Max+FF was retained as the strongest comparison configuration.
