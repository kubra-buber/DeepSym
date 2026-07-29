# Railroad MCTS Notes

We tested Railroad MCTS in the DeepSym probabilistic stacking domain and collected the following notes while debugging and adapting the planner.

- Package: `railroad 0.2.0`
- Tested commit: `6bc74416138eed331cde6615421a0b75dc04a8ed`
- DeepSym repository: <https://github.com/alper111/DeepSym>

DeepSym requires multi-step plans in which later actions are enabled by earlier actions and their probabilistic outcomes. In the stacking domain, a plan typically alternates between selecting a base, updating symbolic height or stack counters, and applying a learned stacking action. The planner must therefore represent actions that become relevant later and distinguish branches with different probabilities of reaching the goal.

## Notes from testing

### Heuristic-failure states could be valued similarly to goal states

In the tested version, the success reward and the value substituted when the heuristic returned infinity were both zero:

```cpp
HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
SUCCESS_REWARD = 0.0;
```

This does not mean that every state with an infinite heuristic is a proven dead end. However, in our controlled tests, replacing infinity with zero could make a non-goal branch receive the same value as a successful branch with the same accumulated cost. Actions with clearly different goal-reachability probabilities could then obtain similar Q values, and selection could depend on action ordering.

For testing, we changed the substituted value to:

```cpp
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 100.0;
```

After this change, the planner generally preferred actions with higher goal-reachability probability in the controlled tests.

### The MCTS action universe was pruned once from the root state

At the start of MCTS, Railroad computes:

```cpp
auto all_actions = get_usable_actions(root_state, all_actions_base);
```

`get_usable_actions()` performs a relaxed reachability analysis starting from the root state. The resulting subset is then reused for every node in the MCTS tree.

This caused a problem in our DeepSym domain. Some actions are needed only after earlier actions or probabilistic outcomes change the state. If such an action is removed during the initial root-based pruning, it can never be considered later in the same search, even when its preconditions eventually become satisfied.

For testing, we disabled this one-time pruning and kept the complete grounded action set:

```cpp
const auto &all_actions = all_actions_base;
```

The existing per-state action generation was then used to determine which actions were actually applicable at each visited state.

### Goal states could still be expanded

In the selection loop, the checks for untried actions and child nodes occurred before the goal check. A state that already satisfied the goal could therefore leave selection for expansion while still marked as a non-goal node.

This allowed additional actions and costs to be added after the goal had already been reached. We moved the goal check before the untried-action and child checks so that goal states were treated as terminal.

### Expansion could go one level beyond `max_depth`

The selection loop respected `max_depth`, but the following expansion block did not have its own depth condition. A search configured with depth 3 could therefore create a depth-4 node.

We added a separate `depth < max_depth` check before expansion.

### `NONE` needed to be handled by the DeepSym wrapper

Railroad deliberately returns the string:

```text
NONE
```

when no root action is selected. Our Python integration initially tried to interpret this sentinel as a grounded action name, which produced a misleading mapping error.

We changed the wrapper to handle it as a planner failure:

```python
if text.upper() == "NONE":
    return None
```

### Additional tracing was added for diagnosis

We added root-level action and outcome statistics, including visit counts, mean rewards, UCB values and probabilistic outcome visits. This helped separate problems caused by action pruning, heuristic values, reward propagation and exploration.

## Heuristic observations

The original setup combined additive and FF heuristic components with a heuristic multiplier of 5:

```text
c = 1.41421356237
heuristic_multiplier = 5
lambda_add = 0.5
lambda_max = 0
lambda_ff = 0.5
```

In our stacking tests, this setup often produced unstable or low-quality plans. Several DeepSym subgoals can be supported by the same physical stacking sequence, while the additive component sums costs over goal fluents and can count shared support more than once.

Using the max component produced better and more stable plans in our tests. We also tested Max+FF, but its performance was almost identical to Max-only. We therefore continued with the simpler Max-only configuration:

```text
c = 3
heuristic_multiplier = 1
lambda_add = 0
lambda_max = 1
lambda_ff = 0
```

These notes describe the behavior observed in our DeepSym experiments and the local changes used to continue testing.
