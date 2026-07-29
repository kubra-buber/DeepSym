# Railroad MCTS Notes

We tested Railroad MCTS in the DeepSym probabilistic stacking domain and collected the following notes while debugging the planner.

- Package: `railroad 0.2.0`
- Upstream commit: `6bc74416138eed331cde6615421a0b75dc04a8ed`
- Domain: DeepSym probabilistic object stacking
- DeepSym repository: <https://github.com/alper111/DeepSym>

DeepSym requires multi-step plans in which some actions become applicable only after earlier actions succeed. A typical sequence alternates between base selection, symbolic counter updates and stacking actions. Because the learned actions are probabilistic, the planner must also distinguish branches with different probabilities of reaching the goal.

## Notes from testing

### Unreachable and successful outcomes could receive similar values

In the tested version, both the success reward and the penalty used when the heuristic could not find a path to the goal were zero:

```cpp
HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
SUCCESS_REWARD = 0.0;
```

This sometimes caused actions with clearly different success probabilities to receive very similar Q values. In simple controlled tests, the selected action could depend on action ordering.

For testing, we changed the unreachable-state penalty to:

```cpp
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 100.0;
```

After this change, the planner generally preferred the action with the higher probability of reaching the goal.

### Actions were filtered only at the root

The tested implementation filtered the grounded action set in the root state and reused that filtered set throughout the tree. Actions that became applicable only after earlier actions could therefore not appear later in the same MCTS search.

This is particularly restrictive for DeepSym because required counter-update and stacking actions become applicable sequentially. During validation, the complete grounded action set was retained:

```cpp
const auto &all_actions = all_actions_base;
```

Applicability was then evaluated independently at each visited state.

### Goal states could still be expanded

A state that already satisfied the goal could still be expanded when other actions remained applicable. This added unnecessary actions and costs after the goal had already been reached.

We changed the selection logic so that the goal condition is checked before untried actions or child nodes are considered. Goal states are therefore treated as terminal.

### Expansion could go one level beyond `max_depth`

The selection loop respected `max_depth`, but the following expansion step could still add another node. In practice, a search configured with depth 3 could generate a depth-4 node.

We added a separate depth check before expansion.

### Applicability checks were not always consistent

In at least one case, `get_usable_actions()` returned an action that was later rejected by `transition()` because its preconditions were not satisfied.

During testing, we treated `transition()` as the final applicability check and discarded actions that raised the precondition error.

### Railroad could return `NONE`

During sequential replanning, Railroad sometimes returned:

```text
NONE
```

The Python wrapper initially tried to interpret this value as a grounded action name, which produced a misleading mapping error.

We changed the wrapper so that `NONE` is recorded as a planner failure:

```python
if text.upper() == "NONE":
    return None
```

## Heuristic observations

The original setup combined the additive and FF heuristics with a relatively large heuristic multiplier. In our tests, it often produced unstable or low-quality stacking plans.

This appears to be related to the structure of the DeepSym domain: several symbolic goals are often achieved by the same physical stacking sequence. The additive heuristic can count the cost of these shared supporting actions more than once.

Using the max heuristic produced noticeably better and more stable plans. We also tested a Max+FF combination, but its results were almost identical to Max-only. We therefore continued with Max-only because it was simpler and did not perform worse in our tests.

The retained configuration was:

```text
c = 3
heuristic_multiplier = 1
lambda_add = 0
lambda_max = 1
lambda_ff = 0
```

These notes describe the behavior we observed in our DeepSym tests and the local changes used to continue the experiments.