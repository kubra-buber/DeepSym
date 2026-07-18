# Railroad review notes

## Tested version

- Package: `railroad 0.2.0`
- Commit: `6bc74416138eed331cde6615421a0b75dc04a8ed`
- Test domain: DeepSym probabilistic stacking domain
- Reference planners:
  - exact finite-horizon expected-reachability planner
  - original DeepSym `mdpsim + mini-gpt` pipeline

## Local validation patches used during testing

The following changes were applied only for diagnosis and validation:

```cpp
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 100.0;
```

and:

```cpp
const auto &all_actions = all_actions_base;
```

The second change replaces the original root-state filtering of the complete grounded action set.

---

## Issue 1 — MCTS can fail to distinguish success probabilities

Two one-step actions with success probabilities `0.80` and `0.20` initially produced equal Q values and order-dependent selection.

The likely cause is that both:

```cpp
HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 0.0;
SUCCESS_REWARD = 0.0;
```

are zero. When the heuristic cannot find a path to the goal, an unreachable outcome can therefore receive the same reward as a successful outcome with the same accumulated action cost.

Temporary validation patch:

```cpp
const double HEURISTIC_CANNOT_FIND_GOAL_PENALTY = 100.0;
```

After rebuilding, the one-step, two-step and controlled full-domain tests selected the higher-reachability action consistently or near-consistently.

### Suggested library fix

Expose the following as planner parameters instead of compile-time constants:

- unreachable-state penalty
- success reward
- heuristic multiplier

The library documentation should also clarify that “the heuristic cannot find a goal” is not necessarily equivalent to a proven dead-end.

---

## Issue 2 — `get_usable_actions()` and `transition()` can disagree

At least one action returned by:

```python
get_usable_actions(state, actions)
```

was rejected by:

```python
transition(state, action)
```

with:

```text
RuntimeError: Precondition not satisfied for applying action
```

### Current workaround

Use `transition()` as the authoritative applicability check and discard actions that raise the precondition error.

### Suggested library fix

Align the two precondition-evaluation paths and add a regression test asserting that every action returned by `get_usable_actions()` can be passed to `transition()` successfully.

---

## Issue 3 — MCTS originally restricts all deeper search to actions usable at the root

The tested MCTS implementation first computes:

```cpp
auto all_actions =
    get_usable_actions(root_state, all_actions_base);
```

and then uses this already-filtered action set when generating actions in deeper states.

This means an action that is not applicable at the root, but becomes applicable after another action or probabilistic outcome, cannot appear later in the same MCTS tree.

This is particularly problematic in DeepSym, where the action sequence alternates between different action classes:

```text
makebase
→ increase_height / increase_stack
→ stack
→ increase_height / increase_stack
→ stack
```

For example, `increase_height3` is not applicable before the preceding stack succeeds, so it is absent from the root-filtered action set even though it is required later.

### Temporary validation patch

```cpp
const auto &all_actions = all_actions_base;
```

Applicability is then evaluated separately at every visited state.

### Result

This patch allowed MCTS to represent actions that become applicable later, but it did not by itself resolve the incorrect H3 root-action selection.

### Suggested library fix

Keep the full grounded action set and filter it independently at each state. Add a multistep regression test in which each action enables an action that was initially inapplicable.

---

## Issue 4 — Goal states can be expanded instead of treated as terminal

In the MCTS selection loop, the checks for untried actions and child nodes occur before the goal test.

A goal state that still has applicable actions can therefore exit selection because it has untried actions while `is_node_goal` remains false. The expansion stage can then add another action below an already successful state.

This behavior was directly visible in the H3 trace:

```text
D:3 ... h=0.00 ... #A=2
   └── Action: stack1 O5 O3
D:4 ...
```

The state at depth 3 had heuristic value zero and already satisfied the goal, but another stack action was still expanded.

### Consequence

Successful branches can receive unnecessary extra action costs after reaching the goal. This can distort their values and can make a high-success action appear worse than an unsuccessful or delayed branch.

### Suggested library fix

Evaluate the goal before checking untried actions or children:

```cpp
while (depth < max_depth) {
    if (goal->evaluate(node->state.fluents())) {
        is_node_goal = true;
        break;
    }

    if (!node->untried_actions.empty()) break;
    if (node->children.empty()) break;

    // selection
}
```

Add a regression test asserting that goal nodes are terminal even when other actions remain applicable.

---

## Issue 5 — `max_depth` can be exceeded by one expansion

With:

```text
max_depth = 3
```

the generated trace reached depth 4:

```text
D:0
D:1
D:2
D:3
D:4
```

The selection loop stops at the depth limit, but the subsequent expansion block can still add one more action because it does not independently check:

```cpp
depth < max_depth
```