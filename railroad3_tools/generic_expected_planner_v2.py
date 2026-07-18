from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from railroad.core import transition


def state_key(state) -> Tuple[str, ...]:
    return tuple(sorted(str(fluent) for fluent in state.fluents))


def merge_outcomes(outcomes):
    merged: Dict[Tuple[str, ...], List] = {}

    for next_state, probability in outcomes:
        probability = float(probability)
        if probability <= 0.0:
            continue

        key = state_key(next_state)
        if key in merged:
            merged[key][1] += probability
        else:
            merged[key] = [next_state, probability]

    total = sum(probability for _, probability in merged.values())
    if total <= 0.0:
        return []

    return [
        (state, probability / total)
        for state, probability in merged.values()
    ]


def safe_transition(state, action):
    """Use transition() as the authoritative applicability check."""
    try:
        return merge_outcomes(transition(state, action))
    except RuntimeError as exc:
        if "precondition not satisfied" in str(exc).lower():
            return []
        raise


class ExactExpectedReachabilityPlanner:
    """Exact finite-horizon goal-reachability planner."""

    def __init__(self, actions: Sequence):
        self.actions = sorted(actions, key=lambda action: action.name)
        self.policy: Dict[Tuple[Tuple[str, ...], int], Optional[str]] = {}
        self.action_values: Dict[
            Tuple[Tuple[str, ...], int], Dict[str, float]
        ] = {}
        self._states: Dict[Tuple[str, ...], object] = {}
        self._transition_cache: Dict[
            Tuple[Tuple[str, ...], str], List
        ] = {}

    def solve(self, initial_state, goal, horizon: int):
        self.policy.clear()
        self.action_values.clear()
        self._states = {state_key(initial_state): initial_state}
        self._transition_cache.clear()

        def cached_transition(state, action):
            key = (state_key(state), action.name)
            if key not in self._transition_cache:
                self._transition_cache[key] = safe_transition(state, action)
            return self._transition_cache[key]

        @lru_cache(maxsize=None)
        def value(key: Tuple[str, ...], depth: int) -> float:
            state = self._states[key]

            if goal.evaluate(state.fluents):
                self.policy[(key, depth)] = None
                self.action_values[(key, depth)] = {}
                return 1.0

            if depth <= 0:
                self.policy[(key, depth)] = None
                self.action_values[(key, depth)] = {}
                return 0.0

            values: Dict[str, float] = {}

            for action in self.actions:
                outcomes = cached_transition(state, action)
                if not outcomes:
                    continue

                expected = 0.0
                for next_state, probability in outcomes:
                    next_key = state_key(next_state)
                    self._states[next_key] = next_state
                    expected += probability * value(next_key, depth - 1)

                values[action.name] = expected

            self.action_values[(key, depth)] = values

            if not values:
                self.policy[(key, depth)] = None
                return 0.0

            best_name, best_value = max(
                values.items(),
                key=lambda item: (item[1], item[0]),
            )
            self.policy[(key, depth)] = best_name
            return best_value

        initial_key = state_key(initial_state)
        probability = value(initial_key, int(horizon))
        action_name = self.policy[(initial_key, int(horizon))]
        root_values = self.action_values[(initial_key, int(horizon))]

        return action_name, probability, root_values