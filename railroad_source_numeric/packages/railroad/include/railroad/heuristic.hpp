#pragma once

// FF heuristic umbrella header.
//
// The heuristic is split across self-contained headers along its dependency
// DAG; this header pulls them together in order and provides the public
// entry point (ff_heuristic) plus the Python/introspection query helpers.
//
//   heuristic_types.hpp    aliases, Achiever, FFForwardResult
//   heuristic_forward.hpp  ff_forward_phase, compute_optimistic_costs
//   heuristic_prob.hpp     probabilistic retry delta
//   heuristic_backward.hpp augment_at_with_found, ff_backward_optimistic
//   goal.hpp               GoalBase (only ff_heuristic / extract_or_branches
//                          need it; included here so it stays out of the
//                          lower-level heuristic headers)

#include "railroad/heuristic_types.hpp"
#include "railroad/heuristic_forward.hpp"
#include "railroad/heuristic_prob.hpp"
#include "railroad/heuristic_backward.hpp"
#include "railroad/goal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

// ============================================================================
//  Python / introspection query helpers
// ============================================================================

// Get usable actions via forward relaxed reachability.
inline const std::vector<Action> get_usable_actions(const State &input_state,
                                                    const std::vector<Action> &all_actions) {
  std::unordered_set<const Action*> feasible_action_set;

  // Pass 1: relaxed transition (processes upcoming effects).
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty()) {
    State relaxed = relaxed_result[0].first;
    std::unordered_set<Fluent> initial_fluents(
        relaxed.fluents().begin(), relaxed.fluents().end());

    auto forward = ff_forward_phase(initial_fluents, all_actions);
    State state_all_known(0.0, forward.known_fluents);
    for (const auto& a : all_actions) {
      if (state_all_known.satisfies_precondition(a, true)) {
        feasible_action_set.insert(&a);
      }
    }
  }

  // Pass 2: also consider current fluents WITHOUT processing upcoming
  // effects — handles cases where upcoming effects would mask actions that
  // remain valid for other robots (e.g., another robot can still move to a
  // location before it is marked visited).
  {
    std::unordered_set<Fluent> current_fluents(
        input_state.fluents().begin(), input_state.fluents().end());

    auto forward_current = ff_forward_phase(current_fluents, all_actions);
    State state_current_known(0.0, forward_current.known_fluents);
    for (const auto& a : all_actions) {
      if (state_current_known.satisfies_precondition(a, true)) {
        feasible_action_set.insert(&a);
      }
    }
  }

  std::vector<Action> feasible_actions;
  feasible_actions.reserve(feasible_action_set.size());
  for (const Action* a : feasible_action_set) {
    feasible_actions.push_back(*a);
  }
  return feasible_actions;
}

// Compute optimistic costs for every reachable fluent from a given state.
// Returns a map from fluent to optimistic cost (0 for initial fluents).
inline std::unordered_map<Fluent, double> get_relaxed_optimistic_costs(
    const State &input_state,
    const std::vector<Action> &all_actions) {

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return {};
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);
  return forward.optimistic_cost;
}

// Debug helper: list the achievers (action_name, wait_cost, exec_cost, prob)
// for `fluent` from the relaxed reachability of `input_state`.
inline std::vector<std::tuple<std::string, double, double, double>> get_achievers_for_fluent(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  std::vector<std::tuple<std::string, double, double, double>> info;

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) return info;
  State relaxed = relaxed_result[0].first;

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);

  auto it = forward.achievers_by_fluent.find(fluent);
  if (it != forward.achievers_by_fluent.end()) {
    for (const auto& a : it->second) {
      info.emplace_back(a.action->name(), a.wait_cost, a.exec_cost, a.probability);
    }
  }
  return info;
}

// Optimistic cost for a single fluent. +inf if unreachable, 0 if already true.
inline double get_relaxed_optimistic_cost(
    const State &input_state,
    const Fluent &fluent,
    const std::vector<Action> &all_actions) {

  auto costs = get_relaxed_optimistic_costs(input_state, all_actions);
  auto it = costs.find(fluent);
  if (it != costs.end()) return it->second;

  // Either already in the initial relaxed state (cost 0) or unreachable.
  auto relaxed_result = transition(input_state, nullptr, true);
  if (!relaxed_result.empty() &&
      relaxed_result[0].first.fluents().count(fluent)) {
    return 0.0;
  }
  return std::numeric_limits<double>::infinity();
}


// ============================================================================
//  Optimistic numeric-goal heuristic
// ============================================================================
//
// This is intentionally smaller in scope than a full numeric FF heuristic.
//
// It estimates how many numeric-progress actions are required by inspecting
// numeric updates that occur anywhere in the grounded action set. It ignores
// action preconditions and outcome probabilities, so it remains optimistic
// with respect to reachability.
//
// For conjunctions we take max(child_cost), rather than sum(child_cost),
// because one physical action may progress multiple variables at once.
// For disjunctions we take min(child_cost).
//
// Example:
//   state: H=0, S=0
//   action: H+=1, S+=1
//   goal: H==2 AND S==2
//
//   estimate = max(2, 2) = 2
//
// Larger updates are handled directly:
//   action: S+=3
//   state: S=0
//   goal: S==3
//
//   estimate = 1

struct NumericHeuristicCapabilities {
  std::unordered_map<std::string, NumericValue> max_increase;
  std::unordered_map<std::string, NumericValue> max_decrease;
  std::unordered_map<std::string, std::vector<NumericValue>> assignments;
};


inline void record_numeric_effect_capabilities(
    const GroundedEffect& effect,
    NumericHeuristicCapabilities& capabilities) {

  // Compose the immediate ordered numeric updates for each variable.
  //
  // A sequence without ASSIGN has the form:
  //     x -> x + delta
  //
  // Once ASSIGN occurs, the remainder has the form:
  //     x -> constant
  struct Transformation {
    bool assigned = false;
    NumericValue value = 0;
  };

  std::unordered_map<std::string, Transformation> transforms;

  for (const auto& update : effect.numeric_updates()) {
    auto& tr = transforms[update.variable];

    switch (update.op) {
    case NumericUpdateOp::ASSIGN:
      tr.assigned = true;
      tr.value = update.value;
      break;

    case NumericUpdateOp::INCREASE:
      tr.value += update.value;
      break;

    case NumericUpdateOp::DECREASE:
      tr.value -= update.value;
      break;
    }
  }

  for (const auto& [variable, tr] : transforms) {
    if (tr.assigned) {
      capabilities.assignments[variable].push_back(tr.value);
      continue;
    }

    if (tr.value > 0) {
      auto& best = capabilities.max_increase[variable];
      best = std::max(best, tr.value);
    } else if (tr.value < 0) {
      auto& best = capabilities.max_decrease[variable];
      best = std::max(best, -tr.value);
    }
  }

  // Numeric updates may live inside probabilistic branches.
  for (const auto& branch : effect.prob_effects()) {
    for (const auto& sub_effect : branch.effects()) {
      if (sub_effect) {
        record_numeric_effect_capabilities(
            *sub_effect,
            capabilities);
      }
    }
  }
}


inline NumericHeuristicCapabilities
collect_numeric_heuristic_capabilities(
    const std::vector<Action>& all_actions) {

  NumericHeuristicCapabilities capabilities;

  for (const auto& action : all_actions) {
    for (const auto& effect : action.effects()) {
      if (effect) {
        record_numeric_effect_capabilities(
            *effect,
            capabilities);
      }
    }
  }

  return capabilities;
}


inline bool numeric_condition_holds_for_value(
    const NumericCondition& condition,
    NumericValue value) {

  switch (condition.op) {
  case NumericCompareOp::EQ:
    return value == condition.value;

  case NumericCompareOp::NE:
    return value != condition.value;

  case NumericCompareOp::LT:
    return value < condition.value;

  case NumericCompareOp::LE:
    return value <= condition.value;

  case NumericCompareOp::GT:
    return value > condition.value;

  case NumericCompareOp::GE:
    return value >= condition.value;
  }

  return false;
}


inline double optimistic_steps_for_delta(
    NumericValue required_delta,
    NumericValue max_step) {

  if (required_delta <= 0) {
    return 0.0;
  }

  if (max_step <= 0) {
    return std::numeric_limits<double>::infinity();
  }

  return std::ceil(
      static_cast<double>(required_delta) /
      static_cast<double>(max_step));
}


inline double numeric_condition_optimistic_steps(
    const NumericCondition& condition,
    const NumericState& numeric_values,
    const NumericHeuristicCapabilities& capabilities) {

  auto current_it =
      numeric_values.find(condition.variable);

  if (current_it == numeric_values.end()) {
    throw std::runtime_error(
        "Numeric state variable not found while evaluating "
        "numeric heuristic: " +
        condition.variable);
  }

  const NumericValue current = current_it->second;

  if (numeric_condition_holds_for_value(
          condition,
          current)) {
    return 0.0;
  }

  double best =
      std::numeric_limits<double>::infinity();

  // A suitable ASSIGN can satisfy the condition in one action.
  auto assign_it =
      capabilities.assignments.find(condition.variable);

  if (assign_it != capabilities.assignments.end()) {
    for (NumericValue assigned :
         assign_it->second) {

      if (numeric_condition_holds_for_value(
              condition,
              assigned)) {
        best = 1.0;
        break;
      }
    }
  }

  const auto increase_it =
      capabilities.max_increase.find(condition.variable);

  const auto decrease_it =
      capabilities.max_decrease.find(condition.variable);

  const NumericValue max_increase =
      increase_it == capabilities.max_increase.end()
          ? 0
          : increase_it->second;

  const NumericValue max_decrease =
      decrease_it == capabilities.max_decrease.end()
          ? 0
          : decrease_it->second;

  double incremental =
      std::numeric_limits<double>::infinity();

  switch (condition.op) {
  case NumericCompareOp::EQ:
    if (current < condition.value) {
      incremental = optimistic_steps_for_delta(
          condition.value - current,
          max_increase);
    } else {
      incremental = optimistic_steps_for_delta(
          current - condition.value,
          max_decrease);
    }
    break;

  case NumericCompareOp::NE:
    // We already know current == target here.
    if (max_increase > 0 || max_decrease > 0) {
      incremental = 1.0;
    }
    break;

  case NumericCompareOp::LT:
    incremental = optimistic_steps_for_delta(
        current - condition.value + 1,
        max_decrease);
    break;

  case NumericCompareOp::LE:
    incremental = optimistic_steps_for_delta(
        current - condition.value,
        max_decrease);
    break;

  case NumericCompareOp::GT:
    incremental = optimistic_steps_for_delta(
        condition.value - current + 1,
        max_increase);
    break;

  case NumericCompareOp::GE:
    incremental = optimistic_steps_for_delta(
        condition.value - current,
        max_increase);
    break;
  }

  return std::min(best, incremental);
}


inline double numeric_goal_optimistic_steps(
    const GoalBase* goal,
    const NumericState& numeric_values,
    const NumericHeuristicCapabilities& capabilities) {

  if (!goal) {
    return 0.0;
  }

  switch (goal->get_type()) {
  case GoalType::TRUE_GOAL:
    return 0.0;

  case GoalType::FALSE_GOAL:
    return std::numeric_limits<double>::infinity();

  case GoalType::LITERAL:
    // Fluent-only requirements are handled by the existing FF heuristic.
    return 0.0;

  case GoalType::NUMERIC: {
    const auto* numeric_goal =
        dynamic_cast<const NumericGoal*>(goal);

    if (!numeric_goal) {
      throw std::runtime_error(
          "GoalType::NUMERIC is not a NumericGoal");
    }

    return numeric_condition_optimistic_steps(
        numeric_goal->condition(),
        numeric_values,
        capabilities);
  }

  case GoalType::AND: {
    double cost = 0.0;

    for (const auto& child : goal->children()) {
      cost = std::max(
          cost,
          numeric_goal_optimistic_steps(
              child.get(),
              numeric_values,
              capabilities));
    }

    return cost;
  }

  case GoalType::OR: {
    double cost =
        std::numeric_limits<double>::infinity();

    for (const auto& child : goal->children()) {
      cost = std::min(
          cost,
          numeric_goal_optimistic_steps(
              child.get(),
              numeric_values,
              capabilities));
    }

    return cost;
  }
  }

  return 0.0;
}


inline double numeric_goal_optimistic_steps(
    const State& state,
    const GoalBase* goal,
    const std::vector<Action>& all_actions) {

  if (!goal) {
    return 0.0;
  }

  const auto capabilities =
      collect_numeric_heuristic_capabilities(
          all_actions);

  return numeric_goal_optimistic_steps(
      goal,
      state.numeric_values(),
      capabilities);
}


// ============================================================================
//  Goal API + main entry point
// ============================================================================

// Pull the cached DNF branches off a goal. Distribution of OR over AND
// (e.g., AND(A, OR(B,C)) -> [{A,B}, {A,C}]) is handled by the goal itself.
inline const std::vector<std::unordered_set<Fluent>>& extract_or_branches(const GoalBase* goal) {
  static const std::vector<std::unordered_set<Fluent>> empty_branches;
  if (!goal) return empty_branches;
  return goal->get_dnf_branches();
}

inline std::size_t hash_action_set_for_heuristic(const std::vector<Action>& all_actions) {
  std::size_t h = all_actions.size();
  std::size_t xor_hash = 0;
  std::size_t sum_hash = 0;

  for (const auto& action : all_actions) {
    std::size_t action_hash = action.hash();
    hash_combine(action_hash, 0);
    xor_hash ^= action_hash;
    sum_hash += action_hash;
  }

  hash_combine(h, xor_hash);
  hash_combine(h, sum_hash);
  return h;
}

inline FFCacheKey make_ff_cache_key(const State& relaxed,
                                    const GoalBase* goal,
                                    const std::vector<Action>& all_actions,
                                    double lambda_add,
                                    double lambda_max,
                                    double lambda_ff,
                                    bool at_implies_found) {
  return {
      relaxed.hash(),
      goal ? goal->hash() : 0,
      hash_action_set_for_heuristic(all_actions),
      std::hash<double>{}(lambda_add),
      std::hash<double>{}(lambda_max),
      std::hash<double>{}(lambda_ff),
      at_implies_found,
  };
}

// Main FF heuristic.
//
// The relaxed-plan extraction produces three component values:
//   h_add: Σ optimistic_cost over goal fluents (classic additive)
//   h_max: max optimistic_cost over goal fluents
//   h_ff:  Σ action_duration over unique actions on the relaxed plan
// These are mixed via the lambda_* weights (free-form, not normalized).
// The probabilistic-retry delta is added once per branch *after* mixing.
// Defaults are an even split between h_add and h_ff (0.5, 0.0, 0.5).
//
// Layout:
//   1. Relaxed transition for fluents (union over outcomes).
//   2. Non-relaxed transition just to read out the time of the next robot
//      completion — gives a tighter dtime lower bound than the relaxed step.
//   3. Forward reachability + optimistic costs.
//   4. For each DNF branch of the goal: backward extraction of all three
//      component values, mix with lambdas, add probabilistic-delta retries
//      for fluents on the relaxed plan that have probabilistic achievers.
//      Take the minimum across branches.
inline double ff_heuristic(const State &input_state,
                           const GoalBase *goal,
                           const std::vector<Action> &all_actions,
                           FFMemory *ff_memory = nullptr,
                           double lambda_add = 0.5,
                           double lambda_max = 0.0,
                           double lambda_ff  = 0.5,
                           bool at_implies_found = true) {
  if (!goal) return 0.0;

  GoalType type = goal->get_type();
  if (type == GoalType::TRUE_GOAL) return 0.0;
  if (type == GoalType::FALSE_GOAL) {
    return std::numeric_limits<double>::infinity();
  }

  const double numeric_cost =
      numeric_goal_optimistic_steps(
          input_state,
          goal,
          all_actions);

  if (numeric_cost ==
      std::numeric_limits<double>::infinity()) {
    return numeric_cost;
  }

  const double t0 = input_state.time();

  auto relaxed_result = transition(input_state, nullptr, true);
  if (relaxed_result.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  State relaxed = relaxed_result[0].first;

  auto nonrelaxed_result = transition(input_state, nullptr, false);
  double dtime = 0.0;
  if (!nonrelaxed_result.empty()) {
    dtime = nonrelaxed_result[0].first.time() - t0;
  }

  // Memoization key: relaxed-state fluents at time 0. The cached value is the
  // already-mixed branch minimum. Include the goal, action universe, lambda
  // weights, and augmentation policy because all of them affect that value.
  relaxed.set_time(0);
  std::optional<FFCacheKey> cache_key;
  if (ff_memory) {
    cache_key = make_ff_cache_key(relaxed, goal, all_actions,
                                  lambda_add, lambda_max, lambda_ff,
                                  at_implies_found);
    auto cached_it = ff_memory->find(*cache_key);
    if (cached_it != ff_memory->end()) {
      return dtime + cached_it->second;
    }
  }

  std::unordered_set<Fluent> initial_fluents(
      relaxed.fluents().begin(), relaxed.fluents().end());

  auto forward = ff_forward_phase(initial_fluents, all_actions);
  compute_optimistic_costs(forward);

  auto branches = extract_or_branches(goal);
  if (branches.empty()) {
    return std::numeric_limits<double>::infinity();  // FalseGoal-like
  }

  double min_cost = std::numeric_limits<double>::infinity();
  for (const auto& branch : branches) {
    auto opt = ff_backward_optimistic(forward, branch, at_implies_found);
    if (opt.h_add == std::numeric_limits<double>::infinity()) continue;  // unreachable branch

    double delta_total = relaxed_plan_prob_delta(forward, opt.on_path);
    double mixed = lambda_add * opt.h_add
                 + lambda_max * opt.h_max
                 + lambda_ff  * opt.h_ff;
    min_cost = std::min(min_cost, mixed + delta_total);
  }

  // The fluent FF estimate and numeric estimate may describe progress
  // achieved by the same physical action, so use max rather than sum.
  // This avoids double-counting joint fluent/numeric progress.
  const double combined_cost =
      std::max(min_cost, numeric_cost);

  if (ff_memory && cache_key) {
    (*ff_memory)[*cache_key] = combined_cost;
  }

  return dtime + combined_cost;
}

} // namespace railroad
