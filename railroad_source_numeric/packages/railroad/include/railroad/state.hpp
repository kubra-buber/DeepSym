#pragma once
#include "railroad/core.hpp"
#include "railroad/constants.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace railroad {

class State {
public:
  using EffectQueue =
      std::vector<std::pair<double, std::shared_ptr<const GroundedEffect>>>;

  using NumericValue = railroad::NumericValue;
  using NumericState = railroad::NumericState;

State(double time = 0, std::unordered_set<Fluent> fluents = {},
      EffectQueue upcoming_effects = {},
      NumericState numeric_values = {})
    : time_(time), fluents_(std::move(fluents)),
      upcoming_effects_(std::move(upcoming_effects)),
      numeric_values_(std::move(numeric_values)),
      cached_hash_(std::nullopt) {}

  double time() const { return time_; }
  const std::unordered_set<Fluent> &fluents() const { return fluents_; }
  const EffectQueue &upcoming_effects() const { return upcoming_effects_; }
  const NumericState &numeric_values() const {
  return numeric_values_;
}

  void set_time(double new_time) {
    time_ = new_time;
    cached_hash_ = std::nullopt;
  }

  bool satisfies_precondition(
      const Action &action,
      bool relax = false) const {

    const auto &pos = action.pos_preconditions();

    bool has_all_positive =
        std::all_of(
            pos.begin(),
            pos.end(),
            [&](const Fluent &f) {
              return fluents_.count(f);
            });

    if (!has_all_positive) {
      return false;
    }

    // IMPORTANT:
    // Numeric conditions are deliberately ignored in relaxed mode
    // for now. The current FF relaxation does not yet model numeric
    // state progression. Numeric heuristic support will be added
    // separately.
    if (relax) {
      return true;
    }

    const auto &neg =
        action.neg_precond_flipped();

    bool disjoint_negative =
        std::none_of(
            neg.begin(),
            neg.end(),
            [&](const Fluent &f) {
              return fluents_.count(f);
            });

    if (!disjoint_negative) {
      return false;
    }

    for (const auto &condition :
        action.numeric_preconditions()) {

      if (!condition.evaluate(numeric_values_)) {
        return false;
      }
    }

    return true;
  }

State copy() const {
  return State(time_, fluents_, upcoming_effects_, numeric_values_);
}
  bool
  update_with_effect(const std::shared_ptr<const GroundedEffect> &new_effect,
                     bool relax = false) {
    cached_hash_ = std::nullopt;
    bool freed_robot = false;
    for (const auto &f : new_effect->pos_fluents()) {
      if (f.is_free()) {
        freed_robot = true;
      }
      fluents_.insert(f);
    }
    if (!relax) {
      for (const auto &f : new_effect->flipped_neg_fluents()) {
        fluents_.erase(f);
      }
    }

    if (!relax) {
      for (const auto &update : new_effect->numeric_updates()) {
        auto it = numeric_values_.find(update.variable);

        if (it == numeric_values_.end()) {
          throw std::runtime_error(
              "Numeric state variable not found: " +
              update.variable);
        }

        switch (update.op) {
        case NumericUpdateOp::ASSIGN:
          it->second = update.value;
          break;

        case NumericUpdateOp::INCREASE:
          it->second += update.value;
          break;

        case NumericUpdateOp::DECREASE:
          it->second -= update.value;
          break;
        }
      }
    }

    return freed_robot;
  }

  bool update_fluents(const std::unordered_set<Fluent> &new_fluents,
                      bool relax = false) {
    cached_hash_ = std::nullopt;
    bool freed_robot = false;
    for (const auto &f : new_fluents) {
      if (f.is_negated()) {
        if (!relax) {
          fluents_.erase(f.invert());
        }
      } else {
        if (f.is_free()) {
          freed_robot = true;
        }
        fluents_.insert(f);
      }
    }
    return freed_robot;
  }

  void queue_effect(const std::shared_ptr<const GroundedEffect> &effect) {
    cached_hash_ = std::nullopt;
    upcoming_effects_.emplace_back(time_ + effect->time(), effect);
    std::push_heap(upcoming_effects_.begin(), upcoming_effects_.end(),
                   effect_cmp);
  }

  void pop_effect() {
    cached_hash_ = std::nullopt;
    std::pop_heap(upcoming_effects_.begin(), upcoming_effects_.end(),
                  effect_cmp);
    upcoming_effects_.pop_back();
  }

  State copy_and_zero_out_time() const {
    EffectQueue new_effects;
    for (const auto &[t, e] : upcoming_effects_) {
      new_effects.emplace_back(t - time_, e);
    }
return State(0, fluents_, new_effects, numeric_values_);  }

  std::size_t hash() const {
    if (cached_hash_) {
      return *cached_hash_;
    }

    std::size_t h_time = std::hash<double>{}(time_);

    // Hash fluents
    std::size_t h_fluents = 0;
    for (const auto &f : fluents_) {
      std::size_t h = f.hash();
      hash_combine(h, 0);
      h_fluents ^= h;
    }

    std::size_t h_upcoming = 0;
    for (const auto &[t, eff] : upcoming_effects_) {
      std::size_t h_effect = 0;
      hash_combine(h_effect, eff->hash());
      hash_combine(h_effect, std::hash<double>{}(t));
      h_upcoming ^= h_effect;
    }

    // Hash numeric state variables.
    // XOR keeps the result independent of unordered_map iteration order.
    std::size_t h_numeric = 0;
    for (const auto &[name, value] : numeric_values_) {
      std::size_t h_item = std::hash<std::string>{}(name);
      hash_combine(h_item, std::hash<NumericValue>{}(value));
      h_numeric ^= h_item;
    }

    std::size_t h = 0;
    hash_combine(h, h_time);
    hash_combine(h, h_fluents);
    hash_combine(h, h_upcoming);
    hash_combine(h, h_numeric);
    cached_hash_ = h;

    return h;
  }

  bool operator==(const State &other) const {
    return this->hash() == other.hash();
  }

  bool operator<(const State &other) const { return time_ < other.time_; }

  std::string str() const {
    std::ostringstream out;
    out << "State<time=" << time_ << ", fluents={";
    bool first = true;
    for (const auto &f : fluents_) {
      if (!first)
        out << ", ";
      out << (f.is_negated() ? "not " : "") << f.name();
      for (const auto &arg : f.args())
        out << " " << arg;
      first = false;
    }

    out << "}, numeric_values={";

    std::vector<std::pair<std::string, NumericValue>> sorted_numeric(
        numeric_values_.begin(), numeric_values_.end());

    std::sort(
        sorted_numeric.begin(),
        sorted_numeric.end(),
        [](const auto &a, const auto &b) {
          return a.first < b.first;
        });

    first = true;
    for (const auto &[name, value] : sorted_numeric) {
      if (!first)
        out << ", ";
      out << name << "=" << value;
      first = false;
    }

    out << "}, upcoming_effects=[";
    for (size_t i = 0; i < upcoming_effects_.size(); ++i) {
      out << "(" << upcoming_effects_[i].first << ", " << upcoming_effects_[i].second->str() << ")";
      if (i + 1 < upcoming_effects_.size())
        out << ", ";
    }
    out << "]>";
    return out.str();
  }


private:
  double time_;
  std::unordered_set<Fluent> fluents_;
  EffectQueue upcoming_effects_;
  NumericState numeric_values_;
  mutable std::optional<std::size_t> cached_hash_;
  static bool effect_cmp(
      const std::pair<double, std::shared_ptr<const GroundedEffect>> &a,
      const std::pair<double, std::shared_ptr<const GroundedEffect>> &b) {
    return a.first > b.first;
  }
};

} // namespace railroad

namespace std {
template <> struct hash<railroad::State> {
  std::size_t operator()(const railroad::State &s) const noexcept {
    return s.hash(); // assumes State::hash() is a valid const method
  }
};
} // namespace std

namespace railroad {

inline void advance_to_terminal(State state, double prob,
                                std::unordered_map<State, double> &outcomes,
                                bool relax = false) {

  bool any_free_robots = false;

  while (!state.upcoming_effects().empty()) {
    auto [scheduled_time, effect] = state.upcoming_effects().front();

    if (scheduled_time > state.time() && !relax &&
        (any_free_robots ||
         std::any_of(state.fluents().begin(), state.fluents().end(),
                     [](const Fluent &f) { return f.is_free(); }))) {
      outcomes[state] += prob;
      return;
    }

    if (scheduled_time > state.time()) {
      state.set_time(scheduled_time);
    }
    state.pop_effect();
    any_free_robots = state.update_with_effect(effect, relax);

    if (effect->is_probabilistic()) {
      for (const auto &branch : effect->prob_effects()) {
        State branched = state;
        for (const auto &e : branch.effects()) {
          branched.queue_effect(e);
        }
        advance_to_terminal(branched, prob * branch.prob(), outcomes, relax);
      }
      return; // stop after branching
    }
  }
  outcomes[state] += prob;
}

inline std::vector<std::pair<State, double>>
transition(const State &state, const Action *action, bool relax = false) {
  if (action && !relax && !state.satisfies_precondition(*action, relax)) {
    throw std::runtime_error("Precondition not satisfied for applying action");
  }

  State new_state = state;
  if (action) {
    for (const auto &effect : action->effects()) {
      new_state.queue_effect(effect);
    }
  }

  std::unordered_map<State, double> outcomes;
  advance_to_terminal(new_state, 1.0, outcomes, relax);

  // Move elements from unordered_map to a vector (mutable)
  std::vector<std::pair<State, double>> vec_outcomes;
  vec_outcomes.reserve(outcomes.size());

  for (auto &entry : outcomes) {
    vec_outcomes.emplace_back(std::move(const_cast<State &>(entry.first)),
                              std::move(entry.second));
  }

  // Resolve any 'waiting' predicates
  for (auto &[out_state, prob] : vec_outcomes) {
    // If a robot is waiting on a now-free robot, 
    // it is now free and no longer waiting.
    std::unordered_set<Fluent> upd_fluents;
    for (auto &waiting_fluent : out_state.fluents()) {
      if (waiting_fluent.is_waiting() &&
          out_state.fluents().count(
              Fluent("free " + waiting_fluent.args().back()))) {
        upd_fluents.insert(Fluent("free " + waiting_fluent.args().front()));
        upd_fluents.insert(waiting_fluent.invert());
      }
    }
    out_state.update_fluents(upd_fluents);

    // If no robots are free, free all waiting robots
    // and add penalty for getting stuck.
    int count_free = 0;
    for (auto &fluent : out_state.fluents()) {
      if (fluent.is_free()) { ++count_free; }
    }
    if (count_free > 0) { continue; }
    for (auto &waiting_fluent : out_state.fluents()) {
      if (waiting_fluent.is_waiting()) {
        upd_fluents.insert(Fluent("free " + waiting_fluent.args().front()));
        upd_fluents.insert(waiting_fluent.invert());
      }
    }
    out_state.update_fluents(upd_fluents);
    if (action) {
      out_state.set_time(out_state.time() + ALL_ROBOTS_WAITING_PENALTY);
    } else {
      out_state.set_time(out_state.time());
    }

  }

  // If all robots are waiting,	clear all waiting actions
  

  return {vec_outcomes.begin(), vec_outcomes.end()};
}

// Implementation of Action::get_relaxed_successors()
// Must be defined here after State and transition are fully defined
inline const std::vector<std::pair<State, double>>& Action::get_relaxed_successors() const {
  if (!relaxed_successors_cache_) {
    State empty_state;
    relaxed_successors_cache_ = transition(empty_state, this, true);
  }
  return *relaxed_successors_cache_;
}

} // namespace railroad
