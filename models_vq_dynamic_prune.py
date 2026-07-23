"""DeepSym with grow/prune/optional-merge EMA-VQ bottlenecks.

This is a drop-in sibling of models_vq_dynamic.py.  It keeps a generously
sized technical codebook capacity, activates codes through the existing
persistent-surprise rule, and can later remove persistently low-occupancy
codes.  Optional merging is deliberately conservative and disabled by default.

Important:
- Pruning uses EMA occupancy on the *training stream*, not the 50 canonical
  visualization objects.
- At most one structural operation is performed at each maintenance check.
- Removed code slots are compacted by moving the final active slot.
- Code indices are therefore permutation-dependent and can change after prune
  or merge.  Compare partitions with ARI/co-assignment, not raw code numbers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import torch

import utils
from blocks import MLP, build_encoder


MODEL_KIND = "dynamic_prune"


def _weights(opts: Dict, key: str, default, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        opts.get(key, default),
        dtype=torch.float32,
        device=device,
    )


class GrowPruneEMAVQLayer(torch.nn.Module):
    """EMA-VQ with supported growth, occupancy pruning, and optional merging."""

    def __init__(
        self,
        max_embeddings: int,
        embedding_dim: int,
        surprise_threshold: float,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        warmup_steps: int = 200,
        growth_interval: int = 100,
        min_support: int = 8,
        min_support_fraction: float = 0.05,
        required_checks: int = 2,
        initial_embeddings: int = 1,
        # Structural maintenance.
        pruning_enabled: bool = True,
        merging_enabled: bool = False,
        maintenance_interval: int = 100,
        structure_warmup_steps: int = 1000,
        structure_cooldown_steps: int = 300,
        minimum_embeddings: int = 1,
        # Pruning.
        prune_min_fraction: float = 0.01,
        prune_patience: int = 3,
        prune_min_age_steps: int = 800,
        # Conservative merging.
        merge_distance_threshold: float = 0.05,
        merge_max_small_fraction: float = 0.05,
        merge_min_age_steps: int = 800,
    ):
        super().__init__()

        self.max_embeddings = max(1, int(max_embeddings))
        self.embedding_dim = int(embedding_dim)
        self.surprise_threshold = float(surprise_threshold)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        self.warmup_steps = max(0, int(warmup_steps))
        self.growth_interval = max(1, int(growth_interval))
        self.min_support = max(1, int(min_support))
        self.min_support_fraction = max(0.0, float(min_support_fraction))
        self.required_checks = max(1, int(required_checks))
        self.initial_embeddings = max(
            1,
            min(int(initial_embeddings), self.max_embeddings),
        )

        self.pruning_enabled = bool(pruning_enabled)
        self.merging_enabled = bool(merging_enabled)
        self.maintenance_interval = max(1, int(maintenance_interval))
        self.structure_warmup_steps = max(0, int(structure_warmup_steps))
        self.structure_cooldown_steps = max(0, int(structure_cooldown_steps))
        self.minimum_embeddings = max(
            1,
            min(int(minimum_embeddings), self.max_embeddings),
        )

        if not 0.0 <= float(prune_min_fraction) < 1.0:
            raise ValueError("prune_min_fraction must be in [0, 1)")
        self.prune_min_fraction = float(prune_min_fraction)
        self.prune_patience = max(1, int(prune_patience))
        self.prune_min_age_steps = max(0, int(prune_min_age_steps))

        if float(merge_distance_threshold) < 0.0:
            raise ValueError("merge_distance_threshold must be non-negative")
        if not 0.0 <= float(merge_max_small_fraction) < 1.0:
            raise ValueError("merge_max_small_fraction must be in [0, 1)")
        self.merge_distance_threshold = float(merge_distance_threshold)
        self.merge_max_small_fraction = float(merge_max_small_fraction)
        self.merge_min_age_steps = max(0, int(merge_min_age_steps))

        self.embedding = torch.nn.Embedding(
            self.max_embeddings,
            self.embedding_dim,
        )
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        # Fixed-size state makes checkpoints simple even though active K changes.
        self.register_buffer(
            "active_embeddings",
            torch.tensor(self.initial_embeddings, dtype=torch.long),
        )
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_check_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer(
            "last_maintenance_step",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "last_structure_step",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "surprise_streak",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "cluster_size",
            torch.zeros(self.max_embeddings),
        )
        self.register_buffer(
            "embed_avg",
            torch.zeros(self.max_embeddings, self.embedding_dim),
        )
        self.register_buffer(
            "code_age",
            torch.zeros(self.max_embeddings, dtype=torch.long),
        )
        self.register_buffer(
            "low_usage_streak",
            torch.zeros(self.max_embeddings, dtype=torch.long),
        )
        self.register_buffer(
            "growth_count",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "prune_count",
            torch.tensor(0, dtype=torch.long),
        )
        self.register_buffer(
            "merge_count",
            torch.tensor(0, dtype=torch.long),
        )

        self.last_vq_loss = torch.tensor(0.0)
        self.last_distance_stats: Dict[str, float] = {}
        self._events: List[Dict[str, float]] = []

    @torch.no_grad()
    def _initialize_from_batch(self, flat_inputs: torch.Tensor) -> None:
        n = int(flat_inputs.shape[0])
        k = int(self.active_embeddings.item())

        if n < 1:
            raise ValueError("Cannot initialize VQ codebook from an empty batch")

        if k == 1:
            initial = flat_inputs.mean(dim=0, keepdim=True).detach()
        else:
            mean = flat_inputs.mean(dim=0, keepdim=True)
            first = ((flat_inputs - mean) ** 2).sum(dim=1).argmin()
            selected = [int(first.item())]

            while len(selected) < min(k, n):
                chosen = flat_inputs[selected]
                distances = (
                    torch.cdist(flat_inputs, chosen)
                    .pow(2)
                    .min(dim=1)
                    .values
                )
                distances[selected] = -1.0
                selected.append(int(distances.argmax().item()))

            positions = torch.as_tensor(
                selected,
                device=flat_inputs.device,
                dtype=torch.long,
            )
            if len(selected) < k:
                repeats = (k + len(selected) - 1) // len(selected)
                positions = positions.repeat(repeats)[:k]
            initial = flat_inputs[positions].detach()

        self.embedding.weight.data[:k].copy_(initial)
        self.embed_avg[:k].copy_(initial)
        self.cluster_size[:k].fill_(1.0)
        self.code_age[:k].zero_()
        self.low_usage_streak[:k].zero_()
        self.initialized.fill_(True)

    def _active_weights(self) -> torch.Tensor:
        return self.embedding.weight[: int(self.active_embeddings.item())]

    def _distances(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        weights = self._active_weights()
        return (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            + weights.pow(2).sum(dim=1)
            - 2.0 * flat_inputs @ weights.t()
        ).clamp_min_(0.0)

    def _occupancy(self) -> torch.Tensor:
        active = int(self.active_embeddings.item())
        mass = self.cluster_size[:active].clamp_min(0.0)
        total = mass.sum()
        if float(total.item()) <= self.epsilon:
            return torch.full_like(mass, 1.0 / max(active, 1))
        return mass / total

    def _cooldown_complete(self, step: int) -> bool:
        return (
            step - int(self.last_structure_step.item())
            >= self.structure_cooldown_steps
        )

    @torch.no_grad()
    def _record_event(self, event: Dict[str, float]) -> None:
        self._events.append(event)
        self.last_structure_step.fill_(int(event["step"]))

    @torch.no_grad()
    def _maybe_grow(
        self,
        flat_inputs: torch.Tensor,
        distances: torch.Tensor,
    ) -> bool:
        step = int(self.step_counter.item())
        active = int(self.active_embeddings.item())

        if active >= self.max_embeddings or step <= self.warmup_steps:
            return False
        if not self._cooldown_complete(step):
            return False
        if step - int(self.last_check_step.item()) < self.growth_interval:
            return False

        self.last_check_step.fill_(step)
        min_distances = distances.min(dim=1).values
        required = max(
            self.min_support,
            int(math.ceil(
                self.min_support_fraction * min_distances.numel()
            )),
        )
        far_mask = min_distances > self.surprise_threshold
        support = int(far_mask.sum().item())

        self.last_distance_stats = {
            "distance_mean": float(min_distances.mean().item()),
            "distance_p90": float(
                torch.quantile(min_distances, 0.90).item()
            ),
            "distance_p95": float(
                torch.quantile(min_distances, 0.95).item()
            ),
            "distance_max": float(min_distances.max().item()),
            "surprise_support": float(support),
        }

        if support < required:
            self.surprise_streak.zero_()
            return False

        self.surprise_streak.add_(1)
        if int(self.surprise_streak.item()) < self.required_checks:
            return False

        surprising = flat_inputs[far_mask]
        center = surprising.mean(dim=0, keepdim=True)
        representative_index = (
            (surprising - center).pow(2).sum(dim=1).argmin()
        )
        candidate = surprising[representative_index].detach()

        new_index = active
        initial_mass = float(max(1, support))
        self.embedding.weight.data[new_index].copy_(candidate)
        self.cluster_size[new_index] = initial_mass
        self.embed_avg[new_index].copy_(candidate * initial_mass)
        self.code_age[new_index].zero_()
        self.low_usage_streak[new_index].zero_()
        self.active_embeddings.add_(1)
        self.growth_count.add_(1)
        self.surprise_streak.zero_()

        event = {
            "event_type": "grow",
            "step": float(step),
            "new_active_codes": float(active + 1),
            "new_code_index": float(new_index),
            "support": float(support),
            "required_support": float(required),
            "required_checks": float(self.required_checks),
            "threshold": float(self.surprise_threshold),
            **self.last_distance_stats,
        }
        self._record_event(event)
        print(
            f"[GrowPrune VQ] step={step} GROW K={active}->{active + 1} "
            f"support={support}/{min_distances.numel()} "
            f"threshold={self.surprise_threshold:.4f}"
        )
        return True

    @torch.no_grad()
    def _clear_slot(self, index: int) -> None:
        self.embedding.weight.data[index].zero_()
        self.cluster_size[index].zero_()
        self.embed_avg[index].zero_()
        self.code_age[index].zero_()
        self.low_usage_streak[index].zero_()

    @torch.no_grad()
    def _remove_code(self, index: int) -> Tuple[int, Optional[int]]:
        """Remove one active slot by moving the final active slot into it.

        Returns:
            removed_index, moved_from_index (or None if the final slot itself
            was removed).
        """
        active = int(self.active_embeddings.item())
        if not 0 <= index < active:
            raise IndexError(f"Active code index out of range: {index}")
        if active <= self.minimum_embeddings:
            raise RuntimeError("Refusing to remove below minimum_embeddings")

        last = active - 1
        moved_from: Optional[int] = None

        if index != last:
            self.embedding.weight.data[index].copy_(
                self.embedding.weight.data[last]
            )
            self.cluster_size[index].copy_(self.cluster_size[last])
            self.embed_avg[index].copy_(self.embed_avg[last])
            self.code_age[index].copy_(self.code_age[last])
            self.low_usage_streak[index].copy_(
                self.low_usage_streak[last]
            )
            moved_from = last

        self._clear_slot(last)
        self.active_embeddings.sub_(1)
        return index, moved_from

    @torch.no_grad()
    def _maybe_prune(self, step: int) -> bool:
        active = int(self.active_embeddings.item())
        if not self.pruning_enabled:
            return False
        if active <= self.minimum_embeddings:
            return False

        occupancy = self._occupancy()
        old_enough = (
            self.code_age[:active] >= self.prune_min_age_steps
        )
        low = (
            occupancy < self.prune_min_fraction
        ) & old_enough

        current_streak = self.low_usage_streak[:active]
        updated_streak = torch.where(
            low,
            current_streak + 1,
            torch.zeros_like(current_streak),
        )
        current_streak.copy_(updated_streak)

        eligible = torch.nonzero(
            self.low_usage_streak[:active] >= self.prune_patience,
            as_tuple=False,
        ).flatten()
        if eligible.numel() == 0:
            return False

        eligible_occupancy = occupancy[eligible]
        candidate = int(
            eligible[eligible_occupancy.argmin()].item()
        )
        candidate_fraction = float(occupancy[candidate].item())
        candidate_mass = float(self.cluster_size[candidate].item())
        before = active

        removed, moved_from = self._remove_code(candidate)
        self.prune_count.add_(1)

        event = {
            "event_type": "prune",
            "step": float(step),
            "old_active_codes": float(before),
            "new_active_codes": float(before - 1),
            "removed_code_index": float(removed),
            "moved_from_index": (
                float(moved_from) if moved_from is not None else -1.0
            ),
            "occupancy_fraction": candidate_fraction,
            "ema_mass": candidate_mass,
            "prune_threshold": float(self.prune_min_fraction),
            "prune_patience": float(self.prune_patience),
        }
        self._record_event(event)
        print(
            f"[GrowPrune VQ] step={step} PRUNE K={before}->{before - 1} "
            f"code={candidate} occupancy={candidate_fraction:.5f}"
        )
        return True

    @torch.no_grad()
    def _best_merge_pair(
        self,
    ) -> Optional[Tuple[int, int, float, float]]:
        active = int(self.active_embeddings.item())
        if active <= self.minimum_embeddings:
            return None

        occupancy = self._occupancy()
        ages = self.code_age[:active]
        weights = self.embedding.weight[:active]

        distances = torch.cdist(weights, weights).pow(2)
        distances.fill_diagonal_(float("inf"))

        # A merge is allowed only when at least one member is a small
        # satellite code.  This prevents two major semantic modes from being
        # merged merely because their latent vectors are close.
        small = occupancy < self.merge_max_small_fraction
        eligible_age = ages >= self.merge_min_age_steps
        pair_mask = (
            (small[:, None] | small[None, :])
            & eligible_age[:, None]
            & eligible_age[None, :]
        )
        distances = distances.masked_fill(~pair_mask, float("inf"))

        flat_index = int(distances.argmin().item())
        value = float(distances.flatten()[flat_index].item())
        if not math.isfinite(value):
            return None
        if value >= self.merge_distance_threshold:
            return None

        first = flat_index // active
        second = flat_index % active
        if first == second:
            return None

        # Keep the code with greater EMA occupancy.
        if float(occupancy[first].item()) >= float(occupancy[second].item()):
            target, source = first, second
        else:
            target, source = second, first

        small_fraction = float(
            torch.minimum(
                occupancy[first],
                occupancy[second],
            ).item()
        )
        return target, source, value, small_fraction

    @torch.no_grad()
    def _maybe_merge(self, step: int) -> bool:
        if not self.merging_enabled:
            return False

        candidate = self._best_merge_pair()
        if candidate is None:
            return False

        target, source, distance, small_fraction = candidate
        active = int(self.active_embeddings.item())
        target_mass = self.cluster_size[target].clamp_min(self.epsilon)
        source_mass = self.cluster_size[source].clamp_min(self.epsilon)
        combined_mass = target_mass + source_mass
        combined_sum = self.embed_avg[target] + self.embed_avg[source]
        combined_vector = combined_sum / combined_mass

        # Save combined state in the target before compaction.
        self.cluster_size[target].copy_(combined_mass)
        self.embed_avg[target].copy_(combined_sum)
        self.embedding.weight.data[target].copy_(combined_vector)
        self.code_age[target].copy_(
            torch.maximum(self.code_age[target], self.code_age[source])
        )
        self.low_usage_streak[target].zero_()

        target_was_last = target == active - 1
        removed, moved_from = self._remove_code(source)

        # If target was the last slot and source was earlier, compaction moved
        # the merged target into source's old index.
        final_target = source if target_was_last and source != target else target

        self.merge_count.add_(1)
        event = {
            "event_type": "merge",
            "step": float(step),
            "old_active_codes": float(active),
            "new_active_codes": float(active - 1),
            "target_code_index_before": float(target),
            "source_code_index_before": float(source),
            "target_code_index_after": float(final_target),
            "removed_code_index": float(removed),
            "moved_from_index": (
                float(moved_from) if moved_from is not None else -1.0
            ),
            "squared_code_distance": float(distance),
            "smaller_occupancy_fraction": float(small_fraction),
            "merge_distance_threshold": float(
                self.merge_distance_threshold
            ),
        }
        self._record_event(event)
        print(
            f"[GrowPrune VQ] step={step} MERGE K={active}->{active - 1} "
            f"pair=({target},{source}) distance={distance:.6f} "
            f"small_occ={small_fraction:.5f}"
        )
        return True

    @torch.no_grad()
    def _maybe_maintain(self) -> bool:
        step = int(self.step_counter.item())
        if step <= self.structure_warmup_steps:
            return False
        if not self._cooldown_complete(step):
            return False
        if (
            step - int(self.last_maintenance_step.item())
            < self.maintenance_interval
        ):
            return False

        self.last_maintenance_step.fill_(step)

        # Perform at most one structural operation per check.  Pruning is
        # intentionally tried first because it has the clearest evidence:
        # persistent low training-stream occupancy.
        if self._maybe_prune(step):
            return True
        return self._maybe_merge(step)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, self.embedding_dim)

        if self.training:
            self.step_counter.add_(1)
            if not bool(self.initialized.item()):
                self._initialize_from_batch(flat_inputs)

        distances = self._distances(flat_inputs)
        if self.training and self._maybe_grow(flat_inputs, distances):
            distances = self._distances(flat_inputs)

        active = int(self.active_embeddings.item())
        indices = distances.argmin(dim=1)
        encodings = torch.nn.functional.one_hot(
            indices,
            num_classes=active,
        ).to(flat_inputs.dtype)
        quantized = (
            encodings @ self.embedding.weight[:active]
        ).view_as(inputs)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0)
                sums = encodings.t() @ flat_inputs

                self.cluster_size[:active].mul_(self.decay).add_(
                    counts,
                    alpha=1.0 - self.decay,
                )
                self.embed_avg[:active].mul_(self.decay).add_(
                    sums,
                    alpha=1.0 - self.decay,
                )

                used = (
                    self.cluster_size[:active]
                    > self.epsilon * 10.0
                )
                updated = (
                    self.embed_avg[:active]
                    / self.cluster_size[:active]
                    .clamp_min(self.epsilon)
                    .unsqueeze(1)
                )
                self.embedding.weight.data[:active][used] = updated[used]
                self.code_age[:active].add_(1)

                self._maybe_maintain()

            commitment = torch.nn.functional.mse_loss(
                inputs,
                quantized.detach(),
            )
            self.last_vq_loss = self.commitment_cost * commitment
        else:
            self.last_vq_loss = inputs.new_zeros(())

        return inputs + (quantized - inputs).detach()

    def get_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        return self._distances(flat_inputs).argmin(dim=1)

    def get_num_codes(self) -> int:
        return int(self.active_embeddings.item())

    def metrics(self) -> Dict[str, float]:
        active = int(self.active_embeddings.item())
        occupancy = self._occupancy()
        used = int(
            (
                self.cluster_size[:active]
                > self.epsilon * 10.0
            ).sum().item()
        )
        return {
            "active_codes": float(active),
            "ema_used_codes": float(used),
            "occupancy_min": (
                float(occupancy.min().item()) if active else 0.0
            ),
            "occupancy_max": (
                float(occupancy.max().item()) if active else 0.0
            ),
            "occupancy_perplexity": (
                float(torch.exp(
                    -(
                        occupancy.clamp_min(self.epsilon)
                        * occupancy.clamp_min(self.epsilon).log()
                    ).sum()
                ).item())
                if active else 0.0
            ),
            "step_counter": float(self.step_counter.item()),
            "last_check_step": float(self.last_check_step.item()),
            "last_maintenance_step": float(
                self.last_maintenance_step.item()
            ),
            "last_structure_step": float(
                self.last_structure_step.item()
            ),
            "surprise_streak": float(self.surprise_streak.item()),
            "growth_count": float(self.growth_count.item()),
            "prune_count": float(self.prune_count.item()),
            "merge_count": float(self.merge_count.item()),
            **self.last_distance_stats,
        }

    def growth_events(self):
        """Compatibility name used by train.py; returns all structure events."""
        return list(self._events)


class EffectRegressorMLP:
    model_kind = MODEL_KIND

    def __init__(self, opts: Dict):
        self.opts = dict(opts)
        self.device = torch.device(opts["device"])

        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)

        common = dict(
            commitment_cost=float(
                opts.get("vq_commitment_cost", 0.25)
            ),
            decay=float(opts.get("vq_decay", 0.99)),
            epsilon=float(opts.get("vq_epsilon", 1e-5)),
            warmup_steps=int(
                opts.get("dynamic_warmup_steps", 200)
            ),
            growth_interval=int(
                opts.get("dynamic_growth_interval", 100)
            ),
            min_support=int(
                opts.get("dynamic_min_support", 8)
            ),
            min_support_fraction=float(
                opts.get("dynamic_min_support_fraction", 0.05)
            ),
            required_checks=int(
                opts.get("dynamic_required_checks", 2)
            ),
            initial_embeddings=int(
                opts.get("dynamic_initial_embeddings", 1)
            ),
            pruning_enabled=bool(
                opts.get("dynamic_pruning_enabled", True)
            ),
            merging_enabled=bool(
                opts.get("dynamic_merging_enabled", False)
            ),
            maintenance_interval=int(
                opts.get("dynamic_maintenance_interval", 100)
            ),
            structure_warmup_steps=int(
                opts.get("dynamic_structure_warmup_steps", 1000)
            ),
            structure_cooldown_steps=int(
                opts.get("dynamic_structure_cooldown_steps", 300)
            ),
            minimum_embeddings=int(
                opts.get("dynamic_minimum_embeddings", 1)
            ),
            prune_patience=int(
                opts.get("dynamic_prune_patience", 3)
            ),
            prune_min_age_steps=int(
                opts.get("dynamic_prune_min_age_steps", 800)
            ),
            merge_max_small_fraction=float(
                opts.get("dynamic_merge_max_small_fraction", 0.05)
            ),
            merge_min_age_steps=int(
                opts.get("dynamic_merge_min_age_steps", 800)
            ),
        )

        max1 = int(
            opts.get(
                "vq_num_embeddings1",
                2 ** int(opts["code1_dim"]),
            )
        )
        max2 = int(
            opts.get(
                "vq_num_embeddings2",
                2 ** int(opts["code2_dim"]),
            )
        )

        self.encoder1[-1] = GrowPruneEMAVQLayer(
            max_embeddings=max1,
            embedding_dim=int(opts["code1_dim"]),
            surprise_threshold=float(
                opts.get("surprise_threshold_1", 1.0)
            ),
            prune_min_fraction=float(
                opts.get(
                    "dynamic_prune_min_fraction_1",
                    opts.get("dynamic_prune_min_fraction", 0.01),
                )
            ),
            merge_distance_threshold=float(
                opts.get(
                    "dynamic_merge_distance_threshold_1",
                    opts.get(
                        "dynamic_merge_distance_threshold",
                        0.05,
                    ),
                )
            ),
            **common,
        ).to(self.device)

        self.encoder2[-1] = GrowPruneEMAVQLayer(
            max_embeddings=max2,
            embedding_dim=int(opts["code2_dim"]),
            surprise_threshold=float(
                opts.get("surprise_threshold_2", 1.0)
            ),
            prune_min_fraction=float(
                opts.get(
                    "dynamic_prune_min_fraction_2",
                    opts.get("dynamic_prune_min_fraction", 0.01),
                )
            ),
            merge_distance_threshold=float(
                opts.get(
                    "dynamic_merge_distance_threshold_2",
                    opts.get(
                        "dynamic_merge_distance_threshold",
                        0.05,
                    ),
                )
            ),
            **common,
        ).to(self.device)

        self.decoder1 = MLP(
            [opts["code1_dim"] + 3]
            + [opts["hidden_dim"]] * opts["depth"]
            + [3]
        ).to(self.device)
        self.decoder2 = MLP(
            [opts["code2_dim"] + opts["code1_dim"] * 2]
            + [opts["hidden_dim"]] * opts["depth"]
            + [6]
        ).to(self.device)

        self.optimizer1 = torch.optim.Adam(
            [
                {"params": self.encoder1.parameters()},
                {"params": self.decoder1.parameters()},
            ],
            lr=opts["learning_rate1"],
            amsgrad=True,
        )
        self.optimizer2 = torch.optim.Adam(
            [
                {"params": self.encoder2.parameters()},
                {"params": self.decoder2.parameters()},
            ],
            lr=opts["learning_rate2"],
            amsgrad=True,
        )

        self.effect_weights1 = _weights(
            opts,
            "effect_weights1",
            [1.0, 1.0, 10.0],
            self.device,
        )
        self.effect_weights2 = _weights(
            opts,
            "effect_weights2",
            [1.0, 1.0, 5.0, 1.0, 1.0, 1.0],
            self.device,
        )
        self.save_path = opts["save"]

    def _weighted_mse(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.nn.functional.mse_loss(
                prediction,
                target,
                reduction="none",
            )
            * weights
        ).mean()

    def loss_components(self, sample, level: int):
        if level == 1:
            observation = sample["observation"].to(self.device)
            effect = sample["effect"].to(self.device)
            action = sample["action"].to(self.device)

            code = self.encoder1(observation)
            prediction = self.decoder1(
                torch.cat([code, action], dim=-1)
            )
            effect_loss = self._weighted_mse(
                prediction,
                effect,
                self.effect_weights1,
            )
            vq_loss = self.encoder1[-1].last_vq_loss

        elif level == 2:
            observation = sample["observation"].to(self.device)
            effect = sample["effect"].to(self.device)

            with torch.no_grad():
                object_codes = self.encoder1(
                    observation.reshape(
                        -1,
                        1,
                        observation.shape[2],
                        observation.shape[3],
                    )
                )
            object_codes = object_codes.reshape(
                observation.shape[0],
                -1,
            )
            relation_code = self.encoder2(observation)
            prediction = self.decoder2(
                torch.cat([object_codes, relation_code], dim=-1)
            )
            effect_loss = self._weighted_mse(
                prediction,
                effect,
                self.effect_weights2,
            )
            vq_loss = self.encoder2[-1].last_vq_loss
        else:
            raise ValueError(level)

        return {
            "total": effect_loss + vq_loss,
            "effect": effect_loss,
            "vq": vq_loss,
        }

    def optimize_batch(self, sample, level):
        optimizer = self.optimizer1 if level == 1 else self.optimizer2
        optimizer.zero_grad(set_to_none=True)
        losses = self.loss_components(sample, level)
        losses["total"].backward()
        optimizer.step()
        return {
            key: float(value.detach().cpu())
            for key, value in losses.items()
        }

    def prepare_level(self, level: int, training: bool) -> None:
        if level == 1:
            self.encoder1.train(training)
            self.decoder1.train(training)
            self.encoder2.eval()
            self.decoder2.eval()
        else:
            # Eval mode prevents Level-1 EMA/growth/prune/merge changes.
            self.encoder1.eval()
            self.decoder1.eval()
            self.encoder2.train(training)
            self.decoder2.train(training)

    def freeze_level1(self) -> None:
        self.encoder1.eval()
        self.decoder1.eval()
        for module in (self.encoder1, self.decoder1):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def additional_metrics(self, level: int):
        layer = self.encoder1[-1] if level == 1 else self.encoder2[-1]
        return layer.metrics()

    def growth_events(self, level: int):
        layer = self.encoder1[-1] if level == 1 else self.encoder2[-1]
        return layer.growth_events()

    def load(self, path: str, ext: str, level: int) -> None:
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        encoder.load_state_dict(
            torch.load(
                os.path.join(
                    path,
                    f"encoder{level}{ext}.ckpt",
                ),
                map_location=self.device,
            )
        )
        decoder.load_state_dict(
            torch.load(
                os.path.join(
                    path,
                    f"decoder{level}{ext}.ckpt",
                ),
                map_location=self.device,
            )
        )
        encoder.to(self.device)
        decoder.to(self.device)

    def save(self, path: str, ext: str, level: int) -> None:
        os.makedirs(path, exist_ok=True)
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2

        torch.save(
            {
                key: value.detach().cpu()
                for key, value in encoder.state_dict().items()
            },
            os.path.join(
                path,
                f"encoder{level}{ext}.ckpt",
            ),
        )
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in decoder.state_dict().items()
            },
            os.path.join(
                path,
                f"decoder{level}{ext}.ckpt",
            ),
        )

    def print_model(self, level: int) -> None:
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2

        print("=" * 10 + " ENCODER " + "=" * 10)
        print(encoder)
        print(
            f"parameter count: {utils.get_parameter_count(encoder)}"
        )
        print("=" * 29)
        print("=" * 10 + " DECODER " + "=" * 10)
        print(decoder)
        print(
            f"parameter count: {utils.get_parameter_count(decoder)}"
        )
        print("=" * 29)