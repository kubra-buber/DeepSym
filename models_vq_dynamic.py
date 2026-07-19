"""DeepSym with a stabilized dynamically growing EMA-VQ bottleneck."""

from __future__ import annotations

import math
import os
from typing import Dict, List

import torch

import utils
from blocks import MLP, build_encoder


MODEL_KIND = "dynamic"


def _weights(opts: Dict, key: str, default, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(opts.get(key, default), dtype=torch.float32, device=device)


class DynamicEMAVQLayer(torch.nn.Module):
    """EMA codebook that grows only after supported, persistent surprise.

    The growth distance is squared Euclidean distance.  A code is added only at
    a growth-check step, only if enough samples in the current batch exceed the
    configured threshold.  The new vector is a medoid-like representative of
    the surprising samples, rather than the single farthest outlier.
    """

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
    ):
        super().__init__()
        self.max_embeddings = int(max_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.surprise_threshold = float(surprise_threshold)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.warmup_steps = int(warmup_steps)
        self.growth_interval = max(1, int(growth_interval))
        self.min_support = max(1, int(min_support))
        self.min_support_fraction = float(min_support_fraction)
        self.required_checks = max(1, int(required_checks))
        self.initial_embeddings = max(1, min(int(initial_embeddings), self.max_embeddings))

        self.embedding = torch.nn.Embedding(self.max_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("active_embeddings", torch.tensor(self.initial_embeddings, dtype=torch.long))
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_check_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_growth_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("surprise_streak", torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("cluster_size", torch.zeros(self.max_embeddings))
        self.register_buffer("embed_avg", torch.zeros(self.max_embeddings, self.embedding_dim))

        self.last_vq_loss = torch.tensor(0.0)
        self.last_distance_stats: Dict[str, float] = {}
        self._growth_events: List[Dict[str, float]] = []

    @torch.no_grad()
    def _initialize_from_batch(self, flat_inputs: torch.Tensor) -> None:
        n = flat_inputs.shape[0]
        k = int(self.active_embeddings.item())
        if k == 1:
            initial = flat_inputs.mean(dim=0, keepdim=True).detach()
        else:
            mean = flat_inputs.mean(dim=0, keepdim=True)
            first = ((flat_inputs - mean) ** 2).sum(dim=1).argmin()
            selected = [int(first.item())]
            while len(selected) < min(k, n):
                chosen = flat_inputs[selected]
                distances = torch.cdist(flat_inputs, chosen).pow(2).min(dim=1).values
                distances[selected] = -1.0
                selected.append(int(distances.argmax().item()))
            positions = torch.as_tensor(selected, device=flat_inputs.device, dtype=torch.long)
            if len(selected) < k:
                positions = positions.repeat((k + len(selected) - 1) // len(selected))[:k]
            initial = flat_inputs[positions].detach()
        self.embedding.weight.data[:k].copy_(initial)
        self.embed_avg[:k].copy_(initial)
        self.cluster_size[:k].fill_(1.0)
        self.initialized.fill_(True)

    def _active_weights(self) -> torch.Tensor:
        return self.embedding.weight[: int(self.active_embeddings.item())]

    def _distances(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        weights = self._active_weights()
        return (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            + weights.pow(2).sum(dim=1)
            - 2.0 * flat_inputs @ weights.t()
        )

    @torch.no_grad()
    def _maybe_grow(self, flat_inputs: torch.Tensor, distances: torch.Tensor) -> bool:
        step = int(self.step_counter.item())
        active = int(self.active_embeddings.item())
        if active >= self.max_embeddings or step <= self.warmup_steps:
            return False
        if step - int(self.last_check_step.item()) < self.growth_interval:
            return False
        self.last_check_step.fill_(step)

        min_distances = distances.min(dim=1).values
        required = max(
            self.min_support,
            int(math.ceil(self.min_support_fraction * min_distances.numel())),
        )
        far_mask = min_distances > self.surprise_threshold
        support = int(far_mask.sum().item())

        self.last_distance_stats = {
            "distance_mean": float(min_distances.mean().item()),
            "distance_p90": float(torch.quantile(min_distances, 0.90).item()),
            "distance_p95": float(torch.quantile(min_distances, 0.95).item()),
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
        representative_idx = ((surprising - center) ** 2).sum(dim=1).argmin()
        candidate = surprising[representative_idx].detach()

        new_index = active
        initial_mass = float(max(1, support))
        self.embedding.weight.data[new_index].copy_(candidate)
        self.cluster_size[new_index] = initial_mass
        self.embed_avg[new_index].copy_(candidate * initial_mass)
        self.active_embeddings.add_(1)
        self.last_growth_step.fill_(step)
        self.surprise_streak.zero_()

        event = {
            "step": float(step),
            "new_active_codes": float(active + 1),
            "support": float(support),
            "required_support": float(required),
            "required_checks": float(self.required_checks),
            "threshold": float(self.surprise_threshold),
            **self.last_distance_stats,
        }
        self._growth_events.append(event)
        print(
            f"[Dynamic VQ] step={step} K={active}->{active + 1} "
            f"support={support}/{min_distances.numel()} threshold={self.surprise_threshold:.4f}"
        )
        return True

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
        encodings = torch.nn.functional.one_hot(indices, num_classes=active).to(flat_inputs.dtype)
        quantized = (encodings @ self.embedding.weight[:active]).view_as(inputs)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0)
                sums = encodings.t() @ flat_inputs
                self.cluster_size[:active].mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.embed_avg[:active].mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
                used = self.cluster_size[:active] > self.epsilon * 10.0
                updated = self.embed_avg[:active] / self.cluster_size[:active].clamp_min(self.epsilon).unsqueeze(1)
                self.embedding.weight.data[:active][used] = updated[used]

            commitment = torch.nn.functional.mse_loss(inputs, quantized.detach())
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
        used = int((self.cluster_size[:active] > self.epsilon * 10.0).sum().item())
        return {
            "active_codes": float(active),
            "ema_used_codes": float(used),
            "step_counter": float(self.step_counter.item()),
            "last_check_step": float(self.last_check_step.item()),
            "surprise_streak": float(self.surprise_streak.item()),
            **self.last_distance_stats,
        }

    def growth_events(self):
        return list(self._growth_events)


class EffectRegressorMLP:
    model_kind = MODEL_KIND

    def __init__(self, opts: Dict):
        self.opts = dict(opts)
        self.device = torch.device(opts["device"])
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)

        common = dict(
            commitment_cost=float(opts.get("vq_commitment_cost", 0.25)),
            decay=float(opts.get("vq_decay", 0.99)),
            epsilon=float(opts.get("vq_epsilon", 1e-5)),
            warmup_steps=int(opts.get("dynamic_warmup_steps", 200)),
            growth_interval=int(opts.get("dynamic_growth_interval", 100)),
            min_support=int(opts.get("dynamic_min_support", 8)),
            min_support_fraction=float(opts.get("dynamic_min_support_fraction", 0.05)),
            required_checks=int(opts.get("dynamic_required_checks", 2)),
            initial_embeddings=int(opts.get("dynamic_initial_embeddings", 1)),
        )
        max1 = int(opts.get("vq_num_embeddings1", 2 ** int(opts["code1_dim"])))
        max2 = int(opts.get("vq_num_embeddings2", 2 ** int(opts["code2_dim"])))
        self.encoder1[-1] = DynamicEMAVQLayer(
            max1,
            int(opts["code1_dim"]),
            float(opts.get("surprise_threshold_1", 1.0)),
            **common,
        ).to(self.device)
        self.encoder2[-1] = DynamicEMAVQLayer(
            max2,
            int(opts["code2_dim"]),
            float(opts.get("surprise_threshold_2", 1.0)),
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
            opts, "effect_weights1", [1.0, 1.0, 10.0], self.device
        )
        self.effect_weights2 = _weights(
            opts, "effect_weights2", [1.0, 1.0, 5.0, 1.0, 1.0, 1.0], self.device
        )
        self.save_path = opts["save"]

    def _weighted_mse(self, pred, target, weights):
        return (torch.nn.functional.mse_loss(pred, target, reduction="none") * weights).mean()

    def loss_components(self, sample, level: int):
        if level == 1:
            obs = sample["observation"].to(self.device)
            effect = sample["effect"].to(self.device)
            action = sample["action"].to(self.device)
            code = self.encoder1(obs)
            pred = self.decoder1(torch.cat([code, action], dim=-1))
            effect_loss = self._weighted_mse(pred, effect, self.effect_weights1)
            vq_loss = self.encoder1[-1].last_vq_loss
        elif level == 2:
            obs = sample["observation"].to(self.device)
            effect = sample["effect"].to(self.device)
            with torch.no_grad():
                object_codes = self.encoder1(
                    obs.reshape(-1, 1, obs.shape[2], obs.shape[3])
                )
            object_codes = object_codes.reshape(obs.shape[0], -1)
            relation_code = self.encoder2(obs)
            pred = self.decoder2(torch.cat([object_codes, relation_code], dim=-1))
            effect_loss = self._weighted_mse(pred, effect, self.effect_weights2)
            vq_loss = self.encoder2[-1].last_vq_loss
        else:
            raise ValueError(level)
        return {"total": effect_loss + vq_loss, "effect": effect_loss, "vq": vq_loss}

    def optimize_batch(self, sample, level):
        optimizer = self.optimizer1 if level == 1 else self.optimizer2
        optimizer.zero_grad(set_to_none=True)
        losses = self.loss_components(sample, level)
        losses["total"].backward()
        optimizer.step()
        return {k: float(v.detach().cpu()) for k, v in losses.items()}

    def prepare_level(self, level: int, training: bool) -> None:
        if level == 1:
            self.encoder1.train(training)
            self.decoder1.train(training)
            self.encoder2.eval()
            self.decoder2.eval()
        else:
            # Critical: eval mode stops level-1 EMA updates and dynamic growth.
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
            torch.load(os.path.join(path, f"encoder{level}{ext}.ckpt"), map_location=self.device)
        )
        decoder.load_state_dict(
            torch.load(os.path.join(path, f"decoder{level}{ext}.ckpt"), map_location=self.device)
        )
        encoder.to(self.device)
        decoder.to(self.device)

    def save(self, path: str, ext: str, level: int) -> None:
        os.makedirs(path, exist_ok=True)
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        torch.save(
            {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
            os.path.join(path, f"encoder{level}{ext}.ckpt"),
        )
        torch.save(
            {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
            os.path.join(path, f"decoder{level}{ext}.ckpt"),
        )

    def print_model(self, level: int) -> None:
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        print("=" * 10 + " ENCODER " + "=" * 10)
        print(encoder)
        print(f"parameter count: {utils.get_parameter_count(encoder)}")
        print("=" * 29)
        print("=" * 10 + " DECODER " + "=" * 10)
        print(decoder)
        print(f"parameter count: {utils.get_parameter_count(decoder)}")
        print("=" * 29)