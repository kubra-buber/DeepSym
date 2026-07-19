"""DeepSym with a fixed EMA vector-quantized bottleneck."""

from __future__ import annotations

import os
from typing import Dict

import torch

import utils
from blocks import MLP, build_encoder


MODEL_KIND = "vq"


def _weights(opts: Dict, key: str, default, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(opts.get(key, default), dtype=torch.float32, device=device)


class EMAVQLayer(torch.nn.Module):
    """EMA VQ with data-dependent initialization and stable unused-code handling."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        if num_embeddings < 1:
            raise ValueError("num_embeddings must be positive")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.decay = float(decay)
        self.epsilon = float(epsilon)

        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.normal_()
        self.embedding.weight.requires_grad = False

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("embed_avg", torch.zeros(self.num_embeddings, self.embedding_dim))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.last_vq_loss = torch.tensor(0.0)

    @torch.no_grad()
    def _initialize_from_batch(self, flat_inputs: torch.Tensor) -> None:
        n = flat_inputs.shape[0]
        if n == 0:
            raise ValueError("Cannot initialize VQ codebook from an empty batch")
        k = self.num_embeddings
        if k == 1:
            initial = flat_inputs.mean(dim=0, keepdim=True).detach()
        else:
            # Deterministic farthest-point initialization is much less sensitive
            # to minibatch order than selecting equally spaced row indices.
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
        self.embedding.weight.data.copy_(initial)
        self.embed_avg.copy_(initial)
        self.cluster_size.fill_(1.0)
        self.initialized.fill_(True)

    def _distances(self, flat_inputs: torch.Tensor) -> torch.Tensor:
        weights = self.embedding.weight
        return (
            flat_inputs.pow(2).sum(dim=1, keepdim=True)
            + weights.pow(2).sum(dim=1)
            - 2.0 * flat_inputs @ weights.t()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        if self.training and not bool(self.initialized.item()):
            self._initialize_from_batch(flat_inputs)

        distances = self._distances(flat_inputs)
        indices = distances.argmin(dim=1)
        encodings = torch.nn.functional.one_hot(
            indices, num_classes=self.num_embeddings
        ).to(flat_inputs.dtype)
        quantized = encodings @ self.embedding.weight
        quantized = quantized.view_as(inputs)

        if self.training:
            with torch.no_grad():
                counts = encodings.sum(dim=0)
                sums = encodings.t() @ flat_inputs
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.embed_avg.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)

                # Do not move truly unused codes to enormous values by dividing by epsilon.
                used = self.cluster_size > self.epsilon * 10.0
                updated = self.embed_avg / self.cluster_size.clamp_min(self.epsilon).unsqueeze(1)
                self.embedding.weight.data[used] = updated[used]

            commitment = torch.nn.functional.mse_loss(inputs, quantized.detach())
            self.last_vq_loss = self.commitment_cost * commitment
        else:
            self.last_vq_loss = inputs.new_zeros(())

        return inputs + (quantized - inputs).detach()

    def get_indices(self, inputs: torch.Tensor) -> torch.Tensor:
        flat_inputs = inputs.reshape(-1, self.embedding_dim)
        return self._distances(flat_inputs).argmin(dim=1)

    def get_num_codes(self) -> int:
        return self.num_embeddings

    def metrics(self) -> Dict[str, float]:
        nonzero = int((self.cluster_size > self.epsilon * 10.0).sum().item())
        return {
            "active_codes": float(self.num_embeddings),
            "ema_used_codes": float(nonzero),
        }


class EffectRegressorMLP:
    model_kind = MODEL_KIND

    def __init__(self, opts: Dict):
        self.opts = dict(opts)
        self.device = torch.device(opts["device"])
        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)

        commitment = float(opts.get("vq_commitment_cost", 0.25))
        decay = float(opts.get("vq_decay", 0.99))
        epsilon = float(opts.get("vq_epsilon", 1e-5))
        num1 = int(opts.get("vq_num_embeddings1", 2 ** int(opts["code1_dim"])))
        num2 = int(opts.get("vq_num_embeddings2", 2 ** int(opts["code2_dim"])))

        self.encoder1[-1] = EMAVQLayer(
            num1, int(opts["code1_dim"]), commitment, decay, epsilon
        ).to(self.device)
        self.encoder2[-1] = EMAVQLayer(
            num2, int(opts["code2_dim"]), commitment, decay, epsilon
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
        raw = torch.nn.functional.mse_loss(pred, target, reduction="none")
        return (raw * weights).mean()

    def loss_components(self, sample: Dict[str, torch.Tensor], level: int):
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
            self.encoder1.eval()  # prevents EMA updates in level 1
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
        return []

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