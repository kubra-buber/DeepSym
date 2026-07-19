"""Original DeepSym straight-through bottleneck with a fair weighted-effect loss.

This file keeps the original encoder/decoder architecture from DeepSym.  The
only intended experimental change is that it uses the same configurable effect
weights as the VQ baselines, so bottlenecks are compared under the same loss.
"""

from __future__ import annotations

import os
from typing import Dict

import torch

import utils
from blocks import MLP, build_encoder


MODEL_KIND = "original"


def _weights(opts: Dict, key: str, default, device: torch.device) -> torch.Tensor:
    values = opts.get(key, default)
    return torch.as_tensor(values, dtype=torch.float32, device=device)


class EffectRegressorMLP:
    model_kind = MODEL_KIND

    def __init__(self, opts: Dict):
        self.opts = dict(opts)
        self.device = torch.device(opts["device"])

        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
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

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raw = torch.nn.functional.mse_loss(pred, target, reduction="none")
        if raw.shape[-1] != weights.numel():
            raise ValueError(
                f"Effect dimension {raw.shape[-1]} does not match weights {weights.tolist()}"
            )
        return (raw * weights).mean()

    def loss_components(self, sample: Dict[str, torch.Tensor], level: int) -> Dict[str, torch.Tensor]:
        if level == 1:
            obs = sample["observation"].to(self.device)
            effect = sample["effect"].to(self.device)
            action = sample["action"].to(self.device)
            code = self.encoder1(obs)
            pred = self.decoder1(torch.cat([code, action], dim=-1))
            effect_loss = self._weighted_mse(pred, effect, self.effect_weights1)
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
        else:
            raise ValueError(f"Unknown level: {level}")

        zero = effect_loss.new_zeros(())
        return {"total": effect_loss, "effect": effect_loss, "vq": zero}

    def optimize_batch(self, sample: Dict[str, torch.Tensor], level: int) -> Dict[str, float]:
        optimizer = self.optimizer1 if level == 1 else self.optimizer2
        optimizer.zero_grad(set_to_none=True)
        losses = self.loss_components(sample, level)
        losses["total"].backward()
        optimizer.step()
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def prepare_level(self, level: int, training: bool) -> None:
        if level == 1:
            self.encoder1.train(training)
            self.decoder1.train(training)
            self.encoder2.eval()
            self.decoder2.eval()
            return

        # Level 1 is a frozen symbolic feature extractor during level-2 training.
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

    def additional_metrics(self, level: int) -> Dict[str, float]:
        return {}

    def growth_events(self, level: int):
        return []

    def load(self, path: str, ext: str, level: int) -> None:
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        encoder_state = torch.load(
            os.path.join(path, f"encoder{level}{ext}.ckpt"),
            map_location=self.device,
        )
        decoder_state = torch.load(
            os.path.join(path, f"decoder{level}{ext}.ckpt"),
            map_location=self.device,
        )
        encoder.load_state_dict(encoder_state)
        decoder.load_state_dict(decoder_state)
        encoder.to(self.device)
        decoder.to(self.device)

    def save(self, path: str, ext: str, level: int) -> None:
        os.makedirs(path, exist_ok=True)
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        encoder_state = {k: v.detach().cpu() for k, v in encoder.state_dict().items()}
        decoder_state = {k: v.detach().cpu() for k, v in decoder.state_dict().items()}
        torch.save(encoder_state, os.path.join(path, f"encoder{level}{ext}.ckpt"))
        torch.save(decoder_state, os.path.join(path, f"decoder{level}{ext}.ckpt"))

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