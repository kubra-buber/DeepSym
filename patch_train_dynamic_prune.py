#!/usr/bin/env python3
"""Patch train.py so --model dynamic_prune loads models_vq_dynamic_prune."""

from pathlib import Path

path = Path("train.py")
text = path.read_text()

needle = '    "dynamic": "models_vq_dynamic",\n'
addition = '    "dynamic_prune": "models_vq_dynamic_prune",\n'

if addition in text:
    print("train.py already supports dynamic_prune")
elif needle not in text:
    raise SystemExit(
        "Could not find MODEL_MODULES dynamic entry in train.py; "
        "edit MODEL_MODULES manually."
    )
else:
    text = text.replace(needle, needle + addition, 1)
    path.write_text(text)
    print("Patched train.py: added dynamic_prune model")