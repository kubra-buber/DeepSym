#!/usr/bin/env python3
from pathlib import Path

path = Path('compare_tim_fast_downward.py')
text = path.read_text()

if 'import re\n' not in text:
    text = text.replace('import random\n', 'import random\nimport re\n', 1)

old = '''def normalize_action_name(name: str) -> str:
    return " ".join(name.lower().strip().strip("()").split())
'''
new = '''def normalize_action_name(name: str) -> str:
    return " ".join(name.lower().strip().strip("()").split())


def logical_action_name(name: str) -> str:
    """Ignore the final learned sample-count suffix, e.g. _c364/_c599."""
    normalized = normalize_action_name(name)
    if not normalized:
        return normalized
    parts = normalized.split()
    parts[0] = re.sub(r"_c\\d+$", "", parts[0])
    return " ".join(parts)
'''

if old not in text:
    raise RuntimeError('normalize_action_name block not found')
text = text.replace(old, new, 1)

text = text.replace(
    'action = action_map.get(normalize_action_name(step))',
    'action = action_map.get(logical_action_name(step))',
)
text = text.replace(
    'return 0.0, f"unknown action: {step}"',
    'return 0.0, f"unknown logical action: {logical_action_name(step)}"',
)

old_map = '''    action_map = {
        normalize_action_name(action.name): action
        for action in actions
    }
'''
new_map = '''    action_map = {}
    for action in actions:
        key = logical_action_name(action.name)
        if key in action_map and action_map[key].name != action.name:
            raise RuntimeError(
                f"Logical alias collision for {key!r}: "
                f"{action_map[key].name!r} vs {action.name!r}"
            )
        action_map[key] = action
'''

if old_map not in text:
    raise RuntimeError('action_map block not found')
text = text.replace(old_map, new_map, 1)

text = text.replace(
    'normalize_action_name(exact_action)',
    'logical_action_name(exact_action)',
)
text = text.replace(
    'normalize_action_name(nominal_first or "")',
    'logical_action_name(nominal_first or "")',
)
text = text.replace(
    'normalize_action_name(sampled_first or "")',
    'logical_action_name(sampled_first or "")',
)
text = text.replace(
    'normalize_action_name(railroad_mcts_action)',
    'logical_action_name(railroad_mcts_action)',
)

path.write_text(text)
print(f'Patched: {path}')