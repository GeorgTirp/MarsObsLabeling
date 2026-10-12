"""Local class-key preferences; shared class identities are never modified."""

import json
from pathlib import Path

from marslabeler.classes import ClassScheme
from marslabeler.io.atomic import atomic_output


def load_shortcuts(path: Path, scheme: ClassScheme) -> ClassScheme:
    if not path.exists():
        return scheme
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("hotkeys"), dict):
        raise ValueError("Invalid shortcut preferences file.")
    return scheme.with_hotkeys({int(cid): key for cid, key in data["hotkeys"].items()})


def save_shortcuts(path: Path, scheme: ClassScheme) -> None:
    scheme.validate()
    data = {"version": 1, "hotkeys": {
        str(c.id): c.hotkey for c in [*scheme.classes.values(), scheme.abstain]
    }}
    with atomic_output(path) as temporary:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
