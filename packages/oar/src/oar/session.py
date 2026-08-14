"""Append-only JSONL session persistence: ~/.oar/sessions/<escaped-cwd>/<uuid7>.jsonl"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class Session:
    def __init__(self, base_dir: Path | None = None):
        base = base_dir or Path.home() / ".oar" / "sessions"
        escaped = str(Path.cwd()).replace("/", "-")
        self.path = base / escaped / f"{uuid.uuid7()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_id: str | None = None

    def append(self, type: str, **data: object) -> None:
        entry: dict[str, object] = {
            "id": str(uuid.uuid7()),
            "parentId": self._last_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": type,
            **data,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._last_id = entry["id"]
