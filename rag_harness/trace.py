from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        if not self.path:
            return
        record = {"ts": round(time.time(), 3), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

