from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .evaluation_v3 import evaluate_v3, quality_gate
from .sources import build_index_from_config, save_index


class SyncLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"sync already running: {self.path}") from exc
        os.write(self.fd, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def sync_index(
    *,
    config_path: str | Path,
    cases_path: str | Path,
    index_path: str | Path,
    report_path: str | Path,
    cache_dir: str | Path,
    semantic_decisions_path: str | Path | None = None,
) -> dict:
    target = Path(index_path)
    work_dir = target.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    lock = work_dir / ".sync.lock"
    candidate = work_dir / f".{target.name}.candidate"
    rejected = work_dir / f"{target.stem}.rejected.json"
    started = time.perf_counter()

    with SyncLock(lock):
        index, manifest = build_index_from_config(config_path, cache_dir=cache_dir)
        if target.exists():
            current = json.loads(target.read_text(encoding="utf-8"))
            current_hash = current.get("manifest", {}).get("manifest_sha256")
            if current_hash == manifest["manifest_sha256"]:
                return {
                    "status": "unchanged",
                    "manifest_sha256": current_hash,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                }

        save_index(index, manifest, candidate)
        try:
            report = evaluate_v3(
                cases_path,
                candidate,
                report_path,
                semantic_decisions_path=semantic_decisions_path,
            )
            passed, failures = quality_gate(report)
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            passed = False
            failures = [f"evaluation_error:{type(exc).__name__}:{exc}"]
        if passed:
            os.replace(candidate, target)
            rejected.unlink(missing_ok=True)
            status = "promoted"
        else:
            os.replace(candidate, rejected)
            status = "rejected"
        return {
            "status": status,
            "manifest_sha256": manifest["manifest_sha256"],
            "quality_gate_passed": passed,
            "quality_gate_failures": failures,
            "report_path": str(report_path) if Path(report_path).exists() else None,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
