from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SENSITIVE_PATTERNS = {
    "windows_path": re.compile(r"[A-Za-z]:\\"),
    "wsl_path": re.compile(r"/(?:home|mnt)/[A-Za-z0-9_.-]+/"),
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "credential": re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]+"),
}


def scan_sensitive_text(text: str) -> list[str]:
    return [name for name, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]


def assert_public_safe(payload: dict | list) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    hits = scan_sensitive_text(text)
    if hits:
        raise ValueError(f"public artifact contains sensitive patterns: {hits}")
    if len(text) > 2_000_000:
        raise ValueError("public artifact is unexpectedly large; raw knowledge text may have leaked")


def public_manifest(manifest: dict) -> dict:
    by_type = Counter(item["source_type"] for item in manifest.get("documents", []))
    by_status = Counter(item["extraction_status"] for item in manifest.get("documents", []))
    return {
        "schema_version": manifest.get("schema_version"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "document_count": manifest.get("document_count", 0),
        "chunk_count": manifest.get("chunk_count", 0),
        "documents_by_source_type": dict(sorted(by_type.items())),
        "documents_by_extraction_status": dict(sorted(by_status.items())),
    }


def export_public_artifacts(
    *,
    cases_path: str | Path,
    report_path: str | Path,
    index_path: str | Path,
    output_cases: str | Path,
    output_report: str | Path,
) -> tuple[Path, Path]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    index_payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    public_cases = []
    for case in cases:
        public_cases.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"annotation_note", "annotated_by", "source_title", "raw_excerpt"}
            }
        )
    public_report = {
        "schema_version": report["schema_version"],
        "index_manifest": public_manifest(index_payload["manifest"]),
        "config": report["config"],
        "summary": report["summary"],
        "cases": [
            {
                key: value
                for key, value in row.items()
                if key not in {"candidate_chunk_ids", "selected_chunk_ids", "failure_reason", "semantic_error"}
            }
            for row in report["cases"]
        ],
        "privacy_note": "No source text, local path, user identifier, credential, or raw trace is included.",
    }
    assert_public_safe(public_cases)
    assert_public_safe(public_report)
    cases_target = Path(output_cases)
    report_target = Path(output_report)
    cases_target.write_text(json.dumps(public_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    report_target.write_text(json.dumps(public_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return cases_target, report_target
