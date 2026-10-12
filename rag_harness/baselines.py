from __future__ import annotations

import hashlib
import json

from .retrieval import Chunk


# Fingerprint the bundled cases and indexed text, not their paths or case count.
_NOVALAB_INPUT_SHA256 = "790e044170d36be6b0001681ac3815510e98fb1c3cef3395e34fed65b4e60690"
_LABELS = {
    "novalab_controlled": "NovaLab controlled regression (Schema 2.0)",
    "custom_schema2": "Custom evaluation (Schema 2.0; not the bundled NovaLab baseline)",
    "real_kb_schema3": "Real KB retrieval evaluation track (Schema 3.0)",
    "unidentified": "Unidentified legacy evaluation",
}
_COMPARISON_WARNING = (
    "NovaLab controlled regression and real-KB Schema 3.0 scores are not directly comparable: "
    "they use different datasets, difficulty levels and scoring rules. "
    "Do not pool them or present their difference as a quality improvement or regression. "
    "Controlled scores do not establish real-KB or production performance. "
    "A baseline label identifies the evaluation track, not the published run; "
    "scores apply only to this report's cases, sources and configuration."
)


def baseline_metadata(baseline_id: str) -> dict:
    return {
        "id": baseline_id,
        "label": _LABELS[baseline_id],
        "comparison_warning": _COMPARISON_WARNING,
    }


def schema2_baseline(cases: list[dict], chunks: list[Chunk]) -> dict:
    inputs = {"cases": cases, "chunks": [(chunk.chunk_id, chunk.text) for chunk in chunks]}
    fingerprint = hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    baseline_id = "novalab_controlled" if fingerprint == _NOVALAB_INPUT_SHA256 else "custom_schema2"
    return baseline_metadata(baseline_id)
