from __future__ import annotations

"""Schema checks and narrow scoring for Feynman feedback.

The evaluator deliberately avoids exact free-text matching.  It checks the
decision category, required issue types, citation allowlist and output shape.
"""

import json
from pathlib import Path


REQUIRED_FIELDS = {
    "status",
    "accurate_parts",
    "omissions",
    "factual_conflicts",
    "logic_gaps",
    "questions",
    "reteach_request",
    "citations",
}
VALID_STATUSES = {"accurate", "has_omission", "factual_error", "insufficient_evidence", "needs_clarification"}


def _citation_key(item: dict) -> tuple[str, str]:
    return item.get("doc_id", ""), item.get("locator", "")


def evaluate_feynman_predictions(cases_path: str | Path, predictions_path: str | Path) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in predictions}
    rows = []
    for case in cases:
        prediction = by_id.get(case["id"], {})
        schema_valid = REQUIRED_FIELDS.issubset(prediction) and prediction.get("status") in VALID_STATUSES
        expected_types = set(case.get("expected_issue_types", []))
        actual_types = set(prediction.get("issue_types", []))
        issue_recall = len(expected_types & actual_types) / len(expected_types) if expected_types else 1.0
        allowed = {_citation_key(item) for item in case.get("acceptable_citations", [])}
        actual_citations = {_citation_key(item) for item in prediction.get("citations", [])}
        citation_valid = actual_citations.issubset(allowed) and (bool(actual_citations) or not allowed)
        status_correct = prediction.get("status") == case["expected_status"]
        rows.append(
            {
                "id": case["id"],
                "schema_valid": schema_valid,
                "status_correct": status_correct,
                "issue_type_recall": round(issue_recall, 4),
                "citation_valid": citation_valid,
                "passed": bool(schema_valid and status_correct and issue_recall == 1.0 and citation_valid),
            }
        )
    count = len(rows)
    mean = lambda key: round(sum(float(row[key]) for row in rows) / count, 4) if count else 1.0
    return {
        "schema_version": "3.0-feynman",
        "summary": {
            "case_count": count,
            "schema_valid_rate": mean("schema_valid"),
            "status_accuracy": mean("status_correct"),
            "issue_type_recall": mean("issue_type_recall"),
            "citation_valid_rate": mean("citation_valid"),
            "task_pass_rate": mean("passed"),
        },
        "cases": rows,
    }
