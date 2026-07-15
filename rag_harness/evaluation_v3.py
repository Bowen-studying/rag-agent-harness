from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .retrieve import retrieve_candidates, validate_semantic_decision
from .sources import load_index


SCHEMA_VERSION = "3.0"


def _mean(values: list[float], empty: float = 0.0) -> float:
    return statistics.mean(values) if values else empty


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _locator_key(item: dict) -> tuple[str, str]:
    return item.get("doc_id", ""), item.get("locator", "")


def _matches(candidate: dict, locator: dict) -> bool:
    if candidate.get("doc_id") != locator.get("doc_id"):
        return False
    expected = locator.get("locator", "")
    actual = candidate.get("locator", "")
    return actual == expected or (locator.get("allow_prefix") and actual.startswith(expected))


def validate_v3_cases(cases: list[dict], manifest_sha256: str) -> None:
    seen: set[str] = set()
    for case in cases:
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"missing or duplicate case id: {case_id!r}")
        seen.add(case_id)
        if case.get("expected_status") not in {"answerable", "insufficient_evidence", "needs_clarification"}:
            raise ValueError(f"{case_id} has invalid expected_status")
        if case["expected_status"] == "answerable" and not case.get("gold_evidence_groups"):
            raise ValueError(f"{case_id} must define gold_evidence_groups")
        case_manifest = case.get("source_manifest_sha256")
        if case_manifest and case_manifest != manifest_sha256:
            raise ValueError(f"{case_id} was annotated against a different source manifest")


def evaluate_v3(
    cases_path: str | Path,
    index_path: str | Path,
    output_path: str | Path,
    *,
    candidate_top_k: int = 20,
    max_candidates: int = 8,
    max_evidence_tokens: int = 3000,
    semantic_decisions_path: str | Path | None = None,
) -> dict:
    load_started = time.perf_counter()
    index, manifest = load_index(index_path)
    index_load_ms = round((time.perf_counter() - load_started) * 1000, 3)
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    validate_v3_cases(cases, manifest["manifest_sha256"])
    semantic_decisions = {}
    if semantic_decisions_path:
        raw_decisions = json.loads(Path(semantic_decisions_path).read_text(encoding="utf-8"))
        semantic_decisions = raw_decisions.get("decisions", raw_decisions)
        if not isinstance(semantic_decisions, dict):
            raise ValueError("semantic decisions must be an object keyed by case id")
    rows: list[dict] = []

    for case in cases:
        started = time.perf_counter()
        try:
            bundle = retrieve_candidates(
                index,
                case["question"],
                aspects=case.get("aspects"),
                candidate_top_k=candidate_top_k,
                max_candidates=max_candidates,
                max_evidence_tokens=max_evidence_tokens,
                manifest_sha256=manifest["manifest_sha256"],
            )
            execution_success = True
            failure_reason = None
        except Exception as exc:  # report failures instead of hiding them
            bundle = {"retrieval": {"status": "failed", "retrieval_ms": 0.0}, "candidates": []}
            execution_success = False
            failure_reason = type(exc).__name__

        candidates = bundle.get("candidates", [])
        lexical_status = bundle["retrieval"]["status"]
        semantic_raw = semantic_decisions.get(case["id"])
        semantic_error = None
        if semantic_raw is not None:
            try:
                semantic = validate_semantic_decision(bundle, semantic_raw)
            except (KeyError, TypeError, ValueError) as exc:
                semantic = None
                semantic_error = str(exc)
        else:
            semantic = None
        if semantic:
            selected_ids = set(semantic["selected_chunk_ids"])
            selected = [item for item in candidates if item["chunk_id"] in selected_ids]
            judged_status = semantic["status"]
        else:
            selected = candidates if case["expected_status"] == "answerable" else []
            judged_status = (
                "insufficient_evidence"
                if lexical_status in {"no_lexical_match", "weak_match"}
                else "answerable"
            )
        expected_status = case["expected_status"]
        if expected_status == "answerable":
            groups = case["gold_evidence_groups"]
            group_rows = []
            allowed = [locator for group in groups for locator in group.get("any_of", [])]
            for group in groups:
                candidate_matches = [item["chunk_id"] for item in candidates if any(_matches(item, loc) for loc in group.get("any_of", []))]
                selected_matches = [item["chunk_id"] for item in selected if any(_matches(item, loc) for loc in group.get("any_of", []))]
                group_rows.append(
                    {
                        "id": group["id"],
                        "candidate_found": bool(candidate_matches),
                        "passed": bool(selected_matches),
                        "matched_chunk_ids": selected_matches,
                    }
                )
            correct = [item for item in selected if any(_matches(item, loc) for loc in allowed)]
            precision = len(correct) / len(selected) if selected else 0.0
            recall = sum(row["passed"] for row in group_rows) / len(group_rows)
            candidate_recall = sum(row["candidate_found"] for row in group_rows) / len(group_rows)
            f1 = _f1(precision, recall)
            boundary_correct = judged_status == "answerable"
            passed = bool(
                execution_success
                and semantic_error is None
                and boundary_correct
                and precision >= 0.9
                and recall == 1.0
            )
        else:
            group_rows = []
            precision = recall = f1 = None
            candidate_recall = None
            boundary_correct = judged_status == expected_status
            passed = bool(execution_success and semantic_error is None and boundary_correct)

        rows.append(
            {
                "id": case["id"],
                "category": case.get("category", "uncategorized"),
                "expected_status": expected_status,
                "retrieval_status": lexical_status,
                "judged_status": judged_status,
                "semantic_decision_present": semantic_raw is not None,
                "semantic_decision_valid": semantic_raw is not None and semantic_error is None,
                "semantic_error": semantic_error,
                "execution_success": execution_success,
                "failure_reason": failure_reason,
                "case_passed": passed,
                "boundary_correct": boundary_correct,
                "evidence_precision": None if precision is None else round(precision, 4),
                "evidence_group_recall": None if recall is None else round(recall, 4),
                "evidence_f1": None if f1 is None else round(f1, 4),
                "candidate_group_recall": None if candidate_recall is None else round(candidate_recall, 4),
                "evidence_groups": group_rows,
                "candidate_chunk_ids": [item["chunk_id"] for item in candidates],
                "selected_chunk_ids": [item["chunk_id"] for item in selected],
                "retrieval_ms": bundle["retrieval"].get("retrieval_ms", 0.0),
                "total_case_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )

    answer_rows = [row for row in rows if row["expected_status"] == "answerable"]
    negative_rows = [row for row in rows if row["expected_status"] != "answerable"]
    known_ids = {chunk.chunk_id for chunk in index.chunks}
    cited_ids = [chunk_id for row in rows for chunk_id in row["candidate_chunk_ids"]]
    locator_valid = sum(chunk_id in known_ids for chunk_id in cited_ids) / len(cited_ids) if cited_ids else 1.0
    summary = {
        "case_count": len(rows),
        "answerable_case_count": len(answer_rows),
        "hard_negative_case_count": len(negative_rows),
        "execution_success_rate": round(_mean([float(row["execution_success"]) for row in rows], 1.0), 4),
        "retrieval_task_pass_rate": round(_mean([float(row["case_passed"]) for row in rows]), 4),
        "candidate_group_recall": round(_mean([row["candidate_group_recall"] for row in answer_rows]), 4),
        "evidence_precision": round(_mean([row["evidence_precision"] for row in answer_rows]), 4),
        "evidence_group_recall": round(_mean([row["evidence_group_recall"] for row in answer_rows]), 4),
        "evidence_f1": round(_mean([row["evidence_f1"] for row in answer_rows]), 4),
        "aspect_evidence_coverage": round(_mean([row["evidence_group_recall"] for row in answer_rows]), 4),
        "hard_negative_accuracy": round(_mean([float(row["boundary_correct"]) for row in negative_rows], 1.0), 4),
        "semantic_decision_coverage": round(
            _mean([float(row["semantic_decision_valid"]) for row in rows]), 4
        ),
        "citation_locator_valid_rate": round(locator_valid, 4),
        "index_load_ms": index_load_ms,
        "p50_retrieval_ms": round(statistics.median(row["retrieval_ms"] for row in rows), 3) if rows else 0.0,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "index_manifest_sha256": manifest["manifest_sha256"],
        "config": {
            "candidate_top_k": candidate_top_k,
            "max_candidates": max_candidates,
            "max_evidence_tokens": max_evidence_tokens,
            "semantic_decisions_supplied": bool(semantic_decisions_path),
        },
        "summary": summary,
        "cases": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def quality_gate(report: dict) -> tuple[bool, list[str]]:
    summary = report["summary"]
    checks = {
        "citation_locator_valid_rate": (summary["citation_locator_valid_rate"], 1.0),
        "candidate_group_recall": (summary["candidate_group_recall"], 0.9),
        "evidence_precision": (summary["evidence_precision"], 0.9),
        "evidence_group_recall": (summary["evidence_group_recall"], 0.9),
        "evidence_f1": (summary["evidence_f1"], 0.9),
        "aspect_evidence_coverage": (summary["aspect_evidence_coverage"], 0.85),
        "hard_negative_accuracy": (summary["hard_negative_accuracy"], 0.9),
        "retrieval_task_pass_rate": (summary["retrieval_task_pass_rate"], 0.8),
        "semantic_decision_coverage": (summary["semantic_decision_coverage"], 1.0),
    }
    failures = [f"{name}={value:.4f} < {threshold:.4f}" for name, (value, threshold) in checks.items() if value < threshold]
    return not failures, failures
