from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .agent import AgentHarness
from .retrieval import DocumentIndex


SCHEMA_VERSION = "2.0"


def _safe_ratio(numerator: int | float, denominator: int | float, *, empty: float = 0.0) -> float:
    return numerator / denominator if denominator else empty


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _validate_cases(cases: list[dict]) -> None:
    seen = set()
    for index, case in enumerate(cases, 1):
        case_id = case.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"case {index} has a missing or duplicate id: {case_id!r}")
        seen.add(case_id)
        if not str(case.get("question", "")).strip():
            raise ValueError(f"{case_id} has an empty question")
        if "expected_citations" not in case:
            raise ValueError(f"{case_id} must define expected_citations")
        if case.get("expected_failure") and case["expected_citations"]:
            raise ValueError(f"{case_id} cannot expect both a failure and citations")


def _score_aspects(aspects: list[dict], answer: str) -> tuple[float, list[dict]]:
    if not aspects:
        return 1.0, []
    answer_lower = answer.lower()
    rows = []
    for aspect in aspects:
        keywords = aspect.get("keywords", [])
        hits = [keyword for keyword in keywords if keyword.lower() in answer_lower]
        missing = [keyword for keyword in keywords if keyword.lower() not in answer_lower]
        rows.append(
            {
                "name": aspect.get("name", "unnamed"),
                "passed": not missing,
                "hits": hits,
                "missing": missing,
            }
        )
    return _safe_ratio(sum(row["passed"] for row in rows), len(rows), empty=1.0), rows


def _score_answer_case(case: dict, result) -> dict:
    expected = set(case["expected_citations"])
    actual = set(result.citations)
    correct = actual & expected
    unexpected = actual - expected
    missing = expected - actual
    precision = _safe_ratio(len(correct), len(actual))
    recall = _safe_ratio(len(correct), len(expected), empty=1.0)
    citation_f1 = _f1(precision, recall)
    aspect_coverage, aspect_rows = _score_aspects(case.get("aspects", []), result.answer)
    keywords = case.get("keywords", [])
    answer_lower = result.answer.lower()
    keyword_hits = [keyword for keyword in keywords if keyword.lower() in answer_lower]
    keyword_score = _safe_ratio(len(keyword_hits), len(keywords), empty=1.0)
    case_passed = bool(result.success and precision == 1.0 and recall == 1.0 and aspect_coverage == 1.0)
    return {
        "case_passed": case_passed,
        "expected_failure_matched": None,
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(citation_f1, 4),
        "exact_citation_match": actual == expected,
        "expected_citations": sorted(expected),
        "missing_citations": sorted(missing),
        "unexpected_citations": sorted(unexpected),
        "aspect_coverage": round(aspect_coverage, 4),
        "aspects": aspect_rows,
        "keyword_score": round(keyword_score, 4),
        "retrieval_hit": bool(correct),
    }


def _score_failure_case(case: dict, result) -> dict:
    matched = bool(not result.success and result.failure_reason == case["expected_failure"])
    return {
        "case_passed": matched,
        "expected_failure_matched": matched,
        "citation_precision": None,
        "citation_recall": None,
        "citation_f1": None,
        "exact_citation_match": not result.citations,
        "expected_citations": [],
        "missing_citations": [],
        "unexpected_citations": list(result.citations),
        "aspect_coverage": None,
        "aspects": [],
        "keyword_score": None,
        "retrieval_hit": False,
    }


def evaluate_cases(
    cases_path: str | Path,
    docs_dir: str | Path,
    output_path: str | Path,
    trace_dir: str | Path | None = None,
    *,
    default_top_k: int = 5,
    min_global_score_ratio: float = 0.45,
    max_evidence_tokens: int = 800,
) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    _validate_cases(cases)
    index = DocumentIndex.from_directory(docs_dir)
    harness = AgentHarness(
        index,
        min_global_score_ratio=min_global_score_ratio,
        max_evidence_tokens=max_evidence_tokens,
    )
    rows = []
    for case in cases:
        trace_path = Path(trace_dir) / f"{case['id']}.jsonl" if trace_dir else None
        if trace_path and trace_path.exists():
            trace_path.unlink()
        result = harness.run(case["question"], top_k=case.get("top_k", default_top_k), trace_path=trace_path)
        scores = _score_failure_case(case, result) if case.get("expected_failure") else _score_answer_case(case, result)
        rows.append(
            {
                "id": case["id"],
                "category": case.get("category", "uncategorized"),
                "question": case["question"],
                "success": result.success,
                "failure_reason": result.failure_reason,
                **scores,
                "citations": result.citations,
                "latency_ms": result.latency_ms,
                "approx_input_tokens": result.approx_input_tokens,
                "approx_output_tokens": result.approx_output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
            }
        )

    answer_rows = [row for row in rows if row["expected_failure_matched"] is None]
    failure_rows = [row for row in rows if row["expected_failure_matched"] is not None]
    category_rows = defaultdict(list)
    for row in rows:
        category_rows[row["category"]].append(row)
    category_summary = {
        category: {
            "case_count": len(items),
            "pass_rate": round(_safe_ratio(sum(item["case_passed"] for item in items), len(items)), 4),
        }
        for category, items in sorted(category_rows.items())
    }
    total = len(rows)
    summary = {
        "case_count": total,
        "answer_case_count": len(answer_rows),
        "expected_failure_case_count": len(failure_rows),
        "task_pass_rate": round(_safe_ratio(sum(row["case_passed"] for row in rows), total), 4),
        "runtime_success_rate": round(_safe_ratio(sum(row["success"] for row in rows), total), 4),
        "expected_failure_pass_rate": round(
            _safe_ratio(sum(row["expected_failure_matched"] for row in failure_rows), len(failure_rows), empty=1.0), 4
        ),
        "citation_precision": round(statistics.mean(row["citation_precision"] for row in answer_rows), 4),
        "citation_recall": round(statistics.mean(row["citation_recall"] for row in answer_rows), 4),
        "citation_f1": round(statistics.mean(row["citation_f1"] for row in answer_rows), 4),
        "exact_citation_match_rate": round(
            _safe_ratio(sum(row["exact_citation_match"] for row in answer_rows), len(answer_rows), empty=1.0), 4
        ),
        "aspect_coverage": round(statistics.mean(row["aspect_coverage"] for row in answer_rows), 4),
        "retrieval_hit_rate": round(
            _safe_ratio(sum(row["retrieval_hit"] for row in answer_rows), len(answer_rows), empty=1.0), 4
        ),
        "keyword_pass_rate": round(
            _safe_ratio(sum(row["keyword_score"] == 1.0 for row in answer_rows), len(answer_rows), empty=1.0), 4
        ),
        "p50_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 3),
        "approx_input_tokens": sum(row["approx_input_tokens"] for row in rows),
        "approx_output_tokens": sum(row["approx_output_tokens"] for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in rows), 8),
        "by_category": category_summary,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "default_top_k": default_top_k,
            "min_global_score_ratio": min_global_score_ratio,
            "max_evidence_tokens": max_evidence_tokens,
            "selection_policy": harness.selection_policy_version,
        },
        "summary": summary,
        "cases": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
