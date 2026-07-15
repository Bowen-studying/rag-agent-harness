from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from .agent import AgentHarness
from .retrieval import DocumentIndex


def evaluate_cases(cases_path: str | Path, docs_dir: str | Path, output_path: str | Path, trace_dir: str | Path | None = None) -> dict:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    index = DocumentIndex.from_directory(docs_dir)
    harness = AgentHarness(index)
    rows = []
    for case in cases:
        trace_path = Path(trace_dir) / f"{case['id']}.jsonl" if trace_dir else None
        result = harness.run(case["question"], top_k=3, trace_path=trace_path)
        sources = {citation.split("#", 1)[0] for citation in result.citations}
        expected_sources = set(case["expected_sources"])
        answer_lower = result.answer.lower()
        keyword_hits = [keyword for keyword in case["keywords"] if keyword.lower() in answer_lower]
        retrieval_hit = bool(sources & expected_sources)
        citation_accuracy = len(sources & expected_sources) / len(sources) if sources else 0.0
        keyword_score = len(keyword_hits) / len(case["keywords"]) if case["keywords"] else 1.0
        rows.append(
            {
                "id": case["id"],
                "question": case["question"],
                "success": result.success,
                "retrieval_hit": retrieval_hit,
                "citation_accuracy": round(citation_accuracy, 4),
                "keyword_score": round(keyword_score, 4),
                "citations": result.citations,
                "latency_ms": result.latency_ms,
                "approx_input_tokens": result.approx_input_tokens,
                "approx_output_tokens": result.approx_output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "failure_reason": result.failure_reason,
            }
        )
    total = len(rows)
    summary = {
        "case_count": total,
        "success_rate": round(sum(row["success"] for row in rows) / total, 4),
        "retrieval_hit_rate": round(sum(row["retrieval_hit"] for row in rows) / total, 4),
        "keyword_pass_rate": round(sum(row["keyword_score"] >= 0.5 for row in rows) / total, 4),
        "citation_accuracy": round(sum(row["citation_accuracy"] for row in rows) / total, 4),
        "p50_latency_ms": round(statistics.median(row["latency_ms"] for row in rows), 3),
        "approx_input_tokens": sum(row["approx_input_tokens"] for row in rows),
        "approx_output_tokens": sum(row["approx_output_tokens"] for row in rows),
        "estimated_cost_usd": round(sum(row["estimated_cost_usd"] for row in rows), 8),
    }
    report = {"summary": summary, "cases": rows}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

