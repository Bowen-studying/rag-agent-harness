from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass

from .retrieval import DocumentIndex, SearchResult, split_query_aspects


RETRIEVAL_SCHEMA_VERSION = "3.0"
SEMANTIC_STATUSES = {"answerable", "insufficient_evidence", "needs_clarification"}


def _approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return max(1, cjk + round(latin * 1.25))


def sanitize_excerpt(text: str, *, limit: int = 1200) -> str:
    text = re.sub(r"(?i)[A-Z]:\\[^\s]+|/home/[^\s]+|/mnt/[a-z]/[^\s]+", "[LOCAL_PATH]", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]", text)
    compact = re.sub(r"\n{3,}", "\n\n", text).strip()
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


@dataclass(frozen=True)
class Candidate:
    chunk_id: str
    doc_id: str
    source_type: str
    locator: str
    score: float
    score_ratio: float
    query_coverage: float
    approx_tokens: int
    excerpt: str
    reasons: list[str]


def retrieve_candidates(
    index: DocumentIndex,
    question: str,
    *,
    aspects: list[str] | None = None,
    candidate_top_k: int = 20,
    max_candidates: int = 8,
    max_evidence_tokens: int = 3000,
    min_score_ratio: float = 0.35,
    weak_coverage_threshold: float = 0.12,
    manifest_sha256: str | None = None,
) -> dict:
    if not question.strip():
        raise ValueError("question must not be empty")
    if not 1 <= max_candidates <= candidate_top_k <= 50:
        raise ValueError("candidate limits must satisfy 1 <= max_candidates <= candidate_top_k <= 50")
    started = time.perf_counter()
    global_results = [item for item in index.search(question, top_k=candidate_top_k) if not item.text.lstrip().startswith("#")]
    if not global_results:
        return {
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
            "query": {"original": question, "aspects": aspects or []},
            "retrieval": {"status": "no_lexical_match", "candidate_count": 0, "retrieval_ms": round((time.perf_counter() - started) * 1000, 3)},
            "candidates": [],
            "index_manifest_sha256": manifest_sha256,
        }

    explicit_aspects = aspects is not None
    query_aspects = aspects if explicit_aspects else split_query_aspects(question)
    top_score = global_results[0].score
    threshold = top_score * min_score_ratio
    selected: dict[str, tuple[SearchResult, list[str]]] = {}
    if not explicit_aspects:
        for item in global_results:
            if item.score >= threshold:
                selected[item.chunk_id] = (item, ["strong_global_match"])

    for aspect in query_aspects:
        aspect_results = [item for item in index.search(aspect, top_k=5) if not item.text.lstrip().startswith("#")]
        if aspect_results:
            item = aspect_results[0]
            if item.chunk_id in selected:
                selected[item.chunk_id][1].append(f"best_for_aspect:{aspect}")
            else:
                selected[item.chunk_id] = (item, [f"best_for_aspect:{aspect}"])

    if explicit_aspects and not selected:
        selected[global_results[0].chunk_id] = (global_results[0], ["global_fallback_after_empty_aspects"])

    ordered: list[Candidate] = []
    used_tokens = 0
    ranked = sorted(selected.values(), key=lambda value: (-value[0].score, value[0].chunk_id))
    for item, reasons in ranked:
        tokens = _approx_tokens(item.text)
        if ordered and used_tokens + tokens > max_evidence_tokens:
            continue
        ordered.append(
            Candidate(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id or item.source,
                source_type=item.source_type,
                locator=item.locator or item.chunk_id.rsplit("#", 1)[-1],
                score=item.score,
                score_ratio=round(item.score / top_score, 4) if top_score else 0.0,
                query_coverage=item.query_coverage,
                approx_tokens=tokens,
                excerpt=sanitize_excerpt(item.text),
                reasons=reasons,
            )
        )
        used_tokens += tokens
        if len(ordered) >= max_candidates:
            break

    top_coverage = global_results[0].query_coverage
    status = "weak_match" if top_coverage < weak_coverage_threshold else "candidates_found"
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "query": {"original": question, "aspects": query_aspects},
        "retrieval": {
            "status": status,
            "candidate_count": len(ordered),
            "top_score": top_score,
            "top_query_coverage": top_coverage,
            "selected_evidence_tokens": used_tokens,
            "retrieval_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "candidates": [asdict(item) for item in ordered],
        "index_manifest_sha256": manifest_sha256,
    }


def validate_semantic_decision(bundle: dict, decision: dict) -> dict:
    status = decision.get("status")
    if status not in SEMANTIC_STATUSES:
        raise ValueError(f"invalid semantic status: {status!r}")
    candidate_ids = {item["chunk_id"] for item in bundle.get("candidates", [])}
    selected = decision.get("selected_chunk_ids", [])
    invalid = sorted(set(selected) - candidate_ids)
    if invalid:
        raise ValueError(f"invalid_semantic_selection: {invalid}")
    return {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "status": status,
        "selected_chunk_ids": list(dict.fromkeys(selected)),
        "covered_aspects": decision.get("covered_aspects", []),
        "reason_codes": decision.get("reason_codes", []),
        "model_id": decision.get("model_id", "unknown"),
        "prompt_version": decision.get("prompt_version", "unknown"),
    }
