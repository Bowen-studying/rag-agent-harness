from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .retrieval import DocumentIndex, SearchResult, split_query_aspects
from .trace import TraceWriter


class HarnessError(RuntimeError):
    def __init__(self, message: str, failure_type: str):
        super().__init__(message)
        self.failure_type = failure_type


@dataclass
class AgentResult:
    run_id: str
    question: str
    answer: str
    citations: list[str]
    evidence: list[dict]
    success: bool
    failure_reason: str | None
    latency_ms: float
    approx_input_tokens: int
    approx_output_tokens: int
    estimated_cost_usd: float
    selection: dict = field(default_factory=dict)
    resumed: bool = False


def approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return max(1, cjk + round(latin * 1.25))


class AgentHarness:
    allowed_tools = {"search_docs"}
    selection_policy_version = "adaptive-v2"

    def __init__(
        self,
        index: DocumentIndex,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
        min_global_score_ratio: float = 0.45,
        max_evidence_tokens: int = 800,
    ):
        if not 0.0 <= min_global_score_ratio <= 1.0:
            raise ValueError("min_global_score_ratio must be between 0 and 1")
        if max_evidence_tokens < 1:
            raise ValueError("max_evidence_tokens must be positive")
        self.index = index
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.min_global_score_ratio = min_global_score_ratio
        self.max_evidence_tokens = max_evidence_tokens

    def run_tool(self, name: str, *, query: str, top_k: int) -> list[SearchResult]:
        if name not in self.allowed_tools:
            raise HarnessError(f"tool is not allowed: {name}", "tool_boundary")
        return self.index.search(query, top_k=top_k)

    @staticmethod
    def _is_heading(result: SearchResult) -> bool:
        return result.text.lstrip().startswith("#")

    def select_evidence(self, question: str, results: list[SearchResult]) -> tuple[list[SearchResult], dict]:
        """Select a variable number of relevant chunks within a token budget.

        Strong full-query matches are kept by relative score. For an explicit
        compound question, the best result for each sub-question is also added.
        This preserves precision on noisy matches without imposing a fixed
        evidence count that would hide missing support on multi-part questions.
        """
        if not results:
            return [], {"policy": self.selection_policy_version, "aspects": [], "decisions": []}

        ranked = [item for item in results if not self._is_heading(item)]
        if not ranked:
            ranked = results[:1]
        top_score = ranked[0].score
        threshold = top_score * self.min_global_score_ratio
        candidates: dict[str, SearchResult] = {}
        reasons: dict[str, list[str]] = {}
        rejected_reasons: dict[str, list[str]] = {}

        for item in results:
            if self._is_heading(item):
                rejected_reasons[item.chunk_id] = ["heading_only"]
            elif item.score >= threshold:
                candidates[item.chunk_id] = item
                reasons.setdefault(item.chunk_id, []).append("strong_global_match")
            else:
                rejected_reasons[item.chunk_id] = ["below_global_threshold"]

        aspect_runs = []
        for aspect in split_query_aspects(question):
            aspect_results = [item for item in self.index.search(aspect, top_k=3) if not self._is_heading(item)]
            if not aspect_results:
                aspect_runs.append({"query": aspect, "selected": None})
                continue
            best = aspect_results[0]
            candidates.setdefault(best.chunk_id, best)
            reasons.setdefault(best.chunk_id, []).append(f"best_for_aspect:{aspect}")
            rejected_reasons.pop(best.chunk_id, None)
            aspect_runs.append({"query": aspect, "selected": best.chunk_id, "score": best.score})

        ordered_ids = []
        for item in results:
            if item.chunk_id in candidates and item.chunk_id not in ordered_ids:
                ordered_ids.append(item.chunk_id)
        for run in aspect_runs:
            chunk_id = run.get("selected")
            if chunk_id and chunk_id not in ordered_ids:
                ordered_ids.append(chunk_id)

        selected: list[SearchResult] = []
        used_tokens = 0
        for chunk_id in ordered_ids:
            item = candidates[chunk_id]
            item_tokens = approx_tokens(item.text)
            if selected and used_tokens + item_tokens > self.max_evidence_tokens:
                rejected_reasons[chunk_id] = ["token_budget_exceeded"]
                continue
            selected.append(item)
            used_tokens += item_tokens

        selected_ids = {item.chunk_id for item in selected}
        decisions = []
        all_items = {item.chunk_id: item for item in results}
        all_items.update(candidates)
        for chunk_id, item in all_items.items():
            is_selected = chunk_id in selected_ids
            decision_reasons = (
                reasons.get(chunk_id, ["selected"])
                if is_selected
                else rejected_reasons.get(chunk_id, reasons.get(chunk_id, ["not_selected"]))
            )
            decisions.append(
                {
                    "chunk_id": chunk_id,
                    "score": item.score,
                    "selected": is_selected,
                    "reasons": decision_reasons,
                    "approx_tokens": approx_tokens(item.text),
                }
            )
        diagnostics = {
            "policy": self.selection_policy_version,
            "top_score": top_score,
            "min_global_score": round(threshold, 6),
            "min_global_score_ratio": self.min_global_score_ratio,
            "max_evidence_tokens": self.max_evidence_tokens,
            "selected_evidence_tokens": used_tokens,
            "aspects": aspect_runs,
            "decisions": decisions,
        }
        return selected, diagnostics

    def run(
        self,
        question: str,
        *,
        top_k: int = 5,
        tool_name: str = "search_docs",
        trace_path: str | Path | None = None,
        timeout_ms: int = 5_000,
        simulate_delay_ms: int = 0,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> AgentResult:
        if not question.strip():
            raise HarnessError("question must not be empty", "validation")
        if not 1 <= top_k <= 10:
            raise HarnessError("top_k must be between 1 and 10", "validation")
        run_id = uuid.uuid4().hex[:12]
        trace = TraceWriter(trace_path)
        started = time.perf_counter()
        fingerprint = "|".join(
            [
                self.selection_policy_version,
                question,
                str(top_k),
                tool_name,
                str(self.min_global_score_ratio),
                str(self.max_evidence_tokens),
            ]
        )
        key = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        checkpoint = Path(checkpoint_dir) / f"{key}.json" if checkpoint_dir else None
        if resume and checkpoint and checkpoint.exists():
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            data["resumed"] = True
            trace.write("checkpoint_hit", run_id=data["run_id"], checkpoint=checkpoint.name)
            return AgentResult(**data)

        trace.write("run_started", run_id=run_id, top_k=top_k, question=question)
        try:
            if simulate_delay_ms:
                time.sleep(simulate_delay_ms / 1000)
            if (time.perf_counter() - started) * 1000 > timeout_ms:
                raise HarnessError("run exceeded timeout before tool call", "timeout")
            trace.write("tool_started", run_id=run_id, tool=tool_name)
            results = self.run_tool(tool_name, query=question, top_k=top_k)
            trace.write(
                "tool_completed",
                run_id=run_id,
                tool=tool_name,
                result_count=len(results),
                sources=[result.chunk_id for result in results],
            )
            if not results:
                raise HarnessError("no relevant evidence found", "no_result")
            if (time.perf_counter() - started) * 1000 > timeout_ms:
                raise HarnessError("run exceeded timeout after tool call", "timeout")

            selected, selection = self.select_evidence(question, results)
            trace.write(
                "evidence_selected",
                run_id=run_id,
                **selection,
            )
            answer_parts = []
            for item in selected:
                clean = re.sub(r"^#+\s*", "", item.text).strip()
                answer_parts.append(clean)
            answer = "\n\n".join(answer_parts)
            citations = [item.chunk_id for item in selected]
            input_text = question + "\n" + "\n".join(item.text for item in selected)
            input_tokens = approx_tokens(input_text)
            output_tokens = approx_tokens(answer)
            cost = (input_tokens * self.input_cost_per_million + output_tokens * self.output_cost_per_million) / 1_000_000
            latency = round((time.perf_counter() - started) * 1000, 3)
            result = AgentResult(
                run_id=run_id,
                question=question,
                answer=answer,
                citations=citations,
                evidence=[asdict(item) for item in selected],
                success=True,
                failure_reason=None,
                latency_ms=latency,
                approx_input_tokens=input_tokens,
                approx_output_tokens=output_tokens,
                estimated_cost_usd=round(cost, 8),
                selection=selection,
            )
            trace.write("answer_completed", run_id=run_id, citations=citations, approx_input_tokens=input_tokens, approx_output_tokens=output_tokens)
            trace.write("run_completed", run_id=run_id, latency_ms=latency, estimated_cost_usd=result.estimated_cost_usd)
            if checkpoint:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except HarnessError as exc:
            latency = round((time.perf_counter() - started) * 1000, 3)
            trace.write("run_failed", run_id=run_id, failure_type=exc.failure_type, message=str(exc), latency_ms=latency)
            return AgentResult(run_id, question, "", [], [], False, exc.failure_type, latency, approx_tokens(question), 0, 0.0)
