from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .retrieval import DocumentIndex, SearchResult
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
    resumed: bool = False


def approx_tokens(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return max(1, cjk + round(latin * 1.25))


class AgentHarness:
    allowed_tools = {"search_docs"}

    def __init__(self, index: DocumentIndex, input_cost_per_million: float = 0.0, output_cost_per_million: float = 0.0):
        self.index = index
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

    def run_tool(self, name: str, *, query: str, top_k: int) -> list[SearchResult]:
        if name not in self.allowed_tools:
            raise HarnessError(f"tool is not allowed: {name}", "tool_boundary")
        return self.index.search(query, top_k=top_k)

    def run(
        self,
        question: str,
        *,
        top_k: int = 3,
        trace_path: str | Path | None = None,
        timeout_ms: int = 5_000,
        simulate_delay_ms: int = 0,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> AgentResult:
        if not question.strip():
            raise HarnessError("question must not be empty", "validation")
        if not 1 <= top_k <= 5:
            raise HarnessError("top_k must be between 1 and 5", "validation")
        run_id = uuid.uuid4().hex[:12]
        trace = TraceWriter(trace_path)
        started = time.perf_counter()
        key = hashlib.sha256(f"{question}|{top_k}".encode()).hexdigest()[:16]
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
            trace.write("tool_started", run_id=run_id, tool="search_docs")
            results = self.run_tool("search_docs", query=question, top_k=top_k)
            trace.write(
                "tool_completed",
                run_id=run_id,
                tool="search_docs",
                result_count=len(results),
                sources=[result.chunk_id for result in results],
            )
            if not results:
                raise HarnessError("no relevant evidence found", "no_result")
            if (time.perf_counter() - started) * 1000 > timeout_ms:
                raise HarnessError("run exceeded timeout after tool call", "timeout")

            selected = results[:2]
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

