from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .agent import AgentHarness
from .evaluation import evaluate_cases
from .retrieval import DocumentIndex


def ask_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the sample RAG agent and optionally write a JSONL trace.")
    parser.add_argument("question")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--trace")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tool", default="search_docs")
    parser.add_argument("--min-score-ratio", type=float, default=0.45)
    parser.add_argument("--max-evidence-tokens", type=int, default=800)
    args = parser.parse_args(argv)
    harness = AgentHarness(
        DocumentIndex.from_directory(args.docs),
        min_global_score_ratio=args.min_score_ratio,
        max_evidence_tokens=args.max_evidence_tokens,
    )
    result = harness.run(args.question, top_k=args.top_k, tool_name=args.tool, trace_path=args.trace)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 2


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic RAG evaluation cases.")
    parser.add_argument("--cases", default="eval_cases.json")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--output", default="artifacts/eval_report.json")
    parser.add_argument("--trace-dir", default="artifacts/traces")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score-ratio", type=float, default=0.45)
    parser.add_argument("--max-evidence-tokens", type=int, default=800)
    parser.add_argument("--fail-under", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.fail_under <= 1.0:
        parser.error("--fail-under must be between 0 and 1")
    report = evaluate_cases(
        args.cases,
        args.docs,
        args.output,
        args.trace_dir,
        default_top_k=args.top_k,
        min_global_score_ratio=args.min_score_ratio,
        max_evidence_tokens=args.max_evidence_tokens,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["task_pass_rate"] >= args.fail_under else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--docs", default="sample_docs")
    ask.add_argument("--trace")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--tool", default="search_docs")
    ask.add_argument("--min-score-ratio", type=float, default=0.45)
    ask.add_argument("--max-evidence-tokens", type=int, default=800)
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cases", default="eval_cases.json")
    evaluate.add_argument("--docs", default="sample_docs")
    evaluate.add_argument("--output", default="artifacts/eval_report.json")
    evaluate.add_argument("--trace-dir", default="artifacts/traces")
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--min-score-ratio", type=float, default=0.45)
    evaluate.add_argument("--max-evidence-tokens", type=int, default=800)
    evaluate.add_argument("--fail-under", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "ask":
        return ask_main(
            [
                args.question,
                "--docs",
                args.docs,
                "--top-k",
                str(args.top_k),
                "--tool",
                args.tool,
                "--min-score-ratio",
                str(args.min_score_ratio),
                "--max-evidence-tokens",
                str(args.max_evidence_tokens),
            ]
            + (["--trace", args.trace] if args.trace else [])
        )
    return eval_main(
        [
            "--cases",
            args.cases,
            "--docs",
            args.docs,
            "--output",
            args.output,
            "--trace-dir",
            args.trace_dir,
            "--top-k",
            str(args.top_k),
            "--min-score-ratio",
            str(args.min_score_ratio),
            "--max-evidence-tokens",
            str(args.max_evidence_tokens),
            "--fail-under",
            str(args.fail_under),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
