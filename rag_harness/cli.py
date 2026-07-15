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
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    result = AgentHarness(DocumentIndex.from_directory(args.docs)).run(args.question, top_k=args.top_k, trace_path=args.trace)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.success else 2


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic RAG evaluation cases.")
    parser.add_argument("--cases", default="eval_cases.json")
    parser.add_argument("--docs", default="sample_docs")
    parser.add_argument("--output", default="artifacts/eval_report.json")
    parser.add_argument("--trace-dir", default="artifacts/traces")
    args = parser.parse_args(argv)
    report = evaluate_cases(args.cases, args.docs, args.output, args.trace_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--docs", default="sample_docs")
    ask.add_argument("--trace")
    ask.add_argument("--top-k", type=int, default=3)
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cases", default="eval_cases.json")
    evaluate.add_argument("--docs", default="sample_docs")
    evaluate.add_argument("--output", default="artifacts/eval_report.json")
    evaluate.add_argument("--trace-dir", default="artifacts/traces")
    args = parser.parse_args(argv)
    if args.command == "ask":
        return ask_main([args.question, "--docs", args.docs, "--top-k", str(args.top_k)] + (["--trace", args.trace] if args.trace else []))
    return eval_main(["--cases", args.cases, "--docs", args.docs, "--output", args.output, "--trace-dir", args.trace_dir])


if __name__ == "__main__":
    raise SystemExit(main())

