from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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
    index_parser = subparsers.add_parser("index", help="Build a persistent page-aware local index")
    index_subparsers = index_parser.add_subparsers(dest="index_command", required=True)
    index_build = index_subparsers.add_parser("build")
    index_build.add_argument("--config", required=True)
    index_build.add_argument("--output", default=".local/index.json")
    index_build.add_argument("--cache-dir", default=".local/source_cache")
    retrieve = subparsers.add_parser("retrieve", help="Return a sanitized candidate bundle for an Agent")
    retrieve.add_argument("question")
    retrieve.add_argument("--index", default=".local/index.json")
    retrieve.add_argument("--aspect", action="append", default=None)
    retrieve.add_argument("--candidate-top-k", type=int, default=20)
    retrieve.add_argument("--max-candidates", type=int, default=8)
    retrieve.add_argument("--max-evidence-tokens", type=int, default=3000)
    evaluate_v3_parser = subparsers.add_parser("eval-v3", help="Evaluate Schema 3.0 evidence groups")
    evaluate_v3_parser.add_argument("--cases", required=True)
    evaluate_v3_parser.add_argument("--index", default=".local/index.json")
    evaluate_v3_parser.add_argument("--output", default=".local/kb_eval_report.json")
    evaluate_v3_parser.add_argument("--semantic-decisions")
    feynman_eval = subparsers.add_parser("eval-feynman", help="Score structured Feynman feedback without exact text matching")
    feynman_eval.add_argument("--cases", default="eval_feynman_cases.public.json")
    feynman_eval.add_argument("--predictions", required=True)
    feynman_eval.add_argument("--output", default="artifacts/feynman_eval_report.local.json")
    sync_parser = subparsers.add_parser("sync", help="Incrementally rebuild, evaluate, and atomically promote an index")
    sync_parser.add_argument("--config", required=True)
    sync_parser.add_argument("--eval-cases", required=True)
    sync_parser.add_argument("--index", default=".local/index.json")
    sync_parser.add_argument("--report", default=".local/kb_eval_report.json")
    sync_parser.add_argument("--cache-dir", default=".local/source_cache")
    sync_parser.add_argument("--semantic-decisions")
    semantic = subparsers.add_parser("validate-semantic", help="Validate an Agent/LLM decision against a candidate bundle")
    semantic.add_argument("--bundle", required=True)
    semantic.add_argument("--decision", required=True)
    public_export = subparsers.add_parser("public-export", help="Write privacy-checked public cases and report")
    public_export.add_argument("--cases", required=True)
    public_export.add_argument("--report", required=True)
    public_export.add_argument("--index", default=".local/index.json")
    public_export.add_argument("--output-cases", default="eval_kb_cases.public.json")
    public_export.add_argument("--output-report", default="artifacts/kb_eval_report.public.json")
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
    if args.command == "eval":
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
    if args.command == "index":
        from .sources import build_index_from_config, save_index

        index, manifest = build_index_from_config(args.config, cache_dir=args.cache_dir)
        save_index(index, manifest, args.output)
        print(json.dumps({"status": "built", "output": args.output, "manifest": manifest}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "retrieve":
        from .retrieve import retrieve_candidates
        from .sources import load_index

        index, manifest = load_index(args.index)
        bundle = retrieve_candidates(
            index,
            args.question,
            aspects=args.aspect,
            candidate_top_k=args.candidate_top_k,
            max_candidates=args.max_candidates,
            max_evidence_tokens=args.max_evidence_tokens,
            manifest_sha256=manifest["manifest_sha256"],
        )
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0
    if args.command == "eval-v3":
        from .evaluation_v3 import evaluate_v3, quality_gate

        report = evaluate_v3(
            args.cases,
            args.index,
            args.output,
            semantic_decisions_path=args.semantic_decisions,
        )
        passed, failures = quality_gate(report)
        print(json.dumps({"summary": report["summary"], "quality_gate_passed": passed, "failures": failures}, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    if args.command == "eval-feynman":
        from .feynman_evaluation import evaluate_feynman_predictions

        report = evaluate_feynman_predictions(args.cases, args.predictions)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return 0 if report["summary"]["task_pass_rate"] == 1.0 else 1
    if args.command == "sync":
        from .sync import sync_index

        result = sync_index(
            config_path=args.config,
            cases_path=args.eval_cases,
            index_path=args.index,
            report_path=args.report,
            cache_dir=args.cache_dir,
            semantic_decisions_path=args.semantic_decisions,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] in {"unchanged", "promoted"} else 1
    if args.command == "validate-semantic":
        from .retrieve import validate_semantic_decision

        bundle = json.loads(Path(args.bundle).read_text(encoding="utf-8"))
        decision = json.loads(Path(args.decision).read_text(encoding="utf-8"))
        print(json.dumps(validate_semantic_decision(bundle, decision), ensure_ascii=False, indent=2))
        return 0
    from .privacy import export_public_artifacts

    outputs = export_public_artifacts(
        cases_path=args.cases,
        report_path=args.report,
        index_path=args.index,
        output_cases=args.output_cases,
        output_report=args.output_report,
    )
    print(json.dumps({"status": "exported", "files": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
