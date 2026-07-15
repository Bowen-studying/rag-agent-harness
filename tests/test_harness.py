import json
import tempfile
import unittest
from pathlib import Path

from rag_harness.agent import AgentHarness, HarnessError
from rag_harness.evaluation import evaluate_cases
from rag_harness.retrieval import DocumentIndex, split_query_aspects


ROOT = Path(__file__).resolve().parents[1]


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = DocumentIndex.from_directory(ROOT / "sample_docs")
        cls.harness = AgentHarness(cls.index)

    def test_retrieval_returns_expected_source(self):
        results = self.index.search("P0 事故 首次响应")
        self.assertTrue(results)
        self.assertEqual(results[0].source, "incident_response.md")

    def test_bm25_prefers_rare_security_term(self):
        results = self.index.search("越权")
        self.assertEqual(results[0].chunk_id, "security_policy.md#p4")

    def test_compound_question_is_split_into_aspects(self):
        question = "工具调用发生越权时系统怎么处理，日志又必须记录哪些字段？"
        aspects = split_query_aspects(question)
        self.assertIn("越权", aspects)
        self.assertGreaterEqual(len(aspects), 3)

    def test_no_result_is_explicit(self):
        result = self.harness.run("火星基地午餐菜单是什么？")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_result")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(HarnessError) as context:
            self.harness.run("发布窗口", top_k=11)
        self.assertEqual(context.exception.failure_type, "validation")

    def test_tool_boundary_blocks_unknown_tool(self):
        with self.assertRaises(HarnessError) as context:
            self.harness.run_tool("delete_database", query="x", top_k=1)
        self.assertEqual(context.exception.failure_type, "tool_boundary")

    def test_tool_boundary_is_observable_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            result = self.harness.run("删除数据库", tool_name="delete_database", trace_path=trace)
            self.assertFalse(result.success)
            self.assertEqual(result.failure_reason, "tool_boundary")
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            failure = next(event for event in events if event["event"] == "run_failed")
            self.assertEqual(failure["failure_type"], "tool_boundary")

    def test_timeout_is_observable(self):
        result = self.harness.run("发布窗口", timeout_ms=1, simulate_delay_ms=5)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "timeout")

    def test_trace_contains_selection_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            result = self.harness.run("P0 首次响应时间", trace_path=trace)
            self.assertTrue(result.success)
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            names = [event["event"] for event in events]
            self.assertIn("tool_completed", names)
            self.assertIn("evidence_selected", names)
            self.assertIn("run_completed", names)
            selection = next(event for event in events if event["event"] == "evidence_selected")
            self.assertEqual(selection["policy"], "adaptive-v2")
            self.assertTrue(selection["decisions"])

    def test_hard_negative_sources_are_filtered(self):
        cases = [
            ("生产发布允许在哪些时间窗口进行？", ["release_policy.md#p2"]),
            ("公共模型网关每分钟允许多少请求？", ["api_limits.md#p2"]),
        ]
        for question, expected in cases:
            with self.subTest(question=question):
                result = self.harness.run(question)
                self.assertEqual(result.citations, expected)

    def test_strong_multi_source_evidence_is_kept(self):
        result = self.harness.run("Agent 是否可以直接修改生产数据库？")
        self.assertEqual(
            set(result.citations),
            {"incident_response.md#p3", "security_policy.md#p2"},
        )

    def test_three_paragraph_question_keeps_all_support(self):
        result = self.harness.run("生产发布的时间窗口、发布前测试和日志保留期限分别是什么？")
        self.assertEqual(
            set(result.citations),
            {"release_policy.md#p2", "release_policy.md#p3", "release_policy.md#p4"},
        )

    def test_cross_document_question_keeps_all_support(self):
        result = self.harness.run(
            "发生 P0 事故且需要紧急发布修复时，首次响应、双重批准和发布前测试分别是什么要求？"
        )
        self.assertEqual(
            set(result.citations),
            {"incident_response.md#p2", "release_policy.md#p2", "release_policy.md#p3"},
        )

    def test_token_budget_rejects_overflow_evidence(self):
        limited = AgentHarness(self.index, max_evidence_tokens=60)
        result = limited.run("生产发布的时间窗口、发布前测试和日志保留期限分别是什么？")
        decisions = result.selection["decisions"]
        self.assertTrue(any("token_budget_exceeded" in item["reasons"] for item in decisions))
        self.assertLessEqual(result.selection["selected_evidence_tokens"], 60)

    def test_checkpoint_fingerprint_includes_selection_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = AgentHarness(self.index, max_evidence_tokens=400).run("发布窗口是什么？", checkpoint_dir=tmp)
            resumed = AgentHarness(self.index, max_evidence_tokens=400).run(
                "发布窗口是什么？", checkpoint_dir=tmp, resume=True
            )
            changed = AgentHarness(self.index, max_evidence_tokens=800).run(
                "发布窗口是什么？", checkpoint_dir=tmp, resume=True
            )
            self.assertTrue(first.success)
            self.assertTrue(resumed.resumed)
            self.assertFalse(changed.resumed)

    def test_full_evaluation_v2_passes_all_fifteen_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            traces = Path(tmp) / "traces"
            report = evaluate_cases(ROOT / "eval_cases.json", ROOT / "sample_docs", output, traces)
            summary = report["summary"]
            self.assertEqual(report["schema_version"], "2.0")
            self.assertEqual(summary["case_count"], 15)
            self.assertEqual(summary["task_pass_rate"], 1.0)
            self.assertEqual(summary["citation_precision"], 1.0)
            self.assertEqual(summary["citation_recall"], 1.0)
            self.assertEqual(summary["aspect_coverage"], 1.0)
            self.assertEqual(summary["expected_failure_pass_rate"], 1.0)
            self.assertTrue(output.exists())

            evaluate_cases(ROOT / "eval_cases.json", ROOT / "sample_docs", output, traces)
            case_trace = traces / "case_01.jsonl"
            run_starts = [
                json.loads(line)
                for line in case_trace.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "run_started"
            ]
            self.assertEqual(len(run_starts), 1)


if __name__ == "__main__":
    unittest.main()
