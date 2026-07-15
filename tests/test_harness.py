import json
import tempfile
import unittest
from pathlib import Path

from rag_harness.agent import AgentHarness
from rag_harness.evaluation import evaluate_cases
from rag_harness.retrieval import DocumentIndex


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

    def test_no_result_is_explicit(self):
        result = self.harness.run("火星基地午餐菜单是什么？")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "no_result")

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(Exception):
            self.harness.run("发布窗口", top_k=9)

    def test_tool_boundary_blocks_unknown_tool(self):
        with self.assertRaises(Exception):
            self.harness.run_tool("delete_database", query="x", top_k=1)

    def test_timeout_is_observable(self):
        result = self.harness.run("发布窗口", timeout_ms=1, simulate_delay_ms=5)
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "timeout")

    def test_trace_contains_completed_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            result = self.harness.run("P0 首次响应时间", trace_path=trace)
            self.assertTrue(result.success)
            events = [json.loads(line)["event"] for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertIn("tool_completed", events)
            self.assertIn("evidence_selected", events)
            self.assertIn("run_completed", events)

    def test_weak_secondary_evidence_is_filtered(self):
        cases = [
            ("生产发布允许在哪些时间窗口进行？", "release_policy.md#p2"),
            ("公共模型网关每分钟允许多少请求？", "api_limits.md#p2"),
        ]
        for question, expected_citation in cases:
            with self.subTest(question=question):
                result = self.harness.run(question)
                self.assertTrue(result.success)
                self.assertEqual(result.citations, [expected_citation])

    def test_strong_multi_source_evidence_is_kept(self):
        result = self.harness.run("Agent 是否可以直接修改生产数据库？")
        sources = {citation.split("#", 1)[0] for citation in result.citations}
        self.assertEqual(sources, {"incident_response.md", "security_policy.md"})

    def test_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.harness.run("发布窗口是什么？", checkpoint_dir=tmp)
            second = self.harness.run("发布窗口是什么？", checkpoint_dir=tmp, resume=True)
            self.assertTrue(first.success)
            self.assertTrue(second.resumed)
            self.assertEqual(first.answer, second.answer)

    def test_full_evaluation_has_ten_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            report = evaluate_cases(ROOT / "eval_cases.json", ROOT / "sample_docs", output)
            self.assertEqual(report["summary"]["case_count"], 10)
            self.assertEqual(report["summary"]["citation_accuracy"], 1.0)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
