import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rag_harness.cli import eval_main, main
from rag_harness.evaluation import evaluate_cases
from rag_harness.evaluation_v3 import evaluate_v3, quality_gate
from rag_harness.privacy import assert_public_safe, export_public_artifacts
from rag_harness.retrieval import Chunk, DocumentIndex
from rag_harness.sources import save_index


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def assert_baseline(self, payload, expected_id):
        baseline = payload["baseline"]
        self.assertEqual(baseline["id"], expected_id)
        self.assertTrue(baseline["label"])
        for phrase in ("not directly comparable", "different datasets", "scoring"):
            self.assertIn(phrase, baseline["comparison_warning"])
        return baseline

    def write_json(self, name, payload):
        path = self.work / name
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def v3_fixture(self):
        chunk_id = "doc_demo#heading=window&chunk=1"
        index_path = self.work / "index.json"
        index = DocumentIndex([
            Chunk(chunk_id, "demo", "发布窗口要求周二", None, "doc_demo", "heading=window", "obsidian", {})
        ])
        manifest = {
            "schema_version": "3.0", "manifest_sha256": "fixture-manifest",
            "document_count": 1, "chunk_count": 1,
            "documents": [{"source_type": "obsidian", "extraction_status": "ok", "relative_path": "private.md"}],
        }
        save_index(index, manifest, index_path)
        cases_path = self.write_json("v3-cases.json", [{
            "id": "c1", "question": "发布窗口", "expected_status": "answerable",
            "source_manifest_sha256": "fixture-manifest",
            "gold_evidence_groups": [{"id": "window", "any_of": [{"doc_id": "doc_demo", "locator": "heading=window"}]}],
        }])
        decisions_path = self.write_json("decisions.json", {"c1": {"status": "answerable", "selected_chunk_ids": [chunk_id]}})
        return cases_path, index_path, decisions_path

    def export_report(self, cases_path, index_path, report):
        report_path = self.write_json("report-to-export.json", report)
        public_cases, public_report = export_public_artifacts(
            cases_path=cases_path, report_path=report_path, index_path=index_path,
            output_cases=self.work / "public-cases.json", output_report=self.work / "public-report.json",
        )
        cases = json.loads(public_cases.read_text(encoding="utf-8"))
        exported = json.loads(public_report.read_text(encoding="utf-8"))
        assert_public_safe(cases)
        assert_public_safe(exported)
        return exported

    def test_default_report_labels_novalab_without_changing_scores(self):
        output = self.work / "report.json"
        report = evaluate_cases(ROOT / "eval_cases.json", ROOT / "sample_docs", output)
        self.assert_baseline(report, "novalab_controlled")
        self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
        self.assertEqual(report["schema_version"], "2.0")
        self.assertEqual(report["summary"]["case_count"], 15)
        self.assertEqual(report["summary"]["task_pass_rate"], 1.0)
        self.assertEqual(report["summary"]["citation_f1"], 1.0)

    def test_copied_inputs_keep_novalab_identity(self):
        cases = json.loads((ROOT / "eval_cases.json").read_text(encoding="utf-8"))
        copied_cases = self.write_json("renamed-cases.json", cases)
        copied_docs = self.work / "renamed-docs"
        shutil.copytree(ROOT / "sample_docs", copied_docs)
        report = evaluate_cases(copied_cases, copied_docs, self.work / "report.json")
        self.assert_baseline(report, "novalab_controlled")
        self.assertEqual(report["summary"]["task_pass_rate"], 1.0)

    def test_changed_cases_or_documents_are_not_labeled_novalab(self):
        cases = json.loads((ROOT / "eval_cases.json").read_text(encoding="utf-8"))
        cases[0]["question"] += " 请回答。"
        custom_cases = self.write_json("custom-cases.json", cases)
        custom_docs = self.work / "sample_docs"
        shutil.copytree(ROOT / "sample_docs", custom_docs)
        document = custom_docs / "incident_response.md"
        document.write_text(document.read_text(encoding="utf-8") + "\n\nAdditional custom policy.\n", encoding="utf-8")
        for cases_path, docs_path in (
            (custom_cases, ROOT / "sample_docs"), (ROOT / "eval_cases.json", custom_docs),
        ):
            with self.subTest(cases=cases_path.name, docs=str(docs_path)):
                report = evaluate_cases(cases_path, docs_path, self.work / "report.json")
                self.assert_baseline(report, "custom_schema2")

    def test_eval_cli_entrypoints_keep_json_metrics_and_exit_codes(self):
        cases = json.loads((ROOT / "eval_cases.json").read_text(encoding="utf-8"))
        cases[0]["expected_citations"] = ["missing.md#p1"]
        failing_cases = self.write_json("failing-cases.json", cases)
        for entrypoint, prefix in ((eval_main, []), (main, ["eval"])):
            for cases_path, threshold, expected_code, baseline_id in (
                (ROOT / "eval_cases.json", "1", 0, "novalab_controlled"),
                (failing_cases, "1", 1, "custom_schema2"),
                (failing_cases, "0", 0, "custom_schema2"),
            ):
                with self.subTest(entrypoint=entrypoint.__name__, cases=cases_path.name, threshold=threshold):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        code = entrypoint(prefix + [
                            "--cases", str(cases_path), "--docs", str(ROOT / "sample_docs"),
                            "--output", str(self.work / "report.json"), "--trace-dir", str(self.work / "traces"),
                            "--fail-under", threshold,
                        ])
                    payload = json.loads(output.getvalue())
                    self.assertEqual(code, expected_code)
                    self.assert_baseline(payload, baseline_id)
                    self.assertEqual(payload["case_count"], 15)
                    self.assertIsInstance(payload["task_pass_rate"], float)
                    self.assertEqual(payload["task_pass_rate"] == 1.0, cases_path == ROOT / "eval_cases.json")

    def test_schema3_reports_keep_scores_and_semantic_quality_gate(self):
        cases, index, decisions = self.v3_fixture()
        for semantic in (None, decisions):
            with self.subTest(semantic=bool(semantic)):
                output = self.work / "v3-report.json"
                report = evaluate_v3(cases, index, output, semantic_decisions_path=semantic)
                self.assert_baseline(report, "real_kb_schema3")
                self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
                self.assertEqual(report["summary"]["case_count"], 1)
                self.assertEqual(report["summary"]["retrieval_task_pass_rate"], 1.0)
                self.assertEqual(report["summary"]["evidence_f1"], 1.0)
                self.assertEqual(report["summary"]["semantic_decision_coverage"], float(bool(semantic)))
                passed, failures = quality_gate(report)
                self.assertEqual(passed, bool(semantic))
                self.assertEqual(failures, [] if semantic else ["semantic_decision_coverage=0.0000 < 1.0000"])

    def test_schema3_cli_preserves_json_shape_and_gate_exit_codes(self):
        cases, index, decisions = self.v3_fixture()
        for semantic in (None, decisions):
            with self.subTest(semantic=bool(semantic)):
                output = io.StringIO()
                args = ["eval-v3", "--cases", str(cases), "--index", str(index), "--output", str(self.work / "report.json")]
                if semantic:
                    args += ["--semantic-decisions", str(semantic)]
                with redirect_stdout(output):
                    code = main(args)
                payload = json.loads(output.getvalue())
                self.assert_baseline(payload, "real_kb_schema3")
                self.assertEqual(code, 0 if semantic else 1)
                self.assertEqual(payload["summary"]["retrieval_task_pass_rate"], 1.0)
                self.assertEqual(payload["quality_gate_passed"], bool(semantic))
                self.assertEqual(bool(payload["failures"]), not bool(semantic))

    def test_public_export_preserves_baseline_and_privacy(self):
        cases, index, decisions = self.v3_fixture()
        report = evaluate_v3(cases, index, self.work / "report.json", semantic_decisions_path=decisions)
        baseline = self.assert_baseline(report, "real_kb_schema3")
        exported = self.export_report(cases, index, report)
        self.assertEqual(exported["baseline"], baseline)
        self.assertEqual(exported["summary"], report["summary"])
        self.assertNotIn("documents", exported["index_manifest"])
        self.assertNotIn("candidate_chunk_ids", exported["cases"][0])

    def test_legacy_schema3_export_gets_track_label(self):
        cases, index, decisions = self.v3_fixture()
        report = evaluate_v3(cases, index, self.work / "report.json", semantic_decisions_path=decisions)
        report.pop("baseline", None)
        exported = self.export_report(cases, index, report)
        self.assert_baseline(exported, "real_kb_schema3")
        self.assertEqual(exported["summary"], report["summary"])

    def test_unknown_schema_export_does_not_invent_baseline_identity(self):
        cases, index, decisions = self.v3_fixture()
        report = evaluate_v3(cases, index, self.work / "report.json", semantic_decisions_path=decisions)
        report.pop("baseline", None)
        report["schema_version"] = "unknown"
        exported = self.export_report(cases, index, report)
        self.assert_baseline(exported, "unidentified")
        self.assertEqual(exported["summary"], report["summary"])

    def test_public_reference_reports_identify_distinct_baselines(self):
        for filename, baseline_id, count, metric, value in (
            ("eval_report.json", "novalab_controlled", 15, "task_pass_rate", 1.0),
            ("kb_eval_report.public.json", "real_kb_schema3", 32, "retrieval_task_pass_rate", 0.9375),
        ):
            with self.subTest(filename=filename):
                report = json.loads((ROOT / "artifacts" / filename).read_text(encoding="utf-8"))
                self.assert_baseline(report, baseline_id)
                self.assertEqual(report["summary"]["case_count"], count)
                self.assertEqual(report["summary"][metric], value)
                assert_public_safe(report)


if __name__ == "__main__":
    unittest.main()
