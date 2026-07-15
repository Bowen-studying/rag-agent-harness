import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from rag_harness.evaluation_v3 import evaluate_v3, quality_gate, validate_v3_cases
from rag_harness.feynman_evaluation import evaluate_feynman_predictions
from rag_harness.privacy import assert_public_safe, public_manifest, scan_sensitive_text
from rag_harness.retrieve import retrieve_candidates, sanitize_excerpt, validate_semantic_decision
from rag_harness.retrieval import Chunk, DocumentIndex
from rag_harness.sources import (
    build_index_from_config,
    extract_obsidian_source,
    extract_pdf_source,
    load_index,
    public_doc_id,
    save_index,
    split_with_overlap,
)
from rag_harness.sync import SyncLock
from rag_harness.sync import sync_index


def chunk(chunk_id: str, text: str, *, doc_id: str = "doc_demo", locator: str = "heading=demo") -> Chunk:
    return Chunk(chunk_id, "demo", text, None, doc_id, locator, "obsidian", {})


class SourceTests(unittest.TestCase):
    def test_public_doc_id_is_stable(self):
        self.assertEqual(public_doc_id("pdf", "A/B.pdf"), public_doc_id("pdf", "a/b.pdf"))

    def test_public_doc_id_separates_source_types(self):
        self.assertNotEqual(public_doc_id("pdf", "x.md"), public_doc_id("obsidian", "x.md"))

    def test_split_long_text_creates_multiple_chunks(self):
        parts = split_with_overlap("。".join(["知识点" * 40] * 8), max_units=100, overlap_units=10)
        self.assertGreater(len(parts), 1)

    def test_obsidian_heading_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "note.md"
            note.write_text("# 第一章\n稳定引用内容\n## 第二节\n另一个知识点", encoding="utf-8")
            chunks, document = extract_obsidian_source(note, root)
            self.assertEqual(document.source_type, "obsidian")
            self.assertTrue(any("heading=%E7%AC%AC%E4%B8%80%E7%AB%A0" in item.chunk_id for item in chunks))

    def test_obsidian_trove_export_removes_embedded_base64_and_reads_escaped_heading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            note = root / "trove.md"
            note.write_text("\\# Phase rule\n证据正文\n![x](data:image/png;base64,AAAA)", encoding="utf-8")
            chunks, _ = extract_obsidian_source(note, root)
            self.assertTrue(any("heading=Phase%20rule" in item.chunk_id for item in chunks))
            self.assertTrue(all("base64" not in item.text for item in chunks))

    def test_build_excludes_obsidian_internal_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            (vault / ".obsidian").mkdir(parents=True)
            (vault / ".obsidian" / "private.md").write_text("should not index", encoding="utf-8")
            (vault / "note.md").write_text("# Note\npublic knowledge", encoding="utf-8")
            config = root / "sources.toml"
            config.write_text(f'[[sources]]\ntype="obsidian"\nroot="{vault.as_posix()}"\n', encoding="utf-8")
            index, manifest = build_index_from_config(config)
            self.assertEqual(manifest["document_count"], 1)
            self.assertTrue(all("should not index" not in item.text for item in index.chunks))

    @unittest.skipUnless(importlib.util.find_spec("pymupdf"), "PyMuPDF optional dependency is not installed")
    def test_pdf_uses_physical_page_locator(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "tiny.pdf"
            document = pymupdf.open()
            for text in ["first page evidence", "second page evidence"]:
                page = document.new_page()
                page.insert_text((72, 72), text)
            document.save(pdf)
            document.close()
            chunks, metadata = extract_pdf_source(pdf, root)
            self.assertEqual(metadata.page_count, 2)
            self.assertTrue(any("pdf-page=2" in item.chunk_id for item in chunks))

    @unittest.skipUnless(importlib.util.find_spec("pymupdf"), "PyMuPDF optional dependency is not installed")
    def test_blank_pdf_page_is_marked_for_ocr(self):
        import pymupdf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "blank.pdf"
            document = pymupdf.open()
            document.new_page()
            document.save(pdf)
            document.close()
            chunks, metadata = extract_pdf_source(pdf, root)
            self.assertFalse(chunks)
            self.assertEqual(metadata.extraction_status, "ocr_required")
            self.assertEqual(metadata.ocr_required_pages, [1])

    def test_index_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "index.json"
            index = DocumentIndex([chunk("doc_demo#heading=demo&chunk=1", "稳定引用")])
            manifest = {"manifest_sha256": "abc", "schema_version": "3.0", "documents": []}
            save_index(index, manifest, path)
            loaded, loaded_manifest = load_index(path)
            self.assertEqual(loaded.chunks[0].chunk_id, index.chunks[0].chunk_id)
            self.assertEqual(loaded_manifest["manifest_sha256"], "abc")

    def test_index_rejects_old_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps({"schema_version": "2.0", "manifest": {}, "chunks": []}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported index schema"):
                load_index(path)

    def test_source_cache_removes_orphan_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            cache = root / "cache"
            vault.mkdir()
            cache.mkdir()
            (vault / "note.md").write_text("# Note\ncontent", encoding="utf-8")
            stale = cache / "stale.json"
            stale.write_text("{}", encoding="utf-8")
            config = root / "sources.toml"
            config.write_text(f'[[sources]]\ntype="obsidian"\nroot="{vault.as_posix()}"\n', encoding="utf-8")
            build_index_from_config(config, cache_dir=cache)
            self.assertFalse(stale.exists())


class RetrievalV3Tests(unittest.TestCase):
    def setUp(self):
        self.index = DocumentIndex(
            [
                chunk("doc_a#heading=alpha&chunk=1", "发布窗口要求周二，并且必须完成测试", doc_id="doc_a", locator="heading=alpha"),
                chunk("doc_b#heading=beta&chunk=1", "日志保存九十天", doc_id="doc_b", locator="heading=beta"),
                chunk("doc_c#heading=gamma&chunk=1", "完全不同的午餐菜单", doc_id="doc_c", locator="heading=gamma"),
            ]
        )

    def test_no_lexical_match_is_explicit(self):
        bundle = retrieve_candidates(self.index, "zzzabc123")
        self.assertEqual(bundle["retrieval"]["status"], "no_lexical_match")

    def test_candidate_bundle_has_schema(self):
        bundle = retrieve_candidates(self.index, "发布窗口测试")
        self.assertEqual(bundle["schema_version"], "3.0")
        self.assertTrue(bundle["candidates"])

    def test_candidate_limit_is_enforced(self):
        bundle = retrieve_candidates(self.index, "发布 日志", max_candidates=1)
        self.assertEqual(len(bundle["candidates"]), 1)

    def test_explicit_aspect_can_add_evidence(self):
        bundle = retrieve_candidates(self.index, "发布要求", aspects=["日志保存"], max_candidates=3)
        ids = {item["doc_id"] for item in bundle["candidates"]}
        self.assertIn("doc_b", ids)

    def test_token_budget_is_respected(self):
        bundle = retrieve_candidates(self.index, "发布 日志", max_evidence_tokens=20)
        self.assertLessEqual(bundle["retrieval"]["selected_evidence_tokens"], 40)

    def test_sanitize_excerpt_removes_sensitive_values(self):
        cleaned = sanitize_excerpt("C:\\Users\\name\\a.txt mail=a@example.com api_key=secret")
        self.assertIn("[LOCAL_PATH]", cleaned)
        self.assertIn("[EMAIL]", cleaned)
        self.assertIn("[REDACTED]", cleaned)

    def test_semantic_decision_accepts_candidate_subset(self):
        bundle = retrieve_candidates(self.index, "发布窗口")
        selected = bundle["candidates"][0]["chunk_id"]
        decision = validate_semantic_decision(bundle, {"status": "answerable", "selected_chunk_ids": [selected]})
        self.assertEqual(decision["selected_chunk_ids"], [selected])

    def test_semantic_decision_rejects_unknown_candidate(self):
        bundle = retrieve_candidates(self.index, "发布窗口")
        with self.assertRaisesRegex(ValueError, "invalid_semantic_selection"):
            validate_semantic_decision(bundle, {"status": "answerable", "selected_chunk_ids": ["unknown"]})

    def test_semantic_decision_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            validate_semantic_decision({"candidates": []}, {"status": "maybe", "selected_chunk_ids": []})


class EvaluationAndPrivacyTests(unittest.TestCase):
    def test_feynman_evaluation_checks_schema_status_and_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = root / "cases.json"
            predictions = root / "predictions.json"
            cases.write_text(json.dumps([{"id": "f1", "expected_status": "accurate", "expected_issue_types": [], "acceptable_citations": [{"doc_id": "d1", "locator": "p1"}]}]), encoding="utf-8")
            predictions.write_text(json.dumps([{"id": "f1", "status": "accurate", "issue_types": [], "accurate_parts": [], "omissions": [], "factual_conflicts": [], "logic_gaps": [], "questions": [], "reteach_request": "", "citations": [{"doc_id": "d1", "locator": "p1"}]}]), encoding="utf-8")
            report = evaluate_feynman_predictions(cases, predictions)
            self.assertEqual(report["summary"]["task_pass_rate"], 1.0)

    def test_case_manifest_mismatch_is_rejected(self):
        cases = [{"id": "x", "question": "q", "expected_status": "answerable", "gold_evidence_groups": [{"id": "g", "any_of": []}], "source_manifest_sha256": "old"}]
        with self.assertRaisesRegex(ValueError, "different source manifest"):
            validate_v3_cases(cases, "new")

    def test_evaluation_v3_scores_group_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.json"
            cases_path = root / "cases.json"
            report_path = root / "report.json"
            index = DocumentIndex([chunk("doc_a#heading=alpha&chunk=1", "发布窗口要求周二", doc_id="doc_a", locator="heading=alpha")])
            manifest = {"manifest_sha256": "m1", "schema_version": "3.0", "documents": []}
            save_index(index, manifest, index_path)
            cases_path.write_text(json.dumps([{"id": "c1", "question": "发布窗口", "expected_status": "answerable", "source_manifest_sha256": "m1", "gold_evidence_groups": [{"id": "window", "any_of": [{"doc_id": "doc_a", "locator": "heading=alpha"}]}]}]), encoding="utf-8")
            report = evaluate_v3(cases_path, index_path, report_path)
            self.assertEqual(report["summary"]["evidence_group_recall"], 1.0)

    def test_evaluation_v3_scores_semantic_selected_subset_not_all_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "index.json"
            cases_path = root / "cases.json"
            decisions_path = root / "decisions.json"
            report_path = root / "report.json"
            index = DocumentIndex([
                chunk("doc_a#heading=alpha&chunk=1", "发布窗口要求周二", doc_id="doc_a", locator="heading=alpha"),
                chunk("doc_b#heading=beta&chunk=1", "发布窗口只是一个无关标题", doc_id="doc_b", locator="heading=beta"),
            ])
            manifest = {"manifest_sha256": "m1", "schema_version": "3.0", "documents": []}
            save_index(index, manifest, index_path)
            cases_path.write_text(json.dumps([{"id": "c1", "question": "发布窗口", "expected_status": "answerable", "source_manifest_sha256": "m1", "gold_evidence_groups": [{"id": "window", "any_of": [{"doc_id": "doc_a", "locator": "heading=alpha"}]}]}]), encoding="utf-8")
            decisions_path.write_text(json.dumps({"decisions": {"c1": {"status": "answerable", "selected_chunk_ids": ["doc_a#heading=alpha&chunk=1"]}}}), encoding="utf-8")
            report = evaluate_v3(cases_path, index_path, report_path, semantic_decisions_path=decisions_path)
            self.assertEqual(report["summary"]["evidence_precision"], 1.0)
            self.assertEqual(report["summary"]["semantic_decision_coverage"], 1.0)

    def test_quality_gate_reports_failures(self):
        report = {"summary": {"citation_locator_valid_rate": 1.0, "candidate_group_recall": 1.0, "evidence_precision": 0.5, "evidence_group_recall": 1.0, "evidence_f1": 0.6, "aspect_evidence_coverage": 1.0, "hard_negative_accuracy": 1.0, "retrieval_task_pass_rate": 0.5, "semantic_decision_coverage": 1.0}}
        passed, failures = quality_gate(report)
        self.assertFalse(passed)
        self.assertTrue(failures)

    def test_privacy_scanner_detects_local_path(self):
        self.assertIn("windows_path", scan_sensitive_text("C:\\Users\\person\\file"))

    def test_public_safe_rejects_email(self):
        with self.assertRaises(ValueError):
            assert_public_safe({"email": "person@example.com"})

    def test_public_manifest_contains_aggregates_only(self):
        public = public_manifest({"schema_version": "3.0", "manifest_sha256": "x", "document_count": 1, "chunk_count": 2, "documents": [{"source_type": "pdf", "extraction_status": "ok", "relative_path": "/private"}]})
        self.assertNotIn("documents", public)
        self.assertEqual(public["documents_by_source_type"], {"pdf": 1})

    def test_sync_lock_prevents_parallel_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock"
            with SyncLock(path):
                with self.assertRaises(RuntimeError):
                    with SyncLock(path):
                        pass

    def test_sync_rejection_keeps_previous_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vault = root / "vault"
            vault.mkdir()
            (vault / "note.md").write_text("# Note\nnew content", encoding="utf-8")
            config = root / "sources.toml"
            config.write_text(f'[[sources]]\ntype="obsidian"\nroot="{vault.as_posix()}"\n', encoding="utf-8")
            target = root / "index.json"
            old = DocumentIndex([chunk("old#heading=x&chunk=1", "old")])
            save_index(old, {"manifest_sha256": "old", "schema_version": "3.0", "documents": []}, target)
            cases = root / "cases.json"
            cases.write_text(json.dumps([{"id": "c1", "question": "new", "expected_status": "answerable", "source_manifest_sha256": "different", "gold_evidence_groups": [{"id": "g", "any_of": [{"doc_id": "x", "locator": "y"}]}]}]), encoding="utf-8")
            result = sync_index(config_path=config, cases_path=cases, index_path=target, report_path=root / "report.json", cache_dir=root / "cache")
            self.assertEqual(result["status"], "rejected")
            _, manifest = load_index(target)
            self.assertEqual(manifest["manifest_sha256"], "old")


if __name__ == "__main__":
    unittest.main()
