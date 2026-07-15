from __future__ import annotations

"""Source connectors and stable citation metadata for local knowledge bases.

The public repository never needs the user's documents.  Connectors create a
local index whose identifiers are derived from source kind + relative path,
while citations retain PDF physical pages or Obsidian headings.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from .retrieval import Chunk, DocumentIndex, tokenize


INDEX_SCHEMA_VERSION = "3.0"
CONNECTOR_VERSION = "3.0.2"
DEFAULT_EXCLUDES = {".obsidian", ".git", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    source_type: str
    display_name: str
    relative_path: str
    content_sha256: str
    page_count: int | None
    chunk_count: int
    extraction_status: str
    ocr_required_pages: list[int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_doc_id(source_type: str, relative_path: str) -> str:
    normalized_path = relative_path.replace("\\", "/").lower()
    normalized = f"{source_type}:{normalized_path}"
    return "doc_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _rough_units(text: str) -> int:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z0-9_]+", text))
    return cjk + latin


def split_with_overlap(text: str, *, max_units: int = 520, overlap_units: int = 60) -> list[str]:
    """Split text without losing page/heading metadata.

    Paragraphs are preferred; very long paragraphs fall back to line/sentence
    pieces.  Overlap is added only between chunks from the same locator.
    """
    text = text.strip()
    if not text:
        return []
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
    if len(paragraphs) == 1 and _rough_units(paragraphs[0]) > max_units:
        paragraphs = [item.strip() for item in re.split(r"(?<=[。！？.!?])\s*|\n+", text) if item.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_units = 0
    for paragraph in paragraphs:
        units = _rough_units(paragraph)
        if current and current_units + units > max_units:
            chunks.append("\n\n".join(current))
            overlap: list[str] = []
            overlap_total = 0
            for previous in reversed(current):
                overlap.insert(0, previous)
                overlap_total += _rough_units(previous)
                if overlap_total >= overlap_units:
                    break
            current = overlap
            current_units = overlap_total
        current.append(paragraph)
        current_units += units
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk(
    *,
    doc_id: str,
    source_type: str,
    display_name: str,
    locator: str,
    chunk_number: int,
    text: str,
    metadata: dict,
) -> Chunk:
    chunk_id = f"{doc_id}#{locator}&chunk={chunk_number}"
    return Chunk(
        chunk_id=chunk_id,
        source=display_name,
        text=text,
        terms=None,
        doc_id=doc_id,
        locator=locator,
        source_type=source_type,
        metadata=metadata,
    )


def extract_pdf_source(path: Path, root: Path, *, source_priority: float = 1.0) -> tuple[list[Chunk], SourceDocument]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - exercised in dependency error paths
        raise RuntimeError("PDF indexing requires PyMuPDF; install rag-agent-harness[pdf]") from exc

    relative = path.relative_to(root).as_posix()
    doc_id = public_doc_id("pdf", relative)
    file_hash = sha256_file(path)
    chunks: list[Chunk] = []
    ocr_pages: list[int] = []
    with pymupdf.open(str(path)) as document:
        page_count = len(document)
        for page_index, page in enumerate(document, 1):
            text = page.get_text("text").strip()
            if not text:
                ocr_pages.append(page_index)
                continue
            for number, part in enumerate(split_with_overlap(text), 1):
                chunks.append(
                    _chunk(
                        doc_id=doc_id,
                        source_type="pdf",
                        display_name=path.stem,
                        locator=f"pdf-page={page_index}",
                        chunk_number=number,
                        text=part,
                        metadata={"pdf_page": page_index, "content_sha256": file_hash, "source_priority": source_priority},
                    )
                )
    status = "ocr_required" if ocr_pages and len(ocr_pages) == page_count else ("partial_text" if ocr_pages else "ok")
    return chunks, SourceDocument(
        doc_id=doc_id,
        source_type="pdf",
        display_name=path.stem,
        relative_path=relative,
        content_sha256=file_hash,
        page_count=page_count,
        chunk_count=len(chunks),
        extraction_status=status,
        ocr_required_pages=ocr_pages,
    )


def _excluded(path: Path, root: Path, extra: set[str]) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part.startswith(".") or part in DEFAULT_EXCLUDES or part in extra for part in relative_parts[:-1])


def _markdown_sections(text: str) -> Iterable[tuple[str, str]]:
    heading = "document"
    body: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\\?(#{1,6})\s+(.+?)\s*$", line)
        if match:
            if any(item.strip() for item in body):
                yield heading, "\n".join(body).strip()
            heading = match.group(2).strip().replace("\\-", "-")
            body = []
        else:
            body.append(line)
    if any(item.strip() for item in body):
        yield heading, "\n".join(body).strip()


def _strip_frontmatter(text: str) -> str:
    return re.sub(r"\A---\s*\n.*?\n---\s*(?:\n|\Z)", "", text, count=1, flags=re.DOTALL)


def _clean_markdown(text: str) -> str:
    text = _strip_frontmatter(text)
    text = re.sub(r"!\[[^\]]*\]\(data:image/[^)]*\)", "[embedded image omitted]", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def extract_obsidian_source(path: Path, root: Path, *, source_priority: float = 0.9) -> tuple[list[Chunk], SourceDocument]:
    relative = path.relative_to(root).as_posix()
    doc_id = public_doc_id("obsidian", relative)
    text = _clean_markdown(path.read_text(encoding="utf-8"))
    file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    chunks: list[Chunk] = []
    for heading, section in _markdown_sections(text):
        locator = "heading=" + quote(heading, safe="")
        for number, part in enumerate(split_with_overlap(section), 1):
            chunks.append(
                _chunk(
                    doc_id=doc_id,
                    source_type="obsidian",
                    display_name=path.stem,
                    locator=locator,
                    chunk_number=number,
                    text=part,
                    metadata={"heading": heading, "content_sha256": file_hash, "source_priority": source_priority},
                )
            )
    return chunks, SourceDocument(
        doc_id=doc_id,
        source_type="obsidian",
        display_name=path.stem,
        relative_path=relative,
        content_sha256=file_hash,
        page_count=None,
        chunk_count=len(chunks),
        extraction_status="ok" if chunks else "empty",
        ocr_required_pages=[],
    )


def read_source_config(path: str | Path) -> dict:
    import tomllib

    config_path = Path(path)
    with config_path.open("rb") as stream:
        data = tomllib.load(stream)
    sources = data.get("sources", [])
    if not sources:
        raise ValueError("source config must define at least one [[sources]] entry")
    return data


def _cache_key(source_type: str, relative: str, content_hash: str, source_priority: float) -> str:
    raw = f"{CONNECTOR_VERSION}:{source_type}:{relative}:{content_hash}:{source_priority}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_cached_source(path: Path) -> tuple[list[Chunk], SourceDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        Chunk(
            chunk_id=item["chunk_id"], source=item["source"], text=item["text"], terms=None,
            doc_id=item["doc_id"], locator=item["locator"], source_type=item["source_type"], metadata=item["metadata"]
        )
        for item in data["chunks"]
    ]
    return chunks, SourceDocument(**data["document"])


def _save_cached_source(path: Path, chunks: list[Chunk], document: SourceDocument) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "document": asdict(document),
        "chunks": [
            {
                "chunk_id": item.chunk_id, "source": item.source, "text": item.text,
                "doc_id": item.doc_id, "locator": item.locator, "source_type": item.source_type,
                "metadata": item.metadata,
            }
            for item in chunks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_index_from_config(config_path: str | Path, *, cache_dir: str | Path | None = None) -> tuple[DocumentIndex, dict]:
    config = read_source_config(config_path)
    all_chunks: list[Chunk] = []
    documents: list[SourceDocument] = []
    cache_root = Path(cache_dir) if cache_dir else None
    active_cache_files: set[Path] = set()
    for source in config["sources"]:
        source_type = source.get("type")
        root = Path(source["root"]).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"configured source does not exist: {root}")
        if source_type == "pdf":
            source_priority = float(source.get("priority", 1.0))
            for path in sorted(root.rglob("*.pdf")):
                relative = path.relative_to(root).as_posix()
                content_hash = sha256_file(path)
                cache_path = cache_root / f"{_cache_key('pdf', relative, content_hash, source_priority)}.json" if cache_root else None
                if cache_path and cache_path.exists():
                    chunks, document = _load_cached_source(cache_path)
                else:
                    chunks, document = extract_pdf_source(path, root, source_priority=source_priority)
                    if cache_path:
                        _save_cached_source(cache_path, chunks, document)
                if cache_path:
                    active_cache_files.add(cache_path)
                all_chunks.extend(chunks)
                documents.append(document)
        elif source_type == "obsidian":
            source_priority = float(source.get("priority", 0.9))
            extra = set(source.get("exclude_directories", []))
            for path in sorted(root.rglob("*.md")):
                if _excluded(path, root, extra):
                    continue
                relative = path.relative_to(root).as_posix()
                content_hash = sha256_file(path)
                cache_path = cache_root / f"{_cache_key('obsidian', relative, content_hash, source_priority)}.json" if cache_root else None
                if cache_path and cache_path.exists():
                    chunks, document = _load_cached_source(cache_path)
                else:
                    chunks, document = extract_obsidian_source(path, root, source_priority=source_priority)
                    if cache_path:
                        _save_cached_source(cache_path, chunks, document)
                if cache_path:
                    active_cache_files.add(cache_path)
                all_chunks.extend(chunks)
                documents.append(document)
        else:
            raise ValueError(f"unsupported source type: {source_type!r}")
    if not all_chunks:
        raise ValueError("configured sources produced no indexable chunks")
    if cache_root and cache_root.exists():
        for stale in cache_root.glob("*.json"):
            if stale not in active_cache_files:
                stale.unlink()

    manifest_core = [
        {
            "doc_id": item.doc_id,
            "source_type": item.source_type,
            "content_sha256": item.content_sha256,
            "page_count": item.page_count,
            "chunk_count": item.chunk_count,
            "extraction_status": item.extraction_status,
        }
        for item in sorted(documents, key=lambda value: value.doc_id)
    ]
    manifest_hash = hashlib.sha256(json.dumps(manifest_core, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "document_count": len(documents),
        "chunk_count": len(all_chunks),
        "documents": [asdict(item) for item in documents],
    }
    return DocumentIndex(all_chunks), manifest


def save_index(index: DocumentIndex, manifest: dict, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "manifest": manifest,
        "chunks": [
            {
                "chunk_id": item.chunk_id,
                "source": item.source,
                "text": item.text,
                "doc_id": item.doc_id,
                "locator": item.locator,
                "source_type": item.source_type,
                "metadata": item.metadata,
            }
            for item in index.chunks
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return target


def load_index(path: str | Path) -> tuple[DocumentIndex, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(f"unsupported index schema: {payload.get('schema_version')!r}")
    chunks = [
        Chunk(
            chunk_id=item["chunk_id"],
            source=item["source"],
            text=item["text"],
            terms=None,
            doc_id=item.get("doc_id", ""),
            locator=item.get("locator", ""),
            source_type=item.get("source_type", "text"),
            metadata=item.get("metadata", {}),
        )
        for item in payload["chunks"]
    ]
    return DocumentIndex(chunks), payload["manifest"]
