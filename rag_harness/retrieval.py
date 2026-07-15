from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ASCII_TOKEN = re.compile(r"[A-Za-z0-9_+-]+")
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
QUESTION_SUFFIX = re.compile(r"(?:分别)?(?:是|为)?什么(?:要求)?$|(?:有|是)?哪些$|怎么处理$")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = ASCII_TOKEN.findall(text)
    for run in CJK_RUN.findall(text):
        tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        if len(run) == 1:
            tokens.append(run)
    return tokens


def split_query_aspects(query: str) -> list[str]:
    """Extract explicit sub-questions from a compound Chinese query.

    This is intentionally conservative: a normal one-part question returns no
    aspects. Commas create clauses, while list separators are expanded only
    when the question explicitly asks for separate answers.
    """
    cleaned = query.strip().rstrip("?？。")
    clauses = [part.strip() for part in re.split(r"[，,；;]", cleaned) if part.strip()]
    aspects: list[str] = []
    for clause in clauses:
        should_expand = "、" in clause or "分别" in clause
        parts = re.split(r"、|以及|并且|和", clause) if should_expand else [clause]
        for part in parts:
            part = QUESTION_SUFFIX.sub("", part.strip()).strip()
            if len(tokenize(part)) >= 2 and part not in aspects:
                aspects.append(part)
        for event in re.findall(r"(?:发生|检测到)(.+?)(?:时|后)", clause):
            event = event.strip()
            if tokenize(event) and event not in aspects:
                aspects.append(event)
    if len(aspects) <= 1 or aspects == [cleaned]:
        return []
    return aspects


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    terms: Counter[str]


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    source: str
    text: str
    score: float


class DocumentIndex:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.document_frequency = Counter()
        for chunk in chunks:
            self.document_frequency.update(set(chunk.terms))
        self.average_document_length = sum(sum(chunk.terms.values()) for chunk in chunks) / len(chunks)

    @classmethod
    def from_directory(cls, directory: str | Path) -> "DocumentIndex":
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"knowledge directory does not exist: {root}")
        chunks: list[Chunk] = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            for idx, paragraph in enumerate(paragraphs, 1):
                relative = path.relative_to(root).as_posix()
                chunks.append(
                    Chunk(
                        chunk_id=f"{relative}#p{idx}",
                        source=relative,
                        text=paragraph,
                        terms=Counter(tokenize(paragraph)),
                    )
                )
        if not chunks:
            raise ValueError(f"no .md or .txt documents found in {root}")
        return cls(chunks)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        query_terms = Counter(tokenize(query))
        scored: list[SearchResult] = []
        total = len(self.chunks)
        for chunk in self.chunks:
            score = 0.0
            document_length = sum(chunk.terms.values())
            length_norm = 1 - self.b + self.b * document_length / self.average_document_length
            for term, qtf in query_terms.items():
                tf = chunk.terms.get(term, 0)
                if not tf:
                    continue
                document_frequency = self.document_frequency[term]
                idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                tf_weight = tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
                score += idf * tf_weight * qtf
            if score > 0:
                scored.append(SearchResult(chunk.chunk_id, chunk.source, chunk.text, round(score, 6)))
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:top_k]
