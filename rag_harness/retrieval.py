from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ASCII_TOKEN = re.compile(r"[A-Za-z0-9_+-]+")
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = ASCII_TOKEN.findall(text)
    for run in CJK_RUN.findall(text):
        tokens.extend(run[i : i + 2] for i in range(max(1, len(run) - 1)))
        if len(run) == 1:
            tokens.append(run)
    return tokens


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
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.document_frequency = Counter()
        for chunk in chunks:
            self.document_frequency.update(set(chunk.terms))

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
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        query_terms = Counter(tokenize(query))
        scored: list[SearchResult] = []
        total = len(self.chunks)
        for chunk in self.chunks:
            score = 0.0
            for term, qtf in query_terms.items():
                tf = chunk.terms.get(term, 0)
                if not tf:
                    continue
                idf = math.log((total + 1) / (self.document_frequency[term] + 1)) + 1
                score += (1 + math.log(tf)) * idf * qtf
            if score > 0:
                scored.append(SearchResult(chunk.chunk_id, chunk.source, chunk.text, round(score, 6)))
        scored.sort(key=lambda item: (-item.score, item.chunk_id))
        return scored[:top_k]

