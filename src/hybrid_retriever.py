"""CPU-local hybrid BM25 + dense vector retriever with RRF fusion."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from src.parser import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_HOME = ROOT / ".cache" / "huggingface"
DEFAULT_ST_HOME = ROOT / ".cache" / "sentence-transformers"
DEFAULT_EMBEDDING_CACHE = ROOT / ".cache" / "retriever" / "official_health_embeddings.npy"
DEFAULT_META_CACHE = ROOT / ".cache" / "retriever" / "official_health_embeddings.meta.json"

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(DEFAULT_ST_HOME))

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "ini", "itu", "dengan", "atau",
    "sebagai", "dalam", "akan", "dapat", "bisa", "harus", "adalah", "ialah", "oleh",
}

QUERY_EXPANSIONS = (
    (r"\bfktp\b", "fktp fasilitas kesehatan tingkat pertama faskes 1 faskes"),
    (r"\bfaskes\b", "faskes fasilitas kesehatan"),
    (r"\bubah\b", "ubah mengubah perubahan ganti mengganti"),
    (r"\bmengubah\b", "mengubah ubah perubahan ganti mengganti"),
    (r"\bfornas\b", "fornas formularium nasional obat keputusan menteri kesehatan"),
)

NOISE_LINE_RE = re.compile(r"^[\W_]+$")
REPEATED_SPACE_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    tokens = re.sub(r"[^a-zA-Z0-9À-ÿ]+", " ", text.lower()).split()
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def chunk_search_text(chunk: Dict) -> str:
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return " ".join(str(value or "") for value in (
        chunk.get("source"),
        chunk.get("title"),
        chunk.get("section") or meta.get("section"),
        chunk.get("subsection") or meta.get("subsection"),
        chunk.get("content"),
        chunk.get("text"),
    ))


def expand_query(query: str) -> str:
    additions = []
    lowered = query.lower()
    for pattern, replacement in QUERY_EXPANSIONS:
        if re.search(pattern, lowered):
            additions.append(replacement)
    return f"{query} {' '.join(additions)}" if additions else query


def reciprocal_rank_fusion(rank_maps: Iterable[Dict[str, int]], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for ranks in rank_maps:
        for chunk_id, rank in ranks.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def approx_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text))


def clean_context_line(line: str) -> str:
    line = REPEATED_SPACE_RE.sub(" ", line).strip()
    if not line or len(line) < 3 or NOISE_LINE_RE.fullmatch(line):
        return ""
    return line


def prune_context(chunks: Sequence[Dict], *, max_tokens: int = 1200, max_chars: int | None = None) -> List[Dict]:
    """Deduplicate layout noise while preserving numeric policy/table facts."""
    pruned: List[Dict] = []
    seen_lines: set[str] = set()
    used_tokens = 0
    char_budget = max_chars if max_chars is not None else max_tokens * 5

    for chunk in chunks:
        text = str(chunk.get("content") or chunk.get("text") or "")
        kept_lines = []
        for raw_line in text.splitlines():
            line = clean_context_line(raw_line)
            if not line:
                continue
            key = line.lower()
            if key in seen_lines:
                continue
            seen_lines.add(key)
            kept_lines.append(line)

        clean = "\n".join(kept_lines)
        if not clean:
            continue

        remaining_tokens = max_tokens - used_tokens
        remaining_chars = char_budget - sum(len(str(c.get("content") or c.get("text") or "")) for c in pruned)
        if remaining_tokens <= 0 or remaining_chars <= 0:
            break

        words = clean.split()
        if len(words) > remaining_tokens:
            clean = " ".join(words[:remaining_tokens])
        if len(clean) > remaining_chars:
            clean = clean[:remaining_chars]
        used_tokens += approx_tokens(clean)
        pruned.append({**chunk, "text": clean, "content": clean})

    return pruned


class HybridRetriever:
    """Hybrid retriever using BM25 ranks, dense cosine ranks, and RRF merging."""

    def __init__(
        self,
        chunks: Sequence[Dict],
        *,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_cache_path: str | Path = DEFAULT_EMBEDDING_CACHE,
        meta_cache_path: str | Path = DEFAULT_META_CACHE,
        qdrant_path: str | Path | None = None,
        collection_name: str | None = None,
        local_files_only: bool | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.model_name = model_name
        self.embedding_cache_path = Path(embedding_cache_path)
        self.meta_cache_path = Path(meta_cache_path)
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name
        if local_files_only is None:
            local_only_env = os.environ.get("RAG_RETRIEVER_LOCAL_ONLY", "1").lower()
            local_files_only = local_only_env not in {"0", "false", "no"}
        self.local_files_only = local_files_only
        self._embeddings = None
        self._load_dependencies()
        self.search_texts = [chunk_search_text(chunk) for chunk in self.chunks]
        self.corpus_tokens = [tokenize(text) for text in self.search_texts]
        self.bm25 = self.BM25Okapi(self.corpus_tokens)
        self.model = self.SentenceTransformer(model_name, local_files_only=self.local_files_only)

    def _load_dependencies(self) -> None:
        try:
            import numpy as np
            from rank_bm25 import BM25Okapi
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "Hybrid retrieval requires numpy, rank_bm25, and sentence-transformers. "
                "Install requirements.txt and cache the embedding model for offline runs."
            ) from exc
        self.np = np
        self.BM25Okapi = BM25Okapi
        self.SentenceTransformer = SentenceTransformer

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "HybridRetriever":
        return cls(read_jsonl(Path(path)), **kwargs)

    def corpus_digest(self) -> str:
        payload = [{"chunk_id": c.get("chunk_id"), "text": t} for c, t in zip(self.chunks, self.search_texts)]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]):
        return self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)

    def cache_meta(self) -> Dict:
        return {
            "model_name": self.model_name,
            "digest": self.corpus_digest(),
            "chunk_ids": [chunk["chunk_id"] for chunk in self.chunks],
        }

    def load_cached_embeddings(self):
        if not self.embedding_cache_path.exists() or not self.meta_cache_path.exists():
            return None
        try:
            meta = json.loads(self.meta_cache_path.read_text(encoding="utf-8"))
            expected = self.cache_meta()
            if meta != expected:
                return None
            embeddings = self.np.load(self.embedding_cache_path)
            if len(embeddings) != len(self.chunks):
                return None
            return embeddings
        except Exception:
            return None

    def build_index(self, *, recreate: bool = False, batch_size: int = 64) -> None:
        if not recreate:
            cached = self.load_cached_embeddings()
            if cached is not None:
                self._embeddings = cached
                return
        vectors = []
        for start in range(0, len(self.search_texts), batch_size):
            vectors.extend(self.encode(self.search_texts[start:start + batch_size]))
        self._embeddings = self.np.asarray(vectors, dtype="float32")
        self.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.np.save(self.embedding_cache_path, self._embeddings)
        self.meta_cache_path.write_text(json.dumps(self.cache_meta(), ensure_ascii=False, indent=2), encoding="utf-8")

    def ensure_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        cached = self.load_cached_embeddings()
        if cached is not None:
            self._embeddings = cached
            return self._embeddings
        self.build_index(recreate=True)
        return self._embeddings

    def bm25_search(self, query: str, top_k: int = 5, *, expand: bool = True) -> List[Dict]:
        """Return lexical BM25 matches with exact-keyword-friendly scores."""
        search_query = expand_query(query) if expand else query
        scores = self.bm25.get_scores(tokenize(search_query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for rank, (index, score) in enumerate(ranked, start=1):
            if score <= 0:
                continue
            chunk = self.chunks[index]
            results.append({**chunk, "bm25_rank": rank, "bm25_score": float(score)})
        return results

    def bm25_ranks(self, query: str, limit: int) -> Dict[str, int]:
        return {row["chunk_id"]: row["bm25_rank"] for row in self.bm25_search(query, limit, expand=False)}

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Return dense semantic matches using cosine similarity on normalized embeddings."""
        embeddings = self.ensure_embeddings()
        query_vector = self.np.asarray(self.encode([query])[0], dtype="float32")
        scores = embeddings @ query_vector
        ranked_indexes = scores.argsort()[::-1][:top_k]
        results = []
        for rank, index in enumerate(ranked_indexes, start=1):
            chunk = self.chunks[int(index)]
            results.append({
                **chunk,
                "vector_rank": rank,
                "semantic_score": float(scores[int(index)]),
            })
        return results

    def vector_ranks(self, query: str, limit: int) -> Dict[str, int]:
        return {row["chunk_id"]: row["vector_rank"] for row in self.semantic_search(query, limit)}

    def retrieve(
        self, 
        query: str, 
        top_k: int = 4, 
        *, 
        candidate_k: int = 20, 
        prune: bool = True,
        filter_chunk_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks with optional chunk type filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            candidate_k: Number of candidates for RRF fusion
            prune: Whether to apply context pruning
            filter_chunk_types: Optional list of chunk types to filter (e.g., ["table"], ["text"])
        
        Returns:
            List of retrieved chunks with ranking metadata
        """
        expanded_query = expand_query(query)
        bm25 = self.bm25_ranks(expanded_query, candidate_k)
        vector = self.vector_ranks(expanded_query, candidate_k)
        fused = reciprocal_rank_fusion([bm25, vector])
        ranked_ids = sorted(fused, key=fused.get, reverse=True)
        
        results = []
        for chunk_id in ranked_ids:
            if len(results) >= top_k:
                break
                
            chunk = self.by_id[chunk_id]
            
            # Filter by chunk type if specified
            if filter_chunk_types is not None:
                chunk_type = chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type")
                if chunk_type not in filter_chunk_types:
                    continue
            
            results.append({
                **chunk,
                "bm25_rank": bm25.get(chunk_id),
                "vector_rank": vector.get(chunk_id),
                "rrf_score": round(fused[chunk_id], 6),
            })
        
        return prune_context(results) if prune else results
