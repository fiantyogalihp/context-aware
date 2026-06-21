"""Compatibility wrapper for the production hybrid retriever.

Older scripts/tests imported `src.vector_retriever`. Keep that surface stable
while the maintained implementation lives in `src.hybrid_retriever`.
"""
from src.hybrid_retriever import (  # noqa: F401
    HybridRetriever,
    chunk_search_text,
    expand_query,
    prune_context,
    reciprocal_rank_fusion,
    tokenize,
)
