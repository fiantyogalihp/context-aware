"""Intent-based query router for RAG system.

Routes queries to appropriate processing paths based on detected intent:
- Computational/Analytic: Queries requiring aggregation, comparison, or computation
- Factual Retrieval: Queries seeking procedural or descriptive information
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class QueryIntent(Enum):
    """Query intent types for routing."""
    COMPUTATIONAL = "computational"  # Aggregation, comparison, computation
    FACTUAL = "factual"              # Procedural, descriptive information
    UNKNOWN = "unknown"              # Cannot determine intent


@dataclass
class QueryAnalysis:
    """Result of query intent analysis."""
    intent: QueryIntent
    confidence: float
    matched_patterns: List[str]
    suggested_chunk_types: List[str]  # ["table"] or ["text"] or ["text", "table"]


class QueryRouter:
    """Deterministic query router using regex and keyword matching."""
    
    # Computational/Analytic patterns (pattern, confidence_weight)
    COMPUTATIONAL_PATTERNS = [
        (r'\b(berapa|jumlah|total|sum|count)\b', 0.9),
        (r'\b(rata-rata|average|mean|median)\b', 0.9),
        (r'\b(bandingkan|compare|perbandingan|comparison|versus|vs)\b', 0.85),
        (r'\b(tertinggi|terendah|maksimal|minimal|terbanyak|paling)\b', 0.8),
        (r'\b(hitung|calculate|kalkulasi|compute)\b', 0.9),
        (r'\b(statistik|data|angka|persentase|persen|%)\b', 0.7),
        (r'\b(lebih banyak|lebih sedikit|lebih tinggi|lebih rendah)\b', 0.75),
        (r'\b(selisih|difference|perbedaan)\b', 0.8),
        (r'\b(ranking|urutan|urutkan|peringkat)\b', 0.75),
    ]
    
    # Factual/Procedural patterns (pattern, confidence_weight)
    FACTUAL_PATTERNS = [
        (r'\b(apa|what|apakah)\b', 0.8),
        (r'\b(bagaimana cara|how to|cara|langkah|step)\b', 0.9),
        (r'\b(syarat|requirement|persyaratan|kondisi|requirements)\b', 0.85),
        (r'\b(prosedur|procedure|proses|alur)\b', 0.85),
        (r'\b(kapan|when|waktu|jadwal)\b', 0.8),
        (r'\b(dimana|where|lokasi|tempat)\b', 0.8),
        (r'\b(siapa|who|pihak)\b', 0.75),
        (r'\b(mengapa|why|alasan|kenapa)\b', 0.75),
        (r'\b(definisi|pengertian|arti|maksud|adalah)\b', 0.85),
        (r'\b(fungsi|kegunaan|manfaat|tujuan)\b', 0.8),
    ]
    
    def __init__(self, *, computational_threshold: float = 0.6, factual_threshold: float = 0.5):
        """
        Initialize query router.
        
        Args:
            computational_threshold: Minimum confidence for computational intent
            factual_threshold: Minimum confidence for factual intent
        """
        self.computational_threshold = computational_threshold
        self.factual_threshold = factual_threshold
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze query intent using deterministic pattern matching.
        
        Args:
            query: User query string
            
        Returns:
            QueryAnalysis with detected intent and metadata
        """
        query_lower = query.lower()
        
        # Check computational patterns
        comp_score = 0.0
        comp_matches = []
        for pattern, weight in self.COMPUTATIONAL_PATTERNS:
            if re.search(pattern, query_lower):
                comp_score = max(comp_score, weight)
                comp_matches.append(pattern)
        
        # Check factual patterns
        fact_score = 0.0
        fact_matches = []
        for pattern, weight in self.FACTUAL_PATTERNS:
            if re.search(pattern, query_lower):
                fact_score = max(fact_score, weight)
                fact_matches.append(pattern)
        
        # Determine intent based on scores
        if comp_score >= self.computational_threshold and comp_score > fact_score:
            return QueryAnalysis(
                intent=QueryIntent.COMPUTATIONAL,
                confidence=comp_score,
                matched_patterns=comp_matches,
                suggested_chunk_types=["table"],  # Prioritize tables for computation
            )
        elif fact_score >= self.factual_threshold:
            return QueryAnalysis(
                intent=QueryIntent.FACTUAL,
                confidence=fact_score,
                matched_patterns=fact_matches,
                suggested_chunk_types=["text", "table"],  # Both types acceptable
            )
        else:
            return QueryAnalysis(
                intent=QueryIntent.UNKNOWN,
                confidence=max(comp_score, fact_score),
                matched_patterns=comp_matches + fact_matches,
                suggested_chunk_types=["text", "table"],  # Default to both
            )
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Route query to appropriate processing path.
        
        Args:
            query: User query string
            
        Returns:
            Dict with routing information:
            {
                "intent": str,  # "computational", "factual", or "unknown"
                "confidence": float,
                "processing_path": str,  # "computational" or "retrieval"
                "filter_chunk_types": List[str],
                "matched_patterns": List[str]
            }
        """
        analysis = self.analyze_query(query)
        
        if analysis.intent == QueryIntent.COMPUTATIONAL:
            processing_path = "computational"
        else:
            processing_path = "retrieval"
        
        return {
            "intent": analysis.intent.value,
            "confidence": analysis.confidence,
            "processing_path": processing_path,
            "filter_chunk_types": analysis.suggested_chunk_types,
            "matched_patterns": analysis.matched_patterns,
        }


# Example usage and testing
if __name__ == "__main__":
    router = QueryRouter()
    
    # Test computational queries
    comp_queries = [
        "Berapa total vaksin yang harus diberikan pada bayi?",
        "Bandingkan jumlah imunisasi wajib dan opsional",
        "Hitung rata-rata biaya persalinan di faskes",
        "Apa perbedaan antara BPJS PBI dan Non-PBI?",
        "Berapa persentase cakupan imunisasi di Indonesia?",
    ]
    
    # Test factual queries
    fact_queries = [
        "Apa syarat pendaftaran BPJS?",
        "Bagaimana cara mengubah faskes di Mobile JKN?",
        "Kapan jadwal imunisasi BCG?",
        "Dimana lokasi faskes terdekat?",
        "Apa fungsi Buku KIA?",
    ]
    
    print("=" * 60)
    print("COMPUTATIONAL QUERIES")
    print("=" * 60)
    for q in comp_queries:
        result = router.route_query(q)
        print(f"\nQuery: {q}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Path: {result['processing_path']}")
        print(f"  Chunk Types: {result['filter_chunk_types']}")
    
    print("\n" + "=" * 60)
    print("FACTUAL QUERIES")
    print("=" * 60)
    for q in fact_queries:
        result = router.route_query(q)
        print(f"\nQuery: {q}")
        print(f"  Intent: {result['intent']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Path: {result['processing_path']}")
        print(f"  Chunk Types: {result['filter_chunk_types']}")
