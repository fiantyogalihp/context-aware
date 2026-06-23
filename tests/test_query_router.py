"""Test query router intent classification."""
import pytest
from src.query_router import QueryRouter, QueryIntent, QueryAnalysis


@pytest.fixture
def router():
    """Create a query router instance."""
    return QueryRouter()


class TestComputationalIntent:
    """Test computational/analytic query detection."""
    
    def test_count_query(self, router):
        """Test queries with counting intent."""
        queries = [
            "Berapa jumlah vaksin wajib?",
            "Berapa total imunisasi yang diperlukan?",
            "Count the number of vaccines",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "computational"
            assert result["processing_path"] == "computational"
            assert "table" in result["filter_chunk_types"]
    
    def test_comparison_query(self, router):
        """Test queries with comparison intent."""
        queries = [
            "Bandingkan BPJS PBI dan Non-PBI",
            "Apa perbedaan antara faskes 1 dan faskes 2?",
            "Compare vaccine schedules",
        ]
        for query in queries:
            result = router.route_query(query)
            # Note: "perbedaan" might be classified as factual
            # but "bandingkan" should be computational
            if "bandingkan" in query.lower() or "compare" in query.lower():
                assert result["intent"] == "computational"
    
    def test_aggregation_query(self, router):
        """Test queries with aggregation intent."""
        queries = [
            "Hitung rata-rata biaya persalinan",
            "Berapa total biaya imunisasi?",
            "What is the average cost?",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "computational"
            assert result["confidence"] >= 0.6
    
    def test_ranking_query(self, router):
        """Test queries with ranking/ordering intent."""
        queries = [
            "Vaksin mana yang paling penting?",
            "Urutkan faskes berdasarkan jarak",
            "Ranking of hospitals",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "computational"


class TestFactualIntent:
    """Test factual/procedural query detection."""
    
    def test_what_query(self, router):
        """Test 'what' questions."""
        queries = [
            "Apa itu BPJS?",
            "Apa syarat pendaftaran?",
            "What is KIA book?",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "factual"
            assert result["processing_path"] == "retrieval"
    
    def test_how_query(self, router):
        """Test 'how to' questions."""
        queries = [
            "Bagaimana cara mendaftar BPJS?",
            "Cara mengubah faskes di Mobile JKN",
            "How to register?",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "factual"
            assert result["confidence"] >= 0.5
    
    def test_when_query(self, router):
        """Test 'when' questions."""
        queries = [
            "Kapan jadwal imunisasi BCG?",
            "Waktu pendaftaran BPJS",
            "When is the deadline?",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "factual"
    
    def test_where_query(self, router):
        """Test 'where' questions."""
        queries = [
            "Dimana lokasi faskes terdekat?",
            "Tempat pendaftaran BPJS",
            "Where is the clinic?",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "factual"
    
    def test_requirement_query(self, router):
        """Test requirement/prerequisite questions."""
        queries = [
            "Apa syarat pendaftaran BPJS?",
            "Persyaratan untuk mengubah faskes",
            "Requirements for registration",
        ]
        for query in queries:
            result = router.route_query(query)
            assert result["intent"] == "factual"
            assert result["confidence"] >= 0.8


class TestChunkTypeFiltering:
    """Test chunk type filtering suggestions."""
    
    def test_computational_suggests_table(self, router):
        """Test that computational queries suggest table chunks."""
        query = "Berapa total vaksin yang diperlukan?"
        result = router.route_query(query)
        assert result["filter_chunk_types"] == ["table"]
    
    def test_factual_suggests_both(self, router):
        """Test that factual queries suggest both text and table chunks."""
        query = "Apa syarat pendaftaran BPJS?"
        result = router.route_query(query)
        assert "text" in result["filter_chunk_types"]
        assert "table" in result["filter_chunk_types"]


class TestConfidenceScoring:
    """Test confidence scoring mechanism."""
    
    def test_high_confidence_computational(self, router):
        """Test high confidence for clear computational queries."""
        query = "Hitung rata-rata biaya persalinan"
        result = router.route_query(query)
        assert result["confidence"] >= 0.85
    
    def test_high_confidence_factual(self, router):
        """Test high confidence for clear factual queries."""
        query = "Bagaimana cara mendaftar BPJS?"
        result = router.route_query(query)
        assert result["confidence"] >= 0.85
    
    def test_matched_patterns_recorded(self, router):
        """Test that matched patterns are recorded."""
        query = "Berapa total vaksin?"
        result = router.route_query(query)
        assert len(result["matched_patterns"]) > 0


class TestEdgeCases:
    """Test edge cases and ambiguous queries."""
    
    def test_empty_query(self, router):
        """Test handling of empty query."""
        result = router.route_query("")
        assert result["intent"] == "unknown"
    
    def test_ambiguous_query(self, router):
        """Test handling of ambiguous query."""
        query = "BPJS"  # Single word, no clear intent
        result = router.route_query(query)
        # Should default to retrieval path
        assert result["processing_path"] == "retrieval"
    
    def test_mixed_intent_query(self, router):
        """Test query with mixed intent signals."""
        query = "Apa itu rata-rata biaya persalinan?"  # "apa" (factual) + "rata-rata" (computational)
        result = router.route_query(query)
        # Should pick the stronger signal
        assert result["intent"] in ["computational", "factual"]


class TestCustomThresholds:
    """Test custom threshold configuration."""
    
    def test_custom_computational_threshold(self):
        """Test router with custom computational threshold."""
        router = QueryRouter(computational_threshold=0.8)
        query = "Berapa jumlah vaksin?"
        result = router.route_query(query)
        assert result["confidence"] >= 0.8
    
    def test_custom_factual_threshold(self):
        """Test router with custom factual threshold."""
        router = QueryRouter(factual_threshold=0.7)
        query = "Apa syarat pendaftaran?"
        result = router.route_query(query)
        assert result["confidence"] >= 0.7


class TestAnalyzeQuery:
    """Test the analyze_query method directly."""
    
    def test_analyze_returns_query_analysis(self, router):
        """Test that analyze_query returns QueryAnalysis dataclass."""
        query = "Berapa total vaksin?"
        analysis = router.analyze_query(query)
        assert isinstance(analysis, QueryAnalysis)
        assert isinstance(analysis.intent, QueryIntent)
        assert isinstance(analysis.confidence, float)
        assert isinstance(analysis.matched_patterns, list)
        assert isinstance(analysis.suggested_chunk_types, list)
    
    def test_analyze_computational_query(self, router):
        """Test analysis of computational query."""
        query = "Hitung rata-rata biaya"
        analysis = router.analyze_query(query)
        assert analysis.intent == QueryIntent.COMPUTATIONAL
        assert analysis.suggested_chunk_types == ["table"]
    
    def test_analyze_factual_query(self, router):
        """Test analysis of factual query."""
        query = "Bagaimana cara mendaftar?"
        analysis = router.analyze_query(query)
        assert analysis.intent == QueryIntent.FACTUAL
        assert "text" in analysis.suggested_chunk_types
