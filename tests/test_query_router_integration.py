"""Test integration between query router and hybrid retriever."""
import pytest
from src.query_router import QueryRouter
from src.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_chunks():
    """Create sample chunks with different types."""
    return [
        {
            "chunk_id": "TEXT_001",
            "chunk_type": "text",
            "content": "Cara mendaftar BPJS adalah dengan mengisi formulir pendaftaran.",
            "title": "Panduan BPJS",
            "section": "Pendaftaran",
        },
        {
            "chunk_id": "TABLE_001",
            "chunk_type": "table",
            "content": "| Vaksin | Usia | Dosis |\n| BCG | 0-1 bulan | 0.05ml |",
            "title": "Jadwal Imunisasi",
            "section": "Tabel Vaksin",
        },
        {
            "chunk_id": "TEXT_002",
            "chunk_type": "text",
            "content": "Imunisasi BCG diberikan pada bayi usia 0-1 bulan.",
            "title": "Panduan Imunisasi",
            "section": "BCG",
        },
        {
            "chunk_id": "TABLE_002",
            "chunk_type": "table",
            "content": "| Faskes | Jumlah Pasien |\n| Puskesmas A | 100 |\n| Puskesmas B | 150 |",
            "title": "Data Faskes",
            "section": "Statistik",
        },
    ]


@pytest.fixture
def router():
    """Create query router instance."""
    return QueryRouter()


@pytest.fixture
def retriever(sample_chunks, tmp_path):
    """Create hybrid retriever with sample chunks."""
    # Use temporary cache path to avoid building real embeddings
    cache_path = tmp_path / "test_embeddings.npy"
    meta_path = tmp_path / "test_embeddings.meta.json"
    retriever = HybridRetriever(
        sample_chunks, 
        local_files_only=True,
        embedding_cache_path=cache_path,
        meta_cache_path=meta_path
    )
    # Don't build index for tests - just use BM25
    return retriever


class TestRouterRetrieverIntegration:
    """Test integration between router and retriever."""
    
    def test_computational_query_filters_tables(self, router, retriever):
        """Test that computational queries filter to table chunks."""
        query = "Berapa total vaksin yang diperlukan?"
        
        # Route query
        routing = router.route_query(query)
        assert routing["intent"] == "computational"
        assert routing["filter_chunk_types"] == ["table"]
        
        # Retrieve with filter
        results = retriever.retrieve(
            query,
            top_k=4,
            filter_chunk_types=routing["filter_chunk_types"]
        )
        
        # All results should be tables
        for result in results:
            chunk_type = result.get("chunk_type") or result.get("metadata", {}).get("chunk_type")
            assert chunk_type == "table"
    
    def test_factual_query_allows_both_types(self, router, retriever):
        """Test that factual queries allow both text and table chunks."""
        query = "Bagaimana cara mendaftar BPJS?"
        
        # Route query
        routing = router.route_query(query)
        assert routing["intent"] == "factual"
        assert "text" in routing["filter_chunk_types"]
        assert "table" in routing["filter_chunk_types"]
        
        # Retrieve with filter (should allow both)
        results = retriever.retrieve(
            query,
            top_k=4,
            filter_chunk_types=routing["filter_chunk_types"]
        )
        
        # Results can be either text or table
        chunk_types = set()
        for result in results:
            chunk_type = result.get("chunk_type") or result.get("metadata", {}).get("chunk_type")
            chunk_types.add(chunk_type)
        
        # At least one type should be present
        assert len(chunk_types) > 0
    
    def test_no_filter_returns_all_types(self, retriever):
        """Test that retrieval without filter returns all chunk types."""
        query = "BPJS"
        
        # Retrieve without filter
        results = retriever.retrieve(query, top_k=4, filter_chunk_types=None)
        
        # Should potentially have both types
        assert len(results) > 0
    
    def test_filter_respects_top_k_limit(self, router, retriever):
        """Test that filtering still respects top_k limit."""
        query = "Berapa jumlah vaksin?"
        
        routing = router.route_query(query)
        results = retriever.retrieve(
            query,
            top_k=2,
            filter_chunk_types=routing["filter_chunk_types"]
        )
        
        # Should return at most top_k results
        assert len(results) <= 2
    
    def test_end_to_end_routing_workflow(self, router, retriever):
        """Test complete workflow from query to filtered retrieval."""
        # Computational query
        comp_query = "Bandingkan jumlah pasien di faskes"
        comp_routing = router.route_query(comp_query)
        comp_results = retriever.retrieve(
            comp_query,
            top_k=3,
            filter_chunk_types=comp_routing["filter_chunk_types"]
        )
        
        assert comp_routing["processing_path"] == "computational"
        assert all(
            r.get("chunk_type") == "table" or r.get("metadata", {}).get("chunk_type") == "table"
            for r in comp_results
        )
        
        # Factual query
        fact_query = "Apa syarat pendaftaran BPJS?"
        fact_routing = router.route_query(fact_query)
        fact_results = retriever.retrieve(
            fact_query,
            top_k=3,
            filter_chunk_types=fact_routing["filter_chunk_types"]
        )
        
        assert fact_routing["processing_path"] == "retrieval"
        assert len(fact_results) > 0


class TestFilterChunkTypesParameter:
    """Test filter_chunk_types parameter behavior."""
    
    def test_filter_single_type(self, retriever):
        """Test filtering for single chunk type."""
        results = retriever.retrieve(
            "vaksin",
            top_k=4,
            filter_chunk_types=["table"]
        )
        
        for result in results:
            chunk_type = result.get("chunk_type") or result.get("metadata", {}).get("chunk_type")
            assert chunk_type == "table"
    
    def test_filter_multiple_types(self, retriever):
        """Test filtering for multiple chunk types."""
        results = retriever.retrieve(
            "BPJS",
            top_k=4,
            filter_chunk_types=["text", "table"]
        )
        
        # Should allow both types
        chunk_types = set()
        for result in results:
            chunk_type = result.get("chunk_type") or result.get("metadata", {}).get("chunk_type")
            chunk_types.add(chunk_type)
        
        # All returned types should be in the filter list
        assert chunk_types.issubset({"text", "table"})
    
    def test_filter_nonexistent_type(self, retriever):
        """Test filtering for non-existent chunk type."""
        results = retriever.retrieve(
            "test",
            top_k=4,
            filter_chunk_types=["nonexistent"]
        )
        
        # Should return empty or very few results
        assert len(results) == 0
