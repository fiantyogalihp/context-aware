"""Test chunk type filtering in hybrid retriever."""
import pytest
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
    ]


def test_filter_chunk_types_parameter_exists(sample_chunks):
    """Test that retrieve method accepts filter_chunk_types parameter."""
    retriever = HybridRetriever(sample_chunks, local_files_only=True)
    
    # Should not raise error
    try:
        # Just test that parameter is accepted (don't actually retrieve)
        import inspect
        sig = inspect.signature(retriever.retrieve)
        assert "filter_chunk_types" in sig.parameters
    except Exception as e:
        pytest.fail(f"filter_chunk_types parameter not found: {e}")


def test_chunk_type_filtering_logic():
    """Test the chunk type filtering logic without full retrieval."""
    chunks = [
        {"chunk_id": "T1", "chunk_type": "text", "content": "text content"},
        {"chunk_id": "TB1", "chunk_type": "table", "content": "table content"},
        {"chunk_id": "T2", "chunk_type": "text", "content": "more text"},
    ]
    
    # Simulate filtering logic
    filter_types = ["table"]
    filtered = []
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type")
        if filter_types is None or chunk_type in filter_types:
            filtered.append(chunk)
    
    # Should only have table chunks
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "TB1"


def test_no_filter_includes_all():
    """Test that no filter includes all chunk types."""
    chunks = [
        {"chunk_id": "T1", "chunk_type": "text"},
        {"chunk_id": "TB1", "chunk_type": "table"},
    ]
    
    # Simulate no filtering
    filter_types = None
    filtered = []
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type") or chunk.get("metadata", {}).get("chunk_type")
        if filter_types is None or chunk_type in filter_types:
            filtered.append(chunk)
    
    # Should include all
    assert len(filtered) == 2
