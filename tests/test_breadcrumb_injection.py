"""Test breadcrumb injection for context grounding."""
import pytest
from src.generator import XiaomiMimoClient


def test_breadcrumb_injection_with_hierarchy_path():
    """Test breadcrumb injection uses hierarchy_path when available."""
    client = XiaomiMimoClient(api_key="dummy", base_url="https://api.xiaomimimo.com/v1")
    
    chunks = [
        {
            "chunk_id": "TEST_001",
            "hierarchy_path": ["Buku KIA", "Bab 1", "Imunisasi"],
            "content": "Imunisasi wajib untuk bayi.",
            "chunk_type": "text",
        }
    ]
    
    result = client.inject_breadcrumb_to_context(chunks)
    
    assert "[Buku KIA > Bab 1 > Imunisasi]" in result
    assert "Imunisasi wajib untuk bayi." in result


def test_breadcrumb_injection_fallback_to_manual():
    """Test breadcrumb injection falls back to title/section/subsection."""
    client = XiaomiMimoClient(api_key="dummy", base_url="https://api.xiaomimimo.com/v1")
    
    chunks = [
        {
            "chunk_id": "TEST_002",
            "title": "Panduan BPJS",
            "section": "Pendaftaran",
            "subsection": "Syarat",
            "content": "Syarat pendaftaran peserta baru.",
            "chunk_type": "text",
        }
    ]
    
    result = client.inject_breadcrumb_to_context(chunks)
    
    assert "[Panduan BPJS > Pendaftaran > Syarat]" in result
    assert "Syarat pendaftaran peserta baru." in result


def test_breadcrumb_injection_table_marker():
    """Test breadcrumb injection adds table marker for table chunks."""
    client = XiaomiMimoClient(api_key="dummy", base_url="https://api.xiaomimimo.com/v1")
    
    chunks = [
        {
            "chunk_id": "TEST_003",
            "hierarchy_path": ["Fornas", "Daftar Obat"],
            "content": "| Nama Obat | Dosis |\n|-----------|-------|\n| Paracetamol | 500mg |",
            "chunk_type": "table",
        }
    ]
    
    result = client.inject_breadcrumb_to_context(chunks)
    
    assert "[Fornas > Daftar Obat]" in result
    assert "Tabel Referensi:" in result
    assert "Paracetamol" in result


def test_breadcrumb_injection_multiple_chunks():
    """Test breadcrumb injection handles multiple chunks with separation."""
    client = XiaomiMimoClient(api_key="dummy", base_url="https://api.xiaomimimo.com/v1")
    
    chunks = [
        {
            "chunk_id": "TEST_004",
            "hierarchy_path": ["Doc A", "Section 1"],
            "content": "Content A",
            "chunk_type": "text",
        },
        {
            "chunk_id": "TEST_005",
            "hierarchy_path": ["Doc B", "Section 2"],
            "content": "Content B",
            "chunk_type": "text",
        },
    ]
    
    result = client.inject_breadcrumb_to_context(chunks)
    
    assert "[Doc A > Section 1]" in result
    assert "[Doc B > Section 2]" in result
    assert "Content A" in result
    assert "Content B" in result
    # Check separation between chunks
    assert result.count("\n\n") >= 1


def test_breadcrumb_injection_empty_hierarchy():
    """Test breadcrumb injection handles chunks with no hierarchy gracefully."""
    client = XiaomiMimoClient(api_key="dummy", base_url="https://api.xiaomimimo.com/v1")
    
    chunks = [
        {
            "chunk_id": "TEST_006",
            "content": "Some content without hierarchy",
            "chunk_type": "text",
        }
    ]
    
    result = client.inject_breadcrumb_to_context(chunks)
    
    assert "[Dokumen]" in result
    assert "Some content without hierarchy" in result
