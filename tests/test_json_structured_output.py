"""Test JSON structured output with exact quote extraction."""
import json
import pytest
from unittest.mock import Mock, patch
from src.generator import XiaomiMimoClient


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    return XiaomiMimoClient(api_key="test-key", base_url="https://api.test.com/v1")


@pytest.fixture
def sample_context():
    """Sample context chunks for testing."""
    return [
        {
            "chunk_id": "TEST_001",
            "hierarchy_path": ["Panduan BPJS", "Pendaftaran"],
            "content": "Untuk mendaftar sebagai peserta BPJS, Anda memerlukan KTP dan Kartu Keluarga.",
            "chunk_type": "text",
        }
    ]


def test_system_prompt_enforces_json_format():
    """Test that system prompt explicitly requires JSON output."""
    from src.generator import SYSTEM_PROMPT
    
    assert "JSON" in SYSTEM_PROMPT
    assert "exact_quote" in SYSTEM_PROMPT
    assert "final_answer" in SYSTEM_PROMPT
    assert "Contoh format output" in SYSTEM_PROMPT


def test_build_messages_includes_json_instruction(mock_client, sample_context):
    """Test that build_messages includes JSON format instruction."""
    messages = mock_client.build_messages("Apa syarat pendaftaran BPJS?", sample_context)
    
    system_content = messages[0]["content"]
    assert "JSON" in system_content
    assert "exact_quote" in system_content
    assert "final_answer" in system_content


@patch('requests.post')
def test_generate_answer_with_quote_parses_json(mock_post, mock_client, sample_context):
    """Test that generate_answer_with_quote correctly parses JSON response."""
    # Mock successful JSON response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "exact_quote": "Untuk mendaftar sebagai peserta BPJS, Anda memerlukan KTP dan Kartu Keluarga.",
                    "final_answer": "Syarat pendaftaran BPJS adalah KTP dan Kartu Keluarga."
                })
            }
        }]
    }
    mock_post.return_value = mock_response
    
    result = mock_client.generate_answer_with_quote("Apa syarat pendaftaran BPJS?", sample_context)
    
    assert isinstance(result, dict)
    assert "exact_quote" in result
    assert "final_answer" in result
    assert "KTP dan Kartu Keluarga" in result["exact_quote"]
    assert "KTP dan Kartu Keluarga" in result["final_answer"]


@patch('requests.post')
def test_generate_answer_with_quote_handles_malformed_json(mock_post, mock_client, sample_context):
    """Test that generate_answer_with_quote handles malformed JSON gracefully."""
    # Mock malformed JSON response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": "This is not valid JSON"
            }
        }]
    }
    mock_post.return_value = mock_response
    
    result = mock_client.generate_answer_with_quote("Test question", sample_context)
    
    assert isinstance(result, dict)
    assert result["exact_quote"] == ""
    assert result["final_answer"] == "This is not valid JSON"


@patch('requests.post')
def test_generate_answer_extracts_final_answer_only(mock_post, mock_client, sample_context):
    """Test that generate_answer returns only final_answer for backward compatibility."""
    # Mock successful JSON response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "exact_quote": "Original text from context",
                    "final_answer": "Processed answer for user"
                })
            }
        }]
    }
    mock_post.return_value = mock_response
    
    result = mock_client.generate_answer("Test question", sample_context)
    
    assert isinstance(result, str)
    assert result == "Processed answer for user"
    assert "exact_quote" not in result


@patch('requests.post')
def test_generate_answer_includes_json_mode_in_payload(mock_post, mock_client, sample_context):
    """Test that generate_answer includes response_format in API payload."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({"exact_quote": "test", "final_answer": "test"})
            }
        }]
    }
    mock_post.return_value = mock_response
    
    mock_client.generate_answer("Test question", sample_context)
    
    # Check that the API call included response_format
    call_args = mock_post.call_args
    payload = call_args[1]["json"]
    assert "response_format" in payload
    assert payload["response_format"] == {"type": "json_object"}


@patch('requests.post')
def test_generate_answer_fallback_on_json_parse_error(mock_post, mock_client, sample_context):
    """Test that generate_answer falls back to raw content if JSON parsing fails."""
    # Mock response with invalid JSON
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": "Plain text response without JSON"
            }
        }]
    }
    mock_post.return_value = mock_response
    
    result = mock_client.generate_answer("Test question", sample_context)
    
    assert result == "Plain text response without JSON"


def test_json_mode_enforces_exact_quote_discipline():
    """Test that the system prompt enforces exact quote discipline."""
    from src.generator import SYSTEM_PROMPT
    
    # Check that prompt requires exact quotes before final answer
    assert 'Salin kutipan kata-per-kata' in SYSTEM_PROMPT
    assert 'HANYA berdasarkan "exact_quote"' in SYSTEM_PROMPT
    assert 'Tidak ada' in SYSTEM_PROMPT  # Fallback for missing info
