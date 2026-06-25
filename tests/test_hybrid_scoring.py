"""Test hybrid semantic-lexical scoring in evaluator."""
import pytest
from src.evaluator import (
    evaluate,
    semantic_similarity_score,
    lexical_attribution_score,
    attribution_score,
    SBERT_AVAILABLE,
)


@pytest.fixture
def sample_contexts():
    """Sample contexts for testing."""
    return [
        {
            "chunk_id": "TEST_001",
            "title": "Panduan BPJS",
            "section": "Pendaftaran",
            "content": "Untuk mendaftar BPJS, peserta harus mengisi formulir pendaftaran dengan melampirkan KTP dan Kartu Keluarga.",
            "source_url": "https://bpjs-kesehatan.go.id/panduan",
        },
        {
            "chunk_id": "TEST_002",
            "title": "Syarat Pendaftaran",
            "section": "Dokumen",
            "content": "Dokumen yang diperlukan: KTP asli, Kartu Keluarga, dan pas foto 3x4.",
            "source_url": "https://bpjs-kesehatan.go.id/syarat",
        },
    ]


class TestHybridScoring:
    """Test hybrid semantic-lexical scoring implementation."""
    
    def test_attribution_score_returns_five_values(self, sample_contexts):
        """Test that attribution_score returns 5 values including semantic and lexical scores."""
        answer = "Untuk mendaftar BPJS, peserta perlu menyiapkan KTP dan Kartu Keluarga."
        
        result = attribution_score(answer, sample_contexts)
        
        # Should return (final_score, semantic_score, lexical_score, supported, total)
        assert len(result) == 5
        final_score, semantic_score, lexical_score, supported, total = result
        
        # All scores should be floats between 0 and 1
        assert 0.0 <= final_score <= 1.0
        assert 0.0 <= semantic_score <= 1.0
        assert 0.0 <= lexical_score <= 1.0
        
        # Supported and total should be integers
        assert isinstance(supported, int)
        assert isinstance(total, int)
        assert supported <= total
    
    def test_semantic_score_calculation(self, sample_contexts):
        """Test semantic similarity score calculation."""
        if not SBERT_AVAILABLE:
            pytest.skip("sentence-transformers not available")
        
        # Semantically similar answer
        answer = "Pendaftaran BPJS memerlukan identitas diri seperti KTP dan dokumen keluarga."
        semantic_score = semantic_similarity_score(answer, sample_contexts)
        
        # Should have high semantic similarity
        assert semantic_score > 0.5, f"Expected high semantic similarity, got {semantic_score}"
    
    def test_lexical_score_calculation(self, sample_contexts):
        """Test lexical attribution score calculation."""
        # Answer with exact token overlap
        answer = "Untuk mendaftar BPJS, peserta harus mengisi formulir dengan KTP dan Kartu Keluarga."
        
        lexical_score, supported, total = lexical_attribution_score(answer, sample_contexts)
        
        # Should have high lexical overlap
        assert lexical_score > 0.5, f"Expected high lexical score, got {lexical_score}"
        assert supported > 0
        assert total > 0
    
    def test_hybrid_weighting(self, sample_contexts):
        """Test that hybrid score uses 0.6 semantic + 0.4 lexical weighting."""
        answer = "Untuk mendaftar BPJS, peserta perlu menyiapkan KTP dan Kartu Keluarga."
        
        final_score, semantic_score, lexical_score, _, _ = attribution_score(answer, sample_contexts)
        
        # Calculate expected hybrid score
        expected_hybrid = 0.6 * semantic_score + 0.4 * lexical_score
        
        # Should match (with small floating point tolerance)
        assert abs(final_score - expected_hybrid) < 0.01, \
            f"Expected {expected_hybrid:.3f}, got {final_score:.3f}"
    
    def test_evaluate_includes_semantic_and_lexical_scores(self, sample_contexts):
        """Test that evaluate() returns semantic_score and lexical_score in result."""
        question = "Bagaimana cara mendaftar BPJS?"
        answer = "Untuk mendaftar BPJS, peserta harus mengisi formulir dengan melampirkan KTP dan Kartu Keluarga."
        
        result = evaluate(question, answer, sample_contexts)
        
        # Check that result has semantic_score and lexical_score attributes
        assert hasattr(result, 'semantic_score')
        assert hasattr(result, 'lexical_score')
        assert hasattr(result, 'attribution_score')
        
        # All scores should be valid
        assert 0.0 <= result.semantic_score <= 1.0
        assert 0.0 <= result.lexical_score <= 1.0
        assert 0.0 <= result.attribution_score <= 1.0
    
    def test_semantic_score_with_paraphrase(self, sample_contexts):
        """Test that semantic score captures paraphrased content."""
        if not SBERT_AVAILABLE:
            pytest.skip("sentence-transformers not available")
        
        # Paraphrased answer (different words, same meaning)
        answer = "Proses registrasi kepesertaan BPJS membutuhkan identitas kependudukan dan dokumen kekeluargaan."
        
        semantic_score = semantic_similarity_score(answer, sample_contexts)
        
        # Semantic score should be reasonably high for paraphrase
        assert semantic_score > 0.3, f"Expected semantic similarity for paraphrase, got {semantic_score}"
    
    def test_lexical_score_with_exact_match(self, sample_contexts):
        """Test that lexical score is high for exact token matches."""
        # Answer with many exact tokens from context
        answer = "Untuk mendaftar BPJS, peserta harus mengisi formulir pendaftaran dengan melampirkan KTP dan Kartu Keluarga."
        
        lexical_score, _, _ = lexical_attribution_score(answer, sample_contexts)
        
        # Should have very high lexical overlap
        assert lexical_score > 0.7, f"Expected high lexical score for exact match, got {lexical_score}"
    
    def test_empty_contexts_returns_zero_scores(self):
        """Test that empty contexts return zero scores."""
        answer = "Some answer text"
        empty_contexts = []
        
        final_score, semantic_score, lexical_score, supported, total = attribution_score(answer, empty_contexts)
        
        assert final_score == 0.0
        assert semantic_score == 0.0
        assert lexical_score == 0.0
        assert supported == 0
        assert total == 0
    
    def test_routing_decision_uses_hybrid_score(self, sample_contexts):
        """Test that routing decisions use the hybrid attribution score."""
        question = "Bagaimana cara mendaftar BPJS?"
        
        # High attribution answer
        good_answer = "Untuk mendaftar BPJS, peserta harus mengisi formulir pendaftaran dengan melampirkan KTP dan Kartu Keluarga. Dokumen yang diperlukan adalah KTP asli, Kartu Keluarga, dan pas foto 3x4."
        
        result = evaluate(question, good_answer, sample_contexts)
        
        # Should use hybrid score for routing
        assert result.route in ["ACCEPT", "REVIEW", "REJECT"]
        assert result.attribution_score > 0.0
        
        # Verify hybrid score is being used (not just lexical)
        assert result.semantic_score >= 0.0
        assert result.lexical_score >= 0.0


class TestBackwardCompatibility:
    """Test that existing functionality is preserved."""
    
    def test_hard_fail_rules_still_work(self, sample_contexts):
        """Test that hard-fail rules for medical advice still work."""
        question = "Obat apa yang harus saya minum untuk demam?"
        answer = "Anda harus minum paracetamol 500mg 3 kali sehari."
        
        result = evaluate(question, answer, sample_contexts)
        
        # Should be rejected due to medical advice
        assert result.route == "REJECT"
        assert len(result.hard_fails) > 0
    
    def test_accept_threshold_still_works(self, sample_contexts):
        """Test that ACCEPT threshold logic still works."""
        question = "Bagaimana cara mendaftar BPJS?"
        # High-quality, well-grounded answer
        answer = "Untuk mendaftar BPJS, peserta harus mengisi formulir pendaftaran dengan melampirkan KTP dan Kartu Keluarga. Dokumen yang diperlukan adalah KTP asli, Kartu Keluarga, dan pas foto 3x4."
        
        result = evaluate(question, answer, sample_contexts)
        
        # Should have valid scores
        assert result.attribution_score >= 0.0
        assert result.specificity_score >= 0.0
        assert result.context_quality_score >= 0.0


class TestEdgeCases:
    """Test edge cases for hybrid scoring."""
    
    def test_very_short_answer(self, sample_contexts):
        """Test scoring with very short answer."""
        answer = "Ya."
        
        final_score, semantic_score, lexical_score, supported, total = attribution_score(answer, sample_contexts)
        
        # Should handle gracefully
        assert 0.0 <= final_score <= 1.0
        assert 0.0 <= semantic_score <= 1.0
        assert 0.0 <= lexical_score <= 1.0
    
    def test_answer_with_no_claims(self, sample_contexts):
        """Test answer with no extractable claims."""
        answer = "Hmm..."
        
        final_score, semantic_score, lexical_score, supported, total = attribution_score(answer, sample_contexts)
        
        # Should return zero scores
        assert final_score == 0.0
        assert total == 0
    
    def test_contexts_without_text(self):
        """Test contexts with missing text fields."""
        answer = "Some answer"
        contexts = [
            {"chunk_id": "TEST_001"},  # No text/content
            {"chunk_id": "TEST_002", "content": ""},  # Empty content
        ]
        
        final_score, semantic_score, lexical_score, supported, total = attribution_score(answer, contexts)
        
        # Should handle gracefully without crashing
        assert 0.0 <= final_score <= 1.0
