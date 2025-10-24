"""Integration tests for AI processing pipeline."""

from pst_email_extractor.ai.pipeline import create_text_pipeline


class TestAIPipeline:
    """Test AI processing with real components where available."""

    def test_regex_sanitizer_always_available(self):
        """Test that regex sanitizer always works."""
        pipeline = create_text_pipeline(enable_sanitize=True)
        assert pipeline.sanitizer is not None

        text = "Contact user@example.com or call 123-456-7890"
        result = pipeline.process(text, sanitize=True, polish=False)
        assert "<EMAIL>" in result
        assert "<PHONE>" in result

    def test_spell_corrector_fallback(self):
        """Test spell correction fallback."""
        pipeline = create_text_pipeline(enable_spell=True)
        # Should work even if SymSpell not available (built-in fallback)
        result = pipeline.process("teh quick brown fox", sanitize=False, polish=True)
        assert isinstance(result, str)

    def test_grammar_polisher_fallback(self):
        """Test grammar polishing fallback when LanguageTool unavailable."""
        pipeline = create_text_pipeline(enable_grammar=True)
        # Should return original text if LanguageTool not available
        text = "This is a test sentence."
        result = pipeline.process(text, sanitize=False, polish=True)
        assert result == text  # Fallback returns original

    def test_neural_polisher_fallback(self):
        """Test neural polishing fallback when models unavailable."""
        pipeline = create_text_pipeline(enable_neural=True, neural_model_dir="/nonexistent")
        text = "This is a test."
        result = pipeline.process(text, sanitize=False, polish=True)
        assert result == text  # Fallback returns original

    def test_combined_processing(self):
        """Test full pipeline with multiple components."""
        pipeline = create_text_pipeline(
            enable_sanitize=True,
            enable_spell=True,
            enable_grammar=True,
            enable_neural=True,
        )

        text = "Contact user@example.com for more info. Teh product is grate."
        result = pipeline.process(text, sanitize=True, polish=True)

        # Should have PII masked
        assert "<EMAIL>" in result
        # Should be a string regardless of component availability
        assert isinstance(result, str)
