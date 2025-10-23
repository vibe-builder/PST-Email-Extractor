"""Comprehensive tests for AI pipeline components."""

from pathlib import Path
from unittest.mock import Mock, patch

from pst_email_extractor.ai.pipeline import (
    TextPipeline,
    _LanguageToolPolisher,
    _NeuralPolisher,
    _RegexSanitizer,
    _SymSpellCorrector,
    create_text_pipeline,
    load_config,
)


class TestRegexSanitizer:
    """Test PII sanitization functionality."""

    def test_basic_sanitization(self):
        """Test basic PII pattern replacement."""
        sanitizer = _RegexSanitizer()
        text = "Contact: user@example.com or call 123-456-7890 from 192.168.1.1"
        result = sanitizer.sanitize(text)
        assert "<EMAIL>" in result
        assert "<PHONE>" in result
        assert "<IP>" in result

    def test_custom_patterns(self):
        """Test custom pattern addition."""
        custom_patterns = {"SSN": r"\d{3}-\d{2}-\d{4}"}
        sanitizer = _RegexSanitizer(custom_patterns)
        text = "SSN: 123-45-6789"
        result = sanitizer.sanitize(text)
        assert "<SSN>" in result

    def test_invalid_custom_pattern(self):
        """Test handling of invalid regex patterns."""
        custom_patterns = {"INVALID": "[invalid regex"}
        sanitizer = _RegexSanitizer(custom_patterns)
        # Should still work with valid default patterns
        text = "Contact: user@example.com"
        result = sanitizer.sanitize(text)
        assert "<EMAIL>" in result

    def test_empty_text(self):
        """Test empty text handling."""
        sanitizer = _RegexSanitizer()
        assert sanitizer.sanitize("") == ""

    def test_pattern_order(self):
        """Test that patterns are applied in correct order."""
        sanitizer = _RegexSanitizer()
        # URL should be processed last to avoid conflicts
        text = "Visit http://example.com/email@test.com"
        result = sanitizer.sanitize(text)
        assert "<URL>" in result
        assert "<EMAIL>" in result


class TestSymSpellCorrector:
    """Test spell correction functionality."""

    def test_initialization_without_dependencies(self):
        """Test graceful fallback when SymSpell is not available."""
        with patch.dict('sys.modules', {'symspellpy': None}):
            corrector = _SymSpellCorrector()
            corrector._initialize()
            assert not corrector._enabled

    @patch('pst_email_extractor.ai.pipeline.Path')
    def test_bundled_dictionary_loading(self, mock_path):
        """Test loading of bundled dictionary."""
        mock_path.return_value.parent = Path("/fake/path")
        mock_path.return_value.exists.return_value = True

        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            corrector = _SymSpellCorrector()
            corrector._initialize()
            # Should attempt to load bundled dictionary

    def test_caching(self):
        """Test that corrections are cached."""
        corrector = _SymSpellCorrector()
        # Mock the initialization
        corrector._enabled = True
        corrector._sym = Mock()
        mock_suggestion = Mock()
        mock_suggestion.term = "corrected"
        corrector._sym.lookup_compound.return_value = [mock_suggestion]

        # First call
        result1 = corrector.correct("teh")
        # Second call should use cache
        result2 = corrector.correct("teh")

        assert result1 == result2 == "corrected"
        assert "teh" in corrector._cache

    def test_cache_limit(self):
        """Test cache size limitation."""
        corrector = _SymSpellCorrector()
        corrector._enabled = True
        corrector._sym = Mock()
        mock_suggestion = Mock()
        mock_suggestion.term = "corrected"
        corrector._sym.lookup_compound.return_value = [mock_suggestion]

        # Fill cache beyond limit
        for i in range(11000):  # Exceeds 10000 limit
            corrector.correct(f"word{i}")

        assert len(corrector._cache) <= 10000

    def test_empty_text(self):
        """Test empty text handling."""
        corrector = _SymSpellCorrector()
        assert corrector.correct("") == ""


class TestLanguageToolPolisher:
    """Test grammar polishing functionality."""

    def test_initialization_without_dependencies(self):
        """Test graceful fallback when LanguageTool is not available."""
        with patch.dict('sys.modules', {'language_tool_python': None}):
            polisher = _LanguageToolPolisher()
            assert not polisher._enabled

    @patch('pst_email_extractor.ai.pipeline.Path')
    def test_jar_not_found(self, mock_path):
        """Test handling when JAR file is not found."""
        mock_path.return_value.exists.return_value = False

        with patch.dict('sys.modules', {'language_tool_python': Mock()}):
            polisher = _LanguageToolPolisher()
            assert not polisher._enabled

    def test_chunking(self):
        """Test text chunking for long texts."""
        polisher = _LanguageToolPolisher()
        polisher._enabled = True
        polisher._tool = Mock()
        polisher._tool.correct.return_value = "corrected "

        long_text = "word " * 300  # Over 1000 characters
        polisher.polish(long_text)

        # Should call correct multiple times due to chunking
        assert polisher._tool.correct.call_count > 1

    def test_empty_text(self):
        """Test empty text handling."""
        polisher = _LanguageToolPolisher()
        assert polisher.polish("") == ""


class TestNeuralPolisher:
    """Test neural polishing functionality."""

    def test_initialization_without_model_dir(self):
        """Test when no model directory is provided."""
        polisher = _NeuralPolisher()
        assert not polisher._enabled

    @patch('pst_email_extractor.ai.pipeline.Path')
    def test_model_not_found(self, mock_path):
        """Test when model file is not found."""
        mock_path.return_value.exists.return_value = False

        polisher = _NeuralPolisher("/fake/model/dir")
        assert not polisher._enabled

    def test_batch_processing(self):
        """Test batch processing functionality."""
        polisher = _NeuralPolisher()
        polisher._enabled = True
        polisher._sess = Mock()
        polisher._tok = Mock()
        polisher._model_type = "t5"

        # Mock tokenizer and session
        polisher._tok.return_value = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        polisher._sess.run.return_value = [[[1, 2, 3]]]  # Mock output
        polisher._tok.decode.return_value = "corrected"

        texts = ["text1", "text2"]
        results = polisher.polish_batch(texts)

        assert len(results) == 2
        polisher._sess.run.assert_called_once()


class TestTextPipeline:
    """Test the main TextPipeline class."""

    def test_caching(self):
        """Test pipeline-level caching."""
        pipeline = TextPipeline(None, None, None, None)

        # Mock sanitizer
        sanitizer = Mock()
        sanitizer.sanitize.return_value = "sanitized"
        pipeline.sanitizer = sanitizer

        # First call
        result1 = pipeline.process("text", sanitize=True, polish=False)
        # Second call should use cache
        result2 = pipeline.process("text", sanitize=True, polish=False)

        assert result1 == result2 == "sanitized"
        sanitizer.sanitize.assert_called_once()  # Should only be called once due to caching

    def test_cache_limit(self):
        """Test cache size limitation."""
        pipeline = TextPipeline(None, None, None, None)

        # Fill cache beyond limit
        for i in range(6000):  # Exceeds 5000 limit
            pipeline.process(f"text{i}", sanitize=False, polish=False)

        assert len(pipeline._cache) <= 5000

    def test_batch_processing(self):
        """Test batch processing with component methods."""
        # Create mock components
        sanitizer = Mock()
        sanitizer.sanitize.side_effect = lambda x: f"sanitized_{x}"

        spell_corrector = Mock()
        spell_corrector.correct.side_effect = lambda x: f"corrected_{x}"

        pipeline = TextPipeline(sanitizer, spell_corrector, None, None)

        texts = ["text1", "text2"]
        results = pipeline.process_batch(texts, sanitize=True, polish=True)

        assert len(results) == 2
        assert results[0] == "corrected_sanitized_text1"

    def test_parallel_processing(self):
        """Test parallel batch processing."""
        pipeline = TextPipeline(None, None, None, None)

        # Mock a simple processor
        pipeline.process = Mock(side_effect=lambda text, **_kwargs: f"processed_{text}")

        texts = ["text1", "text2", "text3"]
        results = pipeline.process_batch_parallel(texts, sanitize=False, polish=False)

        assert len(results) == 3
        assert all("processed_" in result for result in results)

    def test_status_method(self):
        """Test status reporting."""
        # Create components with different enabled states
        sanitizer = Mock()
        spell_corrector = Mock()
        spell_corrector._enabled = True
        grammar_polisher = Mock()
        grammar_polisher._enabled = False
        neural_polisher = Mock()
        neural_polisher._enabled = True

        pipeline = TextPipeline(sanitizer, spell_corrector, grammar_polisher, neural_polisher)

        status = pipeline.status()
        assert status["sanitizer"] is True
        assert status["spell_corrector"] is True
        assert status["grammar_polisher"] is False
        assert status["neural_polisher"] is True


class TestCreateTextPipeline:
    """Test pipeline creation function."""

    def test_create_minimal_pipeline(self):
        """Test creating pipeline with minimal components."""
        pipeline = create_text_pipeline(enable_sanitize=False, enable_spell=False, enable_grammar=False, enable_neural=False)
        assert pipeline.sanitizer is None
        assert pipeline.spell_corrector is None
        assert pipeline.grammar_polisher is None
        assert pipeline.neural_polisher is None

    def test_create_full_pipeline(self):
        """Test creating pipeline with all components enabled."""
        # Just test that the function can be called with all flags enabled
        # The actual instantiation will fail gracefully if dependencies are missing
        pipeline = create_text_pipeline(
            enable_sanitize=True,
            enable_spell=True,
            enable_grammar=True,
            enable_neural=True,
            neural_model_dir="/fake/model"
        )
        # Pipeline should be created even if components fail to initialize
        assert isinstance(pipeline, TextPipeline)

    @patch('pst_email_extractor.ai.pipeline.Path')
    def test_config_loading(self, mock_path):
        """Test configuration loading from file."""
        mock_path.return_value.exists.return_value = True

        mock_config = {
            "enable_sanitize": True,
            "enable_spell": False,
            "language": "en-GB"
        }

        with patch('builtins.open', create=True), \
             patch('json.load', return_value=mock_config):
            config = load_config("/fake/config.json")
            assert config["enable_sanitize"] is True
            assert config["enable_spell"] is False
            assert config["language"] == "en-GB"


class TestIntegration:
    """Integration tests for the complete pipeline."""

    def test_end_to_end_processing(self):
        """Test complete pipeline processing."""
        # Create a pipeline with mock components
        sanitizer = Mock()
        sanitizer.sanitize.side_effect = lambda x: x.replace("secret@email.com", "<EMAIL>")

        spell_corrector = Mock()
        spell_corrector.correct.side_effect = lambda x: x.replace("teh", "the")

        pipeline = TextPipeline(sanitizer, spell_corrector, None, None)

        text = "Contact teh secret@email.com for details"
        result = pipeline.process(text, sanitize=True, polish=True)

        assert "<EMAIL>" in result
        assert "the" in result
        assert "teh" not in result

    def test_stream_processing(self):
        """Test stream processing."""
        pipeline = TextPipeline(None, None, None, None)

        texts = ["text1", "text2", "text3"]
        results = list(pipeline.process_stream(texts, sanitize=False, polish=False))

        assert len(results) == 3
        assert results == texts

    def test_empty_inputs(self):
        """Test handling of empty or None inputs."""
        pipeline = TextPipeline(None, None, None, None)

        # Test various empty inputs
        assert pipeline.process("", sanitize=False, polish=False) == ""
        assert pipeline.process(None, sanitize=False, polish=False) == ""

        # Test batch processing with empty list
        assert pipeline.process_batch([], sanitize=False, polish=False) == []

        # Test stream processing with empty iterable
        assert list(pipeline.process_stream([], sanitize=False, polish=False)) == []
