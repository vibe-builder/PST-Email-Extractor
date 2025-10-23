"""
Lightweight, CPU-friendly text sanitization and polishing pipeline.

All third-party dependencies are optional. When unavailable, the pipeline
falls back to fast regex-based sanitization and returns the original text
for polishing steps. This keeps runtime fully offline and resilient.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol

logger = logging.getLogger(__name__)

# Improved regex patterns for better PII detection
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
_URL_RE = re.compile(r"https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^?@<\s]*(?:\?[^@<\s]*)?)?")
_IP_RE = re.compile(r"(?:\b(?:\d{1,3}\.){3}\d{1,3}\b)|(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}")


class TextSanitizer(Protocol):
    def sanitize(self, text: str) -> str:  # pragma: no cover - protocol
        ...


class TextPolisher(Protocol):
    def polish(self, text: str) -> str:  # pragma: no cover - protocol
        ...


class _RegexSanitizer:
    """Fast, dependency-free sanitizer for common PII patterns.

    Replaces:
    - Emails -> <EMAIL>
    - Phone numbers -> <PHONE>
    - URLs -> <URL>
    - IPv4/IPv6 -> <IP>
    - Custom patterns -> <CUSTOM_KEY>
    """

    def __init__(self, custom_patterns: dict[str, str] | None = None):
        # Define patterns in order of specificity to avoid overlaps
        default_patterns = [
            ("EMAIL", _EMAIL_RE),
            ("PHONE", _PHONE_RE),
            ("IP", _IP_RE),
            ("URL", _URL_RE),  # URL last due to broad matching
        ]

        # Validate and compile custom patterns
        custom = []
        for key, pattern in (custom_patterns or {}).items():
            try:
                custom.append((key, re.compile(pattern)))
                logger.debug(f"Added custom pattern: {key}")
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {key}: {e}")

        self.patterns = dict(default_patterns + custom)

    def sanitize(self, text: str) -> str:
        if not text:
            return ""
        masked = text
        for key, pattern in self.patterns.items():
            masked = pattern.sub(f"<{key}>", masked)
        return masked


class _SymSpellCorrector:
    """Optional SymSpell-based corrector with lazy initialization and caching."""

    def __init__(self, language: str = "en", dictionary_path: str | None = None) -> None:
        self._language = language
        self._dictionary_path = dictionary_path
        self._sym = None
        self._enabled = False
        self._cache = {}  # Cache for frequent corrections

    def _initialize(self):
        """Lazy initialization to reduce startup time."""
        if self._sym is not None:
            return

        try:
            from symspellpy import SymSpell, Verbosity  # type: ignore
            self._sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            # Try to load external dictionary first, fallback to built-in
            dict_loaded = False
            if self._dictionary_path and Path(self._dictionary_path).exists():
                try:
                    self._sym.load_dictionary(str(self._dictionary_path), term_index=0, count_index=1)
                    dict_loaded = True
                    logger.info(f"Loaded external dictionary: {self._dictionary_path}")
                except Exception as e:
                    logger.warning(f"Failed to load external dictionary: {e}")

            if not dict_loaded:
                # Try to load bundled full dictionary
                bundled_dict_path = Path(__file__).parent / "data" / f"frequency_dictionary_{self._language}_82_765.txt"
                if bundled_dict_path.exists():
                    try:
                        self._sym.load_dictionary(str(bundled_dict_path), term_index=0, count_index=1)
                        dict_loaded = True
                        logger.info(f"Loaded bundled dictionary: {bundled_dict_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load bundled dictionary: {e}")

            if not dict_loaded:
                # Load built-in dictionary with email-specific terms
                common_words = {
                    # General common words
                    "the": 1000, "and": 800, "is": 600, "in": 500, "to": 450, "of": 400,
                    "a": 350, "that": 300, "it": 250, "with": 200, "as": 180, "for": 170,
                    "was": 160, "on": 150, "are": 140, "be": 130, "this": 120, "have": 110,
                    "or": 100, "by": 90, "one": 80, "had": 70, "not": 60, "but": 50,
                    "what": 40, "all": 30, "can": 20, "her": 15, "will": 10, "up": 5,
                    # Email-specific terms
                    "email": 350, "attachment": 300, "inbox": 250, "subject": 200, "sent": 180,
                    "from": 170, "recipient": 160, "cc": 150, "bcc": 140, "forward": 130,
                    "reply": 120, "message": 110, "mail": 100, "sender": 90, "received": 80,
                    "date": 70, "time": 60, "meeting": 50, "calendar": 40, "invite": 30,
                    "notification": 20, "unread": 15, "read": 10, "archive": 5
                }
                for word, freq in common_words.items():
                    self._sym.create_dictionary_entry(word, freq)
                logger.info("Using built-in dictionary with email-specific terms")

            self._enabled = True
        except Exception as e:
            logger.error(f"SymSpell initialization failed: {e}")
            self._enabled = False

    def correct(self, text: str) -> str:
        if not text:
            return text

        # Check cache first
        if text in self._cache:
            return self._cache[text]

        self._initialize()
        if not self._enabled or not self._sym:
            return text

        try:
            # Use lookup_compound for better contextual corrections
            suggestions = self._sym.lookup_compound(text, max_edit_distance=2)
            corrected = suggestions[0].term if suggestions else text
            # Cache the result (limit cache size to prevent memory issues)
            if len(self._cache) < 10000:  # Arbitrary limit
                self._cache[text] = corrected
            return corrected
        except Exception as e:
            logger.error(f"Spell correction failed: {e}")
            return text


class _LanguageToolPolisher:
    """Optional LanguageTool-based grammar/style polisher."""

    def __init__(self, language: str = "en-US", lt_jar_path: str | None = None, disabled_rules: list[str] | None = None) -> None:
        self._enabled = False
        self._tool = None
        try:
            import language_tool_python  # type: ignore

            lt_jar_path = lt_jar_path or str(Path(__file__).parent / "lib" / "languagetool.jar")
            if not Path(lt_jar_path).exists():
                logger.warning(f"LanguageTool JAR not found at {lt_jar_path}")
                return

            self._tool = language_tool_python.LanguageTool(language, lt_jar_path=lt_jar_path)
            if disabled_rules:
                self._tool.disable_rules(disabled_rules)
            self._enabled = True
            logger.info("LanguageTool initialized successfully")
        except AttributeError:
            # language_tool_python module not available or Java not found
            logger.warning("LanguageTool dependencies not available")
        except Exception as e:
            logger.error(f"LanguageTool initialization failed: {e}")

    def polish(self, text: str) -> str:
        if not self._enabled or not text or not self._tool:
            return text
        try:
            # Process in chunks of max 1000 characters to avoid LanguageTool slowdown
            from textwrap import wrap
            chunks = wrap(text, 1000, break_long_words=False, replace_whitespace=False)
            polished = [self._tool.correct(chunk) for chunk in chunks if chunk.strip()]
            return "".join(polished)
        except Exception as e:
            logger.error(f"Grammar polishing failed: {e}")
            return text


class _NeuralPolisher:
    """Optional ONNX Runtime-backed neural paraphraser/grammar corrector.

    Supports T5-small or similar seq2seq models for text polishing.
    Falls back gracefully when dependencies or models are unavailable.
    """

    def __init__(self, model_dir: str | None = None, model_type: str = "t5") -> None:
        self._enabled = False
        self._sess = None
        self._tok = None
        self._model_type = model_type
        if not model_dir:
            return

        try:
            from pathlib import Path
            import onnxruntime as ort  # type: ignore

            model_path = Path(model_dir) / "model.onnx"
            if not model_path.exists():
                logger.warning(f"ONNX model not found at {model_path}")
                return

            # Try CUDA first, fallback to CPU
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._sess = ort.InferenceSession(str(model_path), providers=providers)

            # Try to load tokenizer
            try:
                from transformers import AutoTokenizer  # type: ignore
                self._tok = AutoTokenizer.from_pretrained(model_dir)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
                return

            self._enabled = True
            logger.info("NeuralPolisher initialized successfully")
        except Exception as e:
            logger.error(f"NeuralPolisher initialization failed: {e}")
            self._enabled = False

    def polish(self, text: str) -> str:
        if not self._enabled or not text or not self._sess or not self._tok:
            return text

        try:
            # Prepare input based on model type
            input_text = f"improve grammar: {text}" if self._model_type == "t5" else text
            inputs = self._tok(input_text, return_tensors="np", padding=True, truncation=True, max_length=512)

            # Run inference
            outputs = self._sess.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask", inputs["input_ids"] != 0)
            })

            # Generalized output handling
            if outputs and len(outputs) > 0:
                output_ids = outputs[0]
                if hasattr(output_ids, 'shape') and len(output_ids.shape) > 1:
                    output_ids = output_ids[0]  # Take first batch
                polished = self._tok.decode(output_ids, skip_special_tokens=True)
                return polished.strip() if polished.strip() else text

            return text
        except Exception as e:
            logger.error(f"Neural polishing failed: {e}")
            return text

    def polish_batch(self, texts: list[str]) -> list[str]:
        """Batch processing for better performance on multiple texts."""
        if not self._enabled or not texts or not self._sess or not self._tok:
            return texts

        try:
            # Prepare inputs based on model type
            input_texts = [f"improve grammar: {text}" if self._model_type == "t5" else text for text in texts]
            inputs = self._tok(input_texts, return_tensors="np", padding=True, truncation=True, max_length=512)

            # Run batch inference
            outputs = self._sess.run(None, {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask", inputs["input_ids"] != 0)
            })

            # Process outputs
            results = []
            for i, text in enumerate(texts):
                if outputs and len(outputs) > 0 and len(outputs[0]) > i:
                    output_ids = outputs[0][i]
                    polished = self._tok.decode(output_ids, skip_special_tokens=True)
                    results.append(polished.strip() if polished.strip() else text)
                else:
                    results.append(text)
            return results
        except Exception as e:
            logger.error(f"Neural batch polishing failed: {e}")
            return texts


@dataclass
class TextPipeline:
    sanitizer: TextSanitizer | None
    spell_corrector: _SymSpellCorrector | None
    grammar_polisher: TextPolisher | None
    neural_polisher: TextPolisher | None

    def __post_init__(self):
        """Initialize cache after dataclass construction."""
        self._cache = {}

    def process(self, text: str, *, sanitize: bool, polish: bool) -> str:
        # Check cache first
        cache_key = (text, sanitize, polish)
        if cache_key in self._cache:
            return self._cache[cache_key]

        content = text or ""
        if sanitize and self.sanitizer:
            content = self.sanitizer.sanitize(content)
            logger.debug("Applied PII sanitization")
        if polish and self.spell_corrector:
            content = self.spell_corrector.correct(content)
            logger.debug("Applied spell correction")
        if polish and self.grammar_polisher:
            content = self.grammar_polisher.polish(content)
            logger.debug("Applied grammar polishing")
        if polish and self.neural_polisher:
            content = self.neural_polisher.polish(content)
            logger.debug("Applied neural polishing")

        # Cache result (limit cache size)
        if len(self._cache) < 5000:  # Arbitrary limit to prevent memory issues
            self._cache[cache_key] = content

        return content

    def process_batch(self, texts: list[str], *, sanitize: bool, polish: bool, progress: bool = False, callback: callable | None = None) -> list[str]:
        """Process multiple texts efficiently with optional progress tracking and GUI callback."""
        # Work with the original list to avoid creating multiple copies
        results = texts[:]  # Shallow copy to avoid modifying the input
        total = len(results)

        # Set up progress tracking
        if progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=total * 4, desc="Processing texts")  # 4 stages max
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None

        try:
            # Process each component in sequence, mutating results in place
            for component in [
                (sanitize, self.sanitizer, "sanitize", "sanitization"),
                (polish, self.spell_corrector, "correct", "spell correction"),
                (polish, self.grammar_polisher, "polish", "grammar polishing"),
                (polish, self.neural_polisher, "polish_batch", "neural polishing")
            ]:
                if component[0] and component[1]:
                    method = getattr(component[1], component[2])
                    if component[2] == "polish_batch":
                        # Use batch method for neural polisher (returns new list)
                        results = method(results)
                    else:
                        # Apply individual method to each text in place
                        for i, text in enumerate(results):
                            results[i] = method(text)
                            if progress_bar:
                                progress_bar.update(1)
                    logger.debug(f"Applied batch {component[3]}")
                elif progress_bar:
                    # Skip this stage, but update progress for consistency
                    progress_bar.update(total if component[2] != "polish_batch" else 1)
        finally:
            if progress_bar:
                progress_bar.close()

        return results

    def process_batch_parallel(self, texts: list[str], *, sanitize: bool, polish: bool, max_workers: int = 4) -> list[str]:
        """Process texts in parallel for better performance on multi-core systems."""
        def process_single(text: str) -> str:
            return self.process(text, sanitize=sanitize, polish=polish)

        with suppress(ImportError):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                return list(executor.map(process_single, texts))

        # Fallback to sequential processing if concurrent.futures not available
        return [process_single(text) for text in texts]

    def process_stream(self, text_stream: Iterable[str], *, sanitize: bool, polish: bool) -> Iterator[str]:
        """Process a stream of texts for memory-efficient handling of large datasets."""
        for text in text_stream:
            yield self.process(text, sanitize=sanitize, polish=polish)

    def status(self) -> dict[str, bool]:
        """Return status of all pipeline components."""
        return {
            "sanitizer": bool(self.sanitizer),  # Regex sanitizer is always enabled if present
            "spell_corrector": bool(self.spell_corrector and getattr(self.spell_corrector, '_enabled', False)),
            "grammar_polisher": bool(self.grammar_polisher and getattr(self.grammar_polisher, '_enabled', False)),
            "neural_polisher": bool(self.neural_polisher and getattr(self.neural_polisher, '_enabled', False)),
        }


def load_config(config_path: str | None) -> dict:
    """Load configuration from JSON file with defaults."""
    default_config = {
        "language": "en-US",
        "enable_sanitize": False,
        "enable_spell": False,
        "enable_grammar": False,
        "enable_neural": False,
        "neural_model_dir": None,
        "model_type": "t5",
        "custom_patterns": {},
        "dictionary_path": None,
        "lt_jar_path": None,
        "disabled_rules": [],
    }
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                return {**default_config, **loaded}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    return default_config


def create_text_pipeline(
    *,
    language: str | None = None,
    enable_sanitize: bool | None = None,
    enable_spell: bool | None = None,
    enable_grammar: bool | None = None,
    enable_neural: bool | None = None,
    neural_model_dir: str | None = None,
    config_path: str | None = None,
) -> TextPipeline:
    """Factory for a best-effort local pipeline.

    All components are optional; unavailable dependencies are skipped.
    """
    config = load_config(config_path)

    # Merge config with explicit parameters (explicit parameters take precedence when provided)
    final_config = {
        "language": language if language is not None else config.get("language", "en-US"),
        "enable_sanitize": enable_sanitize if enable_sanitize is not None else config.get("enable_sanitize", False),
        "enable_spell": enable_spell if enable_spell is not None else config.get("enable_spell", False),
        "enable_grammar": enable_grammar if enable_grammar is not None else config.get("enable_grammar", False),
        "enable_neural": enable_neural if enable_neural is not None else config.get("enable_neural", False),
        "neural_model_dir": neural_model_dir or config.get("neural_model_dir"),
        "model_type": config.get("model_type", "t5"),
        "custom_patterns": config.get("custom_patterns", {}),
        "dictionary_path": config.get("dictionary_path"),
        "lt_jar_path": config.get("lt_jar_path"),
        "disabled_rules": config.get("disabled_rules", []),
    }

    sanitizer = _RegexSanitizer(final_config["custom_patterns"]) if final_config["enable_sanitize"] else None
    spell = _SymSpellCorrector(final_config["language"].split("-")[0], final_config["dictionary_path"]) if final_config["enable_spell"] else None
    grammar = _LanguageToolPolisher(final_config["language"], final_config["lt_jar_path"], final_config["disabled_rules"]) if final_config["enable_grammar"] else None
    neural = _NeuralPolisher(final_config["neural_model_dir"], final_config["model_type"]) if final_config["enable_neural"] and final_config["neural_model_dir"] else None
    return TextPipeline(sanitizer, spell, grammar, neural)


