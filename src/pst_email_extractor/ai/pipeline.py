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
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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

    def __init__(self, language: str = "en", dictionary_path: str | None = None, domain_dictionaries: list[str] | None = None) -> None:
        self._language = language
        self._dictionary_path = dictionary_path
        self._domain_dictionaries = domain_dictionaries or []
        self._sym = None
        self._enabled = False
        self._cache = {}  # Cache for frequent corrections
        self._loaded_dictionaries = set()  # Track loaded dictionaries for runtime updates
        self._custom_entries = {}  # Store runtime-added custom entries

    def _initialize(self):
        """Lazy initialization with clinical trial management dictionary support."""
        if self._sym is not None:
            return

        try:
            from symspellpy import SymSpell  # type: ignore
            self._sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

            dict_loaded = False
            total_entries = 0
            self._loaded_dictionaries = set()

            # Priority 1: Load external dictionary if provided
            if self._dictionary_path and Path(self._dictionary_path).exists():
                is_valid, error_msg = validate_dictionary_file(self._dictionary_path)
                if is_valid:
                    try:
                        count = self._sym.load_dictionary(str(self._dictionary_path), term_index=0, count_index=1)
                        total_entries += count
                        dict_loaded = True
                        self._loaded_dictionaries.add("external")
                        logger.info(f"✓ Loaded external dictionary: {self._dictionary_path} ({count} entries)")
                    except Exception as e:
                        logger.warning(f"Failed to load external dictionary: {e}")
                else:
                    logger.warning(f"Skipping invalid external dictionary {self._dictionary_path}: {error_msg}")

            # Load bundled frequency dictionary (support multiple languages)
            if not dict_loaded or self._dictionary_path is None:
                dict_variants = [
                    f"frequency_dictionary_{self._language}_82_765.txt",  # Full English
                    f"frequency_dictionary_{self._language}.txt",         # Chinese or other
                ]

                for dict_file in dict_variants:
                    bundled_dict_path = Path(__file__).parent / "data" / dict_file
                    if bundled_dict_path.exists():
                        # Validate bundled dictionary file before loading
                        is_valid, error_msg = validate_dictionary_file(str(bundled_dict_path))
                        if is_valid:
                            try:
                                count = self._sym.load_dictionary(str(bundled_dict_path), term_index=0, count_index=1)
                                total_entries += count
                                dict_loaded = True
                                lang_name = "english" if self._language.startswith("en") else self._language.replace("-", "_")
                                self._loaded_dictionaries.add(f"{lang_name}_frequency")
                                logger.info(f"✓ Loaded {lang_name} frequency dictionary: {bundled_dict_path} ({count} entries)")
                                break  # Use first available dictionary
                            except Exception as e:
                                logger.warning(f"Failed to load {dict_file}: {e}")
                        else:
                            logger.warning(f"Skipping invalid frequency dictionary {dict_file}: {error_msg}")

            # Priority 2: Load clinical trial management dictionary (highest priority for CTM/CRA use)
            if self._language in ["en", "en-US", "zh-CN"]:
                clinical_dict_path = Path(__file__).parent / "data" / "frequency_dictionary_clinical_en.txt"
                if clinical_dict_path.exists():
                    is_valid, error_msg = validate_dictionary_file(str(clinical_dict_path))
                    if is_valid:
                        try:
                            count = self._sym.load_dictionary(str(clinical_dict_path), term_index=0, count_index=1)
                            total_entries += count
                            self._loaded_dictionaries.add("clinical_trial")
                            logger.info(f"✓ Loaded clinical trial management dictionary: {clinical_dict_path} ({count} entries)")
                            logger.info("  Clinical terms loaded: ICH GCP, CTM/CRA terminology, abbreviations, emerging terms")
                        except Exception as e:
                            logger.warning(f"Failed to load clinical trial dictionary: {e}")
                    else:
                        logger.warning(f"Clinical trial dictionary validation failed: {error_msg}")

            # Load domain-specific dictionaries
            base_domain_dicts = [
                "email_terminology",
                "abbreviations"
            ]

            domain_dicts = []
            for base_name in base_domain_dicts:
                # Try language-specific version first
                lang_specific = f"{base_name}_{self._language}.txt"
                domain_dicts.append(lang_specific)
                # Then generic version
                domain_dicts.append(f"{base_name}.txt")

            # Add custom domain dictionaries (avoid duplicates)
            custom_dicts = [Path(d).name for d in self._domain_dictionaries if Path(d).exists()]
            domain_dicts.extend([d for d in custom_dicts if d not in domain_dicts])

            # Remove duplicates while preserving order
            seen = set()
            domain_dicts = [d for d in domain_dicts if not (d in seen or seen.add(d))]

            for dict_name in domain_dicts:
                dict_path = Path(__file__).parent / "data" / dict_name
                dict_loaded_for_name = False

                if dict_path.exists():
                    # Validate dictionary file before loading
                    is_valid, error_msg = validate_dictionary_file(str(dict_path))
                    if is_valid:
                        try:
                            count = self._sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
                            total_entries += count
                            dict_loaded_for_name = True
                            logger.info(f"Loaded domain dictionary: {dict_name} ({count} entries)")
                        except Exception as e:
                            logger.warning(f"Failed to load domain dictionary {dict_name}: {e}")
                    else:
                        logger.warning(f"Skipping invalid domain dictionary {dict_name}: {error_msg}")

                elif dict_name in [Path(d).name for d in self._domain_dictionaries]:
                    # Try loading from custom domain dictionary path
                    for custom_path in self._domain_dictionaries:
                        if Path(custom_path).name == dict_name and Path(custom_path).exists():
                            # Validate custom dictionary file before loading
                            is_valid, error_msg = validate_dictionary_file(custom_path)
                            if is_valid:
                                try:
                                    count = self._sym.load_dictionary(custom_path, term_index=0, count_index=1)
                                    total_entries += count
                                    dict_loaded_for_name = True
                                    logger.info(f"Loaded custom domain dictionary: {custom_path} ({count} entries)")
                                    break  # Found and loaded the custom dictionary
                                except Exception as e:
                                    logger.warning(f"Failed to load custom domain dictionary {custom_path}: {e}")
                            else:
                                logger.warning(f"Skipping invalid custom domain dictionary {custom_path}: {error_msg}")

                if not dict_loaded_for_name and dict_name in [Path(d).name for d in self._domain_dictionaries]:
                    logger.warning(f"Could not load domain dictionary: {dict_name} (checked both bundled and custom paths)")

            # Load enhanced built-in dictionary as final fallback
            if total_entries == 0:
                self._load_enhanced_builtin_dictionary()
                logger.info("Using enhanced built-in dictionary as fallback")
            else:
                logger.info(f"Dictionary initialization complete. Total entries: {total_entries}")

            self._enabled = True

        except Exception as e:
            logger.error(f"SymSpell initialization failed: {e}")
            self._enabled = False

    def _load_enhanced_builtin_dictionary(self):
        """Load comprehensive built-in dictionary with domain-specific terms based on language."""
        # Language-specific core vocabulary
        core_vocabularies = {
            "en": {
                "the": 1000000, "and": 800000, "is": 600000, "in": 500000, "to": 450000,
                "of": 400000, "a": 350000, "that": 300000, "it": 250000, "with": 200000,
                "as": 180000, "for": 170000, "was": 160000, "on": 150000, "are": 140000,
                "be": 130000, "this": 120000, "have": 110000, "or": 100000, "by": 90000,
                "one": 80000, "had": 70000, "not": 60000, "but": 50000, "what": 40000,
                "all": 30000, "can": 20000, "her": 15000, "will": 10000, "up": 5000,
                "if": 4500, "so": 4000, "out": 3500, "about": 3000, "who": 2500, "get": 2000,
                "which": 1500, "go": 1000, "me": 500
            },
            "es": {
                "de": 1000000, "la": 800000, "que": 600000, "el": 500000, "en": 450000,
                "y": 400000, "a": 350000, "los": 300000, "se": 250000, "del": 200000,
                "las": 180000, "un": 170000, "por": 160000, "con": 150000, "no": 140000,
                "una": 130000, "su": 120000, "para": 110000, "es": 100000, "al": 90000,
                "lo": 80000, "como": 70000, "más": 60000, "o": 50000, "pero": 40000,
                "sus": 30000, "le": 20000, "ya": 15000, "cuando": 10000, "muy": 5000,
                "sin": 4500, "sobre": 4000, "también": 3500, "me": 3000, "hasta": 2500,
                "hay": 2000, "donde": 1500, "quien": 1000, "desde": 500
            },
            "fr": {
                "de": 1000000, "la": 800000, "et": 600000, "le": 500000, "à": 450000,
                "les": 400000, "du": 350000, "un": 300000, "il": 250000, "une": 200000,
                "dans": 180000, "qui": 170000, "par": 160000, "pour": 150000, "des": 140000,
                "au": 130000, "sur": 120000, "avec": 110000, "ce": 100000, "en": 90000,
                "est": 80000, "pas": 70000, "que": 60000, "vous": 50000, "nous": 40000,
                "elle": 30000, "mais": 20000, "je": 15000, "son": 10000, "se": 5000,
                "si": 4500, "plus": 4000, "tout": 3500, "fait": 3000, "faire": 2500,
                "comme": 2000, "être": 1500, "sans": 1000, "après": 500
            }
        }

        # Language-specific email terms
        email_vocabularies = {
            "en": {
                "email": 50000, "attachment": 45000, "subject": 40000, "inbox": 35000,
                "sent": 32000, "recipient": 30000, "sender": 28000, "cc": 25000,
                "bcc": 24000, "forward": 23000, "reply": 22000, "message": 20000,
                "mail": 18000, "received": 17000, "date": 16000, "time": 15000,
                "meeting": 14000, "calendar": 13000, "invite": 12000, "notification": 11000,
                "unread": 10000, "read": 9500, "archive": 9000, "draft": 8500,
                "spam": 8000, "folder": 7000, "signature": 5000, "priority": 4500,
                "urgent": 4000, "confidential": 3500
            },
            "zh": {
                "邮件": 50000, "附件": 45000, "主题": 40000, "收件箱": 35000,
                "已发送": 32000, "收件人": 30000, "发件人": 28000, "抄送": 25000,
                "密送": 24000, "转发": 23000, "回复": 22000, "消息": 20000,
                "邮件": 18000, "已收到": 17000, "日期": 16000, "时间": 15000,
                "会议": 14000, "日历": 13000, "邀请": 12000, "通知": 11000,
                "未读": 10000, "已读": 9500, "存档": 9000, "草稿": 8500,
                "垃圾邮件": 8000, "文件夹": 7000, "签名": 5000, "优先级": 4500,
                "紧急": 4000, "机密": 3500
            }
        }

        # Get language-specific dictionaries (support English and Chinese)
        lang = self._language[:2] if len(self._language) >= 2 else "en"
        if lang not in ["en", "zh"]:
            lang = "en"  # Default to English for unsupported languages

        core_words = core_vocabularies.get(lang, core_vocabularies["en"])
        email_terms = email_vocabularies.get(lang, email_vocabularies["en"])

        # Additional common terms for email processing
        additional_terms = {
            "company": 45000, "business": 40000, "client": 35000, "customer": 32000,
            "project": 26000, "meeting": 14000, "report": 10000, "system": 18000,
            "user": 16000, "support": 800, "service": 700, "error": 700, "data": 600,
            "information": 500, "please": 400, "thank": 350, "regards": 300, "best": 250
        }

        # Common abbreviations (inspired by clinical abbreviations in reference code)
        abbreviations = {
            # Email abbreviations (English)
            "bcc": 50000, "cc": 45000, "fwd": 40000, "re": 35000, "att": 30000,
            "eom": 25000, "fyi": 20000, "asap": 18000, "tbd": 16000, "tba": 14000,
            "etc": 9000, "ie": 8000, "eg": 7000, "vs": 6000, "aka": 10000,
            # Business abbreviations
            "corp": 40000, "inc": 38000, "ltd": 36000, "co": 34000, "dept": 32000,
            "mgr": 30000, "sr": 28000, "jr": 26000, "dr": 24000, "mr": 22000,
            "mrs": 20000, "ms": 18000, "ceo": 14000, "cfo": 12000, "cto": 10000,
            "vp": 9000, "dir": 8000, "rep": 4000, "acct": 3000, "admin": 2000,
            # Technical abbreviations
            "api": 30000, "url": 28000, "ip": 26000, "dns": 24000, "vpn": 22000,
            "ssl": 20000, "http": 16000, "https": 14000, "ftp": 12000, "ssh": 10000,
            "tcp": 9000, "udp": 8000, "cpu": 1800, "gpu": 1600, "ram": 1400,
            "ssd": 1000, "hdd": 800, "os": 500,
            # Common abbreviations
            "id": 45000, "ok": 40000, "am": 18000, "pm": 16000, "min": 14000,
            "sec": 12000, "hr": 10000, "hrs": 9000, "wk": 8000, "mo": 6000,
            "yr": 4000, "kb": 2500, "mb": 2000, "gb": 1500,

            # Chinese clinical abbreviations
            "不良反应": 40000, "临床试验": 38000, "伦理委员会": 36000, "知情同意": 34000,
            "不良事件": 32000, "严重不良": 30000, "药物不良": 28000, "因果关系": 26000,
            "安全性": 24000, "有效性": 22000, "随机对照": 20000, "双盲试验": 18000,
            "单盲试验": 16000, "开放试验": 14000, "临床研究": 12000
        }

        # Merge all dictionaries
        all_terms = {**core_words, **email_terms, **additional_terms, **abbreviations}

        for word, freq in all_terms.items():
            self._sym.create_dictionary_entry(word, freq)

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

    def get_loaded_dictionaries(self) -> set[str]:
        """Get set of loaded dictionary types for debugging."""
        return self._loaded_dictionaries.copy()

    def get_custom_entries_count(self) -> int:
        """Get count of runtime-added custom entries."""
        return len(self._custom_entries)


class _LanguageToolPolisher:
    """Optional LanguageTool-based grammar/style polisher."""

    def __init__(self, language: str = "en-US", lt_jar_path: str | None = None, disabled_rules: list[str] | None = None) -> None:
        self._enabled = False
        self._tool = None
        try:
            import language_tool_python  # type: ignore

            lt_jar_path = lt_jar_path or str(Path(__file__).parent / "lib" / "languagetool.jar")
            if not Path(lt_jar_path).exists():
                logger.info(f"LanguageTool JAR not found at {lt_jar_path}; grammar polishing disabled.")
                return

            self._tool = language_tool_python.LanguageTool(language, lt_jar_path=lt_jar_path)
            if disabled_rules:
                self._tool.disable_rules(disabled_rules)
            self._enabled = True
            logger.info("LanguageTool initialized successfully")
        except AttributeError:
            # language_tool_python module not available or Java not found
            logger.info("LanguageTool dependencies not available; grammar polishing disabled.")
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
                logger.info(f"Transformers tokenizer unavailable; neural polishing disabled: {e}")
                return

            self._enabled = True
            logger.info("NeuralPolisher initialized successfully")
        except Exception as e:
            logger.info(f"NeuralPolisher unavailable; neural polishing disabled: {e}")
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

    def process_batch(self, texts: list[str], *, sanitize: bool, polish: bool, progress: bool = False, _callback: callable | None = None) -> list[str]:
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
        "domain_dictionaries": [],
    }
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                loaded = json.load(f)
                return {**default_config, **loaded}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    return default_config


def validate_dictionary_file(file_path: str) -> tuple[bool, str]:
    """Validate a dictionary file format and content.

    Returns (is_valid, error_message). If valid, error_message is empty.
    """
    if not Path(file_path).exists():
        return False, f"Dictionary file does not exist: {file_path}"

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if not lines:
            return False, "Dictionary file is empty"

        valid_lines = 0
        invalid_lines = 0

        for i, line in enumerate(lines[:1000]):  # Check first 1000 lines for validation
            line = line.strip()
            if not line or line.startswith('#'):  # Skip comments and empty lines
                continue

            parts = line.split()
            if len(parts) != 2:
                invalid_lines += 1
                continue

            word, freq_str = parts
            try:
                freq = int(freq_str)
                if freq < 0:
                    invalid_lines += 1
                    continue
                valid_lines += 1
            except ValueError:
                invalid_lines += 1
                continue

        if valid_lines == 0:
            return False, "No valid word-frequency pairs found"

        if invalid_lines > valid_lines * 0.1:  # More than 10% invalid lines
            return False, f"Too many invalid lines: {invalid_lines}/{valid_lines + invalid_lines}"

        return True, ""

    except Exception as e:
        return False, f"Error reading dictionary file: {e}"


def merge_dictionaries(*dict_paths: str, output_path: str | None = None) -> dict[str, int]:
    """Merge multiple dictionary files into a single dictionary.

    Returns the merged dictionary, optionally saves to output_path.
    """
    merged = {}

    for dict_path in dict_paths:
        if not Path(dict_path).exists():
            logger.warning(f"Dictionary file not found: {dict_path}")
            continue

        is_valid, error_msg = validate_dictionary_file(dict_path)
        if not is_valid:
            logger.warning(f"Skipping invalid dictionary {dict_path}: {error_msg}")
            continue

        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if len(parts) == 2:
                        word, freq_str = parts
                        try:
                            freq = int(freq_str)
                            # Merge by taking the maximum frequency
                            merged[word] = max(merged.get(word, 0), freq)
                        except ValueError:
                            continue

            logger.info(f"Merged dictionary: {dict_path} ({len(merged)} total entries)")

        except Exception as e:
            logger.warning(f"Failed to merge dictionary {dict_path}: {e}")

    if output_path:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Merged dictionary\n")
                f.write("# Format: word frequency\n")
                for word, freq in sorted(merged.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{word} {freq}\n")
            logger.info(f"Saved merged dictionary to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save merged dictionary: {e}")

    return merged


def create_text_pipeline(
    *,
    language: str | None = None,
    enable_sanitize: bool | None = None,
    enable_spell: bool | None = None,
    enable_grammar: bool | None = None,
    enable_neural: bool | None = None,
    neural_model_dir: str | None = None,
    config_path: str | None = None,
    domain_dictionaries: list[str] | None = None,
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
        "domain_dictionaries": domain_dictionaries if domain_dictionaries is not None else config.get("domain_dictionaries", []),
    }

    sanitizer = _RegexSanitizer(final_config["custom_patterns"]) if final_config["enable_sanitize"] else None
    spell = _SymSpellCorrector(final_config["language"].split("-")[0], final_config["dictionary_path"], final_config["domain_dictionaries"]) if final_config["enable_spell"] else None
    grammar = _LanguageToolPolisher(final_config["language"], final_config["lt_jar_path"], final_config["disabled_rules"]) if final_config["enable_grammar"] else None
    neural = _NeuralPolisher(final_config["neural_model_dir"], final_config["model_type"]) if final_config["enable_neural"] and final_config["neural_model_dir"] else None
    return TextPipeline(sanitizer, spell, grammar, neural)


