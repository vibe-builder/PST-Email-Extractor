# AI Pipeline Resources

This directory contains resources for the AI text processing pipeline. These resources are bundled with the application for offline functionality.

## Directory Structure

```
ai/
├── data/                    # Dictionary and data files
│   ├── frequency_dictionary_en_82_765.txt           # English frequency dictionary
│   ├── frequency_dictionary_zh_cn.txt               # Chinese Simplified frequency dictionary
│   ├── frequency_dictionary_clinical_en.txt         # Clinical trial management (English)
│   ├── email_terminology.txt                         # Email communication terms
│   └── abbreviations.txt                             # Common abbreviations
├── lib/                     # Library files (JARs, etc.)
│   └── languagetool.jar
└── model_dir/              # Neural model files
    ├── model.onnx
    ├── tokenizer.json
    └── config.json
```

## Setup Instructions

### 1. Frequency Dictionaries

Download the frequency dictionaries for supported languages:

```bash
# English (82,765 words - high quality)
curl -o data/frequency_dictionary_en_82_765.txt \
  https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt

# Chinese Simplified (clinical and general terms included)
# Note: Chinese dictionary is pre-configured with clinical research terms
```

### 2. Domain-Specific Dictionaries

Domain dictionaries are automatically included and provide specialized terminology for:

- **Email/Communication**: email, attachment, subject, inbox, recipient, sender, etc.
- **Abbreviations**: bcc, cc, fwd, asap, ceo, api, http, etc.
- **Clinical Trial Management**: specialized CTM/CRA terminology, ICH GCP, emerging terms (decentralized trials, AI monitoring)

**Clinical Research Support**: Comprehensive dictionary for biotech, pharma, and CRO professionals including clinical trial terminology, safety reporting, regulatory submissions, and emerging technologies.

These are pre-configured and included in the repository.

### 3. LanguageTool JAR (Optional)

Download the LanguageTool standalone JAR for grammar checking:

```bash
# Download languagetool.jar (replace VERSION with latest)
curl -L -o lib/languagetool.jar \
  https://github.com/languagetool-org/languagetool/releases/download/v6.4/languagetool-standalone.jar
```

### 4. Neural Model (Optional)

Convert T5-small to ONNX format for advanced text polishing:

```bash
# Install optimum
pip install optimum[onnxruntime]

# Export T5-small to ONNX
optimum-cli export onnx --model t5-small --task seq2seq-lm model_dir/
```

## Configuration

Configure the pipeline in your config file or via function parameters:

```json
{
  "enable_sanitize": true,
  "enable_spell": true,
  "enable_grammar": true,
  "enable_neural": true,
  "language": "en-US",
  "model_type": "t5",
  "dictionary_path": null,
  "domain_dictionaries": [],
  "custom_patterns": {},
  "lt_jar_path": null,
  "disabled_rules": []
}
```

### Clinical Trial Management Support

The pipeline includes comprehensive support for Clinical Trial Managers (CTM) and Clinical Research Associates (CRA) working at biotechs, pharma companies, and CROs:

- **ICH GCP Terminology**: protocol, amendment, investigator, informed consent, adverse event, deviation, compliance
- **Clinical Abbreviations**: AE (adverse event), SAE (serious adverse event), SUSAR (suspected unexpected serious adverse reaction), CRO (contract research organization), IRB/IEC (institutional review board/ethics committee)
- **Regulatory Submissions**: IND, NDA, FDA, EMA, ICH, pre-IND, end-of-phase-2, rolling submission
- **Safety & Pharmacovigilance**: MedDRA coding, signal detection, aggregate reporting, causality assessment
- **Study Design**: randomized controlled trials, double-blind, adaptive design, superiority/non-inferiority
- **Emerging Terms**: decentralized trials, digital health, AI-driven monitoring, real world evidence, pragmatic trials
- **CRO & Vendor Management**: contract negotiation, qualification, oversight, KPIs, outsourcing models

**Biotech/Pharma/CRO Usage**:
```python
# Configure for clinical research environment
pipeline = create_text_pipeline(
    language="en-US",
    enable_spell=True,
    enable_sanitize=True
)

# Process clinical trial email
clinical_email = "protocal deviaton reported in ICH GCP complience with SAE monitoring plan"
processed = pipeline.process(clinical_email, sanitize=True, polish=True)
# Result: "protocol deviation reported in ICH GCP compliance with SAE monitoring plan"
```

### Chinese Language Support

The pipeline supports Chinese Simplified (zh-CN) for clinical research communications:

- **Chinese Frequency Dictionary**: 600+ common Chinese words and clinical research terms
- **Chinese Email Terms**: 邮件 (email), 附件 (attachment), 主题 (subject), 收件箱 (inbox), etc.
- **Chinese Clinical Abbreviations**: 不良反应 (adverse reaction), 临床试验 (clinical trial), 伦理委员会 (ethics committee), 知情同意 (informed consent)
- **Bi-directional Support**: Works with both English and Chinese clinical documents

**Chinese Clinical Usage**:
```python
# Configure for Chinese clinical research
pipeline = create_text_pipeline(
    language="zh-CN",
    enable_spell=True,
    enable_sanitize=True
)

# Process Chinese clinical email
chinese_email = "临床试验协议偏离报告的不良事件监测计划"
processed = pipeline.process(chinese_email, sanitize=True, polish=True)
```

### Custom Dictionaries

Add custom domain dictionaries by placing them in the `data/` directory and configuring:

```json
{
  "domain_dictionaries": [
    "/path/to/custom/domain_terms.txt"
  ]
}
```

## Dictionary Loading Priority

Dictionaries are loaded in this order:

1. **External dictionary** (if `dictionary_path` specified)
2. **Clinical trial management dictionary** (frequency_dictionary_clinical_en.txt - for CTM/CRA terms)
3. **Language frequency dictionary** (frequency_dictionary_en_82_765.txt or frequency_dictionary_zh_cn.txt)
4. **Domain dictionaries** (email_terminology.txt, abbreviations.txt)
5. **Enhanced built-in fallback** (core English/Chinese terms with clinical abbreviations)

## Notes

- All resources are optional; the pipeline gracefully falls back when resources are unavailable
- The application remains fully functional offline once resources are bundled
- For development, resources are loaded from these directories
- PyInstaller automatically bundles these resources into the executable
- Dictionary loading is optimized with lazy initialization and caching
