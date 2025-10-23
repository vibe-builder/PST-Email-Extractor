# AI Pipeline Resources

This directory contains resources for the AI text processing pipeline. These resources are bundled with the application for offline functionality.

## Directory Structure

```
ai/
├── data/                    # Dictionary and data files
│   └── frequency_dictionary_en_82_765.txt
├── lib/                     # Library files (JARs, etc.)
│   └── languagetool.jar
└── model_dir/              # Neural model files
    ├── model.onnx
    ├── tokenizer.json
    └── config.json
```

## Setup Instructions

### 1. SymSpell Dictionary

Download the frequency dictionary from the [SymSpell repository](https://github.com/wolfgarbe/SymSpell):

```bash
# Download frequency_dictionary_en_82_765.txt
curl -o data/frequency_dictionary_en_82_765.txt \
  https://raw.githubusercontent.com/wolfgarbe/SymSpell/master/SymSpell/frequency_dictionary_en_82_765.txt
```

### 2. LanguageTool JAR

Download the LanguageTool standalone JAR:

```bash
# Download languagetool.jar (replace VERSION with latest)
curl -L -o lib/languagetool.jar \
  https://github.com/languagetool-org/languagetool/releases/download/v6.4/languagetool-standalone.jar
```

### 3. Neural Model (T5-small)

Convert T5-small to ONNX format:

```bash
# Install optimum
pip install optimum[onnxruntime]

# Export T5-small to ONNX
optimum-cli export onnx --model t5-small --task seq2seq-lm model_dir/
```

## Configuration

Configure the pipeline in your config file:

```json
{
  "enable_sanitize": true,
  "enable_spell": true,
  "enable_grammar": true,
  "enable_neural": true,
  "language": "en-US",
  "model_type": "t5"
}
```

## Notes

- All resources are optional; the pipeline gracefully falls back when resources are unavailable
- The application remains fully functional offline once resources are bundled
- For development, resources are loaded from these directories
- PyInstaller automatically bundles these resources into the executable
