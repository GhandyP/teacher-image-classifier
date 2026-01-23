# Image Classifier (Gemini)

Python CLI for image classification using the Google Gemini SDK. Built for evaluating high school exam images with consistent, JSON-only outputs.

## Requirements

- Python 3.10+
- Gemini API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp config.yaml.example config.yaml
```

Set `GEMINI_API_KEY` in `.env` or the environment.

## CLI

```bash
python cli.py list-prompts
python cli.py classify --source "tests/fixtures/sample.ppm" --prompt nsfw --format pretty
python cli.py batch --input-file inputs.txt --prompt quality --format json
```

### Output formats

- `json`: machine-readable output
- `pretty`: human-friendly text
- `csv`: CSV rows (flattened)

## Project layout

```
image-classifier-gemini/
├── src/
│   ├── __init__.py
│   ├── classifier.py
│   ├── prompts.py
│   ├── image_loader.py
│   └── config.py
├── cli.py
├── tests/
│   ├── test_classifier.py
│   ├── test_prompts.py
│   └── fixtures/
├── .env.example
├── config.yaml.example
├── requirements.txt
├── pyproject.toml
└── README.md
```
```
