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

### Transcribe

```bash
python cli.py transcribe --source tests/fixtures/sample.ppm --format csv
```

Transcription is a review-first workflow: it preserves model text and uncertainty for teacher correction and produces no grade. Use `--session path.json` to resume completed images and retry failed ones. URLs require `image.allow_remote_urls: true` in the configuration.

## Privacy notes

- Images and transcriptions stay local by default.
- Transcription sends image bytes to Google Gemini; review the provider's data-use terms before using real student work.
- Prefer anonymized student references such as locally generated tokens.
- The review, import, export, and UI workflow makes no network calls.
- `transcribe` URL inputs stay disabled by default.
- Session JSON and CSV files contain student work; do not commit them.
- Add `reviews/` to `.gitignore` for local review and session artifacts.

## CI

GitHub Actions runs the test suite in [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### Review workflow

1. Export transcription rows to CSV: `python cli.py review export --docs transcription.json --format csv --out review.csv`.
2. Edit `teacher_correction` and `item_review_status` in a spreadsheet; `document_review_status` is exported separately.
3. Import the edited spreadsheet: `python cli.py review import --docs transcription.json --csv review.csv`.
4. Check progress: `python cli.py review status --docs transcription.json`.
5. Re-export the reviewed documents with `review export` when ready.
6. Model transcription remains unchanged; corrections stay in a separate field. Export warns about pending items by default; use `--require-reviewed` to block exports until every item is reviewed (or `--only-reviewed` to filter them).

#### Browser review UI (optional)

Install with `pip install -e .[ui]` (or `pip install streamlit`).
Run `streamlit run ui/app.py`, load the session JSON from `transcribe --session`, review/edit items, and save.

### Legacy classification

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
