# ODD Feature: Teacher Evaluation Transcription MVP

## Objective
Revive the repository as a local-first assistant that converts photographed or scanned handwritten evaluations into a structured, teacher-reviewable table. The teacher remains responsible for correcting and approving the transcription.

## Problem
The current project is a generic Gemini image-classification CLI. It has reusable image loading, prompting, JSON parsing, retry, rate limiting, and formatting, but it has no evaluation domain model, transcription workflow, review state, persistence, or teacher-oriented output.

## Why
A trustworthy transcription-and-review workflow is the smallest useful product aligned with the original teacher-support idea. Automatic grading is intentionally deferred until transcription quality and review behavior are proven.

## Scope
- Local-first workflow using the existing Gemini integration.
- Initial input assumption: one evaluation page/document per image.
- Local image paths are the primary input; remote URLs remain optional and disabled by default for the MVP.
- Structured transcription grouped by question/item.
- Explicit uncertainty markers and confidence metadata.
- Separate original model transcription from teacher correction.
- JSON and CSV exports first; a local browser review UI follows the stable data contract.
- Existing generic `classify` behavior is preserved temporarily while `transcribe` is introduced.

## Constraints and safety
- No autonomous grading, ranking, pass/fail decision, or student inference in the MVP.
- Never treat model confidence as proof of correctness.
- Preserve source-image traceability for every extracted item.
- Do not put image bytes, handwritten text, or raw model responses in normal logs.
- Keep student references minimal and support anonymized identifiers.
- Preserve the existing test suite and avoid live API calls in tests.
- Keep each implementation task a coherent work unit with tests and documentation where applicable.

## TDD and verification resolution
- TDD mode: unknown; no repository or session setting was found that enables strict TDD.
- Test runner: `pytest`, inferred from `pyproject.toml` (`testpaths = ["tests"]`, `addopts = "-q"`).
- Until an explicit TDD setting is provided, use ordinary functional checks and report the unresolved TDD mode honestly. Do not claim strict TDD evidence.

## Work units

- [x] ODD-01 — Establish a reproducible baseline and privacy-safe fixture contract.
  - Checks: capture Git status; run the existing test suite; document observed baseline; define a representative anonymized fixture shape without committing real student data.
  - Fixture contract: use the existing synthetic `tests/fixtures/sample.ppm` for image inputs; represent extracted data with `evaluation_id = "eval-demo-001"`, `student_reference = "anon-001"`, question `"1"`, synthetic response text, confidence in `[0, 1]`, and an explicit review status. Never add real student content.
  - Acceptance: baseline evidence is recorded; existing behavior is understood; no real student data is added.

- [x] ODD-02 — Add typed evaluation and transcription domain models.
  - Likely surfaces: `src/models.py`, `tests/test_models.py`.
  - Checks: unit tests for valid documents, uncertain items, review states, source metadata, and invalid confidence/status values.
  - Acceptance: model transcription, teacher correction, provenance, and review status are represented separately and serializable.
  - Evidence: `tests/test_models.py` 8 passed; full suite 17 passed (2 pre-existing warnings: `_UnionGenericAlias` deprecation and `LoggingConfig.json` shadow warning).

- [x] ODD-03 — Harden and normalize image ingestion.
  - Likely surfaces: `src/image_loader.py`, `src/config.py`, `tests/test_image_loader.py`.
  - Checks: unit tests for local size limits, supported formats, malformed Base64, hashes, and actionable errors.
  - Acceptance: accepted images have stable metadata and hashes; oversized/invalid inputs fail clearly; local-first privacy defaults are enforced.
  - Evidence: worker completed the bounded change; parent spot check re-ran the full suite: 23 passed, 2 pre-existing warnings. Added `LoadedImage.sha256`, local 10 MiB cap, data-URL comma error, EXIF orientation normalization (JPEG re-encode only when EXIF orientation != 1), URL scheme allowlist http/https + image content-type policy, `ImageConfig` with `IMAGE_MAX_BYTES` env (CLI wiring deferred to ODD-05).

- [x] ODD-04 — Implement a dedicated transcription service and strict schema.
  - Likely surfaces: `src/transcriber.py`, `src/prompts.py`, `src/classifier.py` or a shared client seam, `tests/test_transcriber.py`, `tests/test_prompts.py`.
  - Checks: mocked Gemini tests for valid output, malformed JSON, invalid fields, uncertain handwriting, retries, and no-grade behavior.
  - Acceptance: transcription returns typed question/items, explicit uncertainty, model/prompt metadata, and controlled raw-response diagnostics; it never emits grades.
  - Evidence: worker completed the bounded change; parent spot check re-ran the full suite: 32 passed, 2 pre-existing warnings. New `src/transcriber.py` reuses the classifier client seam without modifying it; strict `RawTranscription`/`RawItem` schema with `extra="forbid"`; `TranscriptionOutcome` carries raw text, errors, attempts, and prompt version; no score/grade field is produced.

- [x] ODD-05 — Add the `transcribe` CLI workflow and usable exports.
  - Likely surfaces: `cli.py`, `src/export.py`, CLI/export tests, README/config examples.
  - Checks: focused CLI tests for single and batch inputs, JSON and CSV output, per-image failure isolation, and dry-run prompt output.
  - Acceptance: a teacher can process a local image or batch and obtain a table with source, question, model transcription, confidence, correction, and review status.
  - Evidence: worker completed the bounded change; parent spot check re-ran the full suite: 42 passed, 2 pre-existing warnings; dry-run CLI emitted the rendered prompt JSON without an API call. New `src/export.py` (deterministic JSON/CSV, no raw text or grades in CSV); `cli.py` gained the `transcribe` subcommand with URL gate honoring `allow_remote_urls`; README gained a Transcribe section and the duplicated closing code fence was fixed.

- [x] ODD-06 — Add local review state and a spreadsheet-first teacher review flow.
  - Likely surfaces: `src/review.py` (new), `cli.py`, `tests/test_review.py`, `tests/test_cli.py`, `README.md`.
  - Decision (user, confirmed): both surfaces in two steps; spreadsheet flow FIRST.
  - Checks: tests for saving/loading review documents, applying teacher corrections and review statuses from an edited CSV, filtering approved rows, and CLI commands with no live API.
  - Acceptance: a teacher can export a CSV, edit corrections/statuses in a spreadsheet, re-import them into the saved JSON, and re-export with review status preserved; corrections never overwrite the model transcription.
  - Evidence: worker completed the bounded change; parent spot check re-ran the full suite: 50 passed, 2 pre-existing warnings. New `src/review.py` (save/load, `apply_csv_corrections`, `only_reviewed_document`, `review_summary`) plus CLI `review import/status/export` and README review section.

- [x] ODD-07 — Add resumability, approval rules, and the local browser review UI.
  - Likely surfaces: review/storage/export modules, a new local UI entry point, and tests.
  - Decision (user, confirmed): browser UI comes AFTER the spreadsheet flow (this task or ODD-08).
  - Checks: retry only failed pages, preserve ordering, reject or flag unreviewed rows on export, delete derived artifacts; UI shows the source image beside items with edit and status controls.
  - Acceptance: interrupted batches can resume without losing review state; exports distinguish model text from teacher-approved corrections; the local UI supports the review loop.
  - Pass (a) evidence: worker completed `--session` resume (reuse by source+hash, retry failed, save session), `review export --require-reviewed` (exit 1 with pending items, no output) and stderr warning without it; parent spot check: 58 passed, 2 pre-existing warnings.
  - Pass (b) evidence: worker completed `src/review_ui.py` pure helpers + `ui/app.py` lazy-import Streamlit shell + `tests/test_review_ui.py` (9 tests) + optional `ui` dependency group (pytest config untouched, streamlit not installed); parent spot check re-ran the full suite: 67 passed, 2 pre-existing warnings.

- [x] ODD-08 — Harden privacy, reliability, cost visibility, and documentation.
  - Likely surfaces: `src/config.py`, `src/classifier.py`, `src/image_loader.py`, `README.md`, config examples, CI/test configuration.
  - Checks: full test suite, log-redaction checks, bounded retry/cancellation checks, documented setup and deletion behavior.
  - Acceptance: the workflow is reproducible, privacy-aware, auditable by source hash/model/prompt version, and documented for a pilot with anonymized data.
  - Evidence: worker completed; parent spot check: 69 passed, 2 pre-existing warnings; `ci.yml` parses OK. Transcription errors are content-free (JSON position + `type@loc` summaries, 300-char cap, no raw payload echo; regression tests with a marker string). New GitHub Actions workflow runs the documented pytest command with uv. README gained Privacy notes + CI reference; `.gitignore` gained `reviews/` and `ui/.streamlit/secrets.toml`; `.env.example` blind-append of commented `IMAGE_MAX_BYTES` (user-authorized without reading); `config.yaml.example` documents `image.max_bytes`/`allow_remote_urls`. Known cosmetic warning `LoggingConfig.json` shadow left as-is (not trivially safe).

## MVP acceptance criteria
- A local batch of supported evaluation images can be processed.
- Each image produces a structured transcription or a clear failure/needs-rescan state.
- Every extracted item links to its source image and evaluation/page.
- Uncertain content is explicitly marked rather than silently guessed.
- Teacher corrections are separate from model output.
- JSON and CSV exports preserve ordering, provenance, and review status.
- No automatic grade, ranking, or pass/fail decision is produced.
- Tests do not require live Gemini credentials.

## Non-goals for this feature
- Autonomous grading or answer correctness inference.
- Student ranking, disciplinary inference, or biometric identification.
- Multi-user collaboration, school-system integration, or hosted accounts.
- Arbitrary multi-page document understanding before the one-image workflow is validated.
- Public URL ingestion by default.

## Open decisions to validate before dependent work
1. Supported languages and handwriting conventions.
2. Whether an image is one page, one full evaluation, or a page in a grouped batch.
3. Preferred review surface: BOTH, in two steps — spreadsheet/CSV flow first (ODD-06), local browser UI second (ODD-07/08). Resolved by user.
4. Student reference policy: name, internal ID, or anonymized token.
5. Gemini data retention and legal/privacy requirements for real evaluations.
6. Desired final export: CSV, XLSX, or LMS-specific format.

- [x] ODD-09 — Fix the three post-delivery audit defects.
  - Authorized by user (all three), CSV design decision: `item_review_status` per row plus `document_review_status`.
  - Evidence: worker completed all three fixes and the parent verified each independently. DEFECT-1: `tests/conftest.py` puts the repo root on `sys.path`, and the exact CI command `uv run --no-project --with-requirements requirements.txt pytest` (no `PYTHONPATH`) passes with 73 passed. DEFECT-2: parent repro script confirmed the new header `...,document_review_status,item_review_status,errors`, export→import preserving `['reviewed','needs_review']`, and the legacy `review_status` fallback. DEFECT-3: `cli.py` now builds `export_outcomes = existing_outcomes + outcomes` for non-dry-run sessions, and `test_transcribe_session_export_includes_previous_documents` passes (3 session tests green). Full suite: 73 passed, 2 pre-existing warnings.

## Verification gaps found after MVP completion (post-delivery audit)

Three defects were confirmed by the parent with executable evidence and have since been FIXED and re-verified under ODD-09 (see that entry for the verification evidence). Original findings:

- **DEFECT-1 (ODD-08, CI) — FIXED:** `.github/workflows/ci.yml` runs `uv run --no-project --with-requirements requirements.txt pytest` without `PYTHONPATH`, so CI fails at collection. Evidence: same command locally returns `9 errors during collection`, exit 2. Local green runs depended on `PYTHONPATH=.`.
- **DEFECT-2 (ODD-06, core teacher flow) — FIXED:** `documents_to_csv` writes the DOCUMENT-level `review_status` on every row, ignoring each item's own status, and `apply_csv_corrections` reads that same `review_status` column per row. Round-trip evidence: original items `['reviewed','needs_review']` re-import from an exported CSV as `['needs_review','needs_review']` — a teacher's per-item approval is silently lost. This breaks the ODD-06 acceptance criterion "re-export with review status preserved".
- **DEFECT-3 (ODD-07, resume) — FIXED:** after a resume run, the CLI exports only the current run's `outcomes`, so previously sessioned documents are silently omitted from `--out`/stdout. Evidence: `cli.py:389,391` use `outcomes`, while the session persists `existing_documents + successful_documents` (`cli.py:383`).

Additional known-open items (no code defect claimed): the transcription path has never been executed against the live Gemini API (all tests are mocked); the Streamlit UI has never actually been launched; no real handwriting fixture exists; unresolved product decisions 1, 2, 4, 5, 6; deferred work (XLSX export, batch concurrency, legacy `min_confidence` unused, `requirements.txt`/`pyproject.toml` duplication, `LoggingConfig.json` cosmetic warning); and nothing is committed yet.

## Progress
- Exploration: complete; repository map and risks recorded in project memory.
- Product direction: accepted in conversation; teacher transcription and human review are the focus.
- ODD tracking: initialized.
- Current task: none. All 8 original work units plus ODD-09 defect fixes are complete and verified; the 3 audit defects are closed.
- Verification evidence: final suite 73 passed under BOTH invocation styles (`uv run --no-project --with-requirements requirements.txt pytest` with no `PYTHONPATH`, and the `PYTHONPATH=.` form), 2 pre-existing warnings only. Progression: ODD-02 17 / ODD-03 23 / ODD-04 32 / ODD-05 42 / ODD-06 50 / ODD-07 67 / ODD-08 69 / ODD-09 73.
- Engram mirror: id 102 saved; the local task file remains authoritative.
- Next step: user decision on delivery (commit/push) and a real-data pilot with anonymized fixtures; optional follow-ups: XLSX export, batch concurrency, and the unused legacy `min_confidence`.
