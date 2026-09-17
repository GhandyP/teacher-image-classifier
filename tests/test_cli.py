import json

import pytest

import cli
from src.config import AppConfig, GeminiConfig
from src.models import EvaluationDocument, EvaluationItem, ReviewStatus, SourceProvenance
from src.transcriber import TranscriptionOutcome


class FakeTranscriber:
    def __init__(self, config):
        self.calls = []

    def transcribe_batch(self, *, images, dry_run=False):
        self.calls.append((images, dry_run))
        outcomes = []
        for image in images:
            source = SourceProvenance(source=image.source, source_kind=image.source_kind, source_hash=image.sha256)
            document = None if dry_run else EvaluationDocument(
                evaluation_id="eval-test", student_reference="anon-001", source=source,
                items=[EvaluationItem(item_id="1", source=source, model_transcription="answer", confidence=0.9)],
                review_status=ReviewStatus.NEEDS_REVIEW,
            )
            outcomes.append(TranscriptionOutcome(
                document=document, raw_text='{"rendered":"prompt"}', errors=[], attempts=0 if dry_run else 1,
                dry_run=dry_run, model="test-model",
            ))
        return outcomes


def _setup(monkeypatch):
    config = AppConfig(gemini=GeminiConfig(api_key="test"))
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(cli, "configure_logging", lambda *args: None)
    fake = FakeTranscriber(config)
    monkeypatch.setattr(cli, "_build_transcriber", lambda config: fake)
    return fake


def test_transcribe_single_source_prints_json(monkeypatch, capsys):
    _setup(monkeypatch)
    assert cli.main(["transcribe", "--source", "tests/fixtures/sample.ppm"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["items"][0]["model_transcription"] == "answer"


def test_transcribe_csv_prints_expected_header(monkeypatch, capsys):
    _setup(monkeypatch)
    cli.main(["transcribe", "--source", "tests/fixtures/sample.ppm", "--format", "csv"])
    assert capsys.readouterr().out.splitlines()[0] == (
        "evaluation_id,source_image,source_kind,source_hash,page_number,student_reference,"
        "question,model_transcription,confidence,uncertain,uncertainty_note,teacher_correction,"
        "document_review_status,item_review_status,errors"
    )


def test_dry_run_does_not_call_model_and_exports_prompt(monkeypatch, capsys):
    fake = _setup(monkeypatch)
    cli.main(["transcribe", "--source", "tests/fixtures/sample.ppm", "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert fake.calls[0][1] is True
    assert payload[0]["prompt"] == {"rendered": "prompt"}
    assert "raw_text" not in payload[0]


def test_out_writes_file(monkeypatch, capsys, tmp_path):
    _setup(monkeypatch)
    output_path = tmp_path / "transcription.json"
    cli.main(["transcribe", "--source", "tests/fixtures/sample.ppm", "--out", str(output_path)])
    assert capsys.readouterr().out == ""
    assert json.loads(output_path.read_text())[0]["evaluation_id"] == "eval-test"


def test_remote_url_is_rejected_by_default(monkeypatch):
    _setup(monkeypatch)
    with pytest.raises(ValueError, match="Remote URLs are disabled"):
        cli.main(["transcribe", "--source", "https://example.test/image.png"])


def test_source_and_input_file_work(monkeypatch, capsys, tmp_path):
    fake = _setup(monkeypatch)
    input_file = tmp_path / "sources.txt"
    input_file.write_text("tests/fixtures/sample.ppm\n")
    cli.main(["transcribe", "--source", "tests/fixtures/sample.ppm", "--input-file", str(input_file)])
    assert len(fake.calls[0][0]) == 2
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_transcribe_session_reuses_existing_sources(monkeypatch, capsys, tmp_path):
    fake = _setup(monkeypatch)
    session = tmp_path / "session.json"
    source = "tests/fixtures/sample.ppm"
    assert cli.main(["transcribe", "--source", source, "--session", str(session)]) == 0
    capsys.readouterr()
    assert len(fake.calls) == 1
    assert cli.main(["transcribe", "--source", source, "--session", str(session)]) == 0
    output = capsys.readouterr().out
    assert len(fake.calls) == 1
    assert "reused=1" in output


def test_transcribe_session_export_includes_previous_documents(monkeypatch, capsys, tmp_path):
    fake = _setup(monkeypatch)
    first = tmp_path / "a.ppm"
    second = tmp_path / "b.ppm"
    ppm_header = b"P6\n1 1\n255\n"
    first.write_bytes(ppm_header + bytes([255, 0, 0]))
    second.write_bytes(ppm_header + bytes([0, 255, 0]))
    session = tmp_path / "session.json"
    output = tmp_path / "export.json"

    assert cli.main(["transcribe", "--source", str(first), "--session", str(session)]) == 0
    capsys.readouterr()
    assert cli.main([
        "transcribe", "--source", str(second), "--session", str(session), "--out", str(output)
    ]) == 0
    capsys.readouterr()

    exported = json.loads(output.read_text())
    assert {entry["source"] for entry in exported} == {str(first), str(second)}
    assert len(fake.calls) == 2
    assert len(fake.calls[1][0]) == 1


def test_transcribe_session_retries_failed_images(monkeypatch, capsys, tmp_path):
    config = AppConfig(gemini=GeminiConfig(api_key="test"))
    monkeypatch.setattr(cli, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(cli, "configure_logging", lambda *args: None)

    class RetryFake(FakeTranscriber):
        def transcribe_batch(self, *, images, dry_run=False):
            self.calls.append((images, dry_run))
            if len(self.calls) == 1:
                return [TranscriptionOutcome(document=None, raw_text="", errors=["failed"], attempts=1, dry_run=False, model="test")]
            outcomes = []
            for image in images:
                source = SourceProvenance(source=image.source, source_kind=image.source_kind, source_hash=image.sha256)
                document = EvaluationDocument(
                    evaluation_id="eval-test", source=source,
                    items=[EvaluationItem(item_id="1", source=source, model_transcription="answer", confidence=0.9)],
                )
                outcomes.append(TranscriptionOutcome(document=document, raw_text="", errors=[], attempts=1, dry_run=False, model="test"))
            return outcomes

    fake = RetryFake(config)
    monkeypatch.setattr(cli, "_build_transcriber", lambda config: fake)
    session = tmp_path / "session.json"
    args = ["transcribe", "--source", "tests/fixtures/sample.ppm", "--session", str(session)]
    cli.main(args)
    assert "failed=1" in capsys.readouterr().out
    cli.main(args)
    assert "transcribed=1" in capsys.readouterr().out
    assert len(fake.calls) == 2


def _write_review_docs(path):
    source = SourceProvenance(source="page.png", source_kind="path", source_hash="hash")
    document = EvaluationDocument(
        evaluation_id="eval-review", source=source,
        items=[
            EvaluationItem(item_id="1", source=source, model_transcription="first", confidence=.9),
            EvaluationItem(item_id="2", source=source, model_transcription="second", confidence=.8),
        ],
    )
    outcome = TranscriptionOutcome(document=document, raw_text="", errors=[], attempts=1, dry_run=False, model="test")
    path.write_text(cli.documents_to_json([outcome]))


def test_review_import_updates_saved_documents(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    corrections = tmp_path / "corrections.csv"
    _write_review_docs(docs)
    corrections.write_text(
        "evaluation_id,question,teacher_correction,review_status\n"
        "eval-review,1,teacher answer,reviewed\n"
    )
    assert cli.main(["review", "import", "--docs", str(docs), "--csv", str(corrections)]) == 0
    assert "eval-review: applied=1 skipped=0" in capsys.readouterr().out
    payload = json.loads(docs.read_text())
    assert payload[0]["items"][0]["teacher_correction"] == "teacher answer"
    assert payload[0]["items"][0]["item_review_status"] == "reviewed"
    assert payload[0]["items"][0]["model_transcription"] == "first"


def test_review_status_outputs_counts(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    _write_review_docs(docs)
    cli.main(["review", "status", "--docs", str(docs)])
    output = capsys.readouterr().out
    assert "total=2" in output
    assert "needs_review=2" in output


def test_review_export_csv_filters_only_reviewed_rows(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    _write_review_docs(docs)
    corrections = tmp_path / "corrections.csv"
    corrections.write_text("evaluation_id,question,teacher_correction,review_status\neval-review,1,,reviewed\n")
    cli.main(["review", "import", "--docs", str(docs), "--csv", str(corrections)])
    capsys.readouterr()
    cli.main(["review", "export", "--docs", str(docs), "--only-reviewed", "--format", "csv"])
    filtered = capsys.readouterr().out
    assert filtered.count("eval-review") == 1
    cli.main(["review", "export", "--docs", str(docs), "--format", "csv"])
    assert capsys.readouterr().out.count("eval-review") == 2


def test_review_export_warns_about_pending_items(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    _write_review_docs(docs)
    output = tmp_path / "export.csv"
    assert cli.main(["review", "export", "--docs", str(docs), "--format", "csv", "--out", str(output)]) == 0
    captured = capsys.readouterr()
    assert "warning:" in captured.err
    assert output.exists()


def test_review_export_require_reviewed_blocks_pending_items(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    _write_review_docs(docs)
    output = tmp_path / "export.json"
    assert cli.main(["review", "export", "--docs", str(docs), "--require-reviewed", "--out", str(output)]) == 1
    assert "error:" in capsys.readouterr().err
    assert not output.exists()


def test_review_export_require_reviewed_allows_reviewed_documents(tmp_path, capsys):
    docs = tmp_path / "docs.json"
    _write_review_docs(docs)
    corrections = tmp_path / "corrections.csv"
    corrections.write_text(
        "evaluation_id,question,teacher_correction,review_status\n"
        "eval-review,1,,reviewed\n"
        "eval-review,2,,reviewed\n"
    )
    cli.main(["review", "import", "--docs", str(docs), "--csv", str(corrections)])
    capsys.readouterr()
    output = tmp_path / "export.json"
    assert cli.main(["review", "export", "--docs", str(docs), "--require-reviewed", "--out", str(output)]) == 0
    assert output.exists()
