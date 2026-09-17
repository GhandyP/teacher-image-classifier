"""Command line interface for image classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import io

from tqdm import tqdm

from src.classifier import GeminiClient, ImageClassifier, RateLimiter
from src.config import AppConfig, ConfigPaths, configure_logging, load_config
from src.export import documents_to_csv, documents_to_json
from src.image_loader import ImageProcessorFactory
from src.transcriber import Transcriber, TranscriptionOutcome
from src.prompts import PromptContext, default_registry
from src.models import EvaluationDocument, EvaluationItem, ReviewStatus
from src.review import (
    apply_csv_corrections,
    load_documents,
    only_reviewed_document,
    pending_items_document,
    review_summary,
    save_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini image classifier")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--log-level", type=str)
    parser.add_argument("--log-json", action="store_true")
    parser.add_argument("--no-log-json", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify", help="Classify a single image")
    classify.add_argument("--source", required=True)
    classify.add_argument("--kind", choices=["path", "url", "base64"])
    classify.add_argument("--prompt", default="nsfw")
    classify.add_argument("--labels", nargs="*")
    classify.add_argument("--format", choices=["json", "pretty", "csv"], default="pretty")
    classify.add_argument("--dry-run", action="store_true")

    batch = subparsers.add_parser("batch", help="Classify a batch of images")
    batch.add_argument("--input-file", type=Path)
    batch.add_argument("--sources", nargs="*")
    batch.add_argument("--kind", choices=["path", "url", "base64"])
    batch.add_argument("--prompt", default="nsfw")
    batch.add_argument("--labels", nargs="*")
    batch.add_argument("--format", choices=["json", "pretty", "csv"], default="json")
    batch.add_argument("--dry-run", action="store_true")

    transcribe = subparsers.add_parser("transcribe", help="Transcribe evaluation images")
    transcribe.add_argument("--source")
    transcribe.add_argument("--sources", nargs="*")
    transcribe.add_argument("--input-file", type=Path)
    transcribe.add_argument("--kind", choices=["path", "url", "base64"])
    transcribe.add_argument("--format", choices=["json", "csv", "pretty"], default="json")
    transcribe.add_argument("--out", type=Path)
    transcribe.add_argument("--session", type=Path)
    transcribe.add_argument("--dry-run", action="store_true")

    list_prompts = subparsers.add_parser("list-prompts", help="List prompts")
    list_prompts.add_argument("--format", choices=["json", "pretty"], default="pretty")

    review = subparsers.add_parser("review", help="Manage local transcription review files")
    review_subparsers = review.add_subparsers(dest="review_command", required=True)

    review_import = review_subparsers.add_parser("import", help="Import spreadsheet corrections")
    review_import.add_argument("--docs", type=Path, required=True)
    review_import.add_argument("--csv", type=Path, required=True)
    review_import.add_argument("--out", type=Path)

    review_status = review_subparsers.add_parser("status", help="Show review status")
    review_status.add_argument("--docs", type=Path, required=True)
    review_status.add_argument("--format", choices=["json", "pretty"], default="pretty")
    review_status.add_argument("--out", type=Path)

    review_export = review_subparsers.add_parser("export", help="Export reviewed documents")
    review_export.add_argument("--docs", type=Path, required=True)
    review_export.add_argument("--only-reviewed", action="store_true")
    review_export.add_argument("--require-reviewed", action="store_true")
    review_export.add_argument("--format", choices=["json", "csv"], default="json")
    review_export.add_argument("--out", type=Path)

    return parser


def _build_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    logging_overrides: Dict[str, Any] = {}
    if args.log_level:
        logging_overrides["level"] = args.log_level
    if args.log_json:
        logging_overrides["json"] = True
    if args.no_log_json:
        logging_overrides["json"] = False
    return {"logging": logging_overrides} if logging_overrides else {}


def _build_classifier(config: AppConfig, prompt_id: str) -> ImageClassifier:
    logger = logging.getLogger("classifier")
    prompt_registry = default_registry(score_weights=config.scoring.weights)
    prompt = prompt_registry.get(prompt_id)
    gemini = GeminiClient(api_key=config.gemini.api_key, logger=logger)
    limiter = RateLimiter(config.rate_limit.requests_per_minute)
    return ImageClassifier(
        config=config,
        gemini=gemini,
        prompt=prompt,
        rate_limiter=limiter,
        logger=logger,
    )


def _build_transcriber(config: AppConfig) -> Transcriber:
    logger = logging.getLogger("transcriber")
    gemini = GeminiClient(api_key=config.gemini.api_key, logger=logger)
    limiter = RateLimiter(config.rate_limit.requests_per_minute)
    return Transcriber(config=config, gemini=gemini, rate_limiter=limiter, logger=logger)


def _load_sources(args: argparse.Namespace) -> List[str]:
    sources: List[str] = []
    if getattr(args, "source", None):
        sources.append(args.source)
    if getattr(args, "sources", None):
        sources.extend(args.sources or [])
    if getattr(args, "input_file", None):
        if not args.input_file.exists():
            raise ValueError("Input file not found")
        sources.extend(
            line.strip() for line in args.input_file.read_text(encoding="utf-8").splitlines()
        )
    sources = [source for source in sources if source]
    if not sources:
        raise ValueError("No sources provided")
    return sources


def _format_results(results: List[Dict[str, Any]], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(results, indent=2)
    if fmt == "csv":
        return _format_csv(results)
    return _format_pretty(results)


def _format_csv(results: List[Dict[str, Any]]) -> str:
    keys = ["source", "prompt_id", "model", "label", "confidence", "score", "errors"]
    rows: List[Dict[str, Any]] = []
    for result in results:
        output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
        rows.append(
            {
                "source": result.get("source"),
                "prompt_id": result.get("prompt_id"),
                "model": result.get("model"),
                "label": output.get("label"),
                "confidence": output.get("confidence"),
                "score": output.get("score"),
                "errors": ";".join(result.get("errors", [])),
            }
        )
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _format_pretty(results: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for result in results:
        output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
        label = output.get("label", "-")
        confidence = output.get("confidence", "-")
        score = output.get("score", "-")
        lines.append(f"{result.get('source')} | {label} | conf={confidence} | score={score}")
        if result.get("errors"):
            lines.append(f"  errors: {', '.join(result['errors'])}")
    return "\n".join(lines)


def _review_documents(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Documents file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("documents JSON must be an array")
        return payload
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Unable to read documents file {path}: {exc}") from exc


def _payload_to_document(payload: Dict[str, Any]) -> EvaluationDocument:
    source = {
        "source": payload.get("source"),
        "source_kind": payload.get("source_kind", "path"),
        "source_hash": payload.get("source_hash"),
        "page_number": payload.get("page_number"),
    }
    items = []
    for raw_item in payload.get("items", []):
        items.append(EvaluationItem(
            item_id=raw_item["question"],
            source=source,
            model_transcription=raw_item.get("model_transcription", ""),
            confidence=raw_item.get("confidence", 0),
            uncertain=raw_item.get("uncertain", False),
            uncertainty_note=raw_item.get("uncertainty_note"),
            teacher_correction=raw_item.get("teacher_correction"),
            review_status=raw_item.get("item_review_status", raw_item.get("review_status", "needs_review")),
        ))
    return EvaluationDocument(
        evaluation_id=payload["evaluation_id"],
        student_reference=payload.get("student_reference"),
        source=source,
        items=items,
        review_status=payload.get("review_status", "needs_review"),
    )


def _document_outcome(document: EvaluationDocument, payload: Dict[str, Any]) -> TranscriptionOutcome:
    return TranscriptionOutcome(
        document=document,
        raw_text=payload.get("raw_text", ""),
        errors=payload.get("errors", []),
        attempts=payload.get("attempts", 0),
        dry_run=payload.get("dry_run", False),
        model=payload.get("model", ""),
        prompt_version=payload.get("prompt_version", ""),
    )


def _write_or_print(output: str, path: Optional[Path]) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding="utf-8")
    else:
        print(output)


def _dispatch_review(args: argparse.Namespace) -> int:
    payloads = _review_documents(args.docs)
    documents = [_payload_to_document(payload) for payload in payloads]
    if args.review_command == "import":
        if not args.csv.exists():
            raise ValueError(f"Corrections file not found: {args.csv}")
        csv_text = args.csv.read_text(encoding="utf-8")
        outcomes = []
        for document, payload in zip(documents, payloads):
            result = apply_csv_corrections(document, csv_text)
            print(f"{document.evaluation_id}: applied={result.applied} skipped={len(result.skipped_rows)}")
            outcomes.append(_document_outcome(document, payload))
        output = documents_to_json(outcomes)
        _write_or_print(output, args.out or args.docs)
        return 0
    if args.review_command == "status":
        summaries = [{"evaluation_id": document.evaluation_id, **review_summary(document)} for document in documents]
        output = json.dumps(summaries, indent=2) if args.format == "json" else "\n".join(
            f"{summary['evaluation_id']}: total={summary['total_items']} needs_review={summary['needs_review']} "
            f"reviewed={summary['reviewed']} needs_rescan={summary['needs_rescan']} "
            f"teacher_corrected={summary['teacher_corrected']} status={summary['review_status']}"
            for summary in summaries
        )
        _write_or_print(output, args.out)
        return 0
    pending_count = sum(len(pending_items_document(document)) for document in documents)
    if args.require_reviewed and pending_count:
        print(
            f"error: {pending_count} item(s) not yet reviewed across "
            f"{sum(bool(pending_items_document(document)) for document in documents)} document(s)",
            file=sys.stderr,
        )
        return 1
    if not args.require_reviewed and pending_count:
        print(
            f"warning: {pending_count} item(s) not yet reviewed across "
            f"{sum(bool(pending_items_document(document)) for document in documents)} document(s); "
            "use --only-reviewed or --require-reviewed",
            file=sys.stderr,
        )
    if args.only_reviewed:
        documents = [only_reviewed_document(document) for document in documents]
    outcomes = [_document_outcome(document, payload) for document, payload in zip(documents, payloads)]
    output = documents_to_json(outcomes) if args.format == "json" else documents_to_csv(outcomes)
    _write_or_print(output, args.out)
    return 0


def _format_prompt_specs(specs: List[Dict[str, Any]]) -> str:
    lines = []
    for spec in specs:
        lines.append(f"{spec.get('id')}: {spec.get('description')}")
    return "\n".join(lines)


def _result_to_dict(result: Any) -> Dict[str, Any]:
    return {
        "prompt_id": result.prompt_id,
        "source": result.source,
        "source_kind": result.source_kind,
        "model": result.model,
        "output": result.output,
        "raw_text": result.raw_text,
        "errors": result.errors,
        "attempts": result.attempts,
        "dry_run": result.dry_run,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "review":
        return _dispatch_review(args)

    overrides = _build_overrides(args)
    if args.command == "transcribe" and args.dry_run and not args.config.exists() and not os.getenv("GEMINI_API_KEY"):
        config = AppConfig(gemini={"api_key": "dry-run"})
    else:
        config = load_config(paths=ConfigPaths(yaml_path=args.config), overrides=overrides)
    configure_logging(config.logging.level, config.logging.json)

    if args.command == "list-prompts":
        registry = default_registry(score_weights=config.scoring.weights)
        specs = [spec.__dict__ for spec in registry.list()]
        output = (
            json.dumps(specs, indent=2)
            if args.format == "json"
            else _format_prompt_specs(specs)
        )
        print(output)
        return 0

    if args.command == "transcribe":
        transcriber = _build_transcriber(config)
        factory = ImageProcessorFactory(user_agent="image-classifier-gemini/1.0")
        sources = _load_sources(args)
        images = []
        for source in sources:
            is_url = args.kind == "url" or source.startswith(("http://", "https://"))
            if is_url and not config.image.allow_remote_urls:
                raise ValueError(
                    "Remote URLs are disabled; set image.allow_remote_urls: true to enable them"
                )
            processor = factory.for_source(source, args.kind)
            images.append(processor.load(source, timeout_s=config.gemini.timeout_s))

        images = list(tqdm(images, disable=len(images) < 2))
        existing_documents = []
        reused = 0
        if args.session and args.session.exists():
            existing_documents = load_documents(args.session)
        existing_sources = {
            (document.source.source, document.source.source_hash)
            for document in existing_documents
            if document.source.source_hash
        }
        pending_images = []
        for image in images:
            if (image.source, image.sha256) in existing_sources:
                reused += 1
            else:
                pending_images.append(image)
        outcomes = (
            transcriber.transcribe_batch(images=pending_images, dry_run=args.dry_run)
            if pending_images
            else []
        )
        successful_documents = [outcome.document for outcome in outcomes if outcome.document is not None and not outcome.errors]
        failed = sum(1 for outcome in outcomes if outcome.errors or outcome.document is None)
        if args.session:
            save_documents(existing_documents + successful_documents, args.session)
            print(
                f"transcribe: total={len(images)} transcribed={len(successful_documents)} "
                f"reused={reused} failed={failed}"
            )
        export_outcomes = outcomes
        if args.session and not args.dry_run:
            existing_outcomes = [
                TranscriptionOutcome(
                    document=document,
                    raw_text="",
                    errors=[],
                    attempts=0,
                    dry_run=False,
                    model=config.gemini.model,
                )
                for document in existing_documents
            ]
            export_outcomes = existing_outcomes + outcomes
        if args.format == "csv":
            output = documents_to_csv(export_outcomes)
        else:
            output = documents_to_json(export_outcomes)
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0

    classifier = _build_classifier(config, args.prompt)
    ctx = PromptContext(labels=args.labels)
    factory = ImageProcessorFactory(user_agent="image-classifier-gemini/1.0")

    sources = _load_sources(args)
    images = []
    for source in sources:
        processor = factory.for_source(source, args.kind)
        images.append(processor.load(source, timeout_s=config.gemini.timeout_s))

    results = []
    for image in tqdm(images, disable=len(images) < 2):
        result = classifier.classify_one(image=image, ctx=ctx, dry_run=args.dry_run)
        results.append(_result_to_dict(result))

    output = _format_results(results, args.format)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
