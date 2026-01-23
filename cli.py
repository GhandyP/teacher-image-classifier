"""Command line interface for image classification."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import io

from tqdm import tqdm

from src.classifier import GeminiClient, ImageClassifier, RateLimiter
from src.config import AppConfig, ConfigPaths, configure_logging, load_config
from src.image_loader import ImageProcessorFactory
from src.prompts import PromptContext, default_registry


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

    list_prompts = subparsers.add_parser("list-prompts", help="List prompts")
    list_prompts.add_argument("--format", choices=["json", "pretty"], default="pretty")

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

    overrides = _build_overrides(args)
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
