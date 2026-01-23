"""Classifier tests."""

import logging
from typing import List

from src.classifier import ImageClassifier, RateLimiter
from src.config import AppConfig, GeminiConfig
from src.image_loader import LoadedImage
from src.prompts import PromptContext, default_registry


class FakeGemini:
    def __init__(self, responses: List[str]):
        self._responses = responses
        self.calls = 0

    def generate_json(self, **kwargs: object) -> str:  # noqa: D401
        self.calls += 1
        value = self._responses[self.calls - 1]
        if isinstance(value, Exception):
            raise value
        return value


def _build_classifier(fake: FakeGemini) -> ImageClassifier:
    config = AppConfig(gemini=GeminiConfig(api_key="test"))
    prompt = default_registry().get("nsfw")
    limiter = RateLimiter(1000)
    logger = logging.getLogger("test")
    return ImageClassifier(
        config=config,
        gemini=fake,
        prompt=prompt,
        rate_limiter=limiter,
        logger=logger,
    )


def test_classifier_parses_json() -> None:
    fake = FakeGemini(["{\"label\": \"safe\", \"confidence\": 0.8}"])
    classifier = _build_classifier(fake)
    image = LoadedImage(data=b"x", mime_type="image/png", source="x", source_kind="base64")
    result = classifier.classify_one(image=image, ctx=PromptContext())
    assert result.output["label"] == "safe"
    assert result.errors == []


def test_classifier_retries_on_timeout() -> None:
    fake = FakeGemini([TimeoutError("timeout"), "{\"label\": \"safe\", \"confidence\": 0.9}"])
    classifier = _build_classifier(fake)
    image = LoadedImage(data=b"x", mime_type="image/png", source="x", source_kind="base64")
    result = classifier.classify_one(image=image, ctx=PromptContext())
    assert fake.calls == 2
    assert result.output["label"] == "safe"
