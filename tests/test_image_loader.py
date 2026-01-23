"""Image loader tests."""

import base64
from pathlib import Path
from unittest import mock

from src.image_loader import Base64Processor, ImageProcessorFactory, LocalFileProcessor


def test_local_file_processor_loads_image(tmp_path: Path) -> None:
    sample = Path("tests/fixtures/sample.ppm")
    target = tmp_path / "sample.ppm"
    target.write_bytes(sample.read_bytes())

    processor = LocalFileProcessor()
    loaded = processor.load(str(target), timeout_s=5.0)
    assert loaded.source_kind == "path"


def test_base64_processor_loads_image() -> None:
    sample = Path("tests/fixtures/sample.ppm").read_bytes()
    encoded = base64.b64encode(sample).decode("utf-8")
    processor = Base64Processor()
    loaded = processor.load(encoded, timeout_s=5.0)
    assert loaded.source_kind == "base64"


def test_factory_uses_url_processor_for_http() -> None:
    factory = ImageProcessorFactory(user_agent="ua")
    processor = factory.for_source("https://example.com/image.png", None)
    assert processor.__class__.__name__ == "UrlProcessor"


def test_url_processor_reads_bytes() -> None:
    factory = ImageProcessorFactory(user_agent="ua")
    processor = factory.for_source("https://example.com/image.png", "url")
    with mock.patch("urllib.request.urlopen") as mocked:
        mocked.return_value.__enter__.return_value.read.return_value = (
            Path("tests/fixtures/sample.ppm").read_bytes()
        )
        mocked.return_value.__enter__.return_value.headers.get_content_type.return_value = (
            "image/x-portable-pixmap"
        )
        loaded = processor.load("https://example.com/image.png", timeout_s=5.0)
        assert loaded.mime_type == "image/x-portable-pixmap"
