"""Image loader tests."""

import base64
import hashlib
import io
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

from src.image_loader import (
    MAX_IMAGE_BYTES,
    Base64Processor,
    ImageLoadError,
    ImageProcessorFactory,
    LocalFileProcessor,
)


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


def test_local_file_processor_rejects_oversized_file(tmp_path: Path) -> None:
    target = tmp_path / "oversized.bin"
    target.write_bytes(b"x" * (MAX_IMAGE_BYTES + 1))

    with pytest.raises(ImageLoadError, match="maximum allowed size"):
        LocalFileProcessor().load(str(target), timeout_s=5.0)


def test_loaded_image_has_sha256_of_original_bytes() -> None:
    sample = Path("tests/fixtures/sample.ppm").read_bytes()

    loaded = Base64Processor().load(base64.b64encode(sample).decode(), timeout_s=5.0)

    assert loaded.sha256 == hashlib.sha256(sample).hexdigest()


def test_base64_processor_rejects_malformed_input() -> None:
    processor = Base64Processor()

    with pytest.raises(ImageLoadError, match="Invalid base64 input"):
        processor.load("not valid base64", timeout_s=5.0)
    with pytest.raises(ImageLoadError, match="missing comma"):
        processor.load("data:image/jpeg;base64", timeout_s=5.0)


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


def test_url_processor_rejects_non_image_content_type() -> None:
    response = mock.Mock()
    response.read.return_value = Path("tests/fixtures/sample.ppm").read_bytes()
    response.headers.get_content_type.return_value = "text/plain"
    with mock.patch("urllib.request.urlopen") as mocked:
        mocked.return_value.__enter__.return_value = response
        with pytest.raises(ImageLoadError, match="content type"):
            ImageProcessorFactory("ua").for_source("https://example.com/x", "url").load(
                "https://example.com/x", timeout_s=5.0
            )


def test_url_processor_rejects_non_http_scheme() -> None:
    with pytest.raises(ImageLoadError, match="http or https"):
        ImageProcessorFactory("ua").for_source("ftp://example.com/x", "url").load(
            "ftp://example.com/x", timeout_s=5.0
        )


def test_exif_orientation_is_normalized_to_jpeg() -> None:
    image = Image.new("RGB", (4, 2), color="red")
    exif = image.getexif()
    exif[274] = 6
    output = io.BytesIO()
    image.save(output, format="JPEG", exif=exif)

    loaded = Base64Processor().load(base64.b64encode(output.getvalue()).decode(), timeout_s=5.0)

    assert loaded.mime_type == "image/jpeg"
    with Image.open(io.BytesIO(loaded.data)) as normalized:
        assert normalized.format == "JPEG"
        assert normalized.size == (2, 4)
