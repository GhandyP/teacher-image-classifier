"""Image loading and normalization."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import mimetypes
import os
import urllib.request
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional, Protocol

from PIL import Image, ImageOps


MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class LoadedImage:
    """Loaded image data."""

    data: bytes
    mime_type: str
    source: str
    source_kind: str
    filename: Optional[str] = None
    sha256: Optional[str] = None


class ImageLoadError(ValueError):
    """Raised when an image cannot be loaded."""


class ImageProcessor(Protocol):
    """Image processor interface."""

    def load(self, source: str, timeout_s: float) -> LoadedImage:
        """Load and return a normalized image."""


class LocalFileProcessor:
    """Loads local images."""

    def load(self, source: str, timeout_s: float) -> LoadedImage:  # noqa: ARG002
        if not os.path.exists(source):
            raise ImageLoadError(f"File not found: {source}")
        with open(source, "rb") as handle:
            original_data = handle.read(MAX_IMAGE_BYTES + 1)
        if len(original_data) > MAX_IMAGE_BYTES:
            raise ImageLoadError(
                f"Image exceeds the maximum allowed size of {MAX_IMAGE_BYTES} bytes"
            )
        _validate_image_bytes(original_data)
        data, normalized_mime_type = _normalize_orientation(original_data)
        mime_type, _ = mimetypes.guess_type(source)
        return LoadedImage(
            data=data,
            mime_type=normalized_mime_type or mime_type or "image/jpeg",
            source=source,
            source_kind="path",
            filename=os.path.basename(source),
            sha256=_sha256(original_data),
        )


class UrlProcessor:
    """Loads images from URLs."""

    def __init__(self, user_agent: str):
        self._user_agent = user_agent

    def load(self, source: str, timeout_s: float) -> LoadedImage:
        if urlparse(source).scheme.lower() not in {"http", "https"}:
            raise ImageLoadError("Image URL must use the http or https scheme")
        request = urllib.request.Request(
            source, headers={"User-Agent": self._user_agent}
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            original_data = response.read(MAX_IMAGE_BYTES + 1)
            if len(original_data) > MAX_IMAGE_BYTES:
                raise ImageLoadError("Image exceeds size limit")
            mime_type = response.headers.get_content_type()
        if not mime_type or not (mime_type.startswith("image/") or mime_type == "application/octet-stream"):
            raise ImageLoadError(f"Unsupported image content type: {mime_type or '<missing>'}")
        _validate_image_bytes(original_data)
        data, normalized_mime_type = _normalize_orientation(original_data)
        return LoadedImage(
            data=data,
            mime_type=normalized_mime_type or mime_type,
            source=source,
            source_kind="url",
            sha256=_sha256(original_data),
        )


class Base64Processor:
    """Loads images from base64 strings."""

    def load(self, source: str, timeout_s: float) -> LoadedImage:  # noqa: ARG002
        mime_type = "image/jpeg"
        payload = source
        if source.startswith("data:"):
            try:
                header, payload = source.split(",", 1)
            except ValueError as exc:
                raise ImageLoadError("Invalid data URL: missing comma separator") from exc
            if ";base64" in header:
                mime_type = header.split(";")[0].replace("data:", "")
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ImageLoadError("Invalid base64 input") from exc
        _validate_image_bytes(data)
        original_data = data
        data, normalized_mime_type = _normalize_orientation(original_data)
        return LoadedImage(
            data=data,
            mime_type=normalized_mime_type or mime_type,
            source="<base64>",
            source_kind="base64",
            sha256=_sha256(original_data),
        )


class ImageProcessorFactory:
    """Factory for image processors."""

    def __init__(self, user_agent: str):
        self._local = LocalFileProcessor()
        self._url = UrlProcessor(user_agent)
        self._b64 = Base64Processor()

    def for_source(self, source: str, kind_hint: Optional[str] = None) -> ImageProcessor:
        if kind_hint == "url" or source.startswith("http://") or source.startswith("https://"):
            return self._url
        if kind_hint == "base64":
            return self._b64
        if kind_hint == "path" or os.path.exists(source):
            return self._local
        return self._b64


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_orientation(data: bytes) -> tuple[bytes, Optional[str]]:
    """Normalize non-default EXIF orientation, returning data and MIME override."""
    with Image.open(io.BytesIO(data)) as image:
        orientation = image.getexif().get(274, 1)
        if orientation == 1:
            return data, None
        normalized = ImageOps.exif_transpose(image)
        if normalized.mode not in {"1", "L", "RGB"}:
            normalized = normalized.convert("RGB")
        output = io.BytesIO()
        normalized.save(output, format="JPEG", quality=92)
        return output.getvalue(), "image/jpeg"


def _validate_image_bytes(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001
        raise ImageLoadError("Invalid image data") from exc

