from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from PIL import Image

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}

TSourceImage = TypeVar("TSourceImage", bound="SourceImage")


@dataclass
class SourceImage:
    path: Path
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, step: str, message: str) -> None:
        self.errors.append(f"{step}: {message}")


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def absolute_path(path: Path) -> Path:
    return path.expanduser().absolute()


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def is_skipped(path: Path, skipped_paths: list[Path]) -> bool:
    path = absolute_path(path)
    for skipped_path in skipped_paths:
        if path == skipped_path or skipped_path in path.parents:
            return True
    return False


def discover_sources(
    arguments: list[str],
    source_type: type[TSourceImage],
    *,
    skipped: list[Path] | None = None,
    reject_unsupported_files: bool = False,
) -> list[TSourceImage]:
    """Expand each directory argument in place into alphabetically sorted images."""
    skipped = skipped or []
    sources: list[TSourceImage] = []
    for argument in arguments:
        path = absolute_path(Path(argument))
        if is_skipped(path, skipped):
            continue
        if not path.exists():
            sources.append(
                source_type(path, errors=["discovery: path does not exist"])
            )
            continue
        if path.is_dir():
            files = sorted(
                (
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and is_supported_image(candidate)
                    and not is_skipped(candidate, skipped)
                ),
                key=lambda candidate: str(candidate).lower(),
            )
            sources.extend(source_type(candidate) for candidate in files)
            continue
        if reject_unsupported_files and not is_supported_image(path):
            sources.append(
                source_type(path, errors=["discovery: unsupported file extension"])
            )
            continue
        sources.append(source_type(path))
    return sources


def read_dpi_pair(image: Image.Image) -> tuple[float, float]:
    dpi = image.info.get("dpi")
    if not dpi or len(dpi) < 2:
        raise ValueError("missing DPI metadata; use --force-source-dpi to override")
    dpi_x, dpi_y = map(float, dpi[:2])
    if dpi_x <= 0 or dpi_y <= 0:
        raise ValueError(f"invalid DPI metadata: {dpi!r}")
    return dpi_x, dpi_y


def ensure_output_directory(path: Path) -> str | None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return str(exc)
    if not path.is_dir():
        return "path exists but is not a directory"
    return None
