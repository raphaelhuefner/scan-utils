#!/usr/bin/env python3
"""Place scanned images at their physical size onto PDF pages."""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0
PAGE_SIZES = {
    "a4": (210.0, 297.0),
    "letter": (215.9, 279.4),
}
EXIF_ORIENTATION_TAG = 274
ROTATION_BY_EXIF_ORIENTATION = {
    1: 0,
    3: 180,
    6: 270,
    8: 90,
}
MIRRORED_EXIF_ORIENTATIONS = {2, 4, 5, 7}


@dataclass
class SourceImage:
    path: Path
    errors: list[str] = field(default_factory=list)
    width_pt: float | None = None
    height_pt: float | None = None
    rotate: int = 0
    normalized_stream: bytes | None = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, step: str, message: str) -> None:
        self.errors.append(f"{step}: {message}")


@dataclass(frozen=True)
class Placement:
    source: SourceImage
    page_index: int
    rect: pymupdf.Rect


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Place JPEG and PNG images at their physical size onto PDF pages. "
            "Directory arguments are scanned recursively."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMG_OR_DIR")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        metavar="FILE.pdf",
        help="new PDF file to create",
    )
    parser.add_argument(
        "--page-size",
        choices=sorted(PAGE_SIZES),
        default="a4",
        help="portrait PDF page size (default: a4)",
    )
    parser.add_argument(
        "--margin-mm",
        default=10.0,
        type=non_negative_float,
        metavar="MARGIN",
        help="page margin in millimeters (default: 10.0)",
    )
    parser.add_argument(
        "--gap-mm",
        default=5.0,
        type=non_negative_float,
        metavar="GAP",
        help="minimum vertical distance between images in millimeters (default: 5.0)",
    )
    parser.add_argument(
        "--force-source-dpi",
        type=positive_float,
        metavar="SOURCE_DPI",
        help="override source DPI metadata for both axes",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="allow the same source image to appear more than once",
    )
    return parser.parse_args(argv)


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


def discover_sources(arguments: list[str]) -> list[SourceImage]:
    """Expand each directory argument in place into alphabetically sorted images."""
    sources: list[SourceImage] = []
    for argument in arguments:
        path = absolute_path(Path(argument))
        if not path.exists():
            sources.append(SourceImage(path, errors=["discovery: path does not exist"]))
            continue
        if path.is_dir():
            files = sorted(
                (
                    candidate
                    for candidate in path.rglob("*")
                    if candidate.is_file()
                    and candidate.suffix.lower() in SUPPORTED_SUFFIXES
                ),
                key=lambda candidate: str(candidate).lower(),
            )
            sources.extend(SourceImage(candidate) for candidate in files)
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            sources.append(
                SourceImage(path, errors=["discovery: unsupported file extension"])
            )
            continue
        sources.append(SourceImage(path))
    return sources


def validate_duplicates(images: list[SourceImage], allow_duplicates: bool) -> None:
    if allow_duplicates:
        return
    seen: set[Path] = set()
    for source in images:
        duplicate_key = source.path.resolve()
        if duplicate_key in seen:
            source.add_error(
                "discovery",
                "duplicate source image; pass --allow-duplicates to include it again",
            )
        seen.add(duplicate_key)


def read_source_images(images: list[SourceImage], forced_dpi: float | None) -> None:
    for source in images:
        if not source.ok:
            continue
        try:
            with Image.open(source.path) as image:
                image.load()
                dpi_x, dpi_y = read_dpi(image, forced_dpi)
                orientation = image.getexif().get(EXIF_ORIENTATION_TAG, 1)
                if orientation not in range(1, 9):
                    raise ValueError(f"invalid EXIF orientation: {orientation!r}")
                set_geometry(source, image.size, dpi_x, dpi_y, orientation)
                if orientation in MIRRORED_EXIF_ORIENTATIONS:
                    source.normalized_stream = normalized_png(image)
        except (OSError, UnidentifiedImageError, ValueError, TypeError) as exc:
            source.add_error("read", str(exc))


def read_dpi(image: Image.Image, forced_dpi: float | None) -> tuple[float, float]:
    if forced_dpi is not None:
        return forced_dpi, forced_dpi
    dpi = image.info.get("dpi")
    if not dpi or len(dpi) < 2:
        raise ValueError("missing DPI metadata; use --force-source-dpi to override")
    dpi_x, dpi_y = map(float, dpi[:2])
    if dpi_x <= 0 or dpi_y <= 0:
        raise ValueError(f"invalid DPI metadata: {dpi!r}")
    return dpi_x, dpi_y


def set_geometry(
    source: SourceImage,
    pixel_size: tuple[int, int],
    dpi_x: float,
    dpi_y: float,
    orientation: int,
) -> None:
    width_px, height_px = pixel_size
    width_pt = width_px / dpi_x * POINTS_PER_INCH
    height_pt = height_px / dpi_y * POINTS_PER_INCH
    if orientation in {5, 6, 7, 8}:
        width_pt, height_pt = height_pt, width_pt
    source.width_pt = width_pt
    source.height_pt = height_pt
    source.rotate = ROTATION_BY_EXIF_ORIENTATION.get(orientation, 0)


def normalized_png(image: Image.Image) -> bytes:
    normalized = ImageOps.exif_transpose(image)
    stream = io.BytesIO()
    normalized.save(stream, format="PNG")
    return stream.getvalue()


def page_size_points(name: str) -> tuple[float, float]:
    width_mm, height_mm = PAGE_SIZES[name]
    return mm_to_points(width_mm), mm_to_points(height_mm)


def mm_to_points(value: float) -> float:
    return value / MM_PER_INCH * POINTS_PER_INCH


def create_layout(
    images: list[SourceImage],
    page_size: tuple[float, float],
    margin_pt: float,
    gap_pt: float,
) -> list[Placement]:
    page_width, page_height = page_size
    usable_width = page_width - 2 * margin_pt
    usable_height = page_height - 2 * margin_pt
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("page margins leave no usable page area")

    placements: list[Placement] = []
    page_index = 0
    next_y = margin_pt
    for source in images:
        if not source.ok:
            continue
        assert source.width_pt is not None
        assert source.height_pt is not None
        if source.width_pt > usable_width or source.height_pt > usable_height:
            source.add_error(
                "layout",
                "image does not fit within page margins at its physical size",
            )
            continue
        if next_y != margin_pt and next_y + source.height_pt > page_height - margin_pt:
            page_index += 1
            next_y = margin_pt
        left = (page_width - source.width_pt) / 2
        placements.append(
            Placement(
                source,
                page_index,
                pymupdf.Rect(
                    left,
                    next_y,
                    left + source.width_pt,
                    next_y + source.height_pt,
                ),
            )
        )
        next_y += source.height_pt + gap_pt
    return placements


def create_pdf(
    output_path: Path,
    page_size: tuple[float, float],
    placements: list[Placement],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    document = pymupdf.open()
    try:
        for _ in range(placements[-1].page_index + 1):
            document.new_page(width=page_size[0], height=page_size[1])
        for placement in placements:
            source = placement.source
            options: dict[str, object]
            if source.normalized_stream is not None:
                options = {"stream": source.normalized_stream}
            else:
                options = {"filename": source.path, "rotate": source.rotate}
            document[placement.page_index].insert_image(
                placement.rect, keep_proportion=False, **options
            )

        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        document.save(temporary_path, garbage=4, deflate=True)
        os.replace(temporary_path, output_path)
    finally:
        document.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def report_errors(images: list[SourceImage], general_errors: list[str]) -> int:
    if not general_errors and not any(source.errors for source in images):
        return 0

    print("Errors:", file=sys.stderr)
    for error in general_errors:
        print(f"- {error}", file=sys.stderr)
    for source in images:
        if not source.errors:
            continue
        print(f"- {source.path}", file=sys.stderr)
        for error in source.errors:
            print(f"  - {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = absolute_path(args.output)
    images = discover_sources(args.images)
    general_errors: list[str] = []

    if output_path.suffix.lower() != ".pdf":
        general_errors.append(f"output file must have a .pdf extension: {output_path}")
    if output_path.exists():
        general_errors.append(f"output already exists: {output_path}")
    if not images:
        general_errors.append("no JPEG or PNG source images found")

    validate_duplicates(images, args.allow_duplicates)
    read_source_images(images, args.force_source_dpi)
    try:
        placements = create_layout(
            images,
            page_size_points(args.page_size),
            mm_to_points(args.margin_mm),
            mm_to_points(args.gap_mm),
        )
    except ValueError as exc:
        general_errors.append(str(exc))
        placements = []

    if report_errors(images, general_errors):
        return 1
    if not placements:
        print("Errors:\n- no valid source images found", file=sys.stderr)
        return 1

    try:
        create_pdf(output_path, page_size_points(args.page_size), placements)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Errors:\n- could not create {output_path}: {exc}", file=sys.stderr)
        return 1
    print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
