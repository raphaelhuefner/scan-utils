#!/usr/bin/env python3
"""Deskew, crop, and downscale scanned document images for email."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from common import (
    SourceImage as BaseSourceImage,
    absolute_path,
    discover_sources as discover_image_sources,
    ensure_output_directory,
    is_skipped,
    non_negative_float,
    positive_float,
    read_dpi_pair,
)

PREVIEW_MAX_DIMENSION = 800
EMAIL_DPI = 300.0
GRABCUT_ITERATIONS = 4


@dataclass
class SourceImage(BaseSourceImage):
    image: Image.Image | None = None
    dpi: float | None = None
    rectangle: np.ndarray | None = None
    deskew_angle: float | None = None
    archive_path: Path | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deskew and crop scans at their original DPI, then create smaller "
            "300 DPI JPEG copies suitable for email."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMG_OR_DIR")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        metavar="ARCHIVE_DIRECTORY",
        help="directory for deskewed and cropped images at source resolution",
    )
    parser.add_argument(
        "--email-dir",
        type=Path,
        metavar="EMAIL_DIRECTORY",
        help="directory for 300 DPI JPEG images",
    )
    parser.add_argument(
        "--crop-margin-mm",
        default=3.0,
        type=non_negative_float,
        metavar="MARGIN",
        help="crop margin around the detected document rectangle, in millimeters (default: 3.0)",
    )
    parser.add_argument(
        "--force-source-dpi",
        type=positive_float,
        metavar="SOURCE_DPI",
        help="override source DPI metadata",
    )
    parser.add_argument(
        "--email-jpeg-quality",
        default=40,
        type=jpeg_quality,
        metavar="QUALITY",
        help="JPEG quality for 300 DPI email images (default: 40)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        type=Path,
        metavar="IMG_OR_DIR",
        help="skip a source file or directory tree; may be repeated",
    )
    args = parser.parse_args(argv)
    apply_default_output_directories(args, parser)
    return args


def apply_default_output_directories(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.archive_dir is not None and args.email_dir is not None:
        return

    if len(args.images) == 1:
        input_path = absolute_path(Path(args.images[0]))
        if input_path.is_dir():
            parent = input_path.parent
            if args.archive_dir is None:
                args.archive_dir = parent / "aligned-original"
            if args.email_dir is None:
                args.email_dir = parent / "email-images"
            return

    missing_options = []
    if args.archive_dir is None:
        missing_options.append("--archive-dir")
    if args.email_dir is None:
        missing_options.append("--email-dir")
    parser.error(
        "the following arguments are required unless there is exactly one input "
        f"directory: {', '.join(missing_options)}"
    )


def jpeg_quality(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return number


def discover_sources(arguments: list[str], skipped: list[Path]) -> list[SourceImage]:
    return discover_image_sources(arguments, SourceImage, skipped=skipped)


def read_source_images(images: list[SourceImage], forced_dpi: float | None) -> None:
    for source in images:
        if not source.ok:
            continue
        try:
            with Image.open(source.path) as opened:
                image = ImageOps.exif_transpose(opened)
                image.load()
                dpi = forced_dpi if forced_dpi is not None else read_dpi(opened)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            source.add_error("read", str(exc))
            continue
        source.image = image.convert("RGB")
        source.dpi = dpi


def read_dpi(image: Image.Image) -> float:
    dpi = image.info.get("dpi")
    dpi_x, dpi_y = read_dpi_pair(image)
    if abs(dpi_x - dpi_y) / max(dpi_x, dpi_y) > 0.01:
        raise ValueError(f"non-square DPI metadata is unsupported: {dpi!r}")
    return (dpi_x + dpi_y) / 2


def detect_rectangles(images: list[SourceImage]) -> None:
    for source in images:
        if not source.ok:
            continue
        assert source.image is not None
        try:
            rectangle, angle = detect_document_rectangle(source.image)
        except (ValueError, cv2.error) as exc:
            source.add_error("detect", str(exc))
            continue
        source.rectangle = rectangle
        source.deskew_angle = angle


def detect_document_rectangle(image: Image.Image) -> tuple[np.ndarray, float]:
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    scale = min(1.0, PREVIEW_MAX_DIMENSION / max(width, height))
    preview = cv2.resize(
        rgb,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )
    image_area = preview.shape[0] * preview.shape[1]
    mask = segment_document_from_background(preview)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("could not confidently find a rectangular document boundary")

    contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(contour)
    rectangle = cv2.minAreaRect(contour)
    (_, _), (rect_width, rect_height), _ = rectangle
    rectangle_area = rect_width * rect_height
    if rectangle_area <= 0:
        raise ValueError("detected document boundary is empty")

    rectangularity = contour_area / rectangle_area
    area_ratio = rectangle_area / image_area
    if rectangularity < 0.78 or not 0.12 <= area_ratio <= 0.98:
        raise ValueError(
            "low-confidence document boundary "
            f"(rectangularity={rectangularity:.2f}, area={area_ratio:.1%})"
        )

    box = cv2.boxPoints(rectangle).astype(np.float32) / scale
    angle = rectangle_deskew_angle(box)
    return box, angle


def segment_document_from_background(preview: np.ndarray) -> np.ndarray:
    """Separate one document from a mostly uniform scanner-lid background."""
    height, width = preview.shape[:2]
    shortest_side = min(height, width)
    definite_border = max(2, round(shortest_side * 0.015))
    probable_border = max(definite_border + 1, round(shortest_side * 0.06))

    mask = np.full((height, width), cv2.GC_PR_FGD, dtype=np.uint8)
    mask[:definite_border, :] = cv2.GC_BGD
    mask[-definite_border:, :] = cv2.GC_BGD
    mask[:, :definite_border] = cv2.GC_BGD
    mask[:, -definite_border:] = cv2.GC_BGD
    mask[
        definite_border:probable_border,
        definite_border:-definite_border,
    ] = cv2.GC_PR_BGD
    mask[
        -probable_border:-definite_border,
        definite_border:-definite_border,
    ] = cv2.GC_PR_BGD
    mask[
        probable_border:-probable_border,
        definite_border:probable_border,
    ] = cv2.GC_PR_BGD
    mask[
        probable_border:-probable_border,
        -probable_border:-definite_border,
    ] = cv2.GC_PR_BGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        preview,
        mask,
        None,
        background_model,
        foreground_model,
        GRABCUT_ITERATIONS,
        cv2.GC_INIT_WITH_MASK,
    )
    document_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    closing_size = max(5, round(shortest_side * 0.035))
    if closing_size % 2 == 0:
        closing_size += 1
    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_size, closing_size)
    )
    document_mask = cv2.morphologyEx(
        document_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2
    )
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(document_mask, cv2.MORPH_OPEN, opening_kernel)


def rectangle_deskew_angle(box: np.ndarray) -> float:
    edges = np.roll(box, -1, axis=0) - box
    longest_edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    angle = float(np.degrees(np.arctan2(longest_edge[1], longest_edge[0])))
    while angle <= -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle


def rotate_crop_and_archive(
    images: list[SourceImage], archive_dir: Path, crop_margin_mm: float
) -> None:
    reserved_paths: set[Path] = set()
    for source in images:
        if not source.ok:
            continue
        assert source.image is not None
        assert source.rectangle is not None
        assert source.deskew_angle is not None
        assert source.dpi is not None

        archive_path = archive_dir / source.path.name
        if archive_path.exists() or archive_path in reserved_paths:
            source.add_error("archive", f"output already exists: {archive_path}")
            continue
        reserved_paths.add(archive_path)

        try:
            rotated, matrix = rotate_expanded(source.image, source.deskew_angle)
            transformed_rectangle = cv2.transform(
                source.rectangle.reshape(1, -1, 2), matrix
            )[0]
            margin_px = crop_margin_mm / 25.4 * source.dpi
            crop_box = crop_box_for_rectangle(
                transformed_rectangle, rotated.size, margin_px
            )
            cropped = rotated.crop(crop_box)
            save_archive_image(cropped, archive_path, source.dpi)
        except (OSError, ValueError, cv2.error) as exc:
            source.add_error("archive", str(exc))
            continue
        source.archive_path = archive_path


def rotate_expanded(image: Image.Image, angle: float) -> tuple[Image.Image, np.ndarray]:
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    output_width = int(np.ceil(height * sine + width * cosine))
    output_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += output_width / 2 - center[0]
    matrix[1, 2] += output_height / 2 - center[1]
    rotated = cv2.warpAffine(
        rgb,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(rotated), matrix


def crop_box_for_rectangle(
    rectangle: np.ndarray, image_size: tuple[int, int], margin_px: float
) -> tuple[int, int, int, int]:
    width, height = image_size
    left = max(0, int(np.floor(np.min(rectangle[:, 0]) - margin_px)))
    top = max(0, int(np.floor(np.min(rectangle[:, 1]) - margin_px)))
    right = min(width, int(np.ceil(np.max(rectangle[:, 0]) + margin_px)))
    bottom = min(height, int(np.ceil(np.max(rectangle[:, 1]) + margin_px)))
    if right <= left or bottom <= top:
        raise ValueError("detected crop rectangle is empty")
    return left, top, right, bottom


def save_archive_image(image: Image.Image, path: Path, dpi: float) -> None:
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0, dpi=(dpi, dpi))
    elif path.suffix.lower() == ".png":
        image.save(path, dpi=(dpi, dpi), optimize=True)
    else:
        raise ValueError(f"unsupported output extension: {path.suffix}")


def create_email_images(
    images: list[SourceImage], email_dir: Path, quality: int
) -> None:
    reserved_paths: set[Path] = set()
    for source in images:
        if not source.ok:
            continue
        assert source.archive_path is not None
        assert source.dpi is not None
        email_path = email_dir / f"{source.path.stem}.jpg"
        if email_path.exists() or email_path in reserved_paths:
            source.add_error("email", f"output already exists: {email_path}")
            continue
        reserved_paths.add(email_path)

        try:
            with Image.open(source.archive_path) as archive_image:
                image = archive_image.convert("RGB")
                scale = min(1.0, EMAIL_DPI / source.dpi)
                if scale < 1.0:
                    size = (
                        max(1, round(image.width * scale)),
                        max(1, round(image.height * scale)),
                    )
                    image = image.resize(size, Image.Resampling.LANCZOS)
                image.save(
                    email_path,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    dpi=(EMAIL_DPI, EMAIL_DPI),
                )
        except OSError as exc:
            source.add_error("email", str(exc))


def report_errors(images: list[SourceImage]) -> int:
    erroneous_images = [source for source in images if source.errors]
    if not erroneous_images:
        print("All images processed successfully.")
        return 0

    print("Erroneous images:", file=sys.stderr)
    for source in erroneous_images:
        print(f"- {source.path}", file=sys.stderr)
        for error in source.errors:
            print(f"  - {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skipped = [absolute_path(path) for path in args.skip]
    images = discover_sources(args.images, skipped)

    archive_dir = absolute_path(args.archive_dir)
    email_dir = absolute_path(args.email_dir)
    directory_errors = {
        archive_dir: ensure_output_directory(archive_dir),
        email_dir: ensure_output_directory(email_dir),
    }
    for directory, error in directory_errors.items():
        if error:
            for source in images:
                source.add_error(
                    "setup", f"cannot use output directory {directory}: {error}"
                )

    read_source_images(images, args.force_source_dpi)
    detect_rectangles(images)
    rotate_crop_and_archive(images, archive_dir, args.crop_margin_mm)
    create_email_images(images, email_dir, args.email_jpeg_quality)
    return report_errors(images)


if __name__ == "__main__":
    raise SystemExit(main())
