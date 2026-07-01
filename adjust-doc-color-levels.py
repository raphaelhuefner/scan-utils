#!/usr/bin/env python3
"""Adjust black and white levels of scanned document images based on histogram analysis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from PIL import Image, ImageOps, UnidentifiedImageError

from common import (
    SourceImage as BaseSourceImage,
    absolute_path,
    discover_sources as discover_image_sources,
    ensure_output_directory,
)

HISTOGRAM_SIGMA = 3.0


@dataclass
class SourceImage(BaseSourceImage):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjust black and white levels of scanned document images. "
            "Detects dark-content and white-paper clusters in the HSV Value histogram "
            "and remaps pixel values so dark content becomes pure black and "
            "white paper becomes pure white."
        )
    )
    parser.add_argument("images", nargs="+", metavar="IMG_OR_DIR")
    parser.add_argument(
        "--output-dir",
        type=Path,
        metavar="OUTPUT_DIRECTORY",
        help="directory for adjusted images",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        type=Path,
        metavar="IMG_OR_DIR",
        help="skip a source file or directory tree; may be repeated",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=90,
        type=jpeg_quality,
        metavar="QUALITY",
        help="JPEG quality for JPEG outputs (default: 90)",
    )
    parser.add_argument(
        "--dark-tolerance",
        default=0.05,
        type=fraction,
        metavar="FRACTION",
        help=(
            "fraction of the dark-spike peak below which the histogram is considered "
            "to have returned to baseline; used to find the black cutoff point "
            "(default: 0.05)"
        ),
    )
    parser.add_argument(
        "--hist-dir",
        type=Path,
        metavar="HISTOGRAM_DIRECTORY",
        help="directory for histogram diagrams; if omitted, no diagrams are saved",
    )
    parser.add_argument(
        "--white-leading-fraction",
        default=0.15,
        type=fraction,
        metavar="FRACTION",
        help=(
            "fraction of the white-spike peak below which the histogram is considered "
            "to be before the white cluster; used to find the white cutoff point "
            "(default: 0.15)"
        ),
    )
    args = parser.parse_args(argv)
    apply_default_output_directory(args, parser)
    return args


def apply_default_output_directory(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.output_dir is not None:
        return
    if len(args.images) == 1:
        input_path = absolute_path(Path(args.images[0]))
        if input_path.is_dir():
            args.output_dir = input_path.parent / "color-adjusted"
            return
    parser.error("--output-dir is required unless there is exactly one input directory")


def jpeg_quality(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 100:
        raise argparse.ArgumentTypeError("must be between 1 and 100")
    return number


def fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return number


def discover_sources(arguments: list[str], skipped: list[Path]) -> list[SourceImage]:
    return discover_image_sources(arguments, SourceImage, skipped=skipped)


def smooth_histogram(histogram: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to a 1D histogram using convolution."""
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(histogram.astype(float), kernel, mode="same")


def compute_smoothed_value_histogram(image_rgb: np.ndarray) -> np.ndarray:
    """Compute the Gaussian-smoothed HSV Value histogram for an image."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2].ravel()
    histogram, _ = np.histogram(v_channel, bins=256, range=(0, 256))
    return smooth_histogram(histogram, HISTOGRAM_SIGMA)


def scaled_log_histogram(histogram: np.ndarray) -> np.ndarray:
    """Natural log of a histogram, scaled so the max value is 1.0."""
    tiny = np.finfo(float).tiny
    above_tiny = histogram[histogram > tiny]
    floor = above_tiny.min() if above_tiny.size else tiny
    log_hist = np.log(np.where(histogram > tiny, histogram, floor))
    max_log = np.max(log_hist)
    min_log = np.min(log_hist)
    return (
        (log_hist - min_log) / (max_log - min_log) if max_log != min_log else log_hist
    )


def sigmoid(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return 0.5 * (1.0 + erf((x - mu) / (sigma * math.sqrt(2))))


def model(x: np.ndarray, a_d, mu_d, sigma_d, a_w, mu_w, sigma_w, m, b) -> np.ndarray:
    dark_step = a_d * sigmoid(x, mu_d, sigma_d)
    white_step = a_w * sigmoid(x, mu_w, sigma_w)
    bridge = m * x + b
    return dark_step + white_step + bridge


def initial_guess(y: np.ndarray) -> list[float]:
    n = len(y)
    dy = np.diff(y, prepend=y[0])
    # Dark step: steepest rise in the first half; white step: steepest rise overall.
    white_step_pos = int(np.argmax(dy))
    dark_region = dy[: max(n // 2, 1)]
    dark_step_pos = int(np.argmax(dark_region)) if len(dark_region) else 0
    return [
        float(y[dark_step_pos]),  # a_d
        float(dark_step_pos),  # mu_d
        5.0,  # sigma_d
        float(y[-1] - y[dark_step_pos]),  # a_w
        (
            float(white_step_pos) if white_step_pos > dark_step_pos else float(n - 20)
        ),  # mu_w
        5.0,  # sigma_w
        0.0,  # m
        float(y[0]),  # b
    ]


def transition_points(params: np.ndarray) -> tuple[float, float]:
    """X positions where each sigmoid's local slope drops to the bridge's slope.

    Beyond these points the linear bridge term dominates the rate of change,
    i.e. the curve visually "looks like" a straight line rather than a step.
    """
    a_d, mu_d, sigma_d, a_w, mu_w, sigma_w, m, _b = params

    def z_at_slope(amplitude: float, sigma: float) -> float:
        amplitude = max(abs(amplitude), 1e-9)
        # peak density of the sigmoid derivative (a Gaussian) is amplitude / (sigma * sqrt(2*pi))
        threshold = abs(m) * sigma / amplitude
        val = threshold * math.sqrt(2 * math.pi)
        if val >= 1.0:
            return 0.0  # bridge slope is never exceeded; use the sigmoid midpoint
        return math.sqrt(-2.0 * math.log(val))

    dark_to_bridge = mu_d + z_at_slope(a_d, sigma_d) * sigma_d
    bridge_to_white = mu_w - z_at_slope(a_w, sigma_w) * sigma_w
    return dark_to_bridge, bridge_to_white


def fit_one(scaled_log_histogram: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.arange(len(scaled_log_histogram), dtype=float)
    p0 = initial_guess(scaled_log_histogram)
    n = len(scaled_log_histogram)
    bounds = (
        [-2, 0, 0.5, -2, 0, 0.5, -0.05, -2],
        [2, n, n, 2, n, n, 0.05, 2],
    )
    params, _ = curve_fit(
        model, x, scaled_log_histogram, p0=p0, bounds=bounds, maxfev=40000
    )
    fitted = model(x, *params)
    mse = float(np.mean((fitted - scaled_log_histogram) ** 2))
    return params, mse


def scaled_gradient(histogram: np.ndarray) -> np.ndarray:
    """Compute the scaled gradient of a histogram."""
    gradient = np.gradient(histogram)
    max_grad = np.max(gradient)
    scaled_grad = gradient / max_grad
    return scaled_grad


def save_histogram_diagram(
    log_histogram: np.ndarray,
    gradient: np.ndarray,
    dark_to_bridge: float,
    bridge_to_white: float,
    output_path: Path,
) -> None:
    """Plot the scaled log histogram and its gradient, and save to output_path."""
    fig, ax = plt.subplots()
    bins = np.arange(len(log_histogram))
    ax.plot(bins, log_histogram, label="scaled log histogram")
    ax.plot(bins, gradient, label="scaled gradient")
    ax.axvline(dark_to_bridge, color="blue", linewidth=0.6, linestyle="--")
    ax.axvline(bridge_to_white, color="green", linewidth=0.6, linestyle="--")
    ax.set_xlabel("HSV Value")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    output_path.with_suffix(".json").write_text(
        json.dumps(list(log_histogram), indent=2)
    )


def find_level_cutoffs(
    smoothed: np.ndarray,
    dark_tolerance: float,
    white_leading_fraction: float,
) -> tuple[int, int]:
    """
    Analyze the HSV Value histogram to find black and white cutoff levels.

    Returns (black_cutoff, white_cutoff) where:
    - pixels with V <= black_cutoff map to pure black
    - pixels with V >= white_cutoff map to pure white
    """
    # White peak: the dominant cluster at the bright end (always the largest for documents)
    white_peak_pos = int(np.argmax(smoothed))
    white_peak_val = smoothed[white_peak_pos]

    # Valley: the minimum between index 0 and the white peak separates content from paper
    valley_pos = int(np.argmin(smoothed[:white_peak_pos])) if white_peak_pos > 0 else 0

    # Black cutoff: trailing edge of the dark-content spike
    black_cutoff = 0
    if valley_pos > 1:
        dark_region = smoothed[: valley_pos + 1]
        dark_peak_pos = int(np.argmax(dark_region))
        dark_peak_val = dark_region[dark_peak_pos]

        # Scan rightward from the dark peak to find where histogram returns to baseline
        trailing = smoothed[dark_peak_pos : valley_pos + 1]
        below_tolerance = np.where(trailing <= dark_tolerance * dark_peak_val)[0]
        if len(below_tolerance) > 0:
            black_cutoff = dark_peak_pos + int(below_tolerance[0])
        else:
            # Spike never falls back to tolerance level; use the valley as fallback
            black_cutoff = valley_pos

    # White cutoff: leading edge of the white-paper spike
    # Scan leftward from the white peak to find where histogram drops below threshold
    white_cutoff = 255
    if white_peak_pos > 0:
        threshold = white_leading_fraction * white_peak_val
        # Last index (from the left) where smoothed is still below the threshold
        below_leading = np.where(smoothed[: white_peak_pos + 1] <= threshold)[0]
        if len(below_leading) > 0:
            white_cutoff = int(below_leading[-1])

    return max(0, black_cutoff), min(255, white_cutoff)


def build_levels_lut(black_cutoff: int, white_cutoff: int) -> np.ndarray:
    """Build a 256-entry lookup table that applies the level adjustment."""
    lut = np.zeros(256, dtype=np.uint8)
    if white_cutoff > black_cutoff:
        span = white_cutoff - black_cutoff
        for i in range(256):
            if i <= black_cutoff:
                lut[i] = 0
            elif i >= white_cutoff:
                lut[i] = 255
            else:
                lut[i] = round((i - black_cutoff) / span * 255)
    else:
        # Degenerate case: clamp everything to black
        lut[:] = 0
    return lut


def apply_levels(
    image_rgb: np.ndarray, black_cutoff: int, white_cutoff: int
) -> np.ndarray:
    """Apply level adjustment in HSV Value space and return adjusted RGB array."""
    lut = build_levels_lut(black_cutoff, white_cutoff)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 2] = lut[hsv[:, :, 2]]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def save_image(
    image_rgb: np.ndarray,
    path: Path,
    original_format: str | None,
    dpi: tuple[float, float] | None,
    jpeg_quality: int,
) -> None:
    pil_image = Image.fromarray(image_rgb)
    suffix = path.suffix.lower()
    save_kwargs: dict = {}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if suffix in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=jpeg_quality, subsampling=0, optimize=True)
        pil_image.save(path, format="JPEG", **save_kwargs)
    elif suffix == ".png":
        save_kwargs["optimize"] = True
        pil_image.save(path, format="PNG", **save_kwargs)
    else:
        raise ValueError(f"unsupported output extension: {path.suffix!r}")
    pil_image.close()


def process_image(
    source: SourceImage,
    output_dir: Path,
    reserved_paths: set[Path],
    jpeg_quality: int,
    dark_tolerance: float,
    white_leading_fraction: float,
    hist_dir: Path | None = None,
) -> None:
    output_path = output_dir / source.path.name
    if output_path.exists() or output_path in reserved_paths:
        source.add_error("process", f"output already exists: {output_path}")
        return
    reserved_paths.add(output_path)

    try:
        with Image.open(source.path) as opened:
            original_format = opened.format
            dpi = opened.info.get("dpi")
            transposed = ImageOps.exif_transpose(opened)
            transposed.load()
            image_rgb = np.asarray(transposed.convert("RGB"))
            transposed.close()
    except (OSError, UnidentifiedImageError) as exc:
        source.add_error("read", str(exc))
        return

    try:
        smoothed = compute_smoothed_value_histogram(image_rgb)
        log_hist = scaled_log_histogram(smoothed)
        params, mse = fit_one(log_hist)
        dark_to_bridge, bridge_to_white = transition_points(params)
    except (ValueError, RuntimeError, TypeError, cv2.error) as exc:
        source.add_error("histogram", str(exc))
        return

    if hist_dir is not None:
        gradient = scaled_gradient(log_hist)
        save_histogram_diagram(
            log_hist,
            gradient,
            dark_to_bridge,
            bridge_to_white,
            hist_dir / source.path.name,
        )

    try:
        adjusted_rgb = apply_levels(
            image_rgb, int(dark_to_bridge), int(bridge_to_white)
        )
    except (ValueError, cv2.error) as exc:
        source.add_error("analyze", str(exc))
        return
    finally:
        del image_rgb

    try:
        save_image(adjusted_rgb, output_path, original_format, dpi, jpeg_quality)
        print(
            f"{source.path.name}: mse={mse:8.4f}, dark_to_bridge={dark_to_bridge:8.4f}, bridge_to_white={bridge_to_white:8.4f} → {output_path.name}"
        )
    except OSError as exc:
        source.add_error("save", str(exc))
    finally:
        del adjusted_rgb


def report_errors(sources: list[SourceImage]) -> int:
    erroneous = [s for s in sources if s.errors]
    if not erroneous:
        print("All images processed successfully.")
        return 0
    print("Erroneous images:", file=sys.stderr)
    for source in erroneous:
        print(f"  {source.path}", file=sys.stderr)
        for error in source.errors:
            print(f"    {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skipped = [absolute_path(p) for p in args.skip]
    sources = discover_sources(args.images, skipped)

    output_dir = absolute_path(args.output_dir)
    error = ensure_output_directory(output_dir)
    if error:
        for source in sources:
            source.add_error(
                "setup", f"cannot use output directory {output_dir}: {error}"
            )
        return report_errors(sources)

    hist_dir = absolute_path(args.hist_dir) if args.hist_dir is not None else None
    if hist_dir is not None:
        error = ensure_output_directory(hist_dir)
        if error:
            for source in sources:
                source.add_error(
                    "setup", f"cannot use histogram directory {hist_dir}: {error}"
                )
            return report_errors(sources)

    reserved_paths: set[Path] = set()
    for source in sources:
        if not source.ok:
            continue
        process_image(
            source,
            output_dir,
            reserved_paths,
            jpeg_quality=args.jpeg_quality,
            dark_tolerance=args.dark_tolerance,
            white_leading_fraction=args.white_leading_fraction,
            hist_dir=hist_dir,
        )

    return report_errors(sources)


if __name__ == "__main__":
    raise SystemExit(main())
