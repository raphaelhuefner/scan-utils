#!/usr/bin/env python3
"""Exploratory curve-fitting of dark-peak/bridge/white-peak model over ./hist/*.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf

HIST_DIR = Path(__file__).resolve().parent.parent / "hist"
OUTPUT_PNG = Path(__file__).resolve().parent / "fit_hist_curves_grid.png"


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
        float(white_step_pos) if white_step_pos > dark_step_pos else float(n - 20),  # mu_w
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


def fit_one(y: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.arange(len(y), dtype=float)
    p0 = initial_guess(y)
    n = len(y)
    bounds = (
        [-2, 0, 0.5, -2, 0, 0.5, -0.05, -2],
        [2, n, n, 2, n, n, 0.05, 2],
    )
    params, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=40000)
    fitted = model(x, *params)
    mse = float(np.mean((fitted - y) ** 2))
    return params, mse


def main() -> None:
    files = sorted(HIST_DIR.glob("*.json"))
    results = []
    for f in files:
        y = np.array(json.load(f.open()), dtype=float)
        try:
            params, mse = fit_one(y)
        except RuntimeError as exc:
            print(f"FIT FAILED: {f.name}: {exc}")
            continue
        results.append((f.name, y, params, mse))

    results.sort(key=lambda r: r[3], reverse=True)

    print(
        f"{'file':40s} {'a_d':>8s} {'mu_d':>7s} {'sig_d':>7s} "
        f"{'a_w':>8s} {'mu_w':>7s} {'sig_w':>7s} {'m':>8s} {'b':>8s} "
        f"{'dark->bridge':>13s} {'bridge->white':>14s} {'mse':>10s}"
    )
    for name, _, params, mse in results:
        a_d, mu_d, sigma_d, a_w, mu_w, sigma_w, m, b = params
        dark_to_bridge, bridge_to_white = transition_points(params)
        print(
            f"{name:40s} {a_d:8.4f} {mu_d:7.1f} {sigma_d:7.2f} "
            f"{a_w:8.4f} {mu_w:7.1f} {sigma_w:7.2f} {m:8.5f} {b:8.4f} "
            f"{dark_to_bridge:13.1f} {bridge_to_white:14.1f} {mse:10.6f}"
        )

    n = len(results)
    cols = 6
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.8))
    axes = np.atleast_2d(axes)
    for i, (name, y, params, mse) in enumerate(results):
        ax = axes[i // cols][i % cols]
        x = np.arange(len(y))
        ax.plot(x, y, color="black", linewidth=0.8)
        ax.plot(x, model(x, *params), color="red", linewidth=0.8)
        dark_to_bridge, bridge_to_white = transition_points(params)
        ax.axvline(dark_to_bridge, color="blue", linewidth=0.6, linestyle="--")
        ax.axvline(bridge_to_white, color="green", linewidth=0.6, linestyle="--")
        ax.set_title(f"{name}\nmse={mse:.4f}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"\nSaved fit grid to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
