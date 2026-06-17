from __future__ import annotations

import numpy as np

from shoebox.preprocess import adaptive_threshold, deskew, preprocess, to_grayscale


def _color_image() -> np.ndarray:
    image = np.full((100, 120, 3), 255, dtype=np.uint8)
    image[40:60, 30:90] = 0
    return image


def test_to_grayscale_reduces_channels() -> None:
    gray = to_grayscale(_color_image())
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_to_grayscale_idempotent_on_gray() -> None:
    gray = to_grayscale(_color_image())
    assert to_grayscale(gray).ndim == 2


def test_adaptive_threshold_is_binary() -> None:
    out = adaptive_threshold(to_grayscale(_color_image()))
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})


def test_deskew_preserves_shape() -> None:
    gray = to_grayscale(_color_image())
    assert deskew(gray).shape == gray.shape


def test_preprocess_returns_single_channel() -> None:
    assert preprocess(_color_image(), crop=False).ndim == 2
