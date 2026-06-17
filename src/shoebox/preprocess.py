"""OpenCV preprocessing for receipt photos.

Pipeline: grayscale -> perspective crop (largest 4-point contour) -> deskew ->
optional adaptive threshold. Each step is a pure function so it can be tested in
isolation and recombined.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]

_MAX_DESKEW_DEGREES = 20.0
_MIN_DOC_AREA_RATIO = 0.2


def load_image(path: Path) -> Image:
    data = cv2.imread(str(path))
    if data is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return data.astype(np.uint8)


def to_grayscale(image: Image) -> Image:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)


def _order_points(pts: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def find_document_quad(gray: Image) -> npt.NDArray[np.float32] | None:
    """Return the 4 corner points of the dominant document, or None if not found."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = _MIN_DOC_AREA_RATIO * float(gray.shape[0] * gray.shape[1])
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def perspective_crop(image: Image, quad: npt.NDArray[np.float32]) -> Image:
    rect = _order_points(quad)
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 10 or height < 10:
        return image
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped.astype(np.uint8)


def deskew(gray: Image) -> Image:
    """Rotate small skew angles upright; skip large angles to avoid mis-rotating."""
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(inverted)
    if coords is None:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) > _MAX_DESKEW_DEGREES:
        return gray
    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated.astype(np.uint8)


def adaptive_threshold(gray: Image) -> Image:
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    ).astype(np.uint8)


def preprocess(
    image: Image,
    *,
    crop: bool = True,
    deskew_image: bool = True,
    threshold: bool = False,
) -> Image:
    """Run the full preprocessing pipeline and return a single-channel image.

    `threshold` is off by default: PP-OCR detection is trained on natural images and
    binarization tends to hurt recall. The step is implemented and toggleable so M3
    can measure the trade-off.
    """
    gray = to_grayscale(image)
    if crop:
        quad = find_document_quad(gray)
        if quad is not None:
            gray = to_grayscale(perspective_crop(image, quad))
    if deskew_image:
        gray = deskew(gray)
    if threshold:
        gray = adaptive_threshold(gray)
    return gray
