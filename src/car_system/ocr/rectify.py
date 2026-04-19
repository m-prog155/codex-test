from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from car_system.datasets.plate_ocr_dataset import warp_plate_from_vertices


DEFAULT_OUTPUT_SIZE = (168, 48)


@dataclass(slots=True)
class RectifyResult:
    image: np.ndarray
    applied: bool
    reason: str


def _ensure_color_image(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    return array


def _resize_plate(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def _quad_diagnostics(points: np.ndarray, image_shape: tuple[int, ...]) -> dict[str, float]:
    image_height, image_width = image_shape[:2]
    image_area = max(1.0, float(image_height * image_width))
    contour_area = max(1.0, float(cv2.contourArea(points.astype(np.float32))))

    rect = cv2.minAreaRect(points.astype(np.float32))
    width, height = rect[1]
    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge <= 0 or long_edge <= 0:
        return {
            "area_ratio": 0.0,
            "rectangularity": 0.0,
            "center_offset": 1.0,
            "aspect_ratio": 0.0,
            "edge_touch_count": 4.0,
        }

    box_area = max(1.0, float(width * height))
    area_ratio = contour_area / image_area
    rectangularity = contour_area / box_area

    center_x = float(points[:, 0].mean())
    center_y = float(points[:, 1].mean())
    diagonal = max(1.0, float(np.hypot(image_width, image_height)))
    center_distance = float(np.hypot(center_x - image_width / 2.0, center_y - image_height / 2.0)) / diagonal

    min_x = float(points[:, 0].min())
    max_x = float(points[:, 0].max())
    min_y = float(points[:, 1].min())
    max_y = float(points[:, 1].max())
    edge_touch_count = sum(
        gap <= 2.0
        for gap in (
            min_x,
            min_y,
            float(image_width - 1) - max_x,
            float(image_height - 1) - max_y,
        )
    )

    return {
        "area_ratio": area_ratio,
        "rectangularity": rectangularity,
        "center_offset": center_distance,
        "aspect_ratio": long_edge / short_edge,
        "edge_touch_count": float(edge_touch_count),
    }


def _quad_score(points: np.ndarray, contour_area: float, image_shape: tuple[int, ...]) -> float:
    diagnostics = _quad_diagnostics(points, image_shape)
    aspect_ratio = diagnostics["aspect_ratio"]
    if aspect_ratio <= 0:
        return -1.0
    if aspect_ratio < 1.05 or aspect_ratio > 6.5:
        return -1.0
    return diagnostics["area_ratio"] * 3.0 + diagnostics["rectangularity"] - diagnostics["center_offset"]


def _find_plate_quad(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    min_contour_area = max(60.0, image_area * 0.12)
    best_points: np.ndarray | None = None
    best_score = -1.0

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        box_area = max(0.0, float(width * height))
        if max(float(contour_area), box_area) < min_contour_area:
            continue

        candidates: list[tuple[str, np.ndarray]] = []
        if len(approx) == 4:
            candidates.append(("approx", approx.reshape(4, 2).astype(np.float32)))

        box = cv2.boxPoints(rect).astype(np.float32)
        candidates.append(("box", box))

        for candidate_kind, candidate in candidates:
            score = _quad_score(candidate, contour_area, image.shape)
            if candidate_kind == "box" and len(approx) == 4:
                score -= 0.35
            if score > best_score:
                best_score = score
                best_points = candidate

    return best_points


def safe_rectify_plate(
    image: Any,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    min_area_ratio: float = 0.12,
    min_rectangularity: float = 0.70,
    max_center_offset: float = 0.35,
) -> RectifyResult:
    color_image = _ensure_color_image(image)
    if color_image.size == 0:
        return RectifyResult(image=color_image, applied=False, reason="empty")

    plate_quad = _find_plate_quad(color_image)
    if plate_quad is None:
        return RectifyResult(image=color_image, applied=False, reason="no_quad")

    diagnostics = _quad_diagnostics(plate_quad, color_image.shape)
    if diagnostics["aspect_ratio"] < 1.1 or diagnostics["aspect_ratio"] > 6.5:
        return RectifyResult(image=color_image, applied=False, reason="low_score")
    if diagnostics["center_offset"] > max_center_offset:
        return RectifyResult(image=color_image, applied=False, reason="center_offset")
    if diagnostics["edge_touch_count"] >= 3.0:
        return RectifyResult(image=color_image, applied=False, reason="low_score")
    if diagnostics["area_ratio"] < min_area_ratio:
        return RectifyResult(image=color_image, applied=False, reason="area_ratio")
    if diagnostics["rectangularity"] < min_rectangularity:
        return RectifyResult(image=color_image, applied=False, reason="rectangularity")

    rectified = warp_plate_from_vertices(color_image, [(int(x), int(y)) for x, y in plate_quad], output_size=output_size)
    if rectified.size == 0:
        return RectifyResult(image=color_image, applied=False, reason="warp_failed")
    return RectifyResult(image=rectified, applied=True, reason="applied")


def rectify_plate(image: Any, output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE) -> Any:
    color_image = _ensure_color_image(image)
    if color_image.size == 0:
        width, height = output_size
        return np.zeros((height, width, 3), dtype=np.uint8)
    result = safe_rectify_plate(color_image, output_size=output_size)
    if not result.applied:
        return _resize_plate(color_image, output_size)
    return result.image
