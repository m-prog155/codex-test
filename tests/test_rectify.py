import cv2
import numpy as np

from car_system.ocr.rectify import rectify_plate, safe_rectify_plate


def _canonical_plate_image() -> np.ndarray:
    image = np.full((48, 168, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (167, 47), (255, 255, 255), thickness=2)
    cv2.rectangle(image, (0, 0), (11, 11), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (156, 0), (167, 11), (0, 255, 0), thickness=-1)
    cv2.rectangle(image, (156, 36), (167, 47), (255, 0, 0), thickness=-1)
    cv2.rectangle(image, (0, 36), (11, 47), (0, 255, 255), thickness=-1)
    cv2.putText(image, "A12345", (24, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return image


def _make_skewed_plate_crop() -> tuple[np.ndarray, np.ndarray]:
    canonical = _canonical_plate_image()
    vertices = np.float32([(44, 38), (185, 24), (193, 96), (32, 109)])
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [167, 0], [167, 47], [0, 47]]),
        vertices,
    )
    skewed = cv2.warpPerspective(canonical, matrix, (224, 140))
    return canonical, skewed


def test_rectify_plate_recovers_skewed_plate_close_to_canonical() -> None:
    canonical, skewed = _make_skewed_plate_crop()

    rectified = rectify_plate(skewed)

    assert rectified.shape == canonical.shape
    mean_abs_diff = np.abs(rectified.astype(np.int16) - canonical.astype(np.int16)).mean()
    assert mean_abs_diff < 35.0


def test_rectify_plate_falls_back_to_fixed_size_resize_when_no_plate_quad_found() -> None:
    image = np.zeros((31, 63, 3), dtype=np.uint8)
    image[:, :32] = (255, 255, 255)

    rectified = rectify_plate(image)

    assert rectified.shape == (48, 168, 3)


def test_safe_rectify_plate_returns_success_for_canonical_skewed_plate() -> None:
    canonical, skewed = _make_skewed_plate_crop()

    result = safe_rectify_plate(skewed)

    assert result.applied is True
    assert result.reason == "applied"
    assert result.image.shape == canonical.shape
    mean_abs_diff = np.abs(result.image.astype(np.int16) - canonical.astype(np.int16)).mean()
    assert mean_abs_diff < 35.0


def test_safe_rectify_plate_falls_back_when_no_valid_quad_found() -> None:
    image = np.zeros((31, 63, 3), dtype=np.uint8)
    image[:, :32] = (255, 255, 255)

    result = safe_rectify_plate(image)

    assert result.applied is False
    assert result.reason in {"no_quad", "low_score", "area_ratio", "rectangularity", "center_offset"}
    assert result.image.shape == image.shape


def test_safe_rectify_plate_rejects_quad_with_large_center_offset() -> None:
    image = np.zeros((48, 168, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 4), (167, 44), (255, 255, 255), thickness=-1)

    result = safe_rectify_plate(image, max_center_offset=0.10)

    assert result.applied is False
    assert result.reason == "center_offset"
