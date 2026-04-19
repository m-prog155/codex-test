from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from car_system.data.ccpd import ADS, ALPHABETS, PROVINCES, decode_ccpd_plate_indices, parse_ccpd_path


def build_plate_full_text(plate_indices: Sequence[int]) -> str:
    return decode_ccpd_plate_indices(list(plate_indices))


def order_plate_vertices(vertices: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(vertices) != 4:
        raise ValueError(f"Expected 4 plate vertices, got {len(vertices)}")

    points = sorted((tuple(map(int, point)) for point in vertices), key=lambda point: (point[1], point[0]))
    top_points = sorted(points[:2], key=lambda point: point[0])
    bottom_points = sorted(points[2:], key=lambda point: point[0])
    top_left, top_right = top_points
    bottom_left, bottom_right = bottom_points
    return [top_left, top_right, bottom_right, bottom_left]


def warp_plate_from_vertices(
    image: np.ndarray,
    vertices: Sequence[tuple[int, int]],
    output_size: tuple[int, int] = (168, 48),
) -> np.ndarray:
    ordered_vertices = np.asarray(order_plate_vertices(vertices), dtype=np.float32)
    width, height = output_size
    destination = np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(ordered_vertices, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def _build_export_name(relative_path: Path) -> str:
    if len(relative_path.parts) == 1:
        return relative_path.name
    prefix = "__".join(relative_path.parts[:-1])
    return f"{prefix}__{relative_path.name}"


def export_recognition_split(
    source_root: str | Path,
    output_root: str | Path,
    split_name: str,
    entries: Sequence[Path],
    output_width: int = 168,
    output_height: int = 48,
) -> int:
    source_root_path = Path(source_root)
    output_root_path = Path(output_root)
    image_root = output_root_path / "images" / split_name
    label_path = output_root_path / f"{split_name}.txt"

    prepared_exports: list[tuple[str, np.ndarray, str]] = []
    for rel_path in entries:
        annotation = parse_ccpd_path(rel_path)
        source_image = source_root_path / annotation.relative_path
        if not source_image.exists():
            raise FileNotFoundError(f"Missing source image: {source_image}")

        image = cv2.imread(str(source_image))
        if image is None:
            raise RuntimeError(f"Failed to read source image: {source_image}")

        export_name = _build_export_name(annotation.relative_path)
        prepared_exports.append(
            (
                export_name,
                warp_plate_from_vertices(
                    image,
                    annotation.vertices,
                    output_size=(output_width, output_height),
                ),
                build_plate_full_text(annotation.plate_indices),
            )
        )

    if image_root.exists():
        shutil.rmtree(image_root)
    image_root.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    label_lines: list[str] = []
    for export_name, warped, plate_text in prepared_exports:
        target_image = image_root / export_name
        if not cv2.imwrite(str(target_image), warped):
            raise RuntimeError(f"Failed to write exported plate image: {target_image}")

        label_lines.append(f"images/{split_name}/{export_name}\t{plate_text}")

    label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return len(prepared_exports)


def write_plate_dictionary(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered_chars: list[str] = []
    seen: set[str] = set()
    for collection in (PROVINCES, ALPHABETS, ADS):
        for char in collection:
            if char == "O" or char in seen:
                continue
            seen.add(char)
            ordered_chars.append(char)

    path.write_text("\n".join(ordered_chars) + "\n", encoding="utf-8")
    return path
