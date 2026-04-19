from pathlib import Path
from typing import Iterator


def load_image(path: str | Path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def save_image(path: str | Path, image) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")
    return output_path


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, object]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        capture.release()


def save_video(path: str | Path, frames: list[object], fps: float = 10.0) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    if not frames:
        raise ValueError("frames must not be empty")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return output_path
