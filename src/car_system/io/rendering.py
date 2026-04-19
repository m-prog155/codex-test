from typing import Any

from car_system.types import FrameResult


def annotate_frame(image: Any, result: FrameResult) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is not installed.") from exc

    frame = image.copy()

    for detection in result.detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        color = (0, 255, 0) if detection.label != "plate" else (0, 165, 255)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for match in result.matches:
        if match.recognition:
            x1, y1, _, _ = [int(value) for value in match.plate.bbox]
            text = match.recognition.text
            cv2.putText(frame, text, (x1, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    return frame
