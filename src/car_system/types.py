from dataclasses import dataclass, field


BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox


@dataclass(slots=True)
class PlateRecognition:
    text: str
    confidence: float
    raw_text: str | None = None
    normalized_text: str | None = None


@dataclass(slots=True)
class PlateDiagnostic:
    status: str
    crop_bbox: tuple[int, int, int, int]
    crop_shape: tuple[int, ...] | None = None
    rectified_shape: tuple[int, ...] | None = None
    confidence: float | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    rectification_mode: str | None = None
    rectification_applied: bool = False
    rectification_reason: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlateMatch:
    plate: Detection
    vehicle: Detection | None
    recognition: PlateRecognition | None = None
    diagnostic: PlateDiagnostic | None = None


@dataclass(slots=True)
class FrameResult:
    source_name: str
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    matches: list[PlateMatch] = field(default_factory=list)
