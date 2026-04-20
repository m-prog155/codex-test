from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass(slots=True)
class DetectorConfig:
    model_path: str
    confidence: float
    labels: list[str]
    device: str | None = None


@dataclass(slots=True)
class OcrConfig:
    language: str
    use_angle_cls: bool
    mode: str = "generic"
    model_dir: str | None = None
    character_dict_path: str | None = None
    min_confidence: float = 0.0
    enable_rectification: bool = True
    rectification_mode: str = "disabled"
    crop_pad_x_ratio: float = 0.08
    crop_pad_y_ratio: float = 0.12
    safe_rect_min_area_ratio: float = 0.12
    safe_rect_min_rectangularity: float = 0.70
    safe_rect_max_center_offset: float = 0.35
    probe: "OcrProbeConfig" = field(default_factory=lambda: OcrProbeConfig(language="ch", use_angle_cls=False))


@dataclass(slots=True)
class OcrProbeConfig:
    language: str = "ch"
    use_angle_cls: bool = False
    enabled: bool = False
    mode: str | None = None
    model_dir: str | None = None
    character_dict_path: str | None = None
    min_confidence: float | None = None
    disagreement_action: str = "veto"
    disagreement_min_confidence: float | None = None
    disagreement_min_gap: float = 0.0
    rescue_min_confidence: float | None = None


@dataclass(slots=True)
class OutputConfig:
    directory: str
    save_images: bool
    save_video: bool


@dataclass(slots=True)
class AppConfig:
    app_name: str
    vehicle_detector: DetectorConfig
    plate_detector: DetectorConfig
    ocr: OcrConfig
    output: OutputConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def resolve_config_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env_path = os.getenv("CAR_SYSTEM_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    raw = _read_yaml(config_path)

    vehicle_detector_raw = raw.get("vehicle_detector", raw.get("detector", {}))
    plate_detector_raw = raw.get(
        "plate_detector",
        {
            "model_path": "weights/plate_yolo.pt",
            "confidence": 0.25,
            "labels": ["plate"],
        },
    )
    ocr_raw = raw.get("ocr", {})
    probe_raw = ocr_raw.get("probe", {})
    output_raw = raw.get("output", {})
    vehicle_device_raw = vehicle_detector_raw.get("device")
    plate_device_raw = plate_detector_raw.get("device")

    vehicle_detector = DetectorConfig(
        model_path=str(vehicle_detector_raw.get("model_path", "weights/vehicle_yolo.pt")),
        confidence=float(vehicle_detector_raw.get("confidence", 0.35)),
        labels=list(vehicle_detector_raw.get("labels", vehicle_detector_raw.get("vehicle_labels", ["car", "bus", "truck", "motorcycle"]))),
        device=str(vehicle_device_raw) if vehicle_device_raw is not None else None,
    )
    plate_detector = DetectorConfig(
        model_path=str(plate_detector_raw.get("model_path", "weights/plate_yolo.pt")),
        confidence=float(plate_detector_raw.get("confidence", 0.25)),
        labels=list(plate_detector_raw.get("labels", [plate_detector_raw.get("plate_label", "plate")])),
        device=str(plate_device_raw) if plate_device_raw is not None else None,
    )
    ocr_language = str(ocr_raw.get("language", "ch"))
    ocr_use_angle_cls = bool(ocr_raw.get("use_angle_cls", False))
    ocr_mode = str(ocr_raw.get("mode", "generic"))
    ocr_min_confidence = float(ocr_raw.get("min_confidence", 0.0))

    ocr = OcrConfig(
        language=ocr_language,
        use_angle_cls=ocr_use_angle_cls,
        mode=ocr_mode,
        model_dir=str(ocr_raw["model_dir"]) if ocr_raw.get("model_dir") else None,
        character_dict_path=str(ocr_raw["character_dict_path"]) if ocr_raw.get("character_dict_path") else None,
        min_confidence=ocr_min_confidence,
        enable_rectification=bool(ocr_raw.get("enable_rectification", True)),
        rectification_mode=str(ocr_raw.get("rectification_mode", "disabled")),
        crop_pad_x_ratio=float(ocr_raw.get("crop_pad_x_ratio", 0.08)),
        crop_pad_y_ratio=float(ocr_raw.get("crop_pad_y_ratio", 0.12)),
        safe_rect_min_area_ratio=float(ocr_raw.get("safe_rect_min_area_ratio", 0.12)),
        safe_rect_min_rectangularity=float(ocr_raw.get("safe_rect_min_rectangularity", 0.70)),
        safe_rect_max_center_offset=float(ocr_raw.get("safe_rect_max_center_offset", 0.35)),
        probe=OcrProbeConfig(
            language=str(probe_raw.get("language", ocr_language)),
            use_angle_cls=bool(probe_raw.get("use_angle_cls", ocr_use_angle_cls)),
            enabled=bool(probe_raw.get("enabled", False)),
            mode=str(probe_raw.get("mode", ocr_mode)),
            model_dir=str(probe_raw["model_dir"]) if probe_raw.get("model_dir") else None,
            character_dict_path=(
                str(probe_raw["character_dict_path"]) if probe_raw.get("character_dict_path") else None
            ),
            min_confidence=float(probe_raw.get("min_confidence", ocr_min_confidence)),
            disagreement_action=str(probe_raw.get("disagreement_action", "veto")),
            disagreement_min_confidence=(
                float(probe_raw["disagreement_min_confidence"])
                if probe_raw.get("disagreement_min_confidence") is not None
                else None
            ),
            disagreement_min_gap=float(probe_raw.get("disagreement_min_gap", 0.0)),
            rescue_min_confidence=(
                float(probe_raw["rescue_min_confidence"])
                if probe_raw.get("rescue_min_confidence") is not None
                else None
            ),
        ),
    )
    output = OutputConfig(
        directory=str(output_raw.get("directory", "outputs")),
        save_images=bool(output_raw.get("save_images", True)),
        save_video=bool(output_raw.get("save_video", True)),
    )
    return AppConfig(
        app_name=str(raw.get("app_name", "vehicle-license-plate-system")),
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        ocr=ocr,
        output=output,
    )
