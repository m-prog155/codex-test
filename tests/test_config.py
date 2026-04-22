from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config
from car_system.config import resolve_config_path


def test_load_config_reads_detector_and_output_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "app_name": "vehicle-lpr",
                "vehicle_detector": {
                    "model_path": "weights/vehicle_yolo.pt",
                    "confidence": 0.35,
                    "device": "cpu",
                    "labels": ["car", "truck"],
                },
                "plate_detector": {
                    "model_path": "weights/plate_yolo.pt",
                    "confidence": 0.25,
                    "labels": ["plate"],
                },
                "ocr": {
                    "language": "en",
                    "use_angle_cls": False,
                },
                "output": {
                    "directory": "outputs",
                    "save_images": True,
                    "save_video": False,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.app_name == "vehicle-lpr"
    assert config.vehicle_detector.model_path == "weights/vehicle_yolo.pt"
    assert config.vehicle_detector.device == "cpu"
    assert config.vehicle_detector.labels == ["car", "truck"]
    assert config.plate_detector.model_path == "weights/plate_yolo.pt"
    assert config.plate_detector.device is None
    assert config.plate_detector.labels == ["plate"]
    assert config.output.save_images is True
    assert config.output.save_video is False


def test_load_config_reads_specialized_ocr_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "mode": "specialized",
                    "model_dir": "weights/plate_rec/inference",
                    "character_dict_path": "weights/plate_rec/dicts/plate_dict.txt",
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.language == "ch"
    assert config.ocr.use_angle_cls is False
    assert config.ocr.mode == "specialized"
    assert config.ocr.model_dir == "weights/plate_rec/inference"
    assert config.ocr.character_dict_path == "weights/plate_rec/dicts/plate_dict.txt"
    assert config.ocr.enable_rectification is True
    assert config.ocr.min_confidence == 0.0
    assert config.ocr.crop_pad_x_ratio == 0.08
    assert config.ocr.crop_pad_y_ratio == 0.12


def test_load_config_reads_ocr_crop_padding_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "crop_pad_x_ratio": 0.0,
                    "crop_pad_y_ratio": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.crop_pad_x_ratio == 0.0
    assert config.ocr.crop_pad_y_ratio == 0.0


def test_load_config_reads_safe_rectification_mode_and_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "rectification_mode": "safe",
                    "safe_rect_min_area_ratio": 0.18,
                    "safe_rect_min_rectangularity": 0.72,
                    "safe_rect_max_center_offset": 0.28,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.rectification_mode == "safe"
    assert config.ocr.safe_rect_min_area_ratio == 0.18
    assert config.ocr.safe_rect_min_rectangularity == 0.72
    assert config.ocr.safe_rect_max_center_offset == 0.28


def test_load_config_reads_probe_ocr_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "mode": "specialized",
                    "model_dir": "weights/plate_rec/inference",
                    "character_dict_path": "weights/plate_rec/dicts/plate_dict.txt",
                    "min_confidence": 0.89,
                    "probe": {
                        "enabled": True,
                        "model_dir": "weights/plate_rec_probe/inference",
                        "character_dict_path": "weights/plate_rec_probe/dicts/plate_dict.txt",
                        "min_confidence": 0.95,
                        "disagreement_action": "keep_higher_confidence",
                        "disagreement_min_confidence": 0.97,
                        "disagreement_min_gap": 0.08,
                        "rescue_min_confidence": 0.99,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.probe.enabled is True
    assert config.ocr.probe.language == "ch"
    assert config.ocr.probe.use_angle_cls is False
    assert config.ocr.probe.mode == "specialized"
    assert config.ocr.probe.model_dir == "weights/plate_rec_probe/inference"
    assert config.ocr.probe.character_dict_path == "weights/plate_rec_probe/dicts/plate_dict.txt"
    assert config.ocr.probe.min_confidence == 0.95
    assert config.ocr.probe.disagreement_action == "keep_higher_confidence"
    assert config.ocr.probe.disagreement_min_confidence == 0.97
    assert config.ocr.probe.disagreement_min_gap == 0.08
    assert config.ocr.probe.rescue_min_confidence == 0.99


def test_load_config_reads_rescue_probe_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "mode": "specialized",
                    "model_dir": "weights/plate_rec/inference",
                    "character_dict_path": "weights/plate_rec/dicts/plate_dict.txt",
                    "rescue_probe": {
                        "enabled": True,
                        "model_dir": "weights/plate_rec_rescue/inference",
                        "character_dict_path": "weights/plate_rec_rescue/dicts/plate_dict.txt",
                        "min_confidence": 0.90,
                        "rescue_requires_any_char": ["D", "T", "Z"],
                        "rescue_require_alpha_count": 2,
                        "rescue_reject_repeated_required_char": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.rescue_probe.enabled is True
    assert config.ocr.rescue_probe.model_dir == "weights/plate_rec_rescue/inference"
    assert config.ocr.rescue_probe.character_dict_path == "weights/plate_rec_rescue/dicts/plate_dict.txt"
    assert config.ocr.rescue_probe.min_confidence == 0.90
    assert config.ocr.rescue_probe.rescue_requires_any_char == ("D", "T", "Z")
    assert config.ocr.rescue_probe.rescue_require_alpha_count == 2
    assert config.ocr.rescue_probe.rescue_reject_repeated_required_char is True


def test_load_config_defaults_safe_rectification_to_disabled_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ocr:\n  language: ch\n  use_angle_cls: false\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.ocr.rectification_mode == "disabled"
    assert config.ocr.safe_rect_min_area_ratio == 0.12
    assert config.ocr.safe_rect_min_rectangularity == 0.7
    assert config.ocr.safe_rect_max_center_offset == 0.35


def test_load_config_uses_consistent_default_vehicle_labels(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("app_name: defaults\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.vehicle_detector.labels == ["car", "bus", "truck", "motorcycle"]


def test_local_specialized_config_vehicle_labels_match_final_scope() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "plate_ocr_specialized.local.yaml")

    assert config.vehicle_detector.labels == ["car", "bus", "truck", "motorcycle"]
    assert config.ocr.enable_rectification is True
    assert config.ocr.rectification_mode == "safe"
    assert config.ocr.min_confidence == 0.93
    assert config.ocr.crop_pad_x_ratio == 0.0
    assert config.ocr.crop_pad_y_ratio == 0.0
    assert config.ocr.safe_rect_min_area_ratio == 0.12
    assert config.ocr.safe_rect_min_rectangularity == 0.7
    assert config.ocr.safe_rect_max_center_offset == 0.35


def test_local_specialized_config_uses_safe_rectification_by_default() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "plate_ocr_specialized.local.yaml")

    assert config.ocr.enable_rectification is True
    assert config.ocr.rectification_mode == "safe"


def test_local_specialized_control_config_preserves_plain_crop_baseline() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "plate_ocr_specialized_control.local.yaml")

    assert config.ocr.enable_rectification is False
    assert config.ocr.rectification_mode == "disabled"


def test_load_config_reads_ocr_rectification_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "enable_rectification": False,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.enable_rectification is False


def test_load_config_reads_ocr_min_confidence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "min_confidence": 0.93,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.min_confidence == 0.93


def test_default_config_vehicle_labels_match_final_scope() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")

    assert config.vehicle_detector.labels == ["car", "bus", "truck", "motorcycle"]


def test_load_config_preserves_numeric_detector_device_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "vehicle_detector": {
                    "model_path": "weights/vehicle_yolo.pt",
                    "confidence": 0.35,
                    "device": 0,
                    "labels": ["car"],
                },
                "plate_detector": {
                    "model_path": "weights/plate_yolo.pt",
                    "confidence": 0.25,
                    "device": 0,
                    "labels": ["plate"],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.vehicle_detector.device == "0"
    assert config.plate_detector.device == "0"


def test_resolve_config_path_uses_environment_override(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "remote_quick.yaml"
    config_path.write_text("app_name: remote\n", encoding="utf-8")
    monkeypatch.setenv("CAR_SYSTEM_CONFIG", str(config_path))

    resolved = resolve_config_path()

    assert resolved == config_path


def test_resolve_config_path_prefers_explicit_argument_over_environment(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / "env.yaml"
    explicit_path = tmp_path / "explicit.yaml"
    env_path.write_text("app_name: env\n", encoding="utf-8")
    explicit_path.write_text("app_name: explicit\n", encoding="utf-8")
    monkeypatch.setenv("CAR_SYSTEM_CONFIG", str(env_path))

    resolved = resolve_config_path(explicit_path)

    assert resolved == explicit_path
