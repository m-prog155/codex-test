from car_system.config import AppConfig
from car_system.detectors.yolo_detector import YoloDetector
from car_system.ocr.plate_ocr import PaddlePlateOCR


def _build_ocr_engine(*, language, use_angle_cls, mode, model_dir, character_dict_path, min_confidence):
    return PaddlePlateOCR(
        language=language,
        use_angle_cls=use_angle_cls,
        mode=mode,
        model_dir=model_dir,
        character_dict_path=character_dict_path,
        min_confidence=min_confidence,
    )


def build_runtime(config: AppConfig):
    vehicle_detector = YoloDetector(
        config.vehicle_detector.model_path,
        confidence=config.vehicle_detector.confidence,
        device=config.vehicle_detector.device,
    )
    plate_detector = YoloDetector(
        config.plate_detector.model_path,
        confidence=config.plate_detector.confidence,
        device=config.plate_detector.device,
    )
    ocr_engine = _build_ocr_engine(
        language=config.ocr.language,
        use_angle_cls=config.ocr.use_angle_cls,
        mode=config.ocr.mode,
        model_dir=config.ocr.model_dir,
        character_dict_path=config.ocr.character_dict_path,
        min_confidence=config.ocr.min_confidence,
    )
    probe_ocr_engine = None
    if config.ocr.probe.enabled:
        probe_ocr_engine = _build_ocr_engine(
            language=config.ocr.probe.language,
            use_angle_cls=config.ocr.probe.use_angle_cls,
            mode=config.ocr.probe.mode or config.ocr.mode,
            model_dir=config.ocr.probe.model_dir,
            character_dict_path=config.ocr.probe.character_dict_path,
            min_confidence=(
                config.ocr.probe.min_confidence
                if config.ocr.probe.min_confidence is not None
                else config.ocr.min_confidence
            ),
        )
    rescue_probe_ocr_engine = None
    if config.ocr.rescue_probe.enabled:
        rescue_probe_ocr_engine = _build_ocr_engine(
            language=config.ocr.rescue_probe.language,
            use_angle_cls=config.ocr.rescue_probe.use_angle_cls,
            mode=config.ocr.rescue_probe.mode or config.ocr.mode,
            model_dir=config.ocr.rescue_probe.model_dir,
            character_dict_path=config.ocr.rescue_probe.character_dict_path,
            min_confidence=(
                config.ocr.rescue_probe.min_confidence
                if config.ocr.rescue_probe.min_confidence is not None
                else config.ocr.min_confidence
            ),
        )
    return vehicle_detector, plate_detector, ocr_engine, probe_ocr_engine, rescue_probe_ocr_engine
