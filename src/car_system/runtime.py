from car_system.config import AppConfig
from car_system.detectors.yolo_detector import YoloDetector
from car_system.ocr.plate_ocr import PaddlePlateOCR


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
    ocr_engine = PaddlePlateOCR(
        language=config.ocr.language,
        use_angle_cls=config.ocr.use_angle_cls,
        mode=config.ocr.mode,
        model_dir=config.ocr.model_dir,
        character_dict_path=config.ocr.character_dict_path,
        min_confidence=config.ocr.min_confidence,
    )
    return vehicle_detector, plate_detector, ocr_engine
