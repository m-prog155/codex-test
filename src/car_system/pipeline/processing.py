from pathlib import Path
from typing import Any

from car_system.config import AppConfig
from car_system.io.media import iter_video_frames, load_image, save_image, save_video
from car_system.io.rendering import annotate_frame
from car_system.io.writers import (
    ensure_output_dir,
    frame_result_to_dict,
    frame_result_to_rows,
    frame_results_to_dict,
    frame_results_to_rows,
    write_csv,
    write_json,
)
from car_system.pipeline.runner import PipelineRunner


def process_image_file(
    config: AppConfig,
    vehicle_detector: Any,
    plate_detector: Any,
    ocr_engine: Any,
    source_path: str | Path,
    probe_ocr_engine: Any | None = None,
    rescue_probe_ocr_engine: Any | None = None,
    secondary_rescue_probe_ocr_engine: Any | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    source = Path(source_path)
    target_dir = ensure_output_dir(output_dir or config.output.directory)
    image = load_image(source)
    runner = PipelineRunner(
        config=config,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        ocr_engine=ocr_engine,
        probe_ocr_engine=probe_ocr_engine,
        rescue_probe_ocr_engine=rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine=secondary_rescue_probe_ocr_engine,
    )
    result = runner.run_frame(image=image, source_name=source.name, frame_index=0)
    rendered = annotate_frame(image, result)

    stem = source.stem
    json_path = write_json(target_dir / f"{stem}.json", frame_result_to_dict(result))
    csv_path = write_csv(target_dir / f"{stem}.csv", frame_result_to_rows(result))
    image_path = save_image(target_dir / f"{stem}_annotated.jpg", rendered)

    return {
        "result": result,
        "json_path": json_path,
        "csv_path": csv_path,
        "image_path": image_path,
        "output_dir": target_dir,
    }


def process_video_file(
    config: AppConfig,
    vehicle_detector: Any,
    plate_detector: Any,
    ocr_engine: Any,
    source_path: str | Path,
    probe_ocr_engine: Any | None = None,
    rescue_probe_ocr_engine: Any | None = None,
    secondary_rescue_probe_ocr_engine: Any | None = None,
    output_dir: str | Path | None = None,
    fps: float = 10.0,
) -> dict[str, object]:
    source = Path(source_path)
    target_dir = ensure_output_dir(output_dir or config.output.directory)
    runner = PipelineRunner(
        config=config,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        ocr_engine=ocr_engine,
        probe_ocr_engine=probe_ocr_engine,
        rescue_probe_ocr_engine=rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine=secondary_rescue_probe_ocr_engine,
    )

    results = []
    rendered_frames = []
    for frame_index, frame in iter_video_frames(source):
        result = runner.run_frame(image=frame, source_name=source.name, frame_index=frame_index)
        results.append(result)
        rendered_frames.append(annotate_frame(frame, result))

    stem = source.stem
    json_path = write_json(target_dir / f"{stem}.json", frame_results_to_dict(results))
    csv_path = write_csv(target_dir / f"{stem}.csv", frame_results_to_rows(results))
    video_path = save_video(target_dir / f"{stem}_annotated.mp4", rendered_frames, fps=fps)

    return {
        "results": results,
        "json_path": json_path,
        "csv_path": csv_path,
        "video_path": video_path,
        "output_dir": target_dir,
    }
