from car_system.io.writers import frame_result_to_rows, frame_results_to_rows
from car_system.types import FrameResult


def frame_result_to_table(result: FrameResult) -> list[dict[str, object]]:
    return frame_result_to_rows(result)


def frame_results_to_table(results: list[FrameResult]) -> list[dict[str, object]]:
    return frame_results_to_rows(results)
