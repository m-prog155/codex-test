import tempfile
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config, resolve_config_path
from car_system.pipeline.processing import process_image_file, process_video_file
from car_system.runtime import build_runtime
from car_system.ui.view_models import frame_result_to_table, frame_results_to_table


st.set_page_config(page_title="Vehicle And Plate System", layout="wide")

st.title("Vehicle And License Plate Detection System")
st.caption("PC-side MVP for vehicle and license plate detection, OCR, visualization, and result export.")

config_path = resolve_config_path()
st.write("Config file:", str(config_path))


@st.cache_resource
def get_runtime(config_file: str) -> tuple[object, object, object, object, object, object, object]:
    config = load_config(config_file)
    (
        vehicle_detector,
        plate_detector,
        ocr_engine,
        probe_ocr_engine,
        rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine,
    ) = build_runtime(config)
    return (
        config,
        vehicle_detector,
        plate_detector,
        ocr_engine,
        probe_ocr_engine,
        rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine,
    )


(
    config,
    vehicle_detector,
    plate_detector,
    ocr_engine,
    probe_ocr_engine,
    rescue_probe_ocr_engine,
    secondary_rescue_probe_ocr_engine,
) = get_runtime(str(config_path))
uploaded = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix.lower()
    session_dir = Path(tempfile.mkdtemp(prefix="car_system_"))
    source_path = session_dir / uploaded.name
    source_path.write_bytes(uploaded.getbuffer())
    output_dir = session_dir / "outputs"

    if st.button("Run Inference", type="primary"):
        with st.spinner("Running pipeline..."):
            if suffix in {".jpg", ".jpeg", ".png"}:
                artifacts = process_image_file(
                    config,
                    vehicle_detector,
                    plate_detector,
                    ocr_engine,
                    probe_ocr_engine=probe_ocr_engine,
                    rescue_probe_ocr_engine=rescue_probe_ocr_engine,
                    secondary_rescue_probe_ocr_engine=secondary_rescue_probe_ocr_engine,
                    source_path,
                    output_dir=output_dir,
                )
                st.image(str(artifacts["image_path"]), caption="Annotated Image", use_container_width=True)
                st.dataframe(pd.DataFrame(frame_result_to_table(artifacts["result"])), use_container_width=True)
                st.write("JSON:", str(artifacts["json_path"]))
                st.write("CSV:", str(artifacts["csv_path"]))
            else:
                artifacts = process_video_file(
                    config,
                    vehicle_detector,
                    plate_detector,
                    ocr_engine,
                    probe_ocr_engine=probe_ocr_engine,
                    rescue_probe_ocr_engine=rescue_probe_ocr_engine,
                    secondary_rescue_probe_ocr_engine=secondary_rescue_probe_ocr_engine,
                    source_path,
                    output_dir=output_dir,
                )
                st.video(str(artifacts["video_path"]))
                st.dataframe(pd.DataFrame(frame_results_to_table(artifacts["results"])), use_container_width=True)
                st.write("JSON:", str(artifacts["json_path"]))
                st.write("CSV:", str(artifacts["csv_path"]))
