# Plate-Specialized PaddleOCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plate-specialized PaddleOCR path by exporting CCPD into a recognition dataset, adding a reproducible training/evaluation workflow, and integrating the specialized recognizer into the existing runtime without breaking the current YOLO-based pipeline.

**Architecture:** Keep detection and pipeline orchestration unchanged. Add a dedicated CCPD-to-PaddleOCR dataset export path, a local training/evaluation wrapper around the official PaddleOCR recognition workflow, and a configuration-driven OCR backend so the current generic recognizer and the new plate-specialized recognizer can be compared side by side before any default switch.

**Tech Stack:** Python, CCPD2019, OpenCV, Pillow, PaddleOCR recognition models, pytest, YAML

---

## File Structure

- Create: `D:\Projects\Car\src\car_system\datasets\plate_ocr_dataset.py`
  Purpose: own CCPD plate-text decoding for recognition data export, vertex ordering, perspective warp, split export, label-list writing, and plate dictionary generation.
- Create: `D:\Projects\Car\scripts\prepare_ccpd_ocr_dataset.py`
  Purpose: provide a CLI that builds a PaddleOCR-ready recognition dataset from CCPD splits.
- Create: `D:\Projects\Car\scripts\train_plate_ocr.py`
  Purpose: provide a thin, reproducible wrapper around official PaddleOCR training/eval/export commands.
- Create: `D:\Projects\Car\scripts\evaluate_plate_ocr_model.py`
  Purpose: run offline evaluation for generic OCR vs specialized OCR on CCPD crops with full-plate and eval-text metrics.
- Create: `D:\Projects\Car\configs\plate_ocr_specialized.local.yaml`
  Purpose: local config example that enables the specialized OCR path without touching remote configs.
- Create: `D:\Projects\Car\assets\plate_ocr\README.md`
  Purpose: record the expected generated assets layout, dictionary file path, and model artifact naming.
- Modify: `D:\Projects\Car\src\car_system\config.py`
  Purpose: extend OCR config so runtime can choose generic vs specialized recognition and pass model/dict paths.
- Modify: `D:\Projects\Car\src\car_system\runtime.py`
  Purpose: instantiate `PaddlePlateOCR` with specialized model settings from config.
- Modify: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
  Purpose: support custom recognition model path, custom character dictionary, and a mode switch while preserving the current baseline behavior.
- Modify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
  Purpose: allow fixed-sample side-by-side evaluation between generic OCR and specialized OCR under the same metric code path.
- Modify: `D:\Projects\Car\scripts\evaluate_ocr_small_sample.py`
  Purpose: add CLI switches so the existing small-sample script can compare generic vs specialized OCR artifacts.
- Test: `D:\Projects\Car\tests\test_plate_ocr_dataset.py`
  Purpose: verify dataset export, perspective warp, label generation, and dictionary contents.
- Modify: `D:\Projects\Car\tests\test_config.py`
  Purpose: cover new OCR config fields.
- Modify: `D:\Projects\Car\tests\test_runtime.py`
  Purpose: verify runtime passes specialized OCR settings through.
- Modify: `D:\Projects\Car\tests\test_plate_ocr.py`
  Purpose: cover generic/specialized model initialization paths.
- Modify: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`
  Purpose: cover specialized OCR comparison in the shared evaluation path.
- Create: `D:\Projects\Car\tests\test_plate_ocr_model_script.py`
  Purpose: verify the new training/evaluation CLI wrappers stay thin and deterministic.

### Task 1: Add CCPD OCR dataset export helpers

**Files:**
- Create: `D:\Projects\Car\src\car_system\datasets\plate_ocr_dataset.py`
- Create: `D:\Projects\Car\tests\test_plate_ocr_dataset.py`
- Read: `D:\Projects\Car\src\car_system\data\ccpd.py`

- [ ] **Step 1: Write the failing tests for plate text, vertex ordering, and perspective warp**

```python
from pathlib import Path

import numpy as np

from car_system.datasets.plate_ocr_dataset import (
    build_plate_full_text,
    order_plate_vertices,
    warp_plate_from_vertices,
)


def test_build_plate_full_text_decodes_complete_plate() -> None:
    assert build_plate_full_text([0, 0, 22, 27, 27, 33, 16]) == "皖AY339S"


def test_order_plate_vertices_returns_top_left_clockwise() -> None:
    ordered = order_plate_vertices([(90, 40), (20, 20), (80, 70), (10, 60)])
    assert ordered == [(20, 20), (90, 40), (80, 70), (10, 60)]


def test_warp_plate_from_vertices_outputs_non_empty_plate_crop() -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[30:90, 40:160] = 255
    warped = warp_plate_from_vertices(
        image=image,
        vertices=[(40, 35), (160, 30), (155, 88), (35, 92)],
        output_size=(168, 48),
    )
    assert warped.shape == (48, 168, 3)
    assert int(warped.mean()) > 0
```

- [ ] **Step 2: Run the new dataset-helper tests to verify they fail for missing module/functions**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_dataset.py -q`
Expected: FAIL with import error or missing attribute error for `car_system.datasets.plate_ocr_dataset`

- [ ] **Step 3: Write the minimal helper implementation in `plate_ocr_dataset.py`**

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from car_system.data.ccpd import decode_ccpd_plate_indices


def build_plate_full_text(plate_indices: list[int]) -> str:
    return decode_ccpd_plate_indices(plate_indices)


def order_plate_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    points = np.array(vertices, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return [(int(x), int(y)) for x, y in ordered]


def warp_plate_from_vertices(
    image: np.ndarray,
    vertices: list[tuple[int, int]],
    output_size: tuple[int, int] = (168, 48),
) -> np.ndarray:
    ordered = np.array(order_plate_vertices(vertices), dtype=np.float32)
    width, height = output_size
    target = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(image, matrix, (width, height))
```

- [ ] **Step 4: Run the dataset-helper tests to verify they pass**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_dataset.py -q`
Expected: PASS

- [ ] **Step 5: Extend tests for split export and dictionary generation**

```python
from pathlib import Path

import cv2
import numpy as np

from car_system.datasets.plate_ocr_dataset import export_recognition_split, write_plate_dictionary


def test_export_recognition_split_writes_label_lines_and_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "ccpd"
    image_path = dataset_root / "ccpd_base" / "sample-0-10&10_110&40-10&10_110&10_110&40_10&40-0_0_22_27_27_33_16-0-0.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_path), np.full((80, 140, 3), 255, dtype=np.uint8))
    output_root = tmp_path / "plate_ocr"
    count = export_recognition_split(
        source_root=dataset_root,
        output_root=output_root,
        split_name="train",
        entries=[Path("ccpd_base/sample-0-10&10_110&40-10&10_110&10_110&40_10&40-0_0_22_27_27_33_16-0-0.jpg")],
    )
    label_file = output_root / "train.txt"
    assert count == 1
    assert label_file.exists()
    assert "皖AY339S" in label_file.read_text(encoding="utf-8")


def test_write_plate_dictionary_contains_provinces_letters_digits(tmp_path: Path) -> None:
    output_path = tmp_path / "plate_dict.txt"
    write_plate_dictionary(output_path)
    payload = output_path.read_text(encoding="utf-8").splitlines()
    assert "皖" in payload
    assert "A" in payload
    assert "0" in payload
    assert "警" in payload
    assert "学" in payload
```

- [ ] **Step 6: Run the new failing split-export tests**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_dataset.py -q`
Expected: FAIL with missing `export_recognition_split` and `write_plate_dictionary`

- [ ] **Step 7: Implement split export and dictionary generation**

```python
from car_system.data.ccpd import ADS, ALPHABETS, PROVINCES, parse_ccpd_path


def export_recognition_split(
    source_root: Path,
    output_root: Path,
    split_name: str,
    entries: list[Path],
    output_size: tuple[int, int] = (168, 48),
) -> int:
    image_dir = output_root / "images" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_lines: list[str] = []
    count = 0
    for rel_path in entries:
        annotation = parse_ccpd_path(rel_path)
        source_image = source_root / annotation.relative_path
        image = cv2.imread(str(source_image))
        if image is None:
            continue
        plate = warp_plate_from_vertices(image, annotation.vertices, output_size=output_size)
        export_name = "__".join(annotation.relative_path.parts)
        target_image = image_dir / export_name
        cv2.imwrite(str(target_image), plate)
        label_lines.append(f"images/{split_name}/{export_name}\t{build_plate_full_text(annotation.plate_indices)}")
        count += 1
    (output_root / f"{split_name}.txt").write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
    return count


def write_plate_dictionary(path: Path) -> Path:
    ordered_chars: list[str] = []
    for char in [*PROVINCES, *ALPHABETS, *ADS]:
        if char == "O":
            continue
        if char not in ordered_chars:
            ordered_chars.append(char)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ordered_chars) + "\n", encoding="utf-8")
    return path
```

- [ ] **Step 8: Re-run the full dataset-export test file**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_dataset.py -q`
Expected: PASS

### Task 2: Add a dataset-preparation CLI for PaddleOCR recognition

**Files:**
- Create: `D:\Projects\Car\scripts\prepare_ccpd_ocr_dataset.py`
- Modify: `D:\Projects\Car\src\car_system\datasets\plate_ocr_dataset.py`
- Test: `D:\Projects\Car\tests\test_plate_ocr_model_script.py`

- [ ] **Step 1: Write the failing CLI tests for dataset export wiring**

```python
from pathlib import Path

from scripts.prepare_ccpd_ocr_dataset import build_parser


def test_prepare_ccpd_ocr_dataset_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.source_root == Path("D:/plate_project/CCPD2019")
    assert args.output_root == Path("outputs/plate_ocr_dataset")
    assert args.output_width == 168
    assert args.output_height == 48
```

```python
from pathlib import Path

from scripts import prepare_ccpd_ocr_dataset as script


def test_prepare_ccpd_ocr_dataset_main_exports_all_three_splits(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(script, "load_split_entries", lambda path: [Path("ccpd_base/sample.jpg")])
    monkeypatch.setattr(
        script,
        "export_recognition_split",
        lambda source_root, output_root, split_name, entries, output_size: calls.append((split_name, len(entries))) or 1,
    )
    monkeypatch.setattr(script, "write_plate_dictionary", lambda path: path)
    monkeypatch.setattr(script, "build_parser", lambda: type("P", (), {"parse_args": lambda self=None: type("A", (), {
        "source_root": tmp_path / "ccpd",
        "output_root": tmp_path / "plate_ocr",
        "output_width": 168,
        "output_height": 48,
    })()})())
    assert script.main() == 0
    assert calls == [("train", 1), ("val", 1), ("test", 1)]
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k prepare_ccpd_ocr_dataset`
Expected: FAIL with import error for `prepare_ccpd_ocr_dataset`

- [ ] **Step 3: Implement the thin dataset-preparation CLI**

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import load_split_entries
from car_system.datasets.plate_ocr_dataset import export_recognition_split, write_plate_dictionary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CCPD into PaddleOCR recognition format.")
    parser.add_argument("--source-root", type=Path, default=Path("D:/plate_project/CCPD2019"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/plate_ocr_dataset"))
    parser.add_argument("--output-width", type=int, default=168)
    parser.add_argument("--output-height", type=int, default=48)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split_root = args.source_root / "splits"
    output_size = (args.output_width, args.output_height)
    for split_name in ("train", "val", "test"):
        entries = load_split_entries(split_root / f"{split_name}.txt")
        export_recognition_split(args.source_root, args.output_root, split_name, entries, output_size=output_size)
    write_plate_dictionary(args.output_root / "dicts" / "plate_dict.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k prepare_ccpd_ocr_dataset`
Expected: PASS

### Task 3: Add PaddleOCR training and export wrappers

**Files:**
- Create: `D:\Projects\Car\scripts\train_plate_ocr.py`
- Create: `D:\Projects\Car\assets\plate_ocr\README.md`
- Test: `D:\Projects\Car\tests\test_plate_ocr_model_script.py`

- [ ] **Step 1: Write the failing tests for command construction**

```python
from pathlib import Path

from scripts.train_plate_ocr import build_parser, build_train_command, build_export_command


def test_train_plate_ocr_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--pretrained-model", "pretrained/PP-OCRv5_mobile_rec"])
    assert args.paddleocr_root == Path("third_party/PaddleOCR")
    assert args.dataset_root == Path("outputs/plate_ocr_dataset")
    assert args.output_dir == Path("outputs/plate_ocr_runs/plate_specialized")


def test_build_train_command_uses_custom_label_files_and_dict() -> None:
    command = build_train_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        dataset_root=Path("outputs/plate_ocr_dataset"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized"),
        pretrained_model=Path("pretrained/PP-OCRv5_mobile_rec"),
        device="gpu",
    )
    command_text = " ".join(command)
    assert "tools/train.py" in command_text
    assert "Train.dataset.label_file_list=['outputs/plate_ocr_dataset/train.txt']" in command_text
    assert "Global.character_dict_path=outputs/plate_ocr_dataset/dicts/plate_dict.txt" in command_text
```

- [ ] **Step 2: Run the training-wrapper tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k train_plate_ocr`
Expected: FAIL with import error for `train_plate_ocr`

- [ ] **Step 3: Implement the thin training/export wrapper**

```python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a plate-specialized PaddleOCR recognizer.")
    parser.add_argument("--paddleocr-root", type=Path, default=Path("third_party/PaddleOCR"))
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/plate_ocr_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plate_ocr_runs/plate_specialized"))
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_train_command(
    paddleocr_root: Path,
    dataset_root: Path,
    output_dir: Path,
    pretrained_model: Path,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        str(paddleocr_root / "tools" / "train.py"),
        "-c",
        str(paddleocr_root / "configs" / "rec" / "PP-OCRv5" / "ch_PP-OCRv5_rec.yml"),
        "-o",
        f"Global.pretrained_model={pretrained_model}",
        f"Global.save_model_dir={output_dir}",
        f"Global.character_dict_path={dataset_root / 'dicts' / 'plate_dict.txt'}",
        f"Train.dataset.data_dir={dataset_root}",
        f"Train.dataset.label_file_list=['{dataset_root / 'train.txt'}']",
        f"Eval.dataset.data_dir={dataset_root}",
        f"Eval.dataset.label_file_list=['{dataset_root / 'val.txt'}']",
        f"Global.use_gpu={'True' if device == 'gpu' else 'False'}",
    ]


def build_export_command(paddleocr_root: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(paddleocr_root / "tools" / "export_model.py"),
        "-c",
        str(paddleocr_root / "configs" / "rec" / "PP-OCRv5" / "ch_PP-OCRv5_rec.yml"),
        "-o",
        f"Global.pretrained_model={output_dir / 'best_accuracy'}",
        f"Global.save_inference_dir={output_dir / 'inference'}",
    ]


def main() -> int:
    args = build_parser().parse_args()
    train_command = build_train_command(
        paddleocr_root=args.paddleocr_root,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        device=args.device,
    )
    export_command = build_export_command(args.paddleocr_root, args.output_dir)
    if args.dry_run:
        print("TRAIN:", " ".join(train_command))
        print("EXPORT:", " ".join(export_command))
        return 0
    subprocess.run(train_command, check=True, cwd=args.paddleocr_root)
    subprocess.run(export_command, check=True, cwd=args.paddleocr_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add the runbook file that fixes the expected asset layout**

```text
# Plate OCR Assets

Expected generated layout:

- outputs/plate_ocr_dataset/train.txt
- outputs/plate_ocr_dataset/val.txt
- outputs/plate_ocr_dataset/test.txt
- outputs/plate_ocr_dataset/dicts/plate_dict.txt
- outputs/plate_ocr_runs/plate_specialized/best_accuracy
- outputs/plate_ocr_runs/plate_specialized/inference
```

- [ ] **Step 5: Run the wrapper tests to verify they pass**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k train_plate_ocr`
Expected: PASS

### Task 4: Extend config and runtime for specialized OCR

**Files:**
- Modify: `D:\Projects\Car\src\car_system\config.py`
- Modify: `D:\Projects\Car\src\car_system\runtime.py`
- Modify: `D:\Projects\Car\tests\test_config.py`
- Modify: `D:\Projects\Car\tests\test_runtime.py`
- Create: `D:\Projects\Car\configs\plate_ocr_specialized.local.yaml`

- [ ] **Step 1: Write the failing config and runtime tests**

```python
def test_load_config_reads_specialized_ocr_fields(tmp_path: Path) -> None:
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
    assert config.ocr.mode == "specialized"
    assert config.ocr.model_dir == "weights/plate_rec/inference"
    assert config.ocr.character_dict_path == "weights/plate_rec/dicts/plate_dict.txt"
```

```python
def test_build_runtime_passes_specialized_ocr_settings(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        "car_system.runtime.PaddlePlateOCR",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or object(),
    )
    config = AppConfig(
        app_name="test-app",
        vehicle_detector=DetectorConfig(model_path="weights/vehicle.pt", confidence=0.25, labels=["car"]),
        plate_detector=DetectorConfig(model_path="weights/plate.pt", confidence=0.25, labels=["plate"]),
        ocr=OcrConfig(
            language="ch",
            use_angle_cls=False,
            mode="specialized",
            model_dir="weights/plate_rec/inference",
            character_dict_path="weights/plate_rec/dicts/plate_dict.txt",
        ),
        output=OutputConfig(directory="outputs", save_images=True, save_video=True),
    )
    build_runtime(config)
    assert captured["kwargs"]["mode"] == "specialized"
    assert captured["kwargs"]["model_dir"] == "weights/plate_rec/inference"
```

- [ ] **Step 2: Run the config/runtime tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_config.py D:\Projects\Car\tests\test_runtime.py -q`
Expected: FAIL with missing `mode`, `model_dir`, and `character_dict_path` fields on `OcrConfig`

- [ ] **Step 3: Implement the config extension**

```python
@dataclass(slots=True)
class OcrConfig:
    language: str
    use_angle_cls: bool
    mode: str = "generic"
    model_dir: str | None = None
    character_dict_path: str | None = None
```

```python
ocr = OcrConfig(
    language=str(ocr_raw.get("language", "ch")),
    use_angle_cls=bool(ocr_raw.get("use_angle_cls", False)),
    mode=str(ocr_raw.get("mode", "generic")),
    model_dir=str(ocr_raw["model_dir"]) if ocr_raw.get("model_dir") else None,
    character_dict_path=str(ocr_raw["character_dict_path"]) if ocr_raw.get("character_dict_path") else None,
)
```

```python
ocr_engine = PaddlePlateOCR(
    language=config.ocr.language,
    use_angle_cls=config.ocr.use_angle_cls,
    mode=config.ocr.mode,
    model_dir=config.ocr.model_dir,
    character_dict_path=config.ocr.character_dict_path,
)
```

- [ ] **Step 4: Add the local specialized config example**

```yaml
app_name: vehicle-license-plate-system-local-specialized-ocr
vehicle_detector:
  model_path: yolo26n.pt
  confidence: 0.35
  labels: [car, bus, truck, motorcycle]
plate_detector:
  model_path: runs/plate_detector/ccpd_yolo26n_mvp/weights/best.pt
  confidence: 0.25
  labels: [plate]
ocr:
  language: ch
  use_angle_cls: false
  mode: specialized
  model_dir: outputs/plate_ocr_runs/plate_specialized/inference
  character_dict_path: outputs/plate_ocr_dataset/dicts/plate_dict.txt
output:
  directory: outputs
  save_images: true
  save_video: true
```

- [ ] **Step 5: Re-run the config/runtime tests**

Run: `python -m pytest D:\Projects\Car\tests\test_config.py D:\Projects\Car\tests\test_runtime.py -q`
Expected: PASS

### Task 5: Extend `PaddlePlateOCR` to load specialized recognition assets

**Files:**
- Modify: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Modify: `D:\Projects\Car\tests\test_plate_ocr.py`

- [ ] **Step 1: Write the failing OCR initialization tests**

```python
from car_system.ocr.plate_ocr import PaddlePlateOCR


def test_specialized_plate_ocr_passes_model_dir_and_character_dict(monkeypatch) -> None:
    captured = {}

    class _FakeTextRecognition:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "paddleocr", type("P", (), {"TextRecognition": _FakeTextRecognition})())

    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="weights/plate_rec/inference",
        character_dict_path="weights/plate_rec/dicts/plate_dict.txt",
    )
    ocr.load()
    assert captured["model_name"] == "PP-OCRv5_mobile_rec"
    assert captured["inference_model_dir"] == "weights/plate_rec/inference"
    assert captured["character_dict_path"] == "weights/plate_rec/dicts/plate_dict.txt"
```

```python
def test_generic_plate_ocr_keeps_current_default_model(monkeypatch) -> None:
    captured = {}

    class _FakeTextRecognition:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(__import__("sys").modules, "paddleocr", type("P", (), {"TextRecognition": _FakeTextRecognition})())

    PaddlePlateOCR(language="ch", use_angle_cls=False).load()
    assert captured["model_name"] == "PP-OCRv5_mobile_rec"
    assert "inference_model_dir" not in captured
```

- [ ] **Step 2: Run the OCR tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr.py -q`
Expected: FAIL with unexpected `__init__` arguments or missing `mode` parameters

- [ ] **Step 3: Implement the specialized model loading path**

```python
class PaddlePlateOCR:
    def __init__(
        self,
        language: str = "ch",
        use_angle_cls: bool = False,
        mode: str = "generic",
        model_dir: str | None = None,
        character_dict_path: str | None = None,
    ) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.mode = mode
        self.model_dir = model_dir
        self.character_dict_path = character_dict_path
        self._ocr: Any | None = None

    def load(self) -> None:
        from paddleocr import TextRecognition

        kwargs: dict[str, Any] = {"model_name": "PP-OCRv5_mobile_rec"}
        if self.mode == "specialized":
            if not self.model_dir:
                raise ValueError("Specialized OCR mode requires model_dir.")
            kwargs["inference_model_dir"] = self.model_dir
            if self.character_dict_path:
                kwargs["character_dict_path"] = self.character_dict_path
        self._ocr = TextRecognition(**kwargs)
```

- [ ] **Step 4: Re-run the OCR test file**

Run: `python -m pytest D:\Projects\Car\tests\test_plate_ocr.py -q`
Expected: PASS

### Task 6: Add offline evaluation for specialized OCR accuracy

**Files:**
- Create: `D:\Projects\Car\scripts\evaluate_plate_ocr_model.py`
- Modify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
- Modify: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`
- Modify: `D:\Projects\Car\tests\test_plate_ocr_model_script.py`

- [ ] **Step 1: Write the failing evaluation tests for full-plate comparison**

```python
from car_system.experiments.ocr_small_sample import compare_plate_texts


def test_compare_plate_texts_supports_full_plate_metrics() -> None:
    result = compare_plate_texts(expected="皖AY339S", predicted="皖AY339S", use_full_text=True)
    assert result["exact_match"] is True
    assert result["char_correct"] == 7
    assert result["char_total"] == 7
```

```python
from pathlib import Path

from scripts.evaluate_plate_ocr_model import build_parser


def test_evaluate_plate_ocr_model_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--specialized-model",
            "outputs/plate_ocr_runs/plate_specialized/inference",
            "--dict-path",
            "outputs/plate_ocr_dataset/dicts/plate_dict.txt",
        ]
    )
    assert args.dataset_root == Path("D:/plate_project/CCPD2019")
    assert args.split_file == Path("D:/plate_project/CCPD2019/splits/test.txt")
    assert args.use_full_text is True
```

- [ ] **Step 2: Run the evaluation tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k "compare_plate_texts or evaluate_plate_ocr_model"`
Expected: FAIL with missing `compare_plate_texts` or missing script

- [ ] **Step 3: Add a shared comparison helper in `ocr_small_sample.py`**

```python
def compare_plate_texts(expected: str, predicted: str | None, use_full_text: bool) -> dict[str, float | int | bool]:
    if not use_full_text:
        expected = normalize_full_plate_for_eval(expected)
        predicted = normalize_full_plate_for_eval(predicted or "") if predicted else ""
    else:
        predicted = predicted or ""
    char_correct, char_total = compute_char_match_counts(expected, predicted)
    return {
        "exact_match": predicted == expected,
        "char_correct": char_correct,
        "char_total": char_total,
        "char_accuracy": (char_correct / char_total) if char_total else 0.0,
    }
```

- [ ] **Step 4: Implement the thin full-test evaluation CLI**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generic vs specialized plate OCR on CCPD.")
    parser.add_argument("--dataset-root", type=Path, default=Path("D:/plate_project/CCPD2019"))
    parser.add_argument("--split-file", type=Path, default=Path("D:/plate_project/CCPD2019/splits/test.txt"))
    parser.add_argument("--generic-model", type=Path, default=None)
    parser.add_argument("--specialized-model", type=Path, required=True)
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.set_defaults(use_full_text=True)
    parser.add_argument("--use-eval-text", action="store_false", dest="use_full_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/plate_ocr_eval"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    generic_ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    specialized_ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir=str(args.specialized_model),
        character_dict_path=str(args.dict_path),
    )
    entries = load_split_entries(args.split_file)
    if args.limit is not None:
        entries = entries[: args.limit]
    rows = []
    for entry in entries:
        annotation = parse_ccpd_path(entry)
        rows.append(
            evaluate_sample(
                dataset_root=args.dataset_root,
                relative_path=annotation.relative_path,
                annotation=annotation,
                baseline_ocr=generic_ocr,
                stabilized_ocr=specialized_ocr,
            )
        )
    summary = build_summary(
        rows=rows,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        subsets=[],
        per_subset=0,
        seed=0,
        skipped=[],
    )
    output_dir = ensure_output_dir(args.output_dir)
    write_csv(output_dir / "samples.csv", rows)
    write_json(output_dir / "summary.json", summary)
    return 0
```

- [ ] **Step 5: Re-run the evaluation tests**

Run: `python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py D:\Projects\Car\tests\test_plate_ocr_model_script.py -q -k "compare_plate_texts or evaluate_plate_ocr_model"`
Expected: PASS

### Task 7: Extend the existing small-sample evaluation flow for specialized OCR

**Files:**
- Modify: `D:\Projects\Car\scripts\evaluate_ocr_small_sample.py`
- Modify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
- Modify: `D:\Projects\Car\tests\test_ocr_small_sample_script.py`

- [ ] **Step 1: Write the failing script tests for specialized OCR arguments**

```python
from scripts.evaluate_ocr_small_sample import build_parser


def test_evaluate_ocr_small_sample_accepts_specialized_model_settings() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--ocr-mode",
            "specialized",
            "--ocr-model-dir",
            "outputs/plate_ocr_runs/plate_specialized/inference",
            "--ocr-dict-path",
            "outputs/plate_ocr_dataset/dicts/plate_dict.txt",
        ]
    )
    assert args.ocr_mode == "specialized"
    assert str(args.ocr_model_dir).endswith("plate_specialized/inference")
```

```python
def test_main_builds_specialized_ocr_when_requested(monkeypatch, tmp_path: Path) -> None:
    from scripts import evaluate_ocr_small_sample as script

    captured = {}
    monkeypatch.setattr(script, "load_split_entries", lambda path: [])
    monkeypatch.setattr(script, "write_csv", lambda path, rows: path)
    monkeypatch.setattr(script, "write_json", lambda path, payload: path)
    monkeypatch.setattr(script, "ensure_output_dir", lambda path: tmp_path)
    monkeypatch.setattr(
        script,
        "PaddlePlateOCR",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or object(),
    )
    monkeypatch.setattr(
        script,
        "build_parser",
        lambda: type("P", (), {"parse_args": lambda self=None: type("A", (), {
            "dataset_root": tmp_path,
            "split_file": tmp_path / "test.txt",
            "subsets": [],
            "per_subset": 10,
            "seed": 42,
            "output_dir": tmp_path,
            "ocr_mode": "specialized",
            "ocr_model_dir": tmp_path / "inference",
            "ocr_dict_path": tmp_path / "plate_dict.txt",
        })()})(),
    )
    assert script.main() == 0
    assert captured["kwargs"]["mode"] == "specialized"
```

- [ ] **Step 2: Run the small-sample script tests to verify they fail**

Run: `python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_script.py -q`
Expected: FAIL with unrecognized `--ocr-mode` arguments or incorrect `PaddlePlateOCR` call

- [ ] **Step 3: Add the specialized OCR CLI options and runtime wiring**

```python
parser.add_argument("--ocr-mode", choices=["generic", "specialized"], default="generic")
parser.add_argument("--ocr-model-dir", type=Path, default=None)
parser.add_argument("--ocr-dict-path", type=Path, default=None)
```

```python
stabilized_ocr = PaddlePlateOCR(
    language="ch",
    use_angle_cls=False,
    mode=args.ocr_mode,
    model_dir=str(args.ocr_model_dir) if args.ocr_model_dir else None,
    character_dict_path=str(args.ocr_dict_path) if args.ocr_dict_path else None,
)
```

- [ ] **Step 4: Re-run the small-sample script tests**

Run: `python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_script.py -q`
Expected: PASS

### Task 8: Run the end-to-end verification sequence

**Files:**
- Verify: `D:\Projects\Car\src\car_system\datasets\plate_ocr_dataset.py`
- Verify: `D:\Projects\Car\scripts\prepare_ccpd_ocr_dataset.py`
- Verify: `D:\Projects\Car\scripts\train_plate_ocr.py`
- Verify: `D:\Projects\Car\scripts\evaluate_plate_ocr_model.py`
- Verify: `D:\Projects\Car\src\car_system\config.py`
- Verify: `D:\Projects\Car\src\car_system\runtime.py`
- Verify: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Verify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
- Verify: `D:\Projects\Car\scripts\evaluate_ocr_small_sample.py`
- Verify: `D:\Projects\Car\tests\test_plate_ocr_dataset.py`
- Verify: `D:\Projects\Car\tests\test_plate_ocr_model_script.py`
- Verify: `D:\Projects\Car\tests\test_config.py`
- Verify: `D:\Projects\Car\tests\test_runtime.py`
- Verify: `D:\Projects\Car\tests\test_plate_ocr.py`
- Verify: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`
- Verify: `D:\Projects\Car\tests\test_ocr_small_sample_script.py`

- [ ] **Step 1: Run the targeted unit/integration tests**

Run:

```powershell
python -m pytest `
  D:\Projects\Car\tests\test_plate_ocr_dataset.py `
  D:\Projects\Car\tests\test_plate_ocr_model_script.py `
  D:\Projects\Car\tests\test_config.py `
  D:\Projects\Car\tests\test_runtime.py `
  D:\Projects\Car\tests\test_plate_ocr.py `
  D:\Projects\Car\tests\test_ocr_small_sample_eval.py `
  D:\Projects\Car\tests\test_ocr_small_sample_script.py -q
```

Expected: all listed tests pass

- [ ] **Step 2: Build a small local OCR dataset artifact**

Run:

```powershell
python D:\Projects\Car\scripts\prepare_ccpd_ocr_dataset.py `
  --source-root D:\plate_project\CCPD2019 `
  --output-root D:\Projects\Car\outputs\plate_ocr_dataset_smoke `
  --output-width 168 `
  --output-height 48
```

Expected:
- `D:\Projects\Car\outputs\plate_ocr_dataset_smoke\train.txt` exists
- `D:\Projects\Car\outputs\plate_ocr_dataset_smoke\val.txt` exists
- `D:\Projects\Car\outputs\plate_ocr_dataset_smoke\test.txt` exists
- `D:\Projects\Car\outputs\plate_ocr_dataset_smoke\dicts\plate_dict.txt` exists

- [ ] **Step 3: Dry-run the training wrapper**

Run:

```powershell
python D:\Projects\Car\scripts\train_plate_ocr.py `
  --paddleocr-root D:\Projects\Car\third_party\PaddleOCR `
  --dataset-root D:\Projects\Car\outputs\plate_ocr_dataset_smoke `
  --output-dir D:\Projects\Car\outputs\plate_ocr_runs\plate_specialized_smoke `
  --pretrained-model D:\Projects\Car\pretrained\PP-OCRv5_mobile_rec `
  --dry-run
```

Expected: prints the official PaddleOCR train/export commands without starting training

- [ ] **Step 4: Run the full local test suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 5: Perform a final plan-to-spec check before implementation completion**

Checklist:
- dataset export uses CCPD split files and perspective warp
- dictionary and full-plate labels are generated
- specialized OCR can be configured without breaking generic OCR
- specialized OCR can be evaluated on fixed sample/full test flows
- no task depends on a second OCR framework

Expected: all checklist items confirmed

---

## Self-Review

- Spec coverage: the plan covers CCPD-to-recognition data export, plate-specific dictionary creation, PaddleOCR training/export workflow, specialized config/runtime wiring, fixed-sample evaluation, and full-test evaluation.
- Placeholder scan: no `TODO`/`TBD` placeholders remain; every task lists explicit files, concrete test code, commands, and expected outcomes.
- Type consistency: `OcrConfig.mode`, `OcrConfig.model_dir`, `OcrConfig.character_dict_path`, `PaddlePlateOCR(mode=..., model_dir=..., character_dict_path=...)`, and the specialized evaluation hooks are named consistently across tasks.
