from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_PADDLEOCR_ROOT = Path("third_party/PaddleOCR")
DEFAULT_DATASET_ROOT = Path("outputs/plate_ocr_dataset")
DEFAULT_OUTPUT_DIR = Path("outputs/plate_ocr_runs/plate_specialized")
DEFAULT_CONFIG_PATH = Path("configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export a plate-specialized PaddleOCR recognizer.")
    parser.add_argument("--paddleocr-root", type=Path, default=DEFAULT_PADDLEOCR_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _to_shell_path(path: Path) -> str:
    return Path(path).as_posix()


def _resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_train_command(
    paddleocr_root: Path,
    dataset_root: Path,
    output_dir: Path,
    pretrained_model: Path,
    device: str,
) -> list[str]:
    train_script = Path("tools") / "train.py"
    config_path = DEFAULT_CONFIG_PATH
    dataset_root = _resolve_project_path(dataset_root)
    output_dir = _resolve_project_path(output_dir)
    pretrained_model = _resolve_project_path(pretrained_model)
    dict_path = dataset_root / "dicts" / "plate_dict.txt"
    train_label_file = dataset_root / "train.txt"
    val_label_file = dataset_root / "val.txt"

    return [
        sys.executable,
        _to_shell_path(train_script),
        "-c",
        _to_shell_path(config_path),
        "-o",
        f"Global.pretrained_model={_to_shell_path(pretrained_model)}",
        f"Global.save_model_dir={_to_shell_path(output_dir)}",
        f"Global.character_dict_path={_to_shell_path(dict_path)}",
        f"Train.dataset.data_dir={_to_shell_path(dataset_root)}",
        f"Train.dataset.label_file_list=['{_to_shell_path(train_label_file)}']",
        f"Eval.dataset.data_dir={_to_shell_path(dataset_root)}",
        f"Eval.dataset.label_file_list=['{_to_shell_path(val_label_file)}']",
        f"Global.use_gpu={'True' if device == 'gpu' else 'False'}",
    ]


def build_export_command(paddleocr_root: Path, dataset_root: Path, output_dir: Path) -> list[str]:
    export_script = Path("tools") / "export_model.py"
    config_path = DEFAULT_CONFIG_PATH
    dataset_root = _resolve_project_path(dataset_root)
    output_dir = _resolve_project_path(output_dir)
    dict_path = dataset_root / "dicts" / "plate_dict.txt"

    return [
        sys.executable,
        _to_shell_path(export_script),
        "-c",
        _to_shell_path(config_path),
        "-o",
        f"Global.pretrained_model={_to_shell_path(output_dir / 'best_accuracy')}",
        f"Global.save_inference_dir={_to_shell_path(output_dir / 'inference')}",
        f"Global.character_dict_path={_to_shell_path(dict_path)}",
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
    export_command = build_export_command(args.paddleocr_root, args.dataset_root, args.output_dir)

    if args.dry_run:
        print("TRAIN:", " ".join(train_command))
        print("EXPORT:", " ".join(export_command))
        return 0

    subprocess.run(train_command, check=True, cwd=args.paddleocr_root)
    subprocess.run(export_command, check=True, cwd=args.paddleocr_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
