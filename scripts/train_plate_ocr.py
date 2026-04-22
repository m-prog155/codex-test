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
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--eval-dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-model", type=Path, required=True)
    parser.add_argument("--train-label-file", type=Path, default=None)
    parser.add_argument("--val-label-file", type=Path, default=None)
    parser.add_argument("--dict-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
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
    train_label_file: Path | None,
    val_label_file: Path | None,
    dict_path: Path | None,
    epochs: int | None,
    config_path: Path = DEFAULT_CONFIG_PATH,
    eval_dataset_root: Path | None = None,
) -> list[str]:
    train_script = Path("tools") / "train.py"
    config_path = Path(config_path)
    dataset_root = _resolve_project_path(dataset_root)
    eval_dataset_root = _resolve_project_path(eval_dataset_root) if eval_dataset_root is not None else dataset_root
    output_dir = _resolve_project_path(output_dir)
    pretrained_model = _resolve_project_path(pretrained_model)
    dict_path = _resolve_project_path(dict_path) if dict_path is not None else dataset_root / "dicts" / "plate_dict.txt"
    train_label_file = (
        _resolve_project_path(train_label_file) if train_label_file is not None else dataset_root / "train.txt"
    )
    val_label_file = (
        _resolve_project_path(val_label_file)
        if val_label_file is not None
        else eval_dataset_root / "val.txt"
    )

    command = [
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
        f"Eval.dataset.data_dir={_to_shell_path(eval_dataset_root)}",
        f"Eval.dataset.label_file_list=['{_to_shell_path(val_label_file)}']",
        f"Global.use_gpu={'True' if device == 'gpu' else 'False'}",
    ]
    if epochs is not None:
        command.append(f"Global.epoch_num={epochs}")
    return command


def build_export_command(
    paddleocr_root: Path,
    dataset_root: Path,
    output_dir: Path,
    dict_path: Path | None,
    checkpoint_path: Path | None = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[str]:
    export_script = Path("tools") / "export_model.py"
    config_path = Path(config_path)
    dataset_root = _resolve_project_path(dataset_root)
    output_dir = _resolve_project_path(output_dir)
    dict_path = _resolve_project_path(dict_path) if dict_path is not None else dataset_root / "dicts" / "plate_dict.txt"
    checkpoint_path = (
        _resolve_project_path(checkpoint_path) if checkpoint_path is not None else output_dir / "best_accuracy"
    )

    return [
        sys.executable,
        _to_shell_path(export_script),
        "-c",
        _to_shell_path(config_path),
        "-o",
        f"Global.pretrained_model={_to_shell_path(checkpoint_path)}",
        f"Global.save_inference_dir={_to_shell_path(output_dir / 'inference')}",
        f"Global.character_dict_path={_to_shell_path(dict_path)}",
    ]


def resolve_export_checkpoint(output_dir: Path) -> Path:
    output_dir = _resolve_project_path(output_dir)
    best_checkpoint = output_dir / "best_accuracy"
    latest_checkpoint = output_dir / "latest"
    if (best_checkpoint.with_suffix(".pdparams")).exists():
        return best_checkpoint
    if (latest_checkpoint.with_suffix(".pdparams")).exists():
        return latest_checkpoint
    raise FileNotFoundError(
        f"No exportable checkpoint found under {output_dir}. "
        "Expected best_accuracy.pdparams or latest.pdparams."
    )


def main() -> int:
    args = build_parser().parse_args()
    config_path = getattr(args, "config_path", DEFAULT_CONFIG_PATH)
    eval_dataset_root = getattr(args, "eval_dataset_root", None)
    train_command = build_train_command(
        paddleocr_root=args.paddleocr_root,
        config_path=config_path,
        dataset_root=args.dataset_root,
        eval_dataset_root=eval_dataset_root,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        device=args.device,
        train_label_file=args.train_label_file,
        val_label_file=args.val_label_file,
        dict_path=args.dict_path,
        epochs=args.epochs,
    )
    export_command = build_export_command(
        paddleocr_root=args.paddleocr_root,
        config_path=config_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dict_path=args.dict_path,
    )

    if args.dry_run:
        print("TRAIN:", " ".join(train_command))
        print("EXPORT:", " ".join(export_command))
        return 0

    subprocess.run(train_command, check=True, cwd=args.paddleocr_root)
    export_command = build_export_command(
        paddleocr_root=args.paddleocr_root,
        config_path=config_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        dict_path=args.dict_path,
        checkpoint_path=resolve_export_checkpoint(args.output_dir),
    )
    subprocess.run(export_command, check=True, cwd=args.paddleocr_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
