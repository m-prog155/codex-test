import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO plate detector on CCPD-derived data.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml generated from CCPD.")
    parser.add_argument("--weights", default="yolo26n.pt", help="Pretrained YOLO weights used for transfer learning.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of fine-tuning epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--project", default="runs/plate_detector", help="Ultralytics project output directory.")
    parser.add_argument("--name", default="ccpd_yolo26n", help="Ultralytics run name.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
