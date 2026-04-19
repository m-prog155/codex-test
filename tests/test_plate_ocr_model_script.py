from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import scripts.train_plate_ocr as train_script

import scripts.prepare_ccpd_ocr_dataset as script
import scripts.evaluate_plate_ocr_model as eval_script


def test_prepare_ccpd_ocr_dataset_build_parser_uses_expected_defaults() -> None:
    args = script.build_parser().parse_args([])

    assert args.source_root == Path("D:/plate_project/CCPD2019")
    assert args.output_root == Path("outputs/plate_ocr_dataset")
    assert args.output_width == 168
    assert args.output_height == 48


def test_prepare_ccpd_ocr_dataset_build_parser_parses_custom_output_size() -> None:
    args = script.build_parser().parse_args(["--output-width", "96", "--output-height", "32"])

    assert args.output_width == 96
    assert args.output_height == 32


def test_prepare_ccpd_ocr_dataset_main_exports_train_val_and_test_splits(monkeypatch, tmp_path: Path) -> None:
    parsed_args = Namespace(
        source_root=tmp_path / "ccpd",
        output_root=tmp_path / "out",
        output_width=96,
        output_height=32,
    )
    source_root = parsed_args.source_root
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    image = np.full((200, 220, 3), 128, dtype=np.uint8)
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(source_image), image)
    split_root = source_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (split_root / f"{split_name}.txt").write_text(f"{relative_path.as_posix()}\n", encoding="utf-8")

    monkeypatch.setattr(script, "build_parser", lambda: SimpleNamespace(parse_args=lambda *args, **kwargs: parsed_args))

    exit_code = script.main()

    expected_shape = (32, 96, 3)
    export_name = f"{relative_path.parent.name}__{relative_path.name}"
    for split_name in ("train", "val", "test"):
        exported_image = cv2.imread(str(parsed_args.output_root / "images" / split_name / export_name))
        assert exported_image is not None
        assert exported_image.shape == expected_shape
        assert (parsed_args.output_root / f"{split_name}.txt").exists()
    assert (parsed_args.output_root / "dicts" / "plate_dict.txt").exists()
    assert exit_code == 0


def test_prepare_ccpd_ocr_dataset_main_writes_plate_dictionary(monkeypatch, tmp_path: Path) -> None:
    parsed_args = Namespace(
        source_root=tmp_path / "ccpd",
        output_root=tmp_path / "out",
        output_width=96,
        output_height=32,
    )
    source_root = parsed_args.source_root
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    image = np.full((200, 220, 3), 128, dtype=np.uint8)
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(source_image), image)
    split_root = source_root / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        (split_root / f"{split_name}.txt").write_text(f"{relative_path.as_posix()}\n", encoding="utf-8")

    monkeypatch.setattr(script, "build_parser", lambda: SimpleNamespace(parse_args=lambda *args, **kwargs: parsed_args))

    script.main()

    assert (parsed_args.output_root / "dicts" / "plate_dict.txt").exists()


def test_train_plate_ocr_build_parser_uses_expected_defaults() -> None:
    args = train_script.build_parser().parse_args(["--pretrained-model", "pretrained/ppocrv5_rec"])

    assert args.paddleocr_root == Path("third_party/PaddleOCR")
    assert args.dataset_root == Path("outputs/plate_ocr_dataset")
    assert args.output_dir == Path("outputs/plate_ocr_runs/plate_specialized")
    assert args.device == "gpu"
    assert args.dry_run is False


def test_train_plate_ocr_build_train_command_includes_key_paddleocr_overrides() -> None:
    command = train_script.build_train_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        dataset_root=Path("outputs/plate_ocr_dataset"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized"),
        pretrained_model=Path("pretrained/ppocrv5_rec"),
        device="cpu",
    )

    command_text = " ".join(command)
    assert "tools/train.py" in command_text
    assert "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml" in command_text
    assert f"Global.pretrained_model={(train_script.PROJECT_ROOT / 'pretrained' / 'ppocrv5_rec').as_posix()}" in command_text
    assert f"Global.save_model_dir={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_runs/plate_specialized').as_posix()}" in command_text
    assert f"Global.character_dict_path={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset/dicts/plate_dict.txt').as_posix()}" in command_text
    assert f"Train.dataset.data_dir={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset').as_posix()}" in command_text
    assert f"Train.dataset.label_file_list=['{(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset/train.txt').as_posix()}']" in command_text
    assert f"Eval.dataset.data_dir={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset').as_posix()}" in command_text
    assert f"Eval.dataset.label_file_list=['{(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset/val.txt').as_posix()}']" in command_text
    assert "Global.use_gpu=False" in command_text


def test_train_plate_ocr_build_export_command_targets_inference_directory() -> None:
    command = train_script.build_export_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        dataset_root=Path("outputs/plate_ocr_dataset"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized"),
    )

    command_text = " ".join(command)
    assert "tools/export_model.py" in command_text
    assert "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml" in command_text
    assert f"Global.pretrained_model={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_runs/plate_specialized/best_accuracy').as_posix()}" in command_text
    assert f"Global.save_inference_dir={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_runs/plate_specialized/inference').as_posix()}" in command_text
    assert f"Global.character_dict_path={(train_script.PROJECT_ROOT / 'outputs/plate_ocr_dataset/dicts/plate_dict.txt').as_posix()}" in command_text


def test_train_plate_ocr_main_dry_run_prints_train_and_export_commands(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        train_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                paddleocr_root=Path("third_party/PaddleOCR"),
                dataset_root=Path("outputs/plate_ocr_dataset"),
                output_dir=Path("outputs/plate_ocr_runs/plate_specialized"),
                pretrained_model=Path("pretrained/ppocrv5_rec"),
                device="gpu",
                dry_run=True,
            )
        ),
    )

    exit_code = train_script.main()
    captured = capsys.readouterr().out.strip().splitlines()

    assert exit_code == 0
    assert len(captured) == 2
    assert captured[0].startswith("TRAIN:")
    assert captured[1].startswith("EXPORT:")


def test_train_plate_ocr_main_runs_train_and_export_with_stable_paths(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        train_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                paddleocr_root=Path("third_party/PaddleOCR"),
                dataset_root=Path("outputs/plate_ocr_dataset"),
                output_dir=Path("outputs/plate_ocr_runs/plate_specialized"),
                pretrained_model=Path("pretrained/ppocrv5_rec"),
                device="gpu",
                dry_run=False,
            )
        ),
    )
    monkeypatch.setattr(
        train_script.subprocess,
        "run",
        lambda command, check, cwd: calls.append({"command": command, "check": check, "cwd": cwd}) or None,
    )

    exit_code = train_script.main()

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0]["cwd"] == Path("third_party/PaddleOCR")
    assert calls[1]["cwd"] == Path("third_party/PaddleOCR")
    assert calls[0]["command"][1] == "tools/train.py"
    assert calls[0]["command"][3] == "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
    assert Path(calls[0]["command"][5].split("=", 1)[1]).is_absolute()
    assert Path(calls[0]["command"][6].split("=", 1)[1]).is_absolute()
    assert Path(calls[0]["command"][7].split("=", 1)[1]).is_absolute()
    assert Path(calls[0]["command"][8].split("=", 1)[1]).is_absolute()
    assert Path(calls[0]["command"][9].split("=", 1)[1].strip("[]'")).is_absolute()
    assert Path(calls[0]["command"][10].split("=", 1)[1]).is_absolute()
    assert Path(calls[0]["command"][11].split("=", 1)[1].strip("[]'")).is_absolute()
    assert calls[1]["command"][1] == "tools/export_model.py"
    assert Path(calls[1]["command"][5].split("=", 1)[1]).is_absolute()
    assert Path(calls[1]["command"][6].split("=", 1)[1]).is_absolute()
    assert Path(calls[1]["command"][7].split("=", 1)[1]).is_absolute()


def test_evaluate_plate_ocr_model_build_parser_uses_expected_defaults() -> None:
    parser = eval_script.build_parser()
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
    assert args.generic_model is None
    assert args.use_full_text is True
    assert args.limit is None
    assert args.output_dir == Path("outputs/plate_ocr_eval")


def test_evaluate_plate_ocr_model_main_passes_specialized_model_and_dict_to_ocr(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        eval_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                dataset_root=tmp_path / "ccpd",
                split_file=tmp_path / "splits" / "test.txt",
                generic_model=None,
                specialized_model=tmp_path / "specialized" / "inference",
                dict_path=tmp_path / "dicts" / "plate_dict.txt",
                use_full_text=True,
                limit=None,
                output_dir=tmp_path / "out",
            )
        ),
    )
    monkeypatch.setattr(eval_script, "load_split_entries", lambda path: [Path("ccpd_base/sample.jpg")])
    monkeypatch.setattr(eval_script, "parse_ccpd_path", lambda entry: SimpleNamespace(relative_path=Path(entry)))
    monkeypatch.setattr(eval_script, "evaluate_sample", lambda **kwargs: {"subset": "ccpd_base"})
    monkeypatch.setattr(eval_script, "build_summary", lambda **kwargs: {"sample_count": 1})
    monkeypatch.setattr(eval_script, "ensure_output_dir", lambda path: path)
    monkeypatch.setattr(eval_script, "write_csv", lambda path, rows: captured.setdefault("csv", (path, rows)) or path)
    monkeypatch.setattr(eval_script, "write_json", lambda path, payload: captured.setdefault("json", (path, payload)) or path)

    class _FakePaddlePlateOCR:
        def __init__(self, **kwargs):
            captured["ocr_kwargs"] = kwargs

        def load(self) -> None:
            captured["loaded"] = True

    monkeypatch.setattr(eval_script, "PaddlePlateOCR", _FakePaddlePlateOCR)

    exit_code = eval_script.main()

    assert exit_code == 0
    assert captured["ocr_kwargs"]["mode"] == "specialized"
    assert captured["ocr_kwargs"]["model_dir"] == str(tmp_path / "specialized" / "inference")
    assert captured["ocr_kwargs"]["character_dict_path"] == str(tmp_path / "dicts" / "plate_dict.txt")


def test_evaluate_plate_ocr_model_main_transfers_default_full_text_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        eval_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                dataset_root=tmp_path / "ccpd",
                split_file=tmp_path / "splits" / "test.txt",
                generic_model=None,
                specialized_model=tmp_path / "specialized" / "inference",
                dict_path=tmp_path / "dicts" / "plate_dict.txt",
                use_full_text=True,
                limit=None,
                output_dir=tmp_path / "out",
            )
        ),
    )
    monkeypatch.setattr(eval_script, "load_split_entries", lambda path: [Path("ccpd_base/sample.jpg")])
    monkeypatch.setattr(eval_script, "parse_ccpd_path", lambda entry: SimpleNamespace(relative_path=Path(entry)))
    monkeypatch.setattr(eval_script, "build_summary", lambda **kwargs: {"sample_count": 1})
    monkeypatch.setattr(eval_script, "ensure_output_dir", lambda path: path)
    monkeypatch.setattr(eval_script, "write_csv", lambda path, rows: path)
    monkeypatch.setattr(eval_script, "write_json", lambda path, payload: path)

    def fake_evaluate_sample(**kwargs):
        captured["use_full_text"] = kwargs["use_full_text"]
        return {"subset": "ccpd_base"}

    monkeypatch.setattr(eval_script, "evaluate_sample", fake_evaluate_sample)

    class _FakePaddlePlateOCR:
        def __init__(self, **kwargs):
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(eval_script, "PaddlePlateOCR", _FakePaddlePlateOCR)

    exit_code = eval_script.main()

    assert exit_code == 0
    assert captured["use_full_text"] is True


def test_evaluate_plate_ocr_model_main_transfers_eval_text_mode(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        eval_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                dataset_root=tmp_path / "ccpd",
                split_file=tmp_path / "splits" / "test.txt",
                generic_model=None,
                specialized_model=tmp_path / "specialized" / "inference",
                dict_path=tmp_path / "dicts" / "plate_dict.txt",
                use_full_text=False,
                limit=None,
                output_dir=tmp_path / "out",
            )
        ),
    )
    monkeypatch.setattr(eval_script, "load_split_entries", lambda path: [Path("ccpd_base/sample.jpg")])
    monkeypatch.setattr(eval_script, "parse_ccpd_path", lambda entry: SimpleNamespace(relative_path=Path(entry)))
    monkeypatch.setattr(eval_script, "build_summary", lambda **kwargs: {"sample_count": 1})
    monkeypatch.setattr(eval_script, "ensure_output_dir", lambda path: path)
    monkeypatch.setattr(eval_script, "write_csv", lambda path, rows: path)
    monkeypatch.setattr(eval_script, "write_json", lambda path, payload: path)

    def fake_evaluate_sample(**kwargs):
        captured["use_full_text"] = kwargs["use_full_text"]
        return {"subset": "ccpd_base"}

    monkeypatch.setattr(eval_script, "evaluate_sample", fake_evaluate_sample)

    class _FakePaddlePlateOCR:
        def __init__(self, **kwargs):
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(eval_script, "PaddlePlateOCR", _FakePaddlePlateOCR)

    exit_code = eval_script.main()

    assert exit_code == 0
    assert captured["use_full_text"] is False
