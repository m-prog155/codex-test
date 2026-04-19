from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_ocr_small_sample.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_ocr_small_sample", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_common_mocks(
    module,
    monkeypatch,
    tmp_path,
    sampled_entries,
    parse_ccpd_path_handler,
    evaluate_sample_handler,
):
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class _FakeOCR:
        def __init__(self, *call_args, **call_kwargs):
            calls.append(("PaddlePlateOCR", call_args, call_kwargs))

    class _FakeBaselineOCR:
        def __init__(self, backend):
            calls.append(("BaselinePlateOCR", (backend,), {}))

    def record(name):
        def _inner(*call_args, **call_kwargs):
            calls.append((name, call_args, call_kwargs))
            if name == "sample_entries_by_subset":
                return list(sampled_entries)
            if name == "parse_ccpd_path":
                return parse_ccpd_path_handler(*call_args, **call_kwargs)
            if name == "evaluate_sample":
                return evaluate_sample_handler(*call_args, **call_kwargs)
            if name == "build_summary":
                return {
                    "sample_count": 1,
                    "skipped": call_kwargs["skipped"],
                }
            if name == "ensure_output_dir":
                return tmp_path / "out"
            if name == "write_csv":
                return call_args[0]
            if name == "write_json":
                return call_args[0]
            return None

        return _inner

    monkeypatch.setattr(module, "PaddlePlateOCR", _FakeOCR)
    monkeypatch.setattr(module, "BaselinePlateOCR", _FakeBaselineOCR)
    monkeypatch.setattr(module, "sample_entries_by_subset", record("sample_entries_by_subset"))
    monkeypatch.setattr(module, "parse_ccpd_path", record("parse_ccpd_path"))
    monkeypatch.setattr(module, "evaluate_sample", record("evaluate_sample"))
    monkeypatch.setattr(module, "build_summary", record("build_summary"))
    monkeypatch.setattr(module, "ensure_output_dir", record("ensure_output_dir"))
    monkeypatch.setattr(module, "write_csv", record("write_csv"))
    monkeypatch.setattr(module, "write_json", record("write_json"))
    monkeypatch.setattr("sys.argv", ["evaluate_ocr_small_sample.py"])
    return calls


def test_main_happy_path_wires_outputs_and_returns_zero(monkeypatch, tmp_path, capsys) -> None:
    module = _load_script_module()

    args = module.build_parser().parse_args([])
    assert args.dataset_root == Path("D:/plate_project/CCPD2019")
    assert args.split_file == Path("D:/plate_project/CCPD2019/splits/test.txt")
    assert args.subsets == [
        "ccpd_base",
        "ccpd_blur",
        "ccpd_db",
        "ccpd_rotate",
        "ccpd_tilt",
        "ccpd_weather",
        "ccpd_challenge",
    ]
    assert args.per_subset == 10
    assert args.seed == 42
    assert args.output_dir == module.PROJECT_ROOT / "outputs" / "ocr_small_sample_eval"

    calls = _configure_common_mocks(
        module,
        monkeypatch,
        tmp_path,
        [Path("ccpd_base/sample_entry.jpg"), Path("ccpd_base/skip.jpg")],
        lambda entry, **kwargs: (
            (_ for _ in ()).throw(ValueError("bad sample"))
            if entry == Path("ccpd_base/skip.jpg")
            else SimpleNamespace(
                relative_path=Path("ccpd_base/sample_annotation.jpg"),
                bbox=(1, 2, 3, 4),
                vertices=[(0, 0)] * 4,
                plate_indices=[0, 0, 0, 0, 0, 0, 0],
            )
        ),
        lambda *call_args, **call_kwargs: {
            "relative_path": "ccpd_base/sample_annotation.jpg",
            "subset": "ccpd_base",
            "baseline_exact_match": True,
            "stabilized_exact_match": True,
            "baseline_char_correct": 1,
            "stabilized_char_correct": 1,
            "char_total": 1,
            "baseline_char_accuracy": 1.0,
            "stabilized_char_accuracy": 1.0,
            "baseline_is_null": False,
            "stabilized_is_null": False,
            "baseline_confidence": 0.9,
            "stabilized_confidence": 0.95,
        },
    )

    exit_code = module.main()

    assert exit_code == 0
    assert any(
        call[0] == "sample_entries_by_subset"
        and call[1][1] == args.subsets
        and call[2] == {"per_subset": 10, "seed": 42}
        for call in calls
    )
    assert any(
        call[0] == "parse_ccpd_path"
        and call[1] == (Path("ccpd_base/sample_entry.jpg"),)
        for call in calls
    )
    assert any(
        call[0] == "PaddlePlateOCR"
        and call[2] == {"language": "ch", "use_angle_cls": False, "mode": "generic"}
        for call in calls
    )
    assert any(call[0] == "BaselinePlateOCR" for call in calls)
    assert any(
        call[0] == "evaluate_sample"
        and call[2]["relative_path"] == Path("ccpd_base/sample_annotation.jpg")
        for call in calls
    )
    assert any(
        call[0] == "build_summary"
        and call[2]["skipped"] == [{"relative_path": "ccpd_base/skip.jpg", "reason": "bad sample"}]
        for call in calls
    )
    assert any(call[0] == "ensure_output_dir" and call[1] == (module.PROJECT_ROOT / "outputs" / "ocr_small_sample_eval",) for call in calls)
    assert any(call[0] == "write_csv" for call in calls)
    assert any(
        call[0] == "write_json"
        and call[1][1]["skipped"] == [{"relative_path": "ccpd_base/skip.jpg", "reason": "bad sample"}]
        for call in calls
    )

    captured = capsys.readouterr()
    assert "samples.csv" in captured.out
    assert "summary.json" in captured.out


def test_main_records_file_not_found_error_and_passes_it_to_summary(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    calls = _configure_common_mocks(
        module,
        monkeypatch,
        tmp_path,
        [Path("ccpd_base/broken.jpg")],
        lambda entry, **kwargs: SimpleNamespace(
            relative_path=Path("ccpd_base/broken.jpg"),
            bbox=(1, 2, 3, 4),
            vertices=[(0, 0)] * 4,
            plate_indices=[0, 0, 0, 0, 0, 0, 0],
        ),
        lambda *call_args, **call_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing image")),
    )

    exit_code = module.main()

    assert exit_code == 0
    assert any(
        call[0] == "build_summary"
        and call[2]["skipped"] == [{"relative_path": "ccpd_base/broken.jpg", "reason": "missing image"}]
        for call in calls
    )
    assert any(
        call[0] == "write_json"
        and call[1][1]["skipped"] == [{"relative_path": "ccpd_base/broken.jpg", "reason": "missing image"}]
        for call in calls
    )


def test_main_raises_runtime_error_from_evaluate_sample(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    _configure_common_mocks(
        module,
        monkeypatch,
        tmp_path,
        [Path("ccpd_base/broken.jpg")],
        lambda entry, **kwargs: SimpleNamespace(
            relative_path=Path("ccpd_base/broken.jpg"),
            bbox=(1, 2, 3, 4),
            vertices=[(0, 0)] * 4,
            plate_indices=[0, 0, 0, 0, 0, 0, 0],
        ),
        lambda *call_args, **call_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        module.main()


def test_main_raises_value_error_from_evaluate_sample(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    _configure_common_mocks(
        module,
        monkeypatch,
        tmp_path,
        [Path("ccpd_base/broken.jpg")],
        lambda entry, **kwargs: SimpleNamespace(
            relative_path=Path("ccpd_base/broken.jpg"),
            bbox=(1, 2, 3, 4),
            vertices=[(0, 0)] * 4,
            plate_indices=[0, 0, 0, 0, 0, 0, 0],
        ),
        lambda *call_args, **call_kwargs: (_ for _ in ()).throw(ValueError("bad evaluation")),
    )

    with pytest.raises(ValueError, match="bad evaluation"):
        module.main()


def test_build_parser_accepts_specialized_ocr_arguments() -> None:
    module = _load_script_module()

    args = module.build_parser().parse_args(
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
    assert args.ocr_model_dir == Path("outputs/plate_ocr_runs/plate_specialized/inference")
    assert args.ocr_dict_path == Path("outputs/plate_ocr_dataset/dicts/plate_dict.txt")


def test_main_rejects_generic_mode_with_specialized_ocr_arguments(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(
                dataset_root=tmp_path,
                split_file=tmp_path / "test.txt",
                subsets=[],
                per_subset=10,
                seed=42,
                output_dir=tmp_path / "out",
                ocr_mode="generic",
                ocr_model_dir=tmp_path / "inference",
                ocr_dict_path=tmp_path / "dict.txt",
            )

        def error(self, message):
            raise SystemExit(message)

    monkeypatch.setattr(module, "build_parser", lambda: _FakeParser())

    with pytest.raises(SystemExit, match="require --ocr-mode specialized"):
        module.main()


def test_main_rejects_specialized_mode_without_model_dir(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(
                dataset_root=tmp_path,
                split_file=tmp_path / "test.txt",
                subsets=[],
                per_subset=10,
                seed=42,
                output_dir=tmp_path / "out",
                ocr_mode="specialized",
                ocr_model_dir=None,
                ocr_dict_path=tmp_path / "dict.txt",
            )

        def error(self, message):
            raise SystemExit(message)

    monkeypatch.setattr(module, "build_parser", lambda: _FakeParser())

    with pytest.raises(SystemExit, match="required when --ocr-mode specialized"):
        module.main()


def test_main_passes_specialized_ocr_arguments_to_backend(monkeypatch, tmp_path) -> None:
    module = _load_script_module()

    captured: dict[str, object] = {}
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    summary_rows: list[list[dict[str, object]]] = []

    class _FakeParser:
        def parse_args(self):
            return SimpleNamespace(
                dataset_root=tmp_path,
                split_file=tmp_path / "test.txt",
                subsets=["ccpd_base"],
                per_subset=10,
                seed=42,
                output_dir=tmp_path / "out",
                ocr_mode="specialized",
                ocr_model_dir=tmp_path / "inference",
                ocr_dict_path=tmp_path / "dict.txt",
            )

    class _FakeOCR:
        def __init__(self, *args, **kwargs):
            calls.append(("PaddlePlateOCR", args, kwargs))
            captured["kwargs"] = kwargs

    monkeypatch.setattr(module, "build_parser", lambda: _FakeParser())
    monkeypatch.setattr(module, "load_split_entries", lambda path: [])
    monkeypatch.setattr(module, "sample_entries_by_subset", lambda *args, **kwargs: [Path("ccpd_base/sample-0-0&0_10&10-0&0_10&0_10&10_0&10-0_0_0_0_0_0_0-0-0.jpg")])
    monkeypatch.setattr(
        module,
        "parse_ccpd_path",
        lambda entry: SimpleNamespace(
            relative_path=Path("ccpd_base/sample-0-0&0_10&10-0&0_10&0_10&10_0&10-0_0_0_0_0_0_0-0-0.jpg"),
            bbox=(1, 2, 3, 4),
            vertices=[(0, 0)] * 4,
            plate_indices=[0, 0, 0, 0, 0, 0, 0],
        ),
    )
    monkeypatch.setattr(
        module,
        "evaluate_sample",
        lambda *call_args, **call_kwargs: (
            calls.append(("evaluate_sample", call_args, call_kwargs))
            or {
                "relative_path": "ccpd_base/sample-0-0&0_10&10-0&0_10&0_10&10_0&10-0_0_0_0_0_0_0-0-0.jpg",
                "subset": "ccpd_base",
                "baseline_exact_match": True,
                "stabilized_exact_match": True,
                "baseline_char_correct": 7,
                "stabilized_char_correct": 7,
                "char_total": 7,
                "baseline_char_accuracy": 1.0,
                "stabilized_char_accuracy": 1.0,
                "baseline_is_null": False,
                "stabilized_is_null": False,
                "baseline_confidence": 0.9,
                "stabilized_confidence": 0.95,
            }
        ),
    )
    monkeypatch.setattr(module, "PaddlePlateOCR", _FakeOCR)
    monkeypatch.setattr(module, "BaselinePlateOCR", lambda backend: backend)
    monkeypatch.setattr(
        module,
        "build_summary",
        lambda *call_args, **call_kwargs: (
            summary_rows.append(list(call_kwargs["rows"]))
            or {
                "sample_count": len(call_kwargs["rows"]),
                "skipped": call_kwargs["skipped"],
            }
        ),
    )
    monkeypatch.setattr(module, "ensure_output_dir", lambda path: path)
    monkeypatch.setattr(module, "write_csv", lambda path, rows: path)
    monkeypatch.setattr(module, "write_json", lambda path, payload: path)
    monkeypatch.setattr("sys.argv", ["evaluate_ocr_small_sample.py"])

    assert module.main() == 0
    assert any(call[0] == "PaddlePlateOCR" for call in calls)
    assert any(call[0] == "evaluate_sample" for call in calls)
    assert captured["kwargs"] == {
        "language": "ch",
        "use_angle_cls": False,
        "mode": "specialized",
        "model_dir": str(tmp_path / "inference"),
        "character_dict_path": str(tmp_path / "dict.txt"),
    }
    assert summary_rows == [[
        {
            "relative_path": "ccpd_base/sample-0-0&0_10&10-0&0_10&0_10&10_0&10-0_0_0_0_0_0_0-0-0.jpg",
            "subset": "ccpd_base",
            "baseline_exact_match": True,
            "stabilized_exact_match": True,
            "baseline_char_correct": 7,
            "stabilized_char_correct": 7,
            "char_total": 7,
            "baseline_char_accuracy": 1.0,
            "stabilized_char_accuracy": 1.0,
            "baseline_is_null": False,
            "stabilized_is_null": False,
            "baseline_confidence": 0.9,
            "stabilized_confidence": 0.95,
        }
    ]]
