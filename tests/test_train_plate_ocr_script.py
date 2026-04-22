from pathlib import Path

import scripts.train_plate_ocr as train_script


def test_build_train_command_supports_custom_config_and_eval_dataset_root() -> None:
    command = train_script.build_train_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        config_path=Path("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"),
        dataset_root=Path("outputs/plate_ocr_incremental_v2_fulltrain"),
        eval_dataset_root=Path("outputs/plate_ocr_independent_eval_v1"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized_ppocrv5_server_v1"),
        pretrained_model=Path("weights/PP-OCRv5_server_rec_pretrained"),
        device="gpu",
        train_label_file=Path("outputs/plate_ocr_incremental_v2_fulltrain/train.txt"),
        val_label_file=Path("outputs/plate_ocr_independent_eval_v1/val.txt"),
        dict_path=Path("outputs/plate_ocr_incremental_v2_fulltrain/dicts/plate_dict.txt"),
        epochs=10,
    )

    joined = " ".join(command)
    assert "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml" in joined
    assert "Train.dataset.data_dir=" in joined
    assert "outputs/plate_ocr_incremental_v2_fulltrain" in joined
    assert "Eval.dataset.data_dir=" in joined
    assert "outputs/plate_ocr_independent_eval_v1" in joined


def test_build_train_command_defaults_eval_label_file_to_eval_dataset_root() -> None:
    command = train_script.build_train_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        config_path=Path("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"),
        dataset_root=Path("outputs/plate_ocr_incremental_v2_fulltrain"),
        eval_dataset_root=Path("outputs/plate_ocr_independent_eval_v1"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized_ppocrv5_server_v1"),
        pretrained_model=Path("weights/PP-OCRv5_server_rec_pretrained"),
        device="gpu",
        train_label_file=None,
        val_label_file=None,
        dict_path=Path("outputs/plate_ocr_incremental_v2_fulltrain/dicts/plate_dict.txt"),
        epochs=10,
    )

    joined = " ".join(command)
    expected_val = (train_script.PROJECT_ROOT / "outputs/plate_ocr_independent_eval_v1/val.txt").as_posix()
    assert f"Eval.dataset.label_file_list=['{expected_val}']" in joined


def test_build_export_command_supports_custom_config_path() -> None:
    command = train_script.build_export_command(
        paddleocr_root=Path("third_party/PaddleOCR"),
        config_path=Path("configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"),
        dataset_root=Path("outputs/plate_ocr_incremental_v2_fulltrain"),
        output_dir=Path("outputs/plate_ocr_runs/plate_specialized_ppocrv5_server_v1"),
        dict_path=Path("outputs/plate_ocr_incremental_v2_fulltrain/dicts/plate_dict.txt"),
        checkpoint_path=Path("outputs/plate_ocr_runs/plate_specialized_ppocrv5_server_v1/best_accuracy"),
    )

    joined = " ".join(command)
    assert "configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml" in joined
    assert "outputs/plate_ocr_runs/plate_specialized_ppocrv5_server_v1/inference" in joined
