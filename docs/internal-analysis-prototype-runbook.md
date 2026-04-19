# Internal Analysis Prototype Runbook

## Server execution

1. Confirm server Python path:
   `/root/miniconda3/bin/python --version`
2. Run the current default fixed review set:
   `/root/miniconda3/bin/python scripts/run_internal_review_set.py --config configs/remote_mvp_specialized.yaml --review-set configs/review_sets/internal_analysis_v1.yaml --output-dir outputs/internal_analysis_v1_safe_consensus`
3. Run the control review set if you need the old plain-crop baseline:
   `/root/miniconda3/bin/python scripts/run_internal_review_set.py --config configs/remote_mvp_specialized_control.yaml --review-set configs/review_sets/internal_analysis_v1.yaml --output-dir outputs/internal_analysis_v1_control`

## Local report build

1. Pull the CSV/JSON/artifact directory from the server.
2. Build the report:
   `python scripts/build_internal_analysis_report.py --input-csv outputs/internal_analysis_v1_safe_consensus/review_results.csv --output-html outputs/internal_analysis_v1_safe_consensus/report.html`
3. Build the control report if you need the old baseline:
   `python scripts/build_internal_analysis_report.py --input-csv outputs/internal_analysis_v1_control/review_results.csv --output-html outputs/internal_analysis_v1_control/report.html`

## Acceptance

- Report exists
- Report shows summary metrics
- Report shows at least one failure table
- Every failure row can be traced back to crop and rectified images

## First diagnostic conclusion

- Dominant error type: `ocr_wrong_text`
- Recommended next investment: `OCR`
- Evidence: 5/5 samples failed exact-plate matching, with 4/5 samples returning non-empty but incorrect OCR text and 1/5 sample returning `ocr_null`; first-pass character accuracy was `31.43%`

## Crop padding checkpoint

- Change: specialized OCR pipeline crop padding was changed from the implicit runner default to `crop_pad_x_ratio: 0.0` and `crop_pad_y_ratio: 0.0`
- Comparison set: `internal_analysis_v1` vs `internal_analysis_v1_plaincrop`
- Exact plate accuracy improved from `0.0` to `0.4`
- Character accuracy improved from `31.43%` to `48.57%`
- Failure rows dropped from `5` to `3`
- Newly correct samples: `blur`, `fn`
- Remaining failures:
  - `db`: `ocr_null`
  - `tilt`: `ocr_null`
  - `challenge`: non-null wrong text

## Current bottleneck decision

- The harmful OCR padding path has been removed for the specialized configuration.
- The next high-value step is true plate rectification, because the model still sees rectangular crops while the OCR training set was built from perspective-warped plate images.

## Current product defaults

- specialized OCR crop padding: `0.0 / 0.0`
- specialized OCR rectification: `enabled`
- specialized OCR rectification mode: `safe`
- specialized OCR disagreement rescue: `when rectified OCR and plain OCR are both non-null but disagree, prefer plain OCR`
- specialized OCR text validation: `standard single-plate only`
- specialized OCR normalization: `conservative (no ambiguous letter-digit substitution)`
- specialized OCR candidate strategy: `original crop only`
- specialized OCR `min_confidence`: `0.93`
- safe rectification thresholds:
  - `min_area_ratio = 0.12`
  - `min_rectangularity = 0.7`
  - `max_center_offset = 0.35`
- diagnostic exports now include `ocr_confidence`

Why these defaults are current:

- zero-padding plain crops improved exact plate accuracy on the fixed review set
- the first heuristic rectification branch reduced nulls but regressed exact plate accuracy
- the older loose-path confidence audit showed that confidence alone was not a safe reject rule
- a 500-sample strict-validation audit cut wrong non-null outputs from `271` to `113` while increasing exact outputs from `93` to `97`
- a 500-sample conservative-normalization audit improved the strict baseline further from `97/113/290` to `108/105/287`
- a 500-sample original-only candidate audit reduced wrong outputs further from `105` to `80` while keeping exact outputs at `105`
- a 500-sample `min_confidence: 0.93` audit on the tightened path reduced wrong outputs again from `80` to `33`, with exact outputs at `91`
- a 500-sample safe-rectification audit reduced wrong outputs again from `33` to `22`, while keeping `blur` and `fn` exact on the fixed review set
- a fresh current-control rerun on the new server measured the real live baseline at `96/29/375`
- the current default `safe + disagreement rescue` path improved that live comparison to `79/21/400`

## Current fixed review behavior

- `db`: `ocr_null`
- `blur`: exact `皖AP0K76`
- `tilt`: `ocr_null`
- `fn`: exact `皖AX0K18`
- `challenge`: `ocr_null`

Current fixed-review metrics:

- exact plate accuracy: `0.4`
- character accuracy: `0.4`
- null rate: `0.6`

## Control And Audit Configs

- Current default remote config:
  - `configs/remote_mvp_specialized.yaml`
- Current default local config:
  - `configs/plate_ocr_specialized.local.yaml`
- Old plain-crop control configs:
- `configs/remote_mvp_specialized_control.yaml`
- `configs/plate_ocr_specialized_control.local.yaml`
- Review CSV and diagnostic JSON include:
  - `rectification_mode`
  - `rectification_applied`
  - `rectification_reason`
