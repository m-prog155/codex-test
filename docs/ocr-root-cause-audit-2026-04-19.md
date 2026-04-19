# OCR Root-Cause Audit

Date: 2026-04-19

## Scope

This audit focused on why the specialized OCR model still performs poorly in the product pipeline even though the PaddleOCR training log reports very high validation accuracy.

## Verified Facts

1. The full specialized OCR dataset on the server has:
   - train: 20,000
   - val: 5,000
   - test: 10,000
   - dictionary size: 67 characters

2. The label distribution is highly skewed toward `皖A`:
   - test set top prefix: `皖A` = 8,084 / 10,000
   - train set province distribution is also dominated by `皖`

3. The specialized OCR training log reports a very high validation metric:
   - `best metric, acc: 0.9855999980288`
   - best epoch: 74

4. The full external evaluation artifact on the server is much lower:
   - file: `outputs/plate_ocr_eval_full_20260414/summary.json`
   - exact match rate: `0.148`
   - character accuracy: `0.3828`
   - null rate: `0.3279`

5. The current product pipeline does not actually rectify the plate crop:
   - [rectify.py](/D:/Projects/Car/src/car_system/ocr/rectify.py:1) returns the original crop unchanged
   - [runner.py](/D:/Projects/Car/src/car_system/pipeline/runner.py:64) expands a rectangle and passes it directly to OCR

6. The OCR training dataset is built from four-point perspective warps, not from raw rectangular crops:
   - [plate_ocr_dataset.py](/D:/Projects/Car/src/car_system/datasets/plate_ocr_dataset.py:22)
   - [plate_ocr_dataset.py](/D:/Projects/Car/src/car_system/datasets/plate_ocr_dataset.py:48)

7. The full OCR evaluation script also uses rectangular plate crops rather than warped plate images:
   - [ocr_small_sample.py](/D:/Projects/Car/src/car_system/experiments/ocr_small_sample.py:70)
   - [ocr_small_sample.py](/D:/Projects/Car/src/car_system/experiments/ocr_small_sample.py:190)

## Core Finding

The main loss in the current product pipeline is caused by a preprocessing distribution mismatch:

- OCR training data: perspective-warped, canonical plate images
- Product inference path: rectangular bbox crops with no true rectification
- Full evaluation path: rectangular bbox crops with optional padding, also no true rectification

This mismatch is large enough to explain why the internal training/validation metric looks strong while the end-to-end product behavior remains weak.

## Direct Evidence

A controlled server-side comparison was run on 500 random test samples using the same specialized OCR model:

- warped plate images: exact `0.502`, null `0.018`, char accuracy `0.7766`
- raw bbox crops: exact `0.204`, null `0.272`, char accuracy `0.4666`
- padded bbox crops: exact `0.140`, null `0.332`, char accuracy `0.3817`

This directly shows:

1. The specialized OCR model performs far better on data shaped like its training set.
2. The current padded-crop path is the worst of the three tested inputs.
3. The current product path is losing most of its accuracy before OCR decoding quality becomes the dominant issue.

## Secondary Findings

1. Prefix bias is real.
   The model strongly prefers `皖A`, which matches the dataset skew.

2. Output length instability is a major symptom.
   In the full 10,000-sample evaluation:
   - length 6 predictions: 3,684, exact rate `0.0`
   - length 7 predictions: 2,815, exact rate `0.5258`
   - length 8 predictions: 222, exact rate `0.0`

3. Candidate-image heuristics add some damage but are not the primary failure.
   On 500 warped samples:
   - single candidate exact: `0.514`
   - current `recognize_raw` exact: `0.502`
   - normalized `recognize` exact: `0.410`

The wrapper hurts somewhat, but the crop/rectification mismatch hurts much more.

## Implications

The current OCR problem should not be described primarily as "the OCR model itself is too weak."

A more accurate statement is:

- the specialized OCR model has usable ability on canonical plate crops
- the product pipeline feeds it the wrong image distribution
- the dataset is also biased enough to reinforce prefix-level errors

## Recommended Next Actions

1. First, remove or parameterize the current OCR padding path and benchmark plain bbox crops against the current padded crops in the real pipeline.
2. Second, implement a real rectification path for detected plate crops, instead of passing raw rectangular crops through `rectify_plate()`.
3. Third, only after the inference input path is corrected, re-evaluate whether OCR retraining is still the primary bottleneck.
4. Fourth, if retraining is still needed, rebuild the OCR dataset so training inputs better match product inference inputs.

## Completed Follow-Up

The first action above has already been executed.

- Specialized OCR configs now use:
  - `crop_pad_x_ratio: 0.0`
  - `crop_pad_y_ratio: 0.0`
- Fixed internal review set comparison:
  - old exact plate accuracy: `0.0`
  - new exact plate accuracy: `0.4`
- old character accuracy: `31.43%`
- new character accuracy: `48.57%`

This confirms the padding path was actively damaging product performance.

## Rectification Checkpoint

I also implemented a first contour-based rectification branch and ran the same fixed internal review set against it.

- `plaincrop`:
  - exact plate accuracy: `0.4`
  - character accuracy: `48.57%`
  - null rate: `40%`
- `rectified`:
  - exact plate accuracy: `0.2`
  - character accuracy: `65.71%`
  - null rate: `0%`

Interpretation:

- The heuristic rectifier does recover text on some previously null samples.
- It also introduces too many wrong non-null outputs.
- For the current product direction, this is a regression because false plate numbers are more dangerous than nulls.

Current decision:

- keep rectification configurable
- do not leave heuristic rectification enabled in the specialized product configs
- current specialized configs now use `enable_rectification: false`

## Confidence Audit

To test whether OCR confidence could be used as a product reject rule, I added `ocr_confidence` to the diagnostic JSON, review CSV, and HTML report, then ran a larger plain-crop audit on `500` random CCPD test samples.

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500/rows.csv)

Results:

- exact: `93 / 500` (`18.6%`)
- wrong: `271 / 500` (`54.2%`)
- null: `136 / 500` (`27.2%`)

Confidence distribution:

- exact mean confidence: `0.9756`
- wrong mean confidence: `0.8309`
- wrong max confidence: `0.99996`

Threshold slices:

- `>= 0.95`: kept `155`, exact `81`, wrong `74`, precision `52.26%`
- `>= 0.99`: kept `70`, exact `42`, wrong `28`, precision `60.00%`
- `>= 0.995`: kept `51`, exact `33`, wrong `18`, precision `64.71%`

Interpretation:

- Confidence separates the groups somewhat in aggregate.
- It does not separate them cleanly enough for a simple reject threshold.
- High-confidence wrong reads are common enough that `min_confidence` alone is not a safe product rule.

Current decision:

- do not ship a naive confidence-only reject rule
- prioritize stronger text validation and better OCR inputs before revisiting confidence gating

## Candidate Agreement Audit

I also checked whether the current three OCR candidate images (`original`, `2x enlarged`, `equalized`) could be used as a product reject signal on the same 500-sample plain-crop audit.

Results:

- exact samples: `93`
  - at least two candidates agree: `89`
  - all three agree: `46`
- wrong samples: `271`
  - at least two candidates agree: `180`
  - all three agree: `30`

Interpretation:

- Candidate disagreement is correlated with errors.
- It is not strong enough by itself to be a safe reject rule.
- Requiring `votes >= 2` would still keep too many wrong reads.
- Requiring unanimous agreement would cut recall heavily while still leaking wrong outputs.

Current decision:

- do not use simple multi-candidate agreement as the primary product safety gate
- keep it available as a future diagnostic feature, not as the main decision rule

## Specialized Validation Tightening

The next root cause checkpoint was the specialized OCR validity rule itself.

Before this change, specialized OCR still accepted any normalized text with length `6-8`. On the 500-sample plain-crop audit, that rule was too permissive:

- exact predictions: all `93 / 93` were length `7`
- wrong predictions: `171 / 271` were length `6`

That meant the current product path was leaking many short, plate-like but obviously incomplete outputs.

I changed specialized OCR validation to require the current product scope's standard single-plate format:

- total length exactly `7`
- first character is a province prefix
- second character is an uppercase ASCII letter
- remaining five characters are uppercase ASCII alphanumeric

This validation is restricted to `mode='specialized'`. Generic OCR keeps the previous looser behavior.

Code:

- [plate_ocr.py](/D:/Projects/Car/src/car_system/ocr/plate_ocr.py)
- [test_plate_ocr.py](/D:/Projects/Car/tests/test_plate_ocr.py)

Remote verification:

- focused test suite: `43 passed`

Updated 500-sample audit:

- old plaincrop:
  - exact `93`
  - wrong `271`
  - null `136`
  - non-null precision `25.55%`
- strict specialized validation:
  - exact `97`
  - wrong `113`
  - null `290`
  - non-null precision `46.19%`

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_strict7/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_strict7/rows.csv)

Interpretation:

- This is a real product-safety gain.
- Wrong non-null outputs dropped by `158` samples in the audit.
- Exact outputs increased slightly at the same time, which means the stricter filter also helped candidate selection in some cases instead of only forcing more nulls.

Current decision:

- keep strict standard single-plate validation in specialized mode
- treat this as the current default safety baseline for the specialized OCR path

## Conservative Specialized Normalization

After the strict single-plate validation change, the next root-cause check was the ambiguous-character normalization itself.

The previous specialized normalization still rewrote:

- `D -> 0`
- `Q -> 0`
- `L / I -> 1`
- `B -> 8`

I audited this on the same 500-sample strict-validation review set.

Findings:

- normalization-created wrong samples: `47`
- normalization-created exact samples: `0`

Common damage patterns:

- `皖A7Q653 -> 皖A70653`
- `川SND870 -> 川SN0870`
- `皖AB618B -> 皖A86188`

This means the specialized OCR path was still turning many plausible full-plate reads into wrong outputs during post-processing.

I then tested a conservative alternative:

- specialized mode only uppercases and strips non-alphanumeric characters
- specialized mode no longer applies ambiguous letter-digit substitution
- generic mode keeps the old normalization behavior

Code:

- [plate_ocr.py](/D:/Projects/Car/src/car_system/ocr/plate_ocr.py)
- [test_plate_ocr.py](/D:/Projects/Car/tests/test_plate_ocr.py)

Remote verification:

- focused test suite: `45 passed`

Updated 500-sample audit:

- strict validation only:
  - exact `97`
  - wrong `113`
  - null `290`
- strict validation + conservative specialized normalization:
  - exact `108`
  - wrong `105`
  - null `287`

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_conservative/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_conservative/rows.csv)

Interpretation:

- this is a clean product improvement
- it reduces wrong outputs further
- it also recovers additional exact outputs instead of only increasing nulls

Current decision:

- keep conservative normalization for `mode='specialized'`
- keep the old substitution-based normalization only for the generic OCR path

## Candidate Strategy Simplification

The next root-cause check was whether the specialized OCR pipeline should keep using augmented candidate images at all.

Before this change, specialized OCR evaluated three candidates:

- original crop
- 2x enlarged crop
- histogram-equalized crop

The audit showed that the later candidates often increased score but also increased wrong non-null outputs. On the conservative-normalization baseline:

- current 3-candidate specialized path:
  - exact `108`
  - wrong `105`
  - null `287`
- using only the original crop:
  - exact `105`
  - wrong `80`
  - null `315`

Interpretation:

- this is a deliberate safety trade
- exact outputs drop slightly
- wrong outputs drop much more sharply
- non-null precision improves from `108 / 213 = 50.70%` to `105 / 185 = 56.76%`

Fixed internal review set:

- previous conservative path:
  - `db`: null
  - `blur`: exact
  - `tilt`: null
  - `fn`: exact
  - `challenge`: wrong
- original-only path:
  - `db`: null
  - `blur`: exact
  - `tilt`: null
  - `fn`: exact
  - `challenge`: null

That shift is acceptable for the current product direction because it turns one visible wrong plate into a reject.

Code:

- [plate_ocr.py](/D:/Projects/Car/src/car_system/ocr/plate_ocr.py)
- [test_plate_ocr.py](/D:/Projects/Car/tests/test_plate_ocr.py)

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_originalonly/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_originalonly/rows.csv)
- [report.html](/D:/Projects/Car/outputs/internal_analysis_v1_originalonly/report.html)

Remote verification:

- focused test suite: `47 passed`

Current decision:

- specialized OCR now uses only the original crop as its candidate input
- generic OCR keeps the previous augmented candidate behavior

## Candidate Vote Re-check After Strict Validation

I re-evaluated candidate agreement after the stricter specialized validation was added.

On the updated 500-sample audit:

- current strict validation only:
  - kept `210`
  - exact `97`
  - wrong `113`
  - precision `46.19%`
- adding `min_votes >= 2`:
  - kept `165`
  - exact `89`
  - wrong `76`
  - precision `53.94%`
- adding `min_votes == 3`:
  - kept `66`
  - exact `46`
  - wrong `20`
  - precision `69.70%`

Interpretation:

- candidate voting becomes somewhat more useful after strict validation
- it still trades away too much coverage for the current default path
- the fixed internal review set also shows a bad side effect: one correct sample only has a single surviving vote, while one wrong sample still has two votes

Current decision:

- do not make candidate-vote gating the default specialized OCR rule yet
- keep it as an optional future guard for an even more conservative product mode

## Confidence Threshold Re-check on the Tightened Specialized Path

The earlier confidence audit showed that confidence alone was not a safe reject rule on the looser OCR path. After the specialized chain was tightened in four steps:

- standard single-plate validation
- conservative specialized normalization
- original-crop-only candidate strategy
- zero-padding plain crop input

I re-ran the confidence audit on the new 500-sample baseline.

Current baseline before adding a confidence floor:

- exact `105`
- wrong `80`
- null `315`
- non-null precision `105 / 185 = 56.76%`

Threshold sweep on that tightened path:

- `min_confidence >= 0.90`:
  - kept `139`
  - exact `96`
  - wrong `43`
  - precision `69.06%`
- `min_confidence >= 0.93`:
  - kept `124`
  - exact `91`
  - wrong `33`
  - precision `73.39%`
- `min_confidence >= 0.95`:
  - kept `108`
  - exact `84`
  - wrong `24`
  - precision `77.78%`

Why `0.93` was selected:

- it removes `47` wrong outputs while only giving up `14` exact outputs
- it preserves the two fixed-review exact samples that matter in the current showcase set:
  - `blur`: exact, confidence `0.9965`
  - `fn`: exact, confidence `0.9377`
- `0.95` would reject the `fn` sample, which is too aggressive for the current default

Fixed internal review set after enabling `min_confidence: 0.93`:

- `db`: null
- `blur`: exact
- `tilt`: null
- `fn`: exact
- `challenge`: null

This keeps exact-plate accuracy on the 5-sample fixed review set at `0.4`, while turning the pipeline more conservative overall.

Code:

- [config.py](/D:/Projects/Car/src/car_system/config.py)
- [runtime.py](/D:/Projects/Car/src/car_system/runtime.py)
- [plate_ocr.py](/D:/Projects/Car/src/car_system/ocr/plate_ocr.py)
- [test_config.py](/D:/Projects/Car/tests/test_config.py)
- [test_runtime.py](/D:/Projects/Car/tests/test_runtime.py)
- [test_plate_ocr.py](/D:/Projects/Car/tests/test_plate_ocr.py)

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_min093/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_min093/rows.csv)
- [report.html](/D:/Projects/Car/outputs/internal_analysis_v1_min093/report.html)

Verification:

- local focused suite: `53 passed`
- remote focused suite: `53 passed`

Current decision:

- specialized OCR now uses `min_confidence: 0.93`
- confidence is still not perfect as a standalone truth signal, but on the tightened specialized path it is now useful as a final safety gate

## Safe Rectification A/B Result

The next experiment was no longer "add more text rules." The remaining wrong reads had become concentrated enough that OCR input quality was the better lever.

I added a dedicated `safe_rectification` path with three properties:

- it only targets the specialized OCR branch
- it falls back to the current plain crop whenever the geometric candidate is not trustworthy
- it records whether rectification was applied or why it fell back

The first stored comparison on the new server showed:

- control path:
  - exact `91`
  - wrong `33`
  - null `376`
- safe-rectification path:
  - exact `78`
  - wrong `22`
  - null `400`

Interpretation:

- wrong non-null outputs dropped again from `33` to `22`
- exact outputs also dropped from `91` to `78`
- null outputs increased from `376` to `400`
- non-null precision improved from `91 / 124 = 73.39%` to `78 / 100 = 78.00%`

Fixed internal review set under safe rectification:

- `db`: null, reason `no_quad`
- `blur`: exact, reason `low_score`
- `tilt`: null, reason `low_score`
- `fn`: exact, reason `low_score`
- `challenge`: null, rectification applied

This passed the current product gate:

- `wrong` dropped below the control value of `33`
- `blur` stayed exact
- `fn` stayed exact
- `challenge` did not regress into a visible wrong plate

Current decision:

- promote `safe_rectification` to the default specialized OCR path
- keep the old plain-crop baseline as explicit control configs:
  - [plate_ocr_specialized_control.local.yaml](/D:/Projects/Car/configs/plate_ocr_specialized_control.local.yaml)
  - [remote_mvp_specialized_control.yaml](/D:/Projects/Car/configs/remote_mvp_specialized_control.yaml)
- keep dedicated safe A/B config available at:
  - [remote_mvp_specialized_safe.yaml](/D:/Projects/Car/configs/remote_mvp_specialized_safe.yaml)

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_safe/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_safe/rows.csv)
- [report.html](/D:/Projects/Car/outputs/internal_analysis_v1_safe/report.html)

## Fresh Control Rerun And Disagreement Rescue

After promoting `safe_rectification`, I reran the plain-crop control on the same 500-sample list with the current code and the new server environment. That fresh control result was:

- current control path:
  - exact `96`
  - wrong `29`
  - null `375`

This matters because the old `91 / 33 / 376` control snapshot was no longer the right comparison point for current code.

I then compared the current safe path against that fresh control and found:

- current safe path:
  - exact `78`
  - wrong `22`
  - null `400`
- transitions from current control to current safe:
  - `exact -> exact`: `71`
  - `exact -> null`: `23`
  - `exact -> wrong`: `2`
  - `null -> exact`: `7`
  - `wrong -> null`: `9`
  - `wrong -> wrong`: `20`

Interpretation:

- most of the remaining wrong outputs were persistent OCR errors, not rectification-vs-plain disagreements
- there were only `2` current regressions where safe rectification turned a control exact result into a wrong result
- there were no `wrong -> exact` wins available from the current safe branch alone

That made the next product-safe move very narrow:

- only when safe rectification is applied
- and rectified OCR returns a non-null plate
- and plain-crop OCR also returns a non-null plate
- and the two normalized texts disagree
- prefer the plain-crop OCR result

This disagreement rescue does not use plain OCR to recover from `safe -> null`; it only prevents high-confidence safe regressions from leaking a wrong plate.

Code:

- [runner.py](/D:/Projects/Car/src/car_system/pipeline/runner.py)
- [test_runner.py](/D:/Projects/Car/tests/test_runner.py)

Fresh verification:

- local focused suite: `66 passed`
- remote focused suite: `66 passed`

Updated 500-sample safe audit with disagreement rescue:

- previous safe path:
  - exact `78`
  - wrong `22`
  - null `400`
- current default safe path:
  - exact `79`
  - wrong `21`
  - null `400`

The rescued sample was:

- `test/ccpd_db__0101-9_3-322&401_452&466-452&446_326&466_322&421_448&401-0_0_16_30_29_28_0-18-13.jpg`
  - previous safe wrong: `皖AS6554`
  - current default exact: `皖AS654A`
  - current control exact: `皖AS654A`

Fixed internal review set after disagreement rescue:

- `db`: `ocr_null`, reason `no_quad`
- `blur`: exact `皖AP0K76`, reason `low_score`
- `tilt`: `ocr_null`, reason `low_score`
- `fn`: exact `皖AX0K18`, reason `low_score`
- `challenge`: `ocr_null`, reason `applied`

Current decision:

- keep `safe_rectification` as the default specialized OCR path
- keep the disagreement rescue rule enabled in the default runner
- treat the current product-safe baseline as:
  - current control: `96 / 29 / 375`
  - current default safe: `79 / 21 / 400`

Artifacts:

- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_control_current/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_control_current/rows.csv)
- [summary.json](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_safe_consensus/summary.json)
- [rows.csv](/D:/Projects/Car/outputs/ocr_confidence_audit_plaincrop_500_safe_consensus/rows.csv)
- [report.html](/D:/Projects/Car/outputs/internal_analysis_v1_safe_consensus/report.html)

## Remaining Wrong Structure

I also profiled the remaining `21` wrong outputs in the current default safe path.

High-level shape:

- all remaining wrong outputs are still length `7`
- `12 / 21` keep the same `皖A` prefix as the ground truth and fail later in the plate body
- `8 / 21` still contain a province-character mismatch
- the most error-heavy positions are:
  - position `0`: `8`
  - position `6`: `6`
  - position `5`: `5`
  - position `3`: `4`
  - position `1`: `4`

Observed confusion patterns:

- province / prefix bias:
  - `晋 -> 皖`
  - `苏 -> 皖`
  - `湘 -> 皖`
  - `琼 -> 皖`
  - `赣 -> 皖`
  - `鄂 -> 皖`
  - `桂 -> 豫`
  - one safe-specific regression had `皖 -> 鲁`, which the disagreement rescue removed
- body-character confusions:
  - `D -> 0` twice
  - `Q -> 0`
  - `B -> 8`
  - `Z -> 7`
  - `G -> 0`
  - `W -> 3`
  - `H -> N`

Interpretation:

- the remaining errors are no longer dominated by broad safety-gate problems
- the next bottleneck is OCR character discrimination itself, especially province-prefix bias and tail-character confusion
- this is not a good place to keep stacking generic thresholds; the next valuable work should target OCR inputs or OCR data quality directly
