# OCR Stability Design

**Date:** 2026-04-12

## Goal

Improve OCR stability in the existing vehicle and license plate system without changing the main detector architecture or introducing a new primary recognition model.

## Scope

### Included

- Small plate crop expansion before OCR
- Lightweight image preprocessing for OCR candidates
- Multiple OCR attempts on a few candidate crops
- Rule-based post-processing for mainland plate-like text
- Confidence-based selection across OCR candidates
- Unit tests for OCR candidate selection and text normalization

### Excluded

- Replacing PaddleOCR as the primary OCR engine
- Training a new OCR model from scratch
- Adding a second heavy OCR model as the default path
- Complex geometric correction beyond the current lightweight hook
- Introducing country-agnostic plate parsing logic

## Why This Design

The current system already works end-to-end, and the main weakness is OCR instability on blurred, tilted, or tightly cropped plates. The lowest-risk improvement is to keep the current `TextRecognition` path and strengthen the quality of the crop and the selection of the OCR output. This keeps the engineering scope small, preserves reproducibility, and creates a clear “before vs after” experiment for the thesis.

## Recommended Approach

The OCR stabilization path should use one recognition backend and three small improvements around it:

1. expand the plate crop slightly before recognition
2. generate a small set of OCR candidate images from the same crop
3. normalize and score OCR outputs using simple plate-format rules

This approach keeps the pipeline understandable and avoids turning OCR into a second model-integration project.

## System Changes

### 1. Crop Expansion

The current crop in [runner.py](D:\Projects\Car\src\car_system\pipeline\runner.py) slices exactly on the detector box. That is fragile because a slightly tight box can cut off edge characters. The runner should expand the plate crop by a small ratio before slicing, while clamping the crop to the image boundary.

Recommended default:

- horizontal padding ratio: small positive value
- vertical padding ratio: slightly smaller positive value

The padding should stay lightweight and deterministic.

### 2. Lightweight OCR Candidate Preprocessing

The OCR module should generate only a few candidate images from the same crop:

- original crop
- enlarged crop
- contrast-enhanced grayscale crop

No more than three candidates should be used in the first version. This keeps inference overhead bounded and makes the behavior easy to explain.

### 3. OCR Result Selection

Each candidate image should be passed to the same OCR recognizer. The OCR module should collect all non-empty recognition results and select the best one using:

- raw OCR confidence
- text length sanity
- simple format plausibility for mainland vehicle plates

If all candidates are weak, the module should still return the highest-confidence candidate rather than fabricating a result.

### 4. Rule-Based Post-Processing

Post-processing should be intentionally conservative. It should only fix obvious ambiguities, for example:

- `O` vs `0`
- `I` vs `1`
- `B` vs `8`

The correction should depend on character position where possible. If the rule engine is not confident, it should keep the original text.

## Data Flow

The updated OCR path should be:

1. detector outputs plate box
2. runner expands and crops the plate region
3. optional lightweight rectification hook runs
4. OCR module builds candidate images
5. OCR runs on each candidate
6. OCR module normalizes and ranks candidate texts
7. best result is attached to the plate match

## Error Handling

- If preprocessing fails for one candidate, skip that candidate and continue.
- If all OCR candidates fail, return `None`.
- If rule-based normalization cannot confidently improve a result, preserve the original OCR text.
- Crop expansion must never produce out-of-bounds indexing.

## Testing Strategy

The first implementation should add tests for:

- candidate OCR result ranking
- normalization of obvious ambiguous characters
- “no valid candidates” returning `None`
- crop expansion clamping to image boundaries

The tests should stay unit-level and avoid introducing real OCR dependencies.

## Thesis Value

This change gives the project a small but defensible system-level optimization point:

- the detector remains unchanged
- the OCR path becomes more stable
- the result can be measured as “baseline OCR vs stabilized OCR”

That is a better graduation-project tradeoff than replacing the OCR model outright.
