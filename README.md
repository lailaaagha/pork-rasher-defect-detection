# Pork Rasher Packaging Defect Detection

MMA3001 (Numerical Methods and Machine Learning) individual project.
Automatically detects and locates meat-quality and packaging defects in
images of packaged pork rasher trays using a convolutional neural network
(YOLOv8), and compares it against alternative models and settings.

## 1. Engineering problem

**Context.** In meat processing plants, packaged trays travel along a conveyor
and are inspected for defects before shipping. Manual visual inspection is slow,
inconsistent and tiring, and missed defects (e.g. unsealed packs) can cause
spoilage, food-safety risk, product recalls and waste.

**Problem.** Given a photo of a packaged pork rasher tray, identify which
defects are present and where they are in the image.

**Why it matters.** An automated inspection step could flag defective trays
consistently at line speed, so operators only need to check flagged trays.

**Intended use.** A decision-support tool for production-line inspection. It
flags trays for human review; it is not intended to replace final human sign-off.

**Scope limitations.**
- Trained on images from one camera setup and lighting; performance on other
  lines, cameras or lighting conditions is unknown.
- Only the 5 defect types labelled in the dataset can be detected.
- Very few defect-free images exist (21 of 1955), so tray-level PASS/FAIL accuracy
  cannot be reliably validated; per-defect detection is the primary output.
- The `wrinkle` class has only 18 labelled instances, so its results are unreliable.

## 2. Inputs and outputs

**Input:** an RGB image of a packaged tray (JPEG/PNG, 8-bit, pixel values 0–255).
Images of any size are resized by the model to a fixed input resolution
(default 640 px on the longest side).

**Outputs** (per image):
| Output | Format | Meaning / intended use |
|---|---|---|
| Detected defects | list of (class, confidence 0–1, bounding box in pixels) | What defect was found, how sure the model is, and where it is |
| Meat-quality flag | OK / DEFECT | DEFECT if `loose-meat` or `twisted-meat` is detected |
| Packaging-quality flag | OK / DEFECT | DEFECT if `packaging-error`, `unsealed` or `wrinkle` is detected |
| Overall decision | PASS / FAIL | FAIL if any defect is detected above the confidence threshold |

**Defect classes:** `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.

**Invalid inputs:** label files are validated (5 values per box, valid class ID,
coordinates in [0, 1]); polygon labels are converted to their enclosing box.
Handling of unreadable or non-image files is described in the code documentation.

## 3. Dataset

- Source: *Pork Rasher Error (Packaging)* dataset, Roboflow (version 2 used here),
  licensed **CC BY 4.0**.
  https://app.roboflow.com/lailaa-agha/pork-rasher-error-packaging-jybag/2
- 1955 images, object-detection labels in YOLO format.
- Split 70 / 20 / 10 → 1369 train / 391 validation / 195 test images
  (rebalanced from the original 90/6/4 split, which left only 76 test images).
- No augmentation was applied in Roboflow (avoids near-duplicate images leaking
  into the test set).

The dataset is **not** stored in this repository (size). To reproduce, download
version 2 in **YOLOv8** format and extract it to `data/` (see Setup).

## 4. Repository structure

```
data/         dataset (not committed — download separately)
notebooks/    Colab notebooks for training and experiments (GPU)
src/rasher/   Python package (data exploration, evaluation, inspection logic)
tests/        pytest unit tests
results/      saved metrics tables and figures
docs/         generated HTML documentation (pdoc)
```

## 5. Setup

```bash
git clone https://github.com/lailaaagha/pork-rasher-defect-detection.git
cd pork-rasher-defect-detection
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Then download the dataset (see Section 3) into `data/`.

## 6. Usage

Dataset exploration (class counts, box sizes, label validation):
```bash
python src/rasher/explore.py
```
Model training and experiments: open the notebook in `notebooks/` in Google Colab
with a GPU runtime.

## 7. Results

## 7. Results

All model selection used the **validation set** (391 images). The **test set**
(195 images) was used **once**, for the final model only.

### Model and resolution comparison (validation, FP32)

| Model | Input | GFLOPs | Latency (ms, batch 1, T4) | mAP50 | AP50 packaging-error | AP50 unsealed |
|---|---|---|---|---|---|---|
| YOLOv8n | 320 | 2.2 | 5.7 | 0.528 | 0.310 | 0.846 |
| **YOLOv8n** | **480** | **4.9** | **5.7** | **0.584** | **0.381** | **0.849** |
| YOLOv8n | 640 | 8.6 | 7.1 | 0.561 | 0.370 | 0.836 |
| YOLOv8s | 640 | 28.4 | 17.1 | 0.543 | 0.355 | 0.860 |

- A larger model (YOLOv8s, 3.5x FLOPs) gave no accuracy gain.
- Lower resolution mainly hurts the smallest defects (packaging-error);
  large defects (unsealed) are unaffected. No gain beyond 480 px.
- Latency is overhead-bound, not compute-bound: 4x more FLOPs costs only ~1.25x time.
- FP16 changed mAP50 by only -0.002 (box round-off ~1-3 px) and gave no speed
  benefit for YOLOv8n, so FP32 was kept.

**Selected design:** YOLOv8n, 480 px, FP32.

### Numerical-methods checks (`python -m rasher.analysis`)

- **AP verification:** AP50 recomputed with our own trapezoidal rule matches
  Ultralytics to within 0.0003 for every class; Simpson's rule is slightly worse
  because the precision-recall curve is not smooth (observed convergence O(h), not O(h²)).
- **Threshold optimisation:** golden-section search (22 evaluations) matched a
  2000-point grid search. With a missed defect costing 5x a false alarm, the
  optimal confidence threshold is **0.137** (default 0.25).

### Final test-set results (YOLOv8n, 480 px)

| Class | AP50 (test) |
|---|---|
| unsealed | 0.801 |
| loose-meat | 0.763 |
| twisted-meat | 0.490 |
| packaging-error | 0.252 |
| wrinkle | 0.000 (1 test instance) |
| **mAP50** | **0.461** (validation: 0.584) |

Tray level (192 defective, 3 defect-free trays):

| Threshold | Defective trays flagged | Missed | Good trays falsely flagged |
|---|---|---|---|
| 0.25 (default) | 151 / 192 (79%) | 41 | 0 / 3 |
| **0.137 (optimised)** | **177 / 192 (92%)** | **15** | 1 / 3 |

The test mAP is lower than validation because the model and threshold were
selected on the validation set (selection bias) and some classes have very few
test instances. With only 3 defect-free test trays, the false-alarm rate cannot
be reliably estimated.

Full tables and figures: `results/` and `results/analysis/`.
Test report: `results/test_report.html`. API docs: `docs/index.html`.

## 8. AI use

AI tools were used during this project; see `AI_USE.md` for details.

## 9. License

Code: MIT License (see `LICENSE`). Dataset: CC BY 4.0 (Roboflow).