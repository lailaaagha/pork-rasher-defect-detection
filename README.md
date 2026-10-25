# Pork Rasher Packaging Defect Detection

MMA3001 (Numerical Methods and Machine Learning) individual project — Lailaa Agha.

An automated inspection system that finds **meat and packaging defects** in photos of
packaged pork rasher trays, using a YOLOv8 convolutional neural network. Numerical
methods written from scratch (trapezoidal rule, Simpson's rule, golden-section search,
linear interpolation) are used to verify the model's accuracy score and to choose the
best confidence threshold.

## 1. Purpose

Trays on a meat-packing line are usually inspected by eye, which is slow and
inconsistent. Missed defects (e.g. unsealed packs) can cause spoilage, food-safety risk
and waste. This project builds a **decision-support tool**: it flags trays that look
defective so an operator only needs to check those. It is not intended to replace
final human sign-off.

**Scope limits:** only the 5 labelled defect types can be detected; all images come from
one camera set-up; only 21 of 1,955 images are defect-free, so the PASS/FAIL decision
cannot be fully validated.

## 2. Inputs and outputs

**Input:** an RGB tray photo (JPEG/PNG, 8-bit, pixel values 0–255). Any size is accepted;
the model resizes it to 480 px on the longest side.

| Output | Format | Meaning |
|---|---|---|
| Detected defects | class, confidence (0–1), box (x1, y1, x2, y2 in pixels) | What was found, how sure the model is, and where |
| Meat-quality flag | OK / DEFECT | DEFECT if `loose-meat` or `twisted-meat` is found |
| Packaging-quality flag | OK / DEFECT | DEFECT if `packaging-error`, `unsealed` or `wrinkle` is found |
| Overall decision | PASS / FAIL | FAIL if any defect is found at or above the threshold (0.137) |

**Invalid inputs:** missing, corrupt or unsupported image files, invalid detections and
malformed label files are rejected with clear errors; polygon labels are converted to
boxes. All of this is covered by unit tests.

## 3. Data

- **Dataset:** *Pork Rasher Error (Packaging)*, Roboflow, licence **CC BY 4.0**, provided as
  Dataset 3 in MMA3001. Public link (use **version 2**):
  https://universe.roboflow.com/lailaa-agha/pork-rasher-error-packaging-jybag
- 1,955 images, 5 defect classes, bounding-box labels in YOLO format.
- Split **70 / 20 / 10** → 1,369 train / 391 validation / 195 test (rebalanced from the
  default 90/6/4, which left only 76 test images). No augmentation added in Roboflow.
- The dataset is **not stored in this repository** (too large). To use it locally,
  download version 2 in **YOLOv8** format and extract it so the folder looks like:
  ```
  data/
  ├── data.yaml
  ├── train/ (images/, labels/)
  ├── valid/ (images/, labels/)
  └── test/  (images/, labels/)
  ```
  In `data.yaml`, set the paths to `train/images`, `valid/images`, `test/images`
  (Roboflow writes `../train/images`, which breaks).

## 4. Repository structure

```
data/                  dataset (not committed — download separately, see Section 3)
docs/                  HTML API documentation generated with pdoc (open docs/index.html)
notebooks/             Colab notebook: data download, training, evaluation, final test
results/               dataset summaries, test report, figures
  analysis/            validation/test metrics, curves, speed/FP16 results, AP and threshold analysis
src/rasher/            Python package
  explore.py           dataset counts, box sizes and label validation
  inspect_tray.py      turns detections into meat/packaging flags and PASS/FAIL; validates inputs
  numerics.py          trapezoidal rule, Simpson's rule, average precision, golden-section search
  analysis.py          AP verification/convergence and confidence-threshold optimisation
tests/                 pytest unit tests (54 tests)
AI_USE.md              log of AI tool use
requirements.txt       Python dependencies
pyproject.toml         package and pytest configuration
```

## 5. Environment and setup

- **Local:** Python 3.10+ (developed with 3.12 on Windows).
- **Training:** Google Colab with a T4 GPU (training on a CPU is very slow).

```bash
git clone https://github.com/lailaaagha/pork-rasher-defect-detection.git
cd pork-rasher-defect-detection
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .                    # installs the rasher package so `python -m rasher...` works
```

Main dependencies: ultralytics (YOLOv8 / PyTorch), scikit-learn, numpy, pandas,
matplotlib, opencv-python, pyyaml, pytest, pytest-html, pdoc.

## 6. Usage

**Explore and validate the dataset** (needs `data/`):
```bash
python src/rasher/explore.py
```

**Train and evaluate the models (Colab):** open
`notebooks/MMA3001_Meat_Defect_Detection.ipynb` in Google Colab, select a T4 GPU, add your
own free Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (key icon in the left
sidebar), then run the cells top to bottom. Results are saved to Google Drive.

**Verify AP and optimise the threshold** (uses curves exported by the notebook):
```bash
python -m rasher.analysis results/analysis/curves_yolov8n_480.npz --out results/analysis
```

**Run the tests and create the test report:**
```bash
pytest
pytest --html=results/test_report.html --self-contained-html
```

**Regenerate the HTML documentation:**
```bash
pdoc rasher --docformat google -o docs
```

## 7. Results

All model choices were made on the **validation set**; the **test set** was used once, for
the final model only.

**Model and resolution comparison (validation):**

| Model | Image size | GFLOPs | Latency (ms, T4) | mAP50 | AP50 packaging-error | AP50 unsealed |
|---|---|---|---|---|---|---|
| YOLOv8n | 320 | 2.2 | 5.7 | 0.528 | 0.310 | 0.846 |
| **YOLOv8n** | **480** | **4.9** | **5.7** | **0.584** | **0.381** | **0.849** |
| YOLOv8n | 640 | 8.6 | 7.1 | 0.561 | 0.370 | 0.836 |
| YOLOv8s | 640 | 28.4 | 17.1 | 0.543 | 0.355 | 0.860 |

- A larger model gave no accuracy gain; small defects need enough resolution (up to ~480 px).
- FP16 changed mAP50 by only −0.002 and gave no speed gain for YOLOv8n, so FP32 was kept.
- My trapezoidal AP matched Ultralytics to within 0.0003; on the real (step-shaped) curve the
  error converged at O(h), not O(h²).
- Golden-section search (missed defect = 5× cost of a false alarm) gave an optimal
  confidence threshold of **0.137**, matching a 2,000-point grid search.

**Final model (YOLOv8n, 480 px, FP32, threshold 0.137) on the test set:**

| Class | AP50 (test) |
|---|---|
| unsealed | 0.801 |
| loose-meat | 0.763 |
| twisted-meat | 0.490 |
| packaging-error | 0.252 |
| wrinkle | 0.000 (1 test example) |
| **mAP50** | **0.461** |

At threshold 0.137 the system flagged **177 of 192 defective trays (92%)**, compared with
151 (79%) at the default 0.25. With only 3 defect-free test trays, the false-alarm rate
cannot be reliably estimated.

Full results and figures: `results/` and `results/analysis/`. Test report:
`results/test_report.html`. API documentation: `docs/index.html`.

## 8. AI use

AI tools (Claude and ChatGPT) were used during this project. See `AI_USE.md` for what they
were used for, how their output was checked, and which decisions remained mine.

## 9. Licence

Code: MIT License (see `LICENSE`). Dataset: CC BY 4.0 (Roboflow).