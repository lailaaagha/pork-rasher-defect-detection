"""Dataset exploration for the pork rasher packaging-defect dataset.

Counts images, bounding boxes per class and box sizes in each split of a
YOLO-format dataset, and saves summary tables and a chart to ``results/``.

Usage (run from the repository root)::

    python src/rasher/explore.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_class_names(yaml_path: Path) -> list[str]:
    """Read the class names from a YOLO ``data.yaml`` file.

    Args:
        yaml_path: Path to the dataset's ``data.yaml``.

    Returns:
        Class names in class-ID order (index 0 = class ID 0).

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        KeyError: If the file has no ``names`` entry.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text())
    if "names" not in data:
        raise KeyError(f"No 'names' entry in {yaml_path}")
    names = data["names"]
    # Some exports store names as {0: 'a', 1: 'b'} instead of a list.
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def parse_label_file(label_path: Path, n_classes: int) -> list[tuple[int, float, float, float, float]]:
    """Parse one YOLO label file into a list of boxes.

    Each non-empty line must be ``class x_centre y_centre width height``,
    with coordinates normalised to [0, 1] by the image size.

    Args:
        label_path: Path to a ``.txt`` label file.
        n_classes: Number of valid classes; IDs must be in ``[0, n_classes)``.

    Returns:
        List of ``(class_id, x, y, w, h)`` tuples. Empty list = no defects.

    Raises:
        ValueError: If a line has the wrong number of values, an unknown
            class ID, or coordinates outside [0, 1].
    """
    boxes = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue  # blank lines are allowed
        if len(parts) != 5:
            raise ValueError(f"{label_path.name} line {line_no}: expected 5 values, got {len(parts)}")
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:])
        if not 0 <= cls < n_classes:
            raise ValueError(f"{label_path.name} line {line_no}: unknown class ID {cls}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            raise ValueError(f"{label_path.name} line {line_no}: coordinates outside [0, 1]")
        boxes.append((cls, x, y, w, h))
    return boxes


def summarise_split(split_dir: Path, names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Collect every bounding box in one split and count problem files.

    Args:
        split_dir: Split folder containing ``images/`` and ``labels/``.
        names: Class names from :func:`load_class_names`.

    Returns:
        A tuple ``(boxes, info)`` where ``boxes`` has one row per box
        (split, image, class, width, height, area_pct) and ``info`` counts
        images, defect-free images, missing labels and invalid labels.

    Raises:
        FileNotFoundError: If ``images/`` or ``labels/`` is missing.
    """
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and labels/ inside {split_dir}")

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    info = Counter(images=len(images))
    rows = []
    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            info["missing_label"] += 1
            continue
        try:
            boxes = parse_label_file(lbl, len(names))
        except ValueError:
            # Record bad files instead of crashing, so we can report them.
            info["invalid_label"] += 1
            continue
        if not boxes:
            info["no_defect"] += 1
        for cls, _x, _y, w, h in boxes:
            rows.append({
                "split": split_dir.name, "image": img.name, "class": names[cls],
                "width": w, "height": h, "area_pct": 100 * w * h,  # % of image area
            })
    return pd.DataFrame(rows), dict(info)


def main(data_dir: Path = Path("data"), out_dir: Path = Path("results")) -> None:
    """Run the exploration and save tables/plots to ``out_dir``."""
    names = load_class_names(data_dir / "data.yaml")
    out_dir.mkdir(exist_ok=True)

    box_frames, info_rows = [], []
    for split in SPLITS:
        boxes, info = summarise_split(data_dir / split, names)
        box_frames.append(boxes)
        info_rows.append({"split": split, **info})
    boxes = pd.concat(box_frames, ignore_index=True)

    # 1. Image-level summary per split
    images = pd.DataFrame(info_rows).set_index("split").fillna(0).astype(int)

    # 2. Number of boxes per class per split
    counts = (boxes.groupby(["class", "split"]).size().unstack(fill_value=0)
              .reindex(index=names, columns=list(SPLITS), fill_value=0))
    counts["total"] = counts.sum(axis=1)

    # 3. Number of images containing each class (matters for rare classes)
    imgs_with = (boxes.groupby(["class", "split"])["image"].nunique().unstack(fill_value=0)
                 .reindex(index=names, columns=list(SPLITS), fill_value=0))

    # 4. Box size: how much of the image each defect covers
    sizes = boxes.groupby("class")["area_pct"].describe()[["min", "50%", "max"]].reindex(names)
    sizes.columns = ["min_area_%", "median_area_%", "max_area_%"]

    print("\n=== Images per split ===\n", images)
    print("\n=== Boxes per class ===\n", counts)
    print("\n=== Images containing each class ===\n", imgs_with)
    print("\n=== Box size (% of image area) ===\n", sizes.round(3))

    images.to_csv(out_dir / "dataset_images_per_split.csv")
    counts.to_csv(out_dir / "dataset_boxes_per_class.csv")
    imgs_with.to_csv(out_dir / "dataset_images_per_class.csv")
    sizes.to_csv(out_dir / "dataset_box_sizes.csv")

    ax = counts[list(SPLITS)].plot.bar(figsize=(8, 4.5), rot=0)
    ax.set_ylabel("Number of bounding boxes")
    ax.set_title("Defect instances per class and split (dataset v2)")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=150)
    print(f"\nSaved tables and chart to {out_dir.resolve()}")


if __name__ == "__main__":
    main()