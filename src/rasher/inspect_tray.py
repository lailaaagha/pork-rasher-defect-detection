"""Turn object-detection outputs into tray-level inspection decisions.

The detector (YOLOv8) outputs a list of boxes, each with a class name and a
confidence. This module groups those detections into a meat-quality flag,
a packaging-quality flag and an overall PASS/FAIL decision, and validates
inputs so that invalid images or detections are rejected with a clear error.

Example::

    from rasher.inspect_tray import Detection, inspect_tray

    dets = [Detection("unsealed", 0.91, (100, 80, 400, 300))]
    report = inspect_tray(dets, conf_threshold=0.25)
    print(report.decision)        # 'FAIL'
    print(report.packaging_ok)    # False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

#: Defect classes that indicate a problem with the meat itself.
MEAT_DEFECTS = frozenset({"loose-meat", "twisted-meat"})
#: Defect classes that indicate a problem with the packaging.
PACKAGING_DEFECTS = frozenset({"packaging-error", "unsealed", "wrinkle"})
#: Every class the model can output.
ALL_CLASSES = MEAT_DEFECTS | PACKAGING_DEFECTS
#: Image file types accepted by :func:`load_image`.
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Detection:
    """A single detected defect.

    Attributes:
        class_name: One of :data:`ALL_CLASSES`.
        confidence: Model confidence in [0, 1].
        box: Bounding box ``(x1, y1, x2, y2)`` in pixels, top-left origin.

    Raises:
        ValueError: If the class is unknown, the confidence is outside
            [0, 1], or the box has zero or negative width/height.
    """

    class_name: str
    confidence: float
    box: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if self.class_name not in ALL_CLASSES:
            raise ValueError(f"Unknown class '{self.class_name}'. Expected one of {sorted(ALL_CLASSES)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        x1, y1, x2, y2 = self.box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Box must have positive width and height, got {self.box}")


@dataclass(frozen=True)
class TrayReport:
    """Inspection result for one tray image.

    Attributes:
        meat_ok: ``False`` if any meat defect was detected.
        packaging_ok: ``False`` if any packaging defect was detected.
        decision: ``"PASS"`` if both flags are OK, otherwise ``"FAIL"``.
        defects: Detections at or above the confidence threshold.
    """

    meat_ok: bool
    packaging_ok: bool
    decision: str
    defects: tuple[Detection, ...] = field(default_factory=tuple)


def inspect_tray(detections: list[Detection], conf_threshold: float = 0.25) -> TrayReport:
    """Decide whether a tray passes inspection.

    Detections below ``conf_threshold`` are ignored. A tray FAILS if any
    remaining detection belongs to a meat or packaging defect class.

    Args:
        detections: Detections for one image (may be empty).
        conf_threshold: Minimum confidence for a detection to count, in [0, 1].
            A lower threshold catches more defects (higher recall) but raises
            more false alarms (lower precision).

    Returns:
        A :class:`TrayReport` with both quality flags and the decision.

    Raises:
        ValueError: If ``conf_threshold`` is outside [0, 1].
    """
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")

    kept = tuple(d for d in detections if d.confidence >= conf_threshold)
    meat_ok = not any(d.class_name in MEAT_DEFECTS for d in kept)
    packaging_ok = not any(d.class_name in PACKAGING_DEFECTS for d in kept)
    decision = "PASS" if (meat_ok and packaging_ok) else "FAIL"
    return TrayReport(meat_ok, packaging_ok, decision, kept)


def load_image(path: str | Path) -> np.ndarray:
    """Load and validate a tray image before it is passed to the model.

    Args:
        path: Path to an image file (.jpg, .jpeg, .png or .bmp).

    Returns:
        The image as a ``uint8`` array of shape (height, width, 3), BGR order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file type is unsupported, the file cannot be
            decoded (corrupt or not an image), or it is not a 3-channel image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Use one of {sorted(SUPPORTED_EXTS)}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # OpenCV returns None instead of raising on bad files
        raise ValueError(f"Could not read image (corrupt or not an image): {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel colour image, got shape {img.shape}")
    return img


def detections_from_yolo(result) -> list[Detection]:
    """Convert one Ultralytics YOLO result into :class:`Detection` objects.

    Args:
        result: A single ``ultralytics.engine.results.Results`` object, e.g.
            ``model.predict(img)[0]``.

    Returns:
        List of detections, one per predicted box.
    """
    names = result.names
    dets = []
    for cls_id, conf, xyxy in zip(result.boxes.cls.tolist(),
                                  result.boxes.conf.tolist(),
                                  result.boxes.xyxy.tolist()):
        dets.append(Detection(names[int(cls_id)], float(conf), tuple(xyxy)))
    return dets