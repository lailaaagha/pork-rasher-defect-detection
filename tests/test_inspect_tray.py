"""Tests for rasher.inspect_tray: decision logic and input validation."""
import cv2
import numpy as np
import pytest

from rasher.inspect_tray import Detection, inspect_tray, load_image

BOX = (10.0, 10.0, 50.0, 50.0)


# ---------- Decision logic ----------

def test_no_detections_passes():
    report = inspect_tray([])
    assert report.decision == "PASS"
    assert report.meat_ok and report.packaging_ok


def test_packaging_defect_fails_packaging_only():
    report = inspect_tray([Detection("unsealed", 0.9, BOX)])
    assert report.decision == "FAIL"
    assert report.meat_ok is True
    assert report.packaging_ok is False


def test_meat_defect_fails_meat_only():
    report = inspect_tray([Detection("twisted-meat", 0.9, BOX)])
    assert report.decision == "FAIL"
    assert report.meat_ok is False
    assert report.packaging_ok is True


def test_low_confidence_detection_is_ignored():
    report = inspect_tray([Detection("unsealed", 0.10, BOX)], conf_threshold=0.25)
    assert report.decision == "PASS"
    assert report.defects == ()


def test_detection_exactly_at_threshold_counts():
    report = inspect_tray([Detection("wrinkle", 0.25, BOX)], conf_threshold=0.25)
    assert report.decision == "FAIL"


def test_only_kept_detections_are_reported():
    dets = [Detection("unsealed", 0.9, BOX), Detection("loose-meat", 0.1, BOX)]
    report = inspect_tray(dets, conf_threshold=0.5)
    assert [d.class_name for d in report.defects] == ["unsealed"]
    assert report.meat_ok is True


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="conf_threshold"):
        inspect_tray([], conf_threshold=threshold)


# ---------- Detection validation ----------

def test_unknown_class_rejected():
    with pytest.raises(ValueError, match="Unknown class"):
        Detection("mould", 0.9, BOX)


@pytest.mark.parametrize("conf", [-0.01, 1.01])
def test_confidence_outside_range_rejected(conf):
    with pytest.raises(ValueError, match="Confidence"):
        Detection("unsealed", conf, BOX)


def test_zero_size_box_rejected():
    with pytest.raises(ValueError, match="positive width"):
        Detection("unsealed", 0.9, (10, 10, 10, 50))


# ---------- Image loading ----------

def test_load_valid_image(tmp_path):
    path = tmp_path / "tray.png"
    cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))
    img = load_image(path)
    assert img.shape == (20, 30, 3)


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "does_not_exist.jpg")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_image(path)


def test_corrupt_image_raises(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"this is not really a jpeg")
    with pytest.raises(ValueError, match="Could not read image"):
        load_image(path)
