"""Tests for rasher.explore: reading data.yaml and validating YOLO label files."""
import pytest

from rasher.explore import load_class_names, parse_label_file


def write(path, text):
    path.write_text(text)
    return path


# ---------- load_class_names ----------

def test_class_names_from_list(tmp_path):
    yaml_file = write(tmp_path / "data.yaml", "names: ['a', 'b', 'c']\n")
    assert load_class_names(yaml_file) == ["a", "b", "c"]


def test_class_names_from_dict(tmp_path):
    yaml_file = write(tmp_path / "data.yaml", "names:\n  1: b\n  0: a\n")
    assert load_class_names(yaml_file) == ["a", "b"]


def test_missing_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_class_names(tmp_path / "missing.yaml")


def test_yaml_without_names_raises(tmp_path):
    yaml_file = write(tmp_path / "data.yaml", "nc: 5\n")
    with pytest.raises(KeyError):
        load_class_names(yaml_file)


# ---------- parse_label_file ----------

def test_parse_normal_boxes(tmp_path):
    lbl = write(tmp_path / "img.txt", "3 0.5 0.4 0.2 0.1\n1 0.25 0.7 0.03 0.03\n")
    boxes = parse_label_file(lbl, n_classes=5)
    assert boxes == [(3, 0.5, 0.4, 0.2, 0.1), (1, 0.25, 0.7, 0.03, 0.03)]


def test_empty_file_means_no_defects(tmp_path):
    lbl = write(tmp_path / "img.txt", "")
    assert parse_label_file(lbl, n_classes=5) == []


def test_polygon_converted_to_enclosing_box(tmp_path):
    # Triangle with corners (0.2,0.2), (0.6,0.2), (0.4,0.8)
    lbl = write(tmp_path / "img.txt", "4 0.2 0.2 0.6 0.2 0.4 0.8\n")
    cls, x, y, w, h = parse_label_file(lbl, n_classes=5)[0]
    assert cls == 4
    assert abs(x - 0.4) < 1e-12
    assert abs(y - 0.5) < 1e-12
    assert abs(w - 0.4) < 1e-12
    assert abs(h - 0.6) < 1e-12


def test_unknown_class_id_rejected(tmp_path):
    lbl = write(tmp_path / "img.txt", "7 0.5 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="unknown class"):
        parse_label_file(lbl, n_classes=5)


def test_coordinates_outside_image_rejected(tmp_path):
    lbl = write(tmp_path / "img.txt", "0 1.2 0.5 0.1 0.1\n")
    with pytest.raises(ValueError, match="outside"):
        parse_label_file(lbl, n_classes=5)


def test_wrong_number_of_values_rejected(tmp_path):
    lbl = write(tmp_path / "img.txt", "0 0.5 0.5 0.1\n")
    with pytest.raises(ValueError, match="expected 4 box values"):
        parse_label_file(lbl, n_classes=5)
