"""Tests for rasher.analysis on small hand-made curves with known answers."""
import math

import numpy as np
import pytest

from rasher.analysis import ap_convergence, counts_at_threshold, inspection_cost, load_curves, optimise_threshold


def make_curves():
    """One class, 10 ground-truth boxes. Recall falls linearly from 1 to 0 as the
    threshold rises; precision is a constant 0.5."""
    conf = np.linspace(0, 1, 101)
    return {
        "conf_grid": conf,
        "p_curve": np.full((1, 101), 0.5),
        "r_curve": (1 - conf)[None, :],
        "n_gt": np.array([10]),
    }


def test_counts_at_threshold_known_values():
    c = make_curves()
    tp, fp, fn = counts_at_threshold(0.3, c["conf_grid"], c["p_curve"], c["r_curve"], c["n_gt"])
    # recall 0.7 -> TP = 7, FN = 3; precision 0.5 -> FP = TP = 7
    assert math.isclose(tp[0], 7.0)
    assert math.isclose(fn[0], 3.0)
    assert math.isclose(fp[0], 7.0)


def test_counts_handle_zero_precision():
    c = make_curves()
    c["p_curve"] = np.zeros((1, 101))
    _, fp, _ = counts_at_threshold(0.5, c["conf_grid"], c["p_curve"], c["r_curve"], c["n_gt"])
    assert fp[0] == 0.0  # no division by zero


def test_inspection_cost_formula():
    c = make_curves()
    # At t = 0.3: FN = 3, FP = 7 -> cost = 5*3 + 1*7 = 22
    assert math.isclose(inspection_cost(0.3, c, w_miss=5, w_false_alarm=1), 22.0)


def test_optimiser_agrees_with_grid_on_unimodal_cost():
    # cost(t) = 5*(10t) + 1*10(1-t) = 40t + 10 -> minimum at the lower bound
    c = make_curves()
    res = optimise_threshold(c, w_miss=5, w_false_alarm=1, bounds=(0.01, 0.9))
    assert abs(res["t_golden"] - res["t_grid"]) < 1e-3
    assert res["golden_minus_grid_cost"] < 1e-2


def test_ap_convergence_reference_is_exact_for_piecewise_linear():
    # precision = 1 - recall: every method and resolution gives exactly 0.5
    recall = np.linspace(0, 1, 1000)
    conv = ap_convergence(recall, 1 - recall, n_points=(11, 101))
    assert np.allclose(conv["trapezoid"], 0.5)
    assert np.allclose(conv["trap_error"], 0.0, atol=1e-12)


def test_load_curves_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_curves(tmp_path / "nope.npz")


def test_load_curves_missing_arrays(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, recall_grid=np.zeros(3))
    with pytest.raises(KeyError, match="missing arrays"):
        load_curves(path)
