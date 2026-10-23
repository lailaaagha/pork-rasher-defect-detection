"""Tests for rasher.numerics.

The integration and optimisation routines are verified against problems with
known exact answers (manufactured solutions), and their convergence rates are
checked against theory: O(h^2) for the trapezoidal rule, O(h^4) for Simpson.
"""
import math

import numpy as np
import pytest

from rasher.numerics import (
    average_precision,
    golden_section_minimise,
    precision_envelope,
    simpson,
    trapezoid,
)


# ---------- Trapezoidal rule ----------

def test_trapezoid_exact_for_straight_line():
    # The trapezoidal rule is exact for linear functions: integral of 2x+1 on [0, 3] = 12
    x = np.linspace(0, 3, 4)
    assert math.isclose(trapezoid(x, 2 * x + 1), 12.0)


def test_trapezoid_handles_uneven_spacing():
    x = np.array([0.0, 0.1, 0.5, 2.0])
    assert math.isclose(trapezoid(x, 2 * x + 1), 6.0)  # exact: x^2 + x on [0, 2]


def test_trapezoid_converges_second_order():
    # Halving h should divide the error by ~4 for a smooth function.
    exact = 2.0  # integral of sin(x) on [0, pi]
    errors = []
    for n in (11, 21, 41):
        x = np.linspace(0, math.pi, n)
        errors.append(abs(trapezoid(x, np.sin(x)) - exact))
    ratio = errors[0] / errors[1]
    assert 3.8 < ratio < 4.2


# ---------- Simpson's rule ----------

def test_simpson_exact_for_cubic():
    # Simpson's rule is exact for polynomials up to degree 3: integral of x^3 on [0, 2] = 4
    x = np.linspace(0, 2, 5)
    assert math.isclose(simpson(x, x**3), 4.0)


def test_simpson_converges_fourth_order():
    # Halving h should divide the error by ~16 for a smooth function.
    exact = 2.0
    errors = []
    for n in (11, 21):
        x = np.linspace(0, math.pi, n)
        errors.append(abs(simpson(x, np.sin(x)) - exact))
    ratio = errors[0] / errors[1]
    assert 15.0 < ratio < 17.0


def test_simpson_more_accurate_than_trapezoid_for_smooth_function():
    x = np.linspace(0, math.pi, 11)
    assert abs(simpson(x, np.sin(x)) - 2) < abs(trapezoid(x, np.sin(x)) - 2)


def test_simpson_rejects_even_number_of_points():
    with pytest.raises(ValueError, match="odd number"):
        simpson(np.linspace(0, 1, 4), np.ones(4))


def test_simpson_rejects_uneven_spacing():
    with pytest.raises(ValueError, match="equally spaced"):
        simpson(np.array([0.0, 0.1, 1.0]), np.ones(3))


# ---------- Input validation (shared) ----------

@pytest.mark.parametrize("x, y, message", [
    ([0, 1, 2], [0, 1], "equal length"),
    ([0], [0], "two points"),
    ([0, 2, 1], [0, 1, 2], "strictly increasing"),
    ([0, 1, 2], [0, np.nan, 2], "NaN"),
])
def test_integration_rejects_invalid_input(x, y, message):
    with pytest.raises(ValueError, match=message):
        trapezoid(x, y)


# ---------- Average precision ----------

def test_precision_envelope_is_non_increasing():
    env = precision_envelope(np.array([1.0, 0.5, 0.8, 0.3, 0.6, 0.0]))
    assert np.all(np.diff(env) <= 0)
    assert np.allclose(env, [1.0, 0.8, 0.8, 0.6, 0.6, 0.0])


def test_ap_perfect_detector_is_close_to_one():
    # Precision 1 at every recall level -> true area = 1. The curve is padded
    # with precision 0 at recall 1 (same convention as Ultralytics), which
    # costs half of the last sampling interval: 1 - 0.5/(n_points-1).
    recall = np.linspace(0, 1, 50)
    ap = average_precision(recall, np.ones(50), n_points=101)
    assert math.isclose(ap, 1 - 0.5 / 100)


def test_ap_diagonal_curve_is_half():
    # precision = 1 - recall is a straight line from (0,1) to (1,0): exact area 0.5.
    # Both rules are exact for a straight line.
    recall = np.linspace(0, 1, 21)
    assert math.isclose(average_precision(recall, 1 - recall), 0.5)
    assert math.isclose(average_precision(recall, 1 - recall, method="simpson"), 0.5)


def test_ap_trapezoid_and_simpson_agree_on_fine_grid():
    rng = np.random.default_rng(0)
    recall = np.sort(rng.uniform(0, 1, 30))
    precision = np.sort(rng.uniform(0, 1, 30))[::-1]
    ap_t = average_precision(recall, precision, n_points=1001, method="trapezoid")
    ap_s = average_precision(recall, precision, n_points=1001, method="simpson")
    assert abs(ap_t - ap_s) < 1e-3


def test_ap_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        average_precision(np.array([0.5]), np.array([1.0]), method="midpoint")


# ---------- Golden-section search ----------

def test_golden_section_finds_parabola_minimum():
    x_min, f_min, _ = golden_section_minimise(lambda x: (x - 0.3) ** 2 + 1, 0, 1, tol=1e-6)
    assert abs(x_min - 0.3) < 1e-5
    assert abs(f_min - 1.0) < 1e-9


def test_golden_section_minimum_at_boundary():
    # Increasing function: minimum is at the left end of the interval
    x_min, _, _ = golden_section_minimise(lambda x: x, 0, 1, tol=1e-6)
    assert x_min < 1e-5


def test_golden_section_bracket_shrinks_by_golden_ratio():
    # Width after k iterations is (b-a) * 0.618^k, so tol=1e-4 on [0,1] needs ~20 iterations
    _, _, iterations = golden_section_minimise(lambda x: (x - 0.5) ** 2, 0, 1, tol=1e-4)
    assert 18 <= iterations <= 21


def test_golden_section_rejects_bad_interval():
    with pytest.raises(ValueError):
        golden_section_minimise(lambda x: x, 1, 0)
