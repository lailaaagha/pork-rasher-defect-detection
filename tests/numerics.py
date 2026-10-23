"""Numerical methods used to evaluate and tune the defect detector.

This module implements, from scratch, methods taught in MMA3001:

* **Numerical integration** (Newton-Cotes): the composite trapezoidal rule and
  composite Simpson's 1/3 rule. These are used to compute Average Precision
  (AP), which is the area under a precision-recall curve, and to check the
  value reported by the Ultralytics library.
* **Optimisation**: golden-section search, used to choose the confidence
  threshold that minimises an inspection cost.

All functions work on plain NumPy arrays so they can be verified against
integrals and minima with known exact answers (see ``tests/test_numerics.py``).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _check_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert sample points for integration.

    Raises:
        ValueError: If ``x`` and ``y`` differ in length, have fewer than two
            points, contain NaN/inf, or ``x`` is not strictly increasing.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError(f"x and y must be 1-D arrays of equal length, got {x.shape} and {y.shape}")
    if x.size < 2:
        raise ValueError("At least two points are needed to integrate")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must not contain NaN or infinite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    return x, y


def trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Integrate sampled data with the composite trapezoidal rule.

    Each interval is approximated by a straight line, so the integral is the
    sum of trapezium areas ``0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])``.
    Works for uneven spacing. The error for a smooth function is O(h^2),
    where h is the spacing.

    Args:
        x: Strictly increasing sample locations.
        y: Function values at ``x``.

    Returns:
        Approximate value of the integral from ``x[0]`` to ``x[-1]``.

    Raises:
        ValueError: If the inputs are invalid (see :func:`_check_xy`).

    Example:
        >>> trapezoid([0, 1, 2], [0, 1, 4])
        3.0
    """
    x, y = _check_xy(x, y)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def simpson(x: np.ndarray, y: np.ndarray) -> float:
    """Integrate sampled data with the composite Simpson's 1/3 rule.

    Each pair of intervals is approximated by a quadratic (parabola) through
    three points. The error for a smooth function is O(h^4), so it usually
    converges much faster than the trapezoidal rule, but only if the function
    is smooth.

    Args:
        x: Equally spaced, strictly increasing sample locations. The number of
            points must be odd (an even number of intervals).
        y: Function values at ``x``.

    Returns:
        Approximate value of the integral from ``x[0]`` to ``x[-1]``.

    Raises:
        ValueError: If the inputs are invalid, the spacing is not uniform, or
            the number of points is even.

    Example:
        >>> simpson([0, 1, 2], [0, 1, 4])  # exact for a quadratic: 8/3
        2.6666666666666665
    """
    x, y = _check_xy(x, y)
    if x.size % 2 == 0:
        raise ValueError(f"Simpson's rule needs an odd number of points, got {x.size}")
    h = np.diff(x)
    if not np.allclose(h, h[0], rtol=1e-9, atol=1e-12):
        raise ValueError("Simpson's rule needs equally spaced x values")
    h = h[0]
    # Weights 1, 4, 2, 4, 2, ..., 4, 1 (times h/3)
    return float(h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1]))


def precision_envelope(precision: np.ndarray) -> np.ndarray:
    """Make a precision curve non-increasing (the standard AP envelope).

    For each recall level, precision is replaced by the highest precision
    achieved at that recall *or any higher recall*. This removes the
    zig-zags in a raw precision-recall curve, as done by Pascal VOC, COCO and
    Ultralytics before computing AP.

    Args:
        precision: Precision values ordered by increasing recall.

    Returns:
        The non-increasing envelope, same shape as ``precision``.
    """
    p = np.asarray(precision, dtype=float)
    return np.flip(np.maximum.accumulate(np.flip(p)))


def average_precision(recall: np.ndarray, precision: np.ndarray,
                      n_points: int = 101, method: str = "trapezoid") -> float:
    """Average Precision (AP): the area under the precision-recall curve.

    The curve is padded to start at recall 0 and end at recall 1, converted to
    its non-increasing envelope, resampled at ``n_points`` equally spaced
    recall values by linear interpolation, then integrated numerically.

    Args:
        recall: Recall values in [0, 1], non-decreasing.
        precision: Precision values in [0, 1] at each recall value.
        n_points: Number of equally spaced recall samples (101 = COCO style).
        method: ``"trapezoid"`` or ``"simpson"``.

    Returns:
        AP in [0, 1].

    Raises:
        ValueError: If ``method`` is unknown, ``n_points`` < 2, or the curves
            have different lengths.
    """
    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    if recall.shape != precision.shape:
        raise ValueError("recall and precision must have the same shape")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = precision_envelope(np.concatenate(([1.0], precision, [0.0])))
    grid = np.linspace(0.0, 1.0, n_points)
    p_grid = np.interp(grid, mrec, mpre)

    if method == "trapezoid":
        return trapezoid(grid, p_grid)
    if method == "simpson":
        return simpson(grid, p_grid)
    raise ValueError(f"Unknown method '{method}', use 'trapezoid' or 'simpson'")


def golden_section_minimise(f: Callable[[float], float], a: float, b: float,
                            tol: float = 1e-4, max_iter: int = 200) -> tuple[float, float, int]:
    """Find the minimum of a 1-D function on [a, b] by golden-section search.

    The bracket is shrunk by the golden ratio (~0.618) each iteration, keeping
    the sub-interval that must contain the minimum. Requires ``f`` to be
    unimodal (one minimum) on [a, b]; otherwise it finds a local minimum.

    Args:
        f: Function of one variable to minimise.
        a: Lower end of the search interval.
        b: Upper end of the search interval (must be > ``a``).
        tol: Stop when the bracket width is below this value (convergence
            criterion).
        max_iter: Safety limit on the number of iterations.

    Returns:
        ``(x_min, f(x_min), iterations)``.

    Raises:
        ValueError: If ``b <= a`` or ``tol <= 0``.
        RuntimeError: If it does not converge within ``max_iter`` iterations.
    """
    if b <= a:
        raise ValueError(f"Need a < b, got a={a}, b={b}")
    if tol <= 0:
        raise ValueError("tol must be positive")

    inv_phi = (math.sqrt(5) - 1) / 2  # 1/phi ~ 0.618
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for i in range(1, max_iter + 1):
        if fc <= fd:          # minimum lies in [a, d]
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:                 # minimum lies in [c, b]
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
        if b - a < tol:
            x_min = (a + b) / 2
            return x_min, f(x_min), i
    raise RuntimeError(f"Golden-section search did not converge in {max_iter} iterations")
