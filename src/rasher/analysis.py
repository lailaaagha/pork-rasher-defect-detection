"""Apply the numerical methods in :mod:`rasher.numerics` to the trained model.

Two analyses, both run on the validation-set curves exported from Colab
(``curves_<run>.npz``, see ``notebooks/``):

1. **AP verification and convergence (numerical integration).**
   AP50 is the area under the precision-recall curve. It is recomputed with
   our own trapezoidal and Simpson's rules at increasing numbers of sample
   points, and compared with the value reported by Ultralytics.

2. **Confidence-threshold optimisation (golden-section search).**
   For a threshold ``t``, the expected numbers of missed defects (false
   negatives, FN) and false alarms (false positives, FP) are reconstructed
   from the precision and recall curves. The threshold minimising
   ``cost = w_miss * FN + w_false_alarm * FP`` is found with golden-section
   search and checked against an exhaustive grid search.

Usage (from the repository root, after ``pip install -e .``)::

    python -m rasher.analysis results/analysis/curves_yolov8n_640.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rasher.numerics import golden_section_minimise, simpson, trapezoid

#: Sample counts used in the integration convergence study (odd, so Simpson works).
N_POINTS = (11, 21, 41, 81, 101, 201, 401, 1001)


def load_curves(path: str | Path) -> dict:
    """Load the curves exported by the Colab evaluation cell.

    Args:
        path: ``.npz`` file written by the notebook.

    Returns:
        Dict with arrays ``recall_grid``, ``pr_precision`` (classes x points),
        ``conf_grid``, ``p_curve``, ``r_curve``, ``ap50``, ``n_gt`` and ``names``.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If an expected array is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Curves file not found: {path}")
    data = np.load(path, allow_pickle=False)
    needed = ["recall_grid", "pr_precision", "conf_grid", "p_curve", "r_curve", "ap50", "n_gt", "names"]
    missing = [k for k in needed if k not in data.files]
    if missing:
        raise KeyError(f"{path.name} is missing arrays: {missing}")
    return {k: data[k] for k in needed}


# ---------------------------------------------------------------------------
# 1. AP verification and convergence
# ---------------------------------------------------------------------------

def ap_convergence(recall_grid: np.ndarray, precision: np.ndarray,
                   n_points=N_POINTS) -> pd.DataFrame:
    """Integrate one precision-recall curve with both rules at several resolutions.

    The reference value is the trapezoidal integral over *all* exported points.
    The exported curve is piecewise linear between those points, so the
    trapezoidal rule on the full grid is exact for it.

    Args:
        recall_grid: Recall values (increasing) at which precision is given.
        precision: Precision at each recall value.
        n_points: Numbers of equally spaced samples to try.

    Returns:
        Table with columns ``n_points``, ``h``, ``trapezoid``, ``simpson``,
        ``trap_error`` and ``simp_error`` (absolute error vs the reference).
    """
    reference = trapezoid(recall_grid, precision)
    rows = []
    for n in n_points:
        grid = np.linspace(recall_grid[0], recall_grid[-1], n)
        p = np.interp(grid, recall_grid, precision)
        t, s = trapezoid(grid, p), simpson(grid, p)
        rows.append({"n_points": n, "h": grid[1] - grid[0], "trapezoid": t, "simpson": s,
                     "trap_error": abs(t - reference), "simp_error": abs(s - reference)})
    return pd.DataFrame(rows)


def verify_ap(curves: dict) -> pd.DataFrame:
    """Compare our AP50 (trapezoid and Simpson, 101 points) with Ultralytics' AP50.

    Args:
        curves: Output of :func:`load_curves`.

    Returns:
        One row per class with the library value, our two values and the
        absolute differences.
    """
    rows = []
    for i, name in enumerate(curves["names"]):
        conv = ap_convergence(curves["recall_grid"], curves["pr_precision"][i], n_points=(101,))
        lib = float(curves["ap50"][i])
        ours_t, ours_s = conv.loc[0, "trapezoid"], conv.loc[0, "simpson"]
        rows.append({"class": str(name), "AP50_ultralytics": lib,
                     "AP50_trapezoid": ours_t, "AP50_simpson": ours_s,
                     "diff_trapezoid": abs(ours_t - lib), "diff_simpson": abs(ours_s - lib)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Confidence-threshold optimisation
# ---------------------------------------------------------------------------

def counts_at_threshold(t: float, conf_grid: np.ndarray, p_curve: np.ndarray,
                        r_curve: np.ndarray, n_gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct TP, FP and FN per class at confidence threshold ``t``.

    Uses ``recall = TP / n_gt`` and ``precision = TP / (TP + FP)``, so
    ``TP = R * n_gt``, ``FN = n_gt - TP`` and ``FP = TP * (1 - P) / P``.

    Args:
        t: Confidence threshold in [0, 1].
        conf_grid: Thresholds at which the curves are sampled.
        p_curve: Precision vs threshold, shape (classes, len(conf_grid)).
        r_curve: Recall vs threshold, same shape.
        n_gt: Number of ground-truth boxes per class.

    Returns:
        ``(tp, fp, fn)`` arrays, one value per class (may be non-integer
        because the curves are interpolated).
    """
    p = np.array([np.interp(t, conf_grid, row) for row in p_curve])
    r = np.array([np.interp(t, conf_grid, row) for row in r_curve])
    tp = r * n_gt
    fn = n_gt - tp
    # Where precision is 0 there are no true positives, so FP cannot be recovered; use 0.
    fp = np.divide(tp * (1 - p), p, out=np.zeros_like(tp), where=p > 0)
    return tp, fp, fn


def inspection_cost(t: float, curves: dict, w_miss: float = 5.0, w_false_alarm: float = 1.0) -> float:
    """Total inspection cost at threshold ``t``, summed over all classes.

    Args:
        t: Confidence threshold.
        curves: Output of :func:`load_curves`.
        w_miss: Cost of one missed defect (false negative).
        w_false_alarm: Cost of one false alarm (false positive).

    Returns:
        ``w_miss * sum(FN) + w_false_alarm * sum(FP)``.
    """
    _, fp, fn = counts_at_threshold(t, curves["conf_grid"], curves["p_curve"],
                                    curves["r_curve"], curves["n_gt"])
    return float(w_miss * fn.sum() + w_false_alarm * fp.sum())


def optimise_threshold(curves: dict, w_miss: float = 5.0, w_false_alarm: float = 1.0,
                       bounds: tuple[float, float] = (0.01, 0.9)) -> dict:
    """Find the cost-minimising threshold by golden-section search, and verify it.

    Args:
        curves: Output of :func:`load_curves`.
        w_miss: Cost of a missed defect.
        w_false_alarm: Cost of a false alarm.
        bounds: Search interval for the threshold.

    Returns:
        Dict with the golden-section result, the grid-search result, the cost
        at the default threshold (0.25) and the number of function evaluations.
    """
    f = lambda t: inspection_cost(t, curves, w_miss, w_false_alarm)
    t_gs, c_gs, iters = golden_section_minimise(f, *bounds, tol=1e-4)

    grid = np.linspace(*bounds, 2000)
    costs = np.array([f(t) for t in grid])
    t_grid = float(grid[costs.argmin()])

    return {"w_miss": w_miss, "w_false_alarm": w_false_alarm,
            "t_golden": t_gs, "cost_golden": c_gs, "golden_iterations": iters,
            "golden_evaluations": iters + 3,
            "t_grid": t_grid, "cost_grid": float(costs.min()), "grid_evaluations": grid.size,
            # > 0 means golden-section stopped in a local minimum (cost curve not unimodal)
            "golden_minus_grid_cost": c_gs - float(costs.min()),
            "cost_default_0.25": f(0.25)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(curves_path: str | Path, out_dir: str | Path = "results") -> None:
    """Run both analyses and save tables and figures to ``out_dir``."""
    curves = load_curves(curves_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [str(n) for n in curves["names"]]

    # --- 1. AP verification ---
    ver = verify_ap(curves)
    ver.to_csv(out_dir / "ap_verification.csv", index=False)
    print("=== AP50: Ultralytics vs our own integration (101 points) ===")
    print(ver.round(5).to_string(index=False))

    # Convergence study for every class
    fig, ax = plt.subplots(figsize=(7, 4.5))
    conv_all = []
    for i, name in enumerate(names):
        conv = ap_convergence(curves["recall_grid"], curves["pr_precision"][i])
        conv.insert(0, "class", name)
        conv_all.append(conv)
        if conv["trap_error"].max() > 0:
            ax.loglog(conv["h"], conv["trap_error"], "o-", label=f"{name} (trapezoid)")
            ax.loglog(conv["h"], conv["simp_error"], "s--", alpha=0.6, label=f"{name} (Simpson)")
    conv_all = pd.concat(conv_all, ignore_index=True)
    conv_all.to_csv(out_dir / "ap_convergence.csv", index=False)
    h = np.array([conv_all["h"].min(), conv_all["h"].max()])
    ax.loglog(h, 0.3 * h, "k--", label="O(h) reference")
    ax.loglog(h, 0.5 * h**2, "k:", label="O(h²) reference")
    ax.set_xlabel("Sample spacing h (recall)")
    ax.set_ylabel("|AP error| vs full-resolution integral")
    ax.set_title("Convergence of AP50 integration")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "ap_convergence.png", dpi=150)
    print(f"\nSaved convergence table and plot to {out_dir}")

    # --- 2. Threshold optimisation ---
    rows = [optimise_threshold(curves, w_miss=w) for w in (1, 2, 5, 10)]
    opt = pd.DataFrame(rows)
    opt.to_csv(out_dir / "threshold_optimisation.csv", index=False)
    print("\n=== Optimal confidence threshold (golden-section vs grid search) ===")
    print(opt.round(4).to_string(index=False))

    ts = np.linspace(0.01, 0.9, 300)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        c = [inspection_cost(t, curves, r["w_miss"]) for t in ts]
        line, = ax.plot(ts, c, label=f"miss cost = {r['w_miss']}×")
        ax.plot(r["t_golden"], r["cost_golden"], "o", color=line.get_color())
    ax.axvline(0.25, color="grey", ls=":", label="default 0.25")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Inspection cost (validation set)")
    ax.set_title("Cost vs threshold; dots = golden-section minimum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_cost.png", dpi=150)
    print(f"\nSaved threshold table and plot to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("curves", help="Path to curves_<run>.npz exported from Colab")
    parser.add_argument("--out", default="results", help="Output folder (default: results)")
    args = parser.parse_args()
    main(args.curves, args.out)
