# %%
"""
Plot per-cell mean resistance vs. SoC for a single EV.

Pure plotting -- no resistance calculation here. mean_R(t), std_R(t) and the
aligned pack SoC(t) are all loaded straight from
extracted_data/resistance_timeseries.pkl, produced by resistance.py (see
that file for the full derivation and sign convention).

Two subplots, same layout as resplot.py: (1) raw mean_R vs. SoC scatter,
(2) mean_R averaged per SoC percent, scattered, with a bathtub-curve fit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/00_.../cr7.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
TIMESERIES_IN = PROJECT_ROOT / "extracted_data" / "resistance_timeseries.pkl"  # from resistance.py

EV_ID = 256

# Bathtub curve fit (see BATHTUB MODEL below): constrains the minimum to fall
# within this SoC range and the flat-vs-steep shape to this power range.
CURVE_FIT_CENTER_BOUNDS = (20.0, 80.0)
CURVE_FIT_POWER_BOUNDS = (2.0, 12.0)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def mean_per_soc(soc: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of `values` per integer SoC value, sorted by SoC. SoC is rounded
    to the nearest percent to group the (continuous) raw samples into
    buckets -- same approach as resplot.py's mean_resistance_per_soc."""
    soc_rounded = np.round(soc).astype(int)
    means = pd.Series(values).groupby(soc_rounded).mean().sort_index()
    return means.index.to_numpy(dtype=float), means.to_numpy()


# ===========================================================================
# BATHTUB MODEL -- R(soc) = r_min + amplitude * (|soc - center| / 50)^power
# (identical to resplot.py's) -- flat around mid-range SoC, rising toward 0%
# and 100%. `center` (bounded to CURVE_FIT_CENTER_BOUNDS) pins the minimum
# inside the 20-80% band; `power` (bounded to CURVE_FIT_POWER_BOUNDS)
# controls how flat the middle is vs. how sharply the ends rise -- higher
# power = flatter middle, steeper ends. `amplitude >= 0` keeps it a single
# trough (never a hump) on both sides of `center`.
# ===========================================================================


def bathtub_model(soc: np.ndarray, r_min: float, amplitude: float,
                   center: float, power: float) -> np.ndarray:
    return r_min + amplitude * (np.abs(soc - center) / 50.0) ** power


def fit_bathtub_curve(soc: np.ndarray, r_values: np.ndarray) -> np.ndarray:
    """Fit the bathtub model to the (soc, r_values) samples passed in;
    returns the fitted params."""
    p0 = [r_values.min(), r_values.max() - r_values.min(), 50.0, 6.0]
    bounds = (
        [0.0, 0.0, CURVE_FIT_CENTER_BOUNDS[0], CURVE_FIT_POWER_BOUNDS[0]],
        [np.inf, np.inf, CURVE_FIT_CENTER_BOUNDS[1], CURVE_FIT_POWER_BOUNDS[1]],
    )
    params, _ = curve_fit(bathtub_model, soc, r_values, p0=p0, bounds=bounds)
    return params


def plot_mean_resistance_vs_soc(mean_R: np.ndarray, soc: np.ndarray,
                                 title: str = "", out_name: str = "cr7_resistance_vs_soc") -> None:
    """2 subplots: (1) raw mean_R vs. SoC scatter, (2) mean_R averaged per
    SoC percent, scattered, with a bathtub-shaped curve fit through it."""
    valid = ~np.isnan(mean_R)
    soc_valid = soc[valid]
    r_valid = mean_R[valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    ax1.scatter(soc_valid, r_valid, s=6, alpha=0.4, color=RWTHColorScales.cycle[0])
    ax1.set_xlabel("SoC (%)")
    ax1.set_ylabel("Mean cell resistance (Ohm)")
    ax1.set_title(title)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    soc_avg, r_avg = mean_per_soc(soc_valid, r_valid)
    ax2.scatter(soc_avg, r_avg, s=20, color=RWTHColorScales.red[0], label="mean_R per SoC")

    r_min, amplitude, center, power = fit_bathtub_curve(soc_avg, r_avg)
    soc_fit = np.linspace(soc_avg.min(), soc_avg.max(), 200)
    ax2.plot(soc_fit, bathtub_model(soc_fit, r_min, amplitude, center, power),
              color=RWTHColorScales.cycle[0], linewidth=1.5,
              label=f"bathtub fit (center={center:.0f}%, power={power:.1f})")

    ax2.set_xlabel("SoC (%)")
    ax2.set_ylabel("Mean cell resistance, averaged per SoC (Ohm)")
    ax2.set_title("Mean cell resistance per SoC")
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    ax2.legend()

    fig.tight_layout()

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    print(f"Processing EV ID: {EV_ID}")

    timeseries_df = pd.read_pickle(TIMESERIES_IN)
    mean_R = timeseries_df.loc[EV_ID, "mean_R"]
    soc = timeseries_df.loc[EV_ID, "soc"]

    plot_mean_resistance_vs_soc(
        mean_R,
        soc,
        title=f"QAS EV {EV_ID} - Mean cell resistance vs. SoC",
        out_name=f"qas_cr7_ev{EV_ID}_resistance_vs_soc",
    )
