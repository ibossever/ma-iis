"""
Plot mean cell internal resistance vs. SoC for a set of selected EVs.

No individual cell values are used past the mean. Mean cell voltage is the
*mean* of the 110 series cell voltages at each timestep (not their sum, so
this tracks a per-cell quantity, not the summed pack voltage -- see
resistance.py's R_cell). Internal resistance is derived from the
time-derivatives of that mean voltage and of the pack current:
R(t) = -(dV/dt) / (dI/dt). Samples are 30 s apart.

Two subplots: (1) raw R vs. SoC scatter, one color per EV; (2) R smoothed
with LOWESS into a single curve per EV -- non-parametric (no assumed shape),
so it only spans each EV's actually-observed SoC range and doesn't
overreact to the handful of points near sparsely-covered SoC extremes.

QAS & DTI both here
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/00_.../plot_vin.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"

EV_IDS = [382, 371, 363, 310, 347, 58, 186]  # EVs to plot -- add/remove ids here

VOLTAGE_COLS = slice(2, 112)  # 110 series cell voltages [V]
SOC_COL = 223   # SoC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], signed: positive = discharge, negative = charge (QAS_data_format.md)

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

# Mask out points where |dI/dt| is below this (near-rest -> unstable R).
# Equivalent to requiring a >=1 A change in current over one 30 s sample gap.
DIDT_THRESHOLD_A_S = 1.0 / SAMPLE_INTERVAL_S

LOWESS_FRAC = 0.3  # fraction of points used in each local LOWESS regression --
                    # higher = flatter/smoother, lower = follows point-to-point noise more closely

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


# ===========================================================================
# ALGORITHM -- step by step:
# 1. mean_cell_voltage(t) = mean of the 110 series cell voltages at each
#    timestep (np.mean, NOT sum -- this tracks a per-cell quantity, not the
#    summed pack voltage; no individual cell values used past this point).
# 2. time_seconds = np.arange(T) * SAMPLE_INTERVAL_S: the real timestamp of
#    every row (30 s apart).
# 3. dV/dt = np.gradient(mean_cell_voltage, time_seconds): time-derivative
#    of mean cell voltage, central differences, same length T.
# 4. dI/dt = np.gradient(current, time_seconds): time-derivative of pack
#    current, same way.
# 5. R(t) = -(dV/dt) / (dI/dt). Current here is signed positive-for-discharge
#    (QAS_data_format.md), so V_terminal = OCV - I*R and R = -dV/dI for a
#    purely ohmic response to come out positive -- same sign convention as
#    resistance.py's R_cell.
# 6. Mask: points where |dI/dt| < DIDT_THRESHOLD_A_S are set to NaN, since
#    dividing by a near-zero dI/dt blows up into spurious huge R values.
# 7. Mask: non-positive R values (R <= 0) are set to NaN -- not physically a
#    valid internal resistance, so dropped rather than kept as noise.
# ===========================================================================


def mean_cell_voltage(data: np.ndarray) -> np.ndarray:
    """Mean of the 110 series cell voltages at each timestep, shape (T,)."""
    return np.mean(data[:, VOLTAGE_COLS], axis=1)


def mean_cell_resistance(data: np.ndarray) -> np.ndarray:
    """R(t) = -(dV/dt) / (dI/dt) for the mean cell voltage, shape (T,). NaN where masked."""
    time_seconds = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S

    voltage = mean_cell_voltage(data)
    current = data[:, CURRENT_COL]

    dv_dt = np.gradient(voltage, time_seconds)
    di_dt = np.gradient(current, time_seconds)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_r_cell = -dv_dt / di_dt

    mean_r_cell[np.abs(di_dt) < DIDT_THRESHOLD_A_S] = np.nan
    mean_r_cell[mean_r_cell <= 0] = np.nan
    return mean_r_cell


def soc_percent(data: np.ndarray) -> np.ndarray:
    """Pack SoC in percent (col 223 is stored as a 0-1 fraction)."""
    return data[:, SOC_COL] * 100.0


def smoothed_resistance_curve(soc: np.ndarray, mean_r_cell: np.ndarray,
                               frac: float = LOWESS_FRAC) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS-smoothed R vs. SoC curve over the raw valid points, sorted by
    SoC. Non-parametric: makes no assumption about the curve's shape (e.g.
    flat middle / rising ends), so it only covers the SoC range this EV
    actually has data for, and naturally stays closer to the underlying
    points -- rather than a fixed shape -- wherever data is sparse."""
    valid = ~np.isnan(mean_r_cell)
    smoothed = lowess(mean_r_cell[valid], soc[valid], frac=frac, return_sorted=True)
    return smoothed[:, 0], smoothed[:, 1]


# ===========================================================================
# END ALGORITHM
# ===========================================================================


def plot_resistance_vs_soc(ev_ids: list[int], title: str = "",
                            out_name: str = "packr_vs_soc") -> None:
    """2 subplots: (1) raw mean cell R vs. SoC scatter per EV, (2) per-EV
    LOWESS-smoothed R-vs-SoC curve."""
    fig, (ax_raw, ax_avg) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)

    for ev_id in ev_ids:
        data = load_vin(ev_id)
        soc = soc_percent(data)
        mean_r_cell = mean_cell_resistance(data)

        valid = ~np.isnan(mean_r_cell)
        ax_raw.scatter(soc[valid], mean_r_cell[valid], s=6, alpha=0.4, label=f"EV {ev_id}")

        soc_smooth, r_smooth = smoothed_resistance_curve(soc, mean_r_cell)
        ax_avg.plot(soc_smooth, r_smooth, linewidth=1.5, label=f"EV {ev_id}")

    ax_raw.set_xlabel("SoC (%)")
    ax_raw.set_ylabel("Mean cell internal resistance (Ohm)")
    ax_raw.set_title(title)
    ax_raw.grid(True, linewidth=0.4, alpha=0.5)
    ax_raw.legend(fontsize=8)

    ax_avg.set_xlabel("SoC (%)")
    ax_avg.set_ylabel("Mean cell internal resistance (Ohm)")
    ax_avg.set_title(f"LOWESS-smoothed (frac={LOWESS_FRAC:g})")
    ax_avg.grid(True, linewidth=0.4, alpha=0.5)
    ax_avg.legend(fontsize=8)

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    print(f"Processing EV IDs: {EV_IDS}")

    plot_resistance_vs_soc(
        EV_IDS,
        title="QAS mean cell internal resistance vs. SoC",
        out_name=f"qas_packr_vs_soc_ev{'_'.join(str(i) for i in EV_IDS)}",
    )
