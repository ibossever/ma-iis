"""
Plot pack internal resistance vs. SoC for a single EV.

Loads one EV's vin_2.pkl, derives pack resistance R = delta_V_pack / delta_I
between consecutive samples (see rithmalgo section below), and scatters R
against the SoC at each point. Samples are 30 s apart.

QAS & DTI both here
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/00_.../plot_vin.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
EV_ID = 216
VIN_FILE = "vin_2.pkl"

VOLTAGE_COLS = slice(2, 112)  # 110 series cell voltages [V]
SOC_COL = 223   # SoC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], signed: positive = charge, negative = discharge

DI_THRESHOLD_A = 1.0  # mask out points where |delta pack current| is below this (near-rest -> unstable R)

# Bathtub curve fit (see BATHTUB MODEL below): constrains the minimum to fall
# within this SoC range and the flat-vs-steep shape to this power range.
CURVE_FIT_CENTER_BOUNDS = (20.0, 80.0)
CURVE_FIT_POWER_BOUNDS = (2.0, 12.0)

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


# ===========================================================================
# RITHMALGO -- pack internal resistance from consecutive samples
# ===========================================================================
# ALGORITHM -- point-to-point resistance: R = delta_V_pack / delta_I, where
# pack voltage is the sum of the 110 series cell voltages. Current here is
# signed positive-for-charge, so V rises with I and R comes out positive
# directly (no sign flip needed). Points where |delta_I| < DI_THRESHOLD_A are
# masked to NaN since dividing by a near-zero delta_I blows up into spurious
# huge R values.
# ===========================================================================


def pack_resistance(data: np.ndarray) -> np.ndarray:
    """R_pack(t) for consecutive samples, shape (T-1,). NaN where masked."""
    current = data[:, CURRENT_COL]  # ensure: charge = positive, discharge = negative
    pack_voltage = np.sum(data[:, VOLTAGE_COLS], axis=1)

    delta_i = np.diff(current)
    delta_v = np.diff(pack_voltage)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_pack = delta_v / delta_i   # V = OCV + I*R convention

    r_pack[np.abs(delta_i) < DI_THRESHOLD_A] = np.nan
    r_pack[r_pack <= 0] = np.nan
    return r_pack


def soc_percent(data: np.ndarray) -> np.ndarray:
    """Pack SoC in percent (col 223 is stored as a 0-1 fraction)."""
    return data[:, SOC_COL] * 100.0


def mean_resistance_per_soc(soc: np.ndarray, r_pack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean R per integer SoC value, sorted by SoC. SoC is rounded to the
    nearest percent to group the (continuous) raw samples into buckets."""
    soc_rounded = np.round(soc).astype(int)
    means = pd.Series(r_pack).groupby(soc_rounded).mean().sort_index()
    return means.index.to_numpy(dtype=float), means.to_numpy()


# ===========================================================================
# BATHTUB MODEL -- R(soc) = r_min + amplitude * (|soc - center| / 50)^power
# ===========================================================================
# Matches the expected physical shape: flat around mid-range SoC, rising
# toward 0% and 100%. `center` (bounded to CURVE_FIT_CENTER_BOUNDS) pins the
# minimum inside the 20-80% band; `power` (bounded to CURVE_FIT_POWER_BOUNDS)
# controls how flat the middle is vs. how sharply the ends rise -- higher
# power = flatter middle, steeper ends. `amplitude >= 0` keeps it a single
# trough (never a hump) on both sides of `center`.
# ===========================================================================


def bathtub_model(soc: np.ndarray, r_min: float, amplitude: float,
                   center: float, power: float) -> np.ndarray:
    return r_min + amplitude * (np.abs(soc - center) / 50.0) ** power


def fit_bathtub_curve(soc: np.ndarray, r_pack: np.ndarray) -> np.ndarray:
    """Fit the bathtub model to the raw (soc, r_pack) samples; returns the
    fitted params. Fit against the raw points rather than the per-SoC means
    so the curve isn't skewed by how densely each SoC value was sampled."""
    p0 = [r_pack.min(), r_pack.max() - r_pack.min(), 50.0, 6.0]
    bounds = (
        [0.0, 0.0, CURVE_FIT_CENTER_BOUNDS[0], CURVE_FIT_POWER_BOUNDS[0]],
        [np.inf, np.inf, CURVE_FIT_CENTER_BOUNDS[1], CURVE_FIT_POWER_BOUNDS[1]],
    )
    params, _ = curve_fit(bathtub_model, soc, r_pack, p0=p0, bounds=bounds)
    return params


def plot_resistance_vs_soc(data: np.ndarray, title: str = "",
                            out_name: str = "resistance_vs_soc") -> None:
    """Scatter pack internal resistance against SoC, plus a subplot of the
    mean R per SoC with a bathtub-shaped curve fit through it."""
    r_pack = pack_resistance(data)
    soc = soc_percent(data)[1:]  # align with delta_* (post-transition SoC)

    valid = ~np.isnan(r_pack)
    soc_valid = soc[valid]
    r_valid = r_pack[valid]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

    ax1.scatter(soc_valid, r_valid, s=6, alpha=0.4,
                color=RWTHColorScales.cycle[0])
    ax1.set_xlabel("SoC (%)")
    ax1.set_ylabel("Pack internal resistance (Ohm)")
    ax1.set_title(title)
    ax1.grid(True, linewidth=0.4, alpha=0.5)

    soc_avg, r_avg = mean_resistance_per_soc(soc_valid, r_valid)
    ax2.scatter(soc_avg, r_avg, s=20, color=RWTHColorScales.red[0], label="mean R per SoC")

    r_min, amplitude, center, power = fit_bathtub_curve(soc_valid, r_valid)
    soc_fit = np.linspace(soc_avg.min(), soc_avg.max(), 200)
    ax2.plot(soc_fit, bathtub_model(soc_fit, r_min, amplitude, center, power),
              color=RWTHColorScales.cycle[0], linewidth=1.5,
              label=f"bathtub fit (center={center:.0f}%, power={power:.1f})")

    ax2.set_xlabel("SoC (%)")
    ax2.set_ylabel("Mean pack internal resistance (Ohm)")
    ax2.set_title("Mean resistance per SoC")
    ax2.grid(True, linewidth=0.4, alpha=0.5)
    ax2.legend()

    fig.tight_layout()

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    print(f"Processing EV ID: {EV_ID}")

    data = load_vin(EV_ID, VIN_FILE)
    print(f"Loaded {VIN_FILE} for EV {EV_ID}: shape {data.shape}")

    plot_resistance_vs_soc(
        data,
        title=f"QAS EV {EV_ID} - pack internal resistance vs. SoC",
        out_name=f"qas_ev{EV_ID}_resistance_vs_soc",
    )
