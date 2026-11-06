"""
Plot pack temperature derivative, SoC, and cell-voltage stats for a
hand-picked EV (single-EV use -- see plot_temp_derivatives docstring).

Loads the selected EV's vin_2.pkl and produces 3 stacked subplots vs. hours
of operation: (1) dT/dt + raw pack temperature, (2) SoC, (3) mean pack
voltage + std of cell voltages per timestep. See plot_qas.py for the full
vin_2 column layout. Samples are 30 s apart.

QAS & DTI both here
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

EV_IDS = [347]  # EVs to plot -- add/remove ids here, e.g. [382, 79, 30]

ZOOM_START_HOUR = 968.5    # start of the zoomed-in window, in hours of operation
ZOOM_END_HOUR = 970.0     # end of the zoomed-in window, in hours of operation

TEMP_COL = 222     # pack temperature [degC]
SOC_COL = 223      # SoC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], positive = discharge
VOLTAGE_COLS = slice(2, 112)  # 110 series cell voltages [V]

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


# ===========================================================================
# ALGORITHM -- pack temperature derivative, SoC, and cell-voltage stats
# ===========================================================================


def temp_derivative(data: np.ndarray) -> np.ndarray:
    """dT/dt for the pack temperature channel, in degC/s."""
    temp = data[:, TEMP_COL]
    time_seconds = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S
    return np.gradient(temp, time_seconds)


def soc_percent(data: np.ndarray) -> np.ndarray:
    """Pack SoC in percent (col 223 is stored as a 0-1 fraction)."""
    return data[:, SOC_COL] * 100.0


def voltage_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of the 110 series cell voltages (cols 2-111), per timestep."""
    ev_mean = np.mean(data[:, VOLTAGE_COLS], axis=1)
    ev_std = np.std(data[:, VOLTAGE_COLS], axis=1)
    return ev_mean, ev_std


# ===========================================================================
# END ALGORITHM
# ===========================================================================


def plot_temp_derivatives(ev_ids: list[int], title: str = "",
                           out_name: str = "temp_derivative",
                           time_range_hours: tuple[float, float] | None = None) -> None:
    """3 stacked subplots vs. hours of operation, for each EV in `ev_ids`:
    (1) dT/dt + raw pack temperature, (2) SoC + pack current, (3) mean pack
    voltage + std of cell voltages. Meant for a single EV -- multiple EVs
    will overlay on the same subplots.

    `time_range_hours`, if given, restricts every subplot to that
    (start, end) window of hours of operation (for a zoomed-in view)."""
    fig, (ax_temp, ax_soc, ax_volt) = plt.subplots(3, 1, figsize=(12, 15), dpi=150, sharex=True)
    ax_dtdt = ax_temp.twinx()
    ax_current = ax_soc.twinx()
    ax_std = ax_volt.twinx()

    for ev_id in ev_ids:
        data = load_vin(ev_id)
        time_hours = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S / 3600.0

        dtdt = temp_derivative(data)
        temp = data[:, TEMP_COL]
        soc = soc_percent(data)
        current = data[:, CURRENT_COL]
        ev_mean, ev_std = voltage_stats(data)

        if time_range_hours is not None:
            start, end = time_range_hours
            mask = (time_hours >= start) & (time_hours <= end)
            time_hours, dtdt, temp, soc, current, ev_mean, ev_std = (
                time_hours[mask], dtdt[mask], temp[mask], soc[mask],
                current[mask], ev_mean[mask], ev_std[mask],
            )

        ax_dtdt.plot(time_hours, dtdt, label=f"EV {ev_id} dT/dt", linewidth=1.0)
        ax_temp.plot(time_hours, temp, label=f"EV {ev_id} temp",
                     color=RWTHColorScales.red[0], linewidth=1.0, linestyle="--", alpha=0.6)

        ax_soc.plot(time_hours, soc, label=f"EV {ev_id} SoC", linewidth=1.0)
        ax_current.plot(time_hours, current, label=f"EV {ev_id} pack current",
                         color=RWTHColorScales.red[0], linewidth=1.0, linestyle="--", alpha=0.6)

        ax_volt.plot(time_hours, ev_mean, label=f"EV {ev_id} mean voltage", linewidth=1.0)
        ax_std.plot(time_hours, ev_std, label=f"EV {ev_id} voltage std",
                    color=RWTHColorScales.red[0], linewidth=1.0, linestyle="--", alpha=0.6)

    ax_temp.set_ylabel("Temperature (degC)")
    ax_dtdt.set_ylabel("dT/dt (degC/s)")
    ax_temp.set_title(title)
    ax_temp.grid(True, linewidth=0.4, alpha=0.5)
    lines1, labels1 = ax_temp.get_legend_handles_labels()
    lines2, labels2 = ax_dtdt.get_legend_handles_labels()
    ax_temp.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax_soc.set_ylabel("SoC (%)")
    ax_current.set_ylabel("Pack current (A)")
    ax_current.set_ylim(-100, 300)
    ax_soc.set_title("State of charge & pack current")
    ax_soc.grid(True, linewidth=0.4, alpha=0.5)
    lines5, labels5 = ax_soc.get_legend_handles_labels()
    lines6, labels6 = ax_current.get_legend_handles_labels()
    ax_soc.legend(lines5 + lines6, labels5 + labels6, fontsize=8)

    ax_volt.set_xlabel("Hours of operation")
    ax_volt.set_ylabel("Voltage (V)")
    ax_std.set_ylabel("Voltage std (V)")
    ax_volt.set_title("Mean pack voltage & cell-voltage std")
    ax_volt.grid(True, linewidth=0.4, alpha=0.5)
    lines3, labels3 = ax_volt.get_legend_handles_labels()
    lines4, labels4 = ax_std.get_legend_handles_labels()
    ax_volt.legend(lines3 + lines4, labels3 + labels4, fontsize=8)

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    print(f"Processing EV IDs: {EV_IDS}")

    ev_suffix = '_'.join(str(i) for i in EV_IDS)

    plot_temp_derivatives(
        EV_IDS,
        title="QAS pack temperature analysis - EV {EV_IDS}",
        out_name=f"qas_temp_derivative_ev{ev_suffix}",
    )

    plot_temp_derivatives(
        EV_IDS,
        title=f"QAS pack temperature analysis - EV {{EV_IDS}} "
              f"(zoomed: {ZOOM_START_HOUR}-{ZOOM_END_HOUR} h)",
        out_name=f"zoomed_qas_temp_derivative_ev{ev_suffix}",
        time_range_hours=(ZOOM_START_HOUR, ZOOM_END_HOUR),
    )
