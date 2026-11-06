"""
Very basic plot script for looking at a vin_*.pkl file's raw columns.

Just loads one .pkl, converts it to a plain numpy array, and plots the
requested columns vs. hours of operation (samples are 30 s apart). No unit
conversion, no scaling -- Y axis shows exactly what's stored in the pickle.

Meant as a quick "what does this data look like" tool, not an analysis.
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
EV_ID = 56
VIN_FILE = "vin_2.pkl"
COLUMNS = [1, 2, 3] # None = plot all columns; or e.g. [1, 2, 3] to plot specific ones

CELL_ID = 2       # which cell-voltage column (1..111) to use in the algorithm below
CURRENT_COL = 224  # pack current [A], signed (see plot_qas.py for the full vin_2 column layout)

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


# ===========================================================================
# ALGORITHM -- playground for deriving new variables from raw columns
#
# Edit this freely: combine columns, take means/diffs/ratios, whatever you
# want to try. Must return a 1D array of length data.shape[0] (one value per
# timestep). The result gets plotted as one extra line alongside the raw
# columns, using the same time axis -- nothing else about the plot changes.
# ===========================================================================
def algorithm(data: np.ndarray) -> np.ndarray:
    """Derive a 1D variable from `data` (shape (T, n_cols)). Voltage / current for cell CELL_ID."""
    voltage = data[:, CELL_ID]
    current = data[:, CURRENT_COL]
    return voltage / current


def plot_columns(data: np.ndarray, columns=None, algorithm_output=None,
                  title: str = "", out_name: str = "plot") -> None:
    """Plot the given columns of `data` against hours of operation. Raw values, no scaling."""
    if columns is None:
        columns = range(data.shape[1])

    time_hours = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S / 3600.0

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in columns:
        ax.plot(time_hours, data[:, col], label=f"col_{col}", linewidth=0.8)

    if algorithm_output is not None:
        ax.plot(time_hours, algorithm_output, label="algorithm_output",
                 color=RWTHColorScales.red[0], linewidth=1.8, zorder=5)

    ax.set_xlabel("Hours of operation")
    ax.set_title(title)
    if len(columns) <= 15:
        ax.legend(fontsize=8)

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    data = load_vin(EV_ID, VIN_FILE)
    print(f"Loaded {VIN_FILE} for EV {EV_ID}: shape {data.shape}")

    derived = algorithm(data)

    plot_columns(
        data,
        columns=COLUMNS,
        algorithm_output=derived,
        title=f"EV #{EV_ID} - {VIN_FILE} (cell {CELL_ID} voltage / current)",
        out_name=f"ev{EV_ID}_{Path(VIN_FILE).stem}_cell{CELL_ID}_v_over_i",
    )
