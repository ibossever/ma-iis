"""
Very basic plot script for looking at a vin_*.pkl file's raw columns.

Just loads one .pkl, converts it to a plain numpy array, and plots the
requested columns vs. hours of operation (samples are 30 s apart). No unit
conversion, no scaling -- Y axis shows exactly what's stored in the pickle.

Meant as a quick "what does this data look like" tool, not an analysis.

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
EV_ID = 382
VIN_FILE = "vin_2.pkl"
COLUMNS = None

print(f"Processing EV ID: {EV_ID}")

TEMP_COL = 222  
SOC_COL = 223   
CURRENT_COL = 224  # pack current [A] for QAS, signed (see plot_qas.py for the full vin_2 column layout)

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


# ===========================================================================
# RITHMALGO -- finding the outlier cell for a given EV's vin_2.pkl file
# ==========================================================================
# ALGORITHM -- playground for deriving new variables from raw columns for 
# highlighted CELL_ID
# ===========================================================================


def rithmalgo(data: np.ndarray) -> tuple[float, int, np.ndarray, np.ndarray]:
    
    num_rows = data.shape[0]
    
    max_cell = 0
    max_dif = 0.0

    ev_mean = np.mean(data[:, 2:112], axis=1)
    ev_std = np.std(data[:, 2:112], axis=1)


    for cell_z in range(1,111):        
        volt_dif = abs(data[:, cell_z + 1] - ev_mean)   
        volt_dif_max = np.max(volt_dif)
        if volt_dif_max > max_dif:
            max_dif = volt_dif_max
            max_cell = cell_z

    return float(max_dif), int(max_cell), ev_mean, ev_std


def algorithm(data: np.ndarray) -> np.ndarray:
    """Derive a 1D variable from `data` (shape (T, n_cols)). Voltage / current for cell CELL_ID."""
    voltage = data[:, CELL_ID + 1]
    current = data[:, CURRENT_COL]
    soc = data[:, SOC_COL]
    temp = data[:, TEMP_COL]
    return voltage 

def plot_columns(data: np.ndarray, columns=None, algorithm_output=None,
                  title: str = "", out_name: str = "plot") -> None:
    """Plot the given columns of `data` against hours of operation. Raw values, no scaling."""
    if columns is None:
        columns = [CELL_ID + 1]
    else:
        columns = columns + [CELL_ID + 1]



    time_hours = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S / 3600.0

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in columns:
        ax.plot(time_hours, data[:, col], label=f"col_{col}", linewidth=0.8)

    #if algorithm_output is not None:
    #    ax.plot(time_hours, algorithm_output, label="algorithm_output",
    #             color=RWTHColorScales.red[0], linewidth=1.2, zorder=5)

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

    # 2. Run rithmalgo now that data exists
    max_dif, CELL_ID, ev_mean, ev_std = rithmalgo(data)
    
    ev_mean_std = np.std(ev_mean)
    ev_std_std = np.std(ev_std)
    print(f"max deviation: {max_dif}, ev_mean_std: {ev_mean_std}, ev_std_std: {ev_std_std}")

    derived = algorithm(data)

    plot_columns(
        data,
        columns=COLUMNS,
        algorithm_output=derived,
        title=f"QAS EV {EV_ID} - cell {CELL_ID} voltage vs. time",
        out_name=f"qas_ev{EV_ID}_cell{COLUMNS}_voltage",
    )
