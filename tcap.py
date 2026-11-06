"""
Scatter plot: per-EV avg_ah_per_100soc (from capac2.py) vs. mean pack
temperature, faulty EVs (ground-truth label, Labels.xls) highlighted in red.

avg_ah_per_100soc is loaded straight from capac2.py's saved output
(extracted_data/whole_features/charge_capacity_per_100soc.pkl) rather than
recomputed here.
Mean pack temperature is the plain mean of the pack temperature channel
(col 222) over each EV's full vin_2.pkl recording.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)

CAPACITY_FEATURES = Path(__file__).resolve().parents[2] / "extracted_data" / "whole_features" / "charge_capacity_per_100soc.pkl"  # from capac2.py

TEMP_COL = 222  # pack temperature [degC]

OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def mean_pack_temp(ev_id: int) -> float:
    """Mean of the pack temperature channel over the EV's full recording."""
    data = load_vin(ev_id)
    return float(np.mean(data[:, TEMP_COL]))


if __name__ == "__main__":
    cap_df = pd.read_pickle(CAPACITY_FEATURES)[["avg_ah_per_100soc"]]

    rows = []
    for ev_id in cap_df.index:
        rows.append({"ev_id": ev_id, "mean_pack_temp": mean_pack_temp(ev_id)})
        print(f"EV {ev_id}: mean_pack_temp={rows[-1]['mean_pack_temp']:.2f} degC")

    df = cap_df.join(pd.DataFrame(rows).set_index("ev_id"))

    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    df = df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")

    faulty = df["Label"] == 1
    healthy = ~faulty

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[healthy, "avg_ah_per_100soc"], df.loc[healthy, "mean_pack_temp"],
               s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax.scatter(df.loc[faulty, "avg_ah_per_100soc"], df.loc[faulty, "mean_pack_temp"],
               s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax.set_xlabel("avg_ah_per_100soc (Ah needed per 100% SoC swing)")
    ax.set_ylabel("mean pack temperature (degC)")
    ax.set_title("EV scatter: avg_ah_per_100soc vs mean pack temperature")
    ax.legend()
    fig.tight_layout()

    out_path = OUT_DIR / "avg_ah_per_100soc_vs_mean_pack_temp.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()
