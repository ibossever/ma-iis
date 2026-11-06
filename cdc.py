"""
Scatter plot: per-EV max charge current vs. max discharge current, faulty EVs
(ground-truth label, Labels.xls) highlighted in red.

Both features are loaded straight from current2.py's saved output
(extracted_data/pack_current_features.pkl) -- no vin_2.pkl re-reading here.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)

CURRENT_FEATURES = PROJECT_ROOT / "extracted_data" / "pack_current_features.pkl"  # from current2.py

OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"


if __name__ == "__main__":
    df = pd.read_pickle(CURRENT_FEATURES)[["max_charge_A", "max_discharge_A"]].reset_index()

    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "car_id"})
    df = df.merge(labels_df[["car_id", "Label"]], on="car_id", how="left")

    faulty = df["Label"] == 1
    healthy = ~faulty

    OUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[healthy, "max_charge_A"], df.loc[healthy, "max_discharge_A"],
               s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax.scatter(df.loc[faulty, "max_charge_A"], df.loc[faulty, "max_discharge_A"],
               s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax.set_xlabel("max charge current [A] (negative = charging)")
    ax.set_ylabel("max discharge current [A]")
    ax.set_title("EV scatter: max charge current vs max discharge current")
    ax.legend()
    fig.tight_layout()

    out_path = OUT_DIR / "max_charge_vs_max_discharge_current.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()
