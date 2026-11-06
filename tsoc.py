"""
Scatter plots comparing 4 per-EV features, faulty EVs (ground-truth label,
Labels.xls) highlighted in red -- same style as tcap.py, but 4 features
means C(4,2) = 6 pairwise plots, laid out as 2 figures of 3 side-by-side
subplots each.

FEATURES:
  - max_dtdt: max(dT/dt) over the EV's full recording (pack temperature,
    col 222).
  - mean_pack_temp: mean pack temperature over the full recording.
  - mean_pack_soc: mean pack SoC over the full recording, in percent.
  - mean_soc_charging_depth: mean delta_soc (in percentage points) across the
    same qualifying stable charging windows capac2.py finds (reuses
    capac2.find_stable_charging_windows on the same normal_hour-sliced data),
    i.e. how deep a charge this EV's typical charging cycle is.

EVs are restricted to those with >= 1 qualifying charging window (the same
set capac2.py ranks, loaded from its saved output), since
mean_soc_charging_depth is undefined otherwise.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

import capac2  # noqa: E402  (reuse its charging-window algorithm + normal_hour data)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)

CAPACITY_FEATURES = Path(__file__).resolve().parents[2] / "extracted_data" / "whole_features" / "charge_capacity_per_100soc.pkl"  # from capac2.py

TEMP_COL = 222     # pack temperature [degC]
SOC_COL = 223      # pack SoC, stored as a 0-1 fraction

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"

FEATURE_LABELS = {
    "max_dtdt": "max dT/dt (degC/s)",
    "mean_pack_temp": "mean pack temperature (degC)",
    "mean_pack_soc": "mean pack SoC (%)",
    "mean_soc_charging_depth": "mean SoC charging depth (pp)",
}


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def max_dtdt(data: np.ndarray) -> float:
    """max(dT/dt) over the EV's full recording."""
    temp = data[:, TEMP_COL]
    time_seconds = np.arange(data.shape[0]) * SAMPLE_INTERVAL_S
    dtdt = np.gradient(temp, time_seconds)
    return float(np.max(dtdt))


def mean_charging_depth(ev_id: int, data: np.ndarray, normal_hours: pd.Series) -> float | None:
    """Mean delta_soc (as a 0-1 fraction) across capac2.py's qualifying
    stable charging windows for this EV. None if there are none."""
    if ev_id not in normal_hours.index:
        return None

    start_idx = int(round(normal_hours.loc[ev_id] * 3600.0 / capac2.SAMPLE_INTERVAL_S))
    sliced = data[start_idx:]
    current = sliced[:, capac2.CURRENT_COL]
    soc = sliced[:, capac2.SOC_COL]

    depths = [
        float(soc[end - 1] - soc[start])
        for start, end in capac2.find_stable_charging_windows(current)
        if soc[end - 1] - soc[start] >= capac2.MIN_SOC_INCREASE
    ]
    return float(np.mean(depths)) if depths else None


def scatter_pair(ax, df: pd.DataFrame, healthy: pd.Series, faulty: pd.Series, x_col: str, y_col: str) -> None:
    ax.scatter(df.loc[healthy, x_col], df.loc[healthy, y_col],
               s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax.scatter(df.loc[faulty, x_col], df.loc[faulty, y_col],
               s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
    ax.set_xlabel(FEATURE_LABELS[x_col])
    ax.set_ylabel(FEATURE_LABELS[y_col])
    ax.set_title(f"{x_col} vs {y_col}")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    cap_df = pd.read_pickle(CAPACITY_FEATURES)[["avg_ah_per_100soc"]]
    normal_hours = pd.read_pickle(capac2.FEATURES_IN)["normal_hour_h"]

    rows = []
    for ev_id in cap_df.index:
        data = load_vin(ev_id)

        depth = mean_charging_depth(ev_id, data, normal_hours)
        if depth is None:
            print(f"EV {ev_id}: no qualifying stable charging window, skipping")
            continue

        row = {
            "ev_id": ev_id,
            "max_dtdt": max_dtdt(data),
            "mean_pack_temp": float(np.mean(data[:, TEMP_COL])),
            "mean_pack_soc": float(np.mean(data[:, SOC_COL])) * 100.0,
            "mean_soc_charging_depth": depth * 100.0,
        }
        rows.append(row)
        print(f"EV {ev_id}: max_dtdt={row['max_dtdt']:.4f} degC/s, "
              f"mean_pack_temp={row['mean_pack_temp']:.2f} degC, "
              f"mean_pack_soc={row['mean_pack_soc']:.2f} %, "
              f"mean_soc_charging_depth={row['mean_soc_charging_depth']:.2f} pp")

    df = pd.DataFrame(rows).set_index("ev_id")

    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    df = df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")

    faulty = df["Label"] == 1
    healthy = ~faulty

    OUT_DIR.mkdir(exist_ok=True)

    features = ["max_dtdt", "mean_pack_temp", "mean_pack_soc", "mean_soc_charging_depth"]
    all_pairs = list(combinations(features, 2))

    # Figure 1: max_dtdt vs each of the other 3 features.
    pairs1 = [p for p in all_pairs if "max_dtdt" in p]
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (x_col, y_col) in zip(axes1, pairs1):
        scatter_pair(ax, df, healthy, faulty, x_col, y_col)
    fig1.suptitle("EV scatter: max dT/dt vs pack temperature / SoC / charging depth")
    fig1.tight_layout()
    out_path1 = OUT_DIR / "tsoc_dtdt_vs_temp_soc_depth.png"
    fig1.savefig(out_path1, dpi=300)
    print(f"Saved figure to: {out_path1}")

    # Figure 2: the remaining 3 pairs among mean_pack_temp / mean_pack_soc /
    # mean_soc_charging_depth.
    pairs2 = [p for p in all_pairs if "max_dtdt" not in p]
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (x_col, y_col) in zip(axes2, pairs2):
        scatter_pair(ax, df, healthy, faulty, x_col, y_col)
    fig2.suptitle("EV scatter: pack temperature vs SoC vs charging depth")
    fig2.tight_layout()
    out_path2 = OUT_DIR / "tsoc_temp_vs_soc_vs_depth.png"
    fig2.savefig(out_path2, dpi=300)
    print(f"Saved figure to: {out_path2}")

    plt.show()
