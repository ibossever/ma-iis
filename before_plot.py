"""
Simple data-analysis playground for spotting outlier cells/EVs in the QAS
vin_2.pkl dataset. No plotting -- just loading + algorithms + printed stats.

Loop DATA_ROOT for all EV ids, load each one's vin_2.pkl, run rithmalgo (or
whatever algorithm you're playing with) on it, and collect results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)
OUT_DIR = Path(__file__).resolve().parent / "figures"


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def iter_evs(vin_file: str = VIN_FILE):
    """Yield (ev_id, data) for every EV under DATA_ROOT that has `vin_file`."""
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        if not ev_dir.is_dir():
            continue
        pkl_path = ev_dir / vin_file
        if not pkl_path.exists():
            continue
        ev_id = int(ev_dir.name)
        yield ev_id, load_vin(ev_id, vin_file)


# ===========================================================================
# RITHMALGO -- finding the outlier cell for a given EV's vin_2.pkl file
# ===========================================================================
# ALGORITHM -- playground for deriving new variables / stats from raw columns
# ===========================================================================


def rithmalgo(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    ev_mean = np.mean(data[:, 2:112], axis=1)
    ev_std = np.std(data[:, 2:112], axis=1)

    # Per-timestep worst-cell deviation from the pack mean (i.e. the deviation
    # of whichever cell is the outlier at that instant, not a fixed cell).
    max_dif = np.max(np.abs(data[:, 2:112] - ev_mean[:, None]), axis=1)

    return max_dif, ev_mean, ev_std


def algorithm(data: np.ndarray) -> np.ndarray:
    """Playground for a new derived variable. Add whatever you're testing here."""
    ev_mean = np.mean(data[:, 2:112], axis=1)
    return ev_mean


def voltage_outlier_features(data: np.ndarray) -> dict:
    """ev_std_std/ev_std_max/max_dif_max from rithmalgo(data) -- same
    formulas the __main__ block below inlines when building `results`,
    factored out (scoped to just the 3 columns saved to
    whole_features/voltage_features.pkl) so they can be called on an
    arbitrary (e.g. windowed) data slice."""
    max_dif, _ev_mean, ev_std = rithmalgo(data)
    return {
        "ev_std_std": float(np.std(ev_std)),
        "ev_std_max": float(np.max(ev_std)),
        "max_dif_max": float(np.max(max_dif)),
    }


if __name__ == "__main__":
    results = []

    for ev_id, data in iter_evs():
        max_dif, ev_mean, ev_std = rithmalgo(data)
        ev_std_std = np.std(ev_std)
        ev_std_max = np.max(ev_std)
        ev_std_mean = np.mean(ev_std)
        max_dif_std = np.std(max_dif)
        max_dif_max = np.max(max_dif)
        max_dif_mean = np.mean(max_dif)

        results.append({
            "ev_id": ev_id,
            "ev_std_std": ev_std_std,
            "ev_std_max": ev_std_max,
            "ev_std_mean": ev_std_mean,
            "max_dif_std": max_dif_std,
            "max_dif_max": max_dif_max,
            "max_dif_mean": max_dif_mean,
        })

        print(f"EV {ev_id}: "
              f"ev_std_std={ev_std_std:.4f}, ev_std_max={ev_std_max:.4f}, "
              f"ev_std_mean={ev_std_mean:.4f}, "
              f"max_dif_std={max_dif_std:.4f}, max_dif_max={max_dif_max:.4f}, "
              f"max_dif_mean={max_dif_mean:.4f}")

    df = pd.DataFrame(results)

    print("\nTop outlier EVs by ev_std_std:")
    print(df.sort_values("ev_std_std", ascending=False).head(35).to_string(index=False))

    print("\nTop outlier EVs by ev_std_max:")
    print(df.sort_values("ev_std_max", ascending=False).head(35).to_string(index=False))

    print("\nTop outlier EVs by max_dif_std:")
    print(df.sort_values("max_dif_std", ascending=False).head(35).to_string(index=False))

    print("\nTop outlier EVs by max_dif_max:")
    print(df.sort_values("max_dif_max", ascending=False).head(35).to_string(index=False))

    print("\nTop outlier EVs by max_dif_mean:")
    print(df.sort_values("max_dif_mean", ascending=False).head(35).to_string(index=False))

    # =======================================================================
    # Save ev_std_std, ev_std_max, max_dif_max per EV to
    # extracted_data/whole_features/voltage_features.pkl
    # =======================================================================
    VOLTAGE_FEATURES_OUT = PROJECT_ROOT / "extracted_data" / "whole_features" / "voltage_features.pkl"
    voltage_features_df = df[["ev_id", "ev_std_std", "ev_std_max", "max_dif_max"]].set_index("ev_id")
    voltage_features_df.to_pickle(VOLTAGE_FEATURES_OUT)
    print(f"\nSaved voltage features to: {VOLTAGE_FEATURES_OUT}")

    # =======================================================================
    # SCATTER -- ev_std_max (x) vs ev_std_std (y), faulty EVs (ground-truth
    # label, Labels.xls) highlighted in red.
    # =======================================================================
    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    df = df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")

    faulty = df["Label"] == 1
    healthy = ~faulty

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[healthy, "ev_std_max"], df.loc[healthy, "ev_std_std"],
               s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax.scatter(df.loc[faulty, "ev_std_max"], df.loc[faulty, "ev_std_std"],
               s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax.set_xlabel("ev_std_max (max of per-timestep cell-voltage std) [V]")
    ax.set_ylabel("ev_std_std (std of per-timestep cell-voltage std) [V]")
    ax.set_title("EV outlier scatter: ev_std_max vs ev_std_std")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ev_std_max_vs_ev_std_std.png", dpi=300)

    # =======================================================================
    # SCATTER -- max_dif_max (x) vs ev_std_std (y), faulty EVs highlighted in
    # red.
    # =======================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(df.loc[healthy, "max_dif_max"], df.loc[healthy, "ev_std_std"],
                s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax2.scatter(df.loc[faulty, "max_dif_max"], df.loc[faulty, "ev_std_std"],
                s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax2.set_xlabel("max_dif_max (max over time of per-timestep worst-cell deviation) [V]")
    ax2.set_ylabel("ev_std_std (std of per-timestep cell-voltage std) [V]")
    ax2.set_title("EV outlier scatter: max_dif_max vs ev_std_std")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "max_dif_max_vs_ev_std_std.png", dpi=300)

    # =======================================================================
    # SCATTER -- ev_std_max (x) vs max_dif_max (y), faulty EVs highlighted in
    # red.
    # =======================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.scatter(df.loc[healthy, "ev_std_max"], df.loc[healthy, "max_dif_max"],
                s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax3.scatter(df.loc[faulty, "ev_std_max"], df.loc[faulty, "max_dif_max"],
                s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax3.set_xlabel("ev_std_max (max of per-timestep cell-voltage std) [V]")
    ax3.set_ylabel("max_dif_max (max over time of per-timestep worst-cell deviation) [V]")
    ax3.set_title("EV outlier scatter: ev_std_max vs max_dif_max")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "ev_std_max_vs_max_dif_max.png", dpi=300)

    # =======================================================================
    # SCATTER -- ev_std_max ("max of std", x) vs max_dif_std ("std of max",
    # y: std over time of the per-timestep worst-cell deviation from the pack
    # mean), faulty EVs highlighted in red.
    # =======================================================================
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.scatter(df.loc[healthy, "ev_std_max"], df.loc[healthy, "max_dif_std"],
                s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax4.scatter(df.loc[faulty, "ev_std_max"], df.loc[faulty, "max_dif_std"],
                s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax4.set_xlabel("ev_std_max (max of per-timestep cell-voltage std) [V]")
    ax4.set_ylabel("max_dif_std (std of per-timestep worst-cell deviation from mean) [V]")
    ax4.set_title("EV outlier scatter: ev_std_max vs max_dif_std")
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(OUT_DIR / "ev_std_max_vs_max_dif_std.png", dpi=300)

    # =======================================================================
    # SCATTER -- ev_std_max (x) vs max_dif_mean (y), faulty EVs highlighted in
    # red.
    # =======================================================================
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(df.loc[healthy, "ev_std_max"], df.loc[healthy, "max_dif_mean"],
                s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax5.scatter(df.loc[faulty, "ev_std_max"], df.loc[faulty, "max_dif_mean"],
                s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")

    ax5.set_xlabel("ev_std_max (max of per-timestep cell-voltage std) [V]")
    ax5.set_ylabel("max_dif_mean (mean over time of per-timestep worst-cell deviation) [V]")
    ax5.set_title("EV outlier scatter: ev_std_max vs max_dif_mean")
    ax5.legend()
    fig5.tight_layout()
    fig5.savefig(OUT_DIR / "ev_std_max_vs_max_dif_mean.png", dpi=300)

    plt.show()
