"""
Simple data-analysis playground for spotting outlier EVs by comparing each
cell's SOC against its own pack's mean cell SOC, using only per-cell SOC
(vin_3.pkl) -- no pack SOC signal involved. No plotting -- just loading +
algorithms + printed stats.

Loop DATA_ROOT for all EV ids, load each one's vin_3.pkl, find the cell that
deviates most from the pack's own mean cell SOC, and collect both that max
deviation and the std of (cell soc - mean cell soc) for that same cell.
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
VIN3_FILE = "vin_3.pkl"
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)

CELL_SOC_COLS = slice(3, 112)  # vin_3.pkl cols 3-111: cells 1-109 (cell 0 / col 2 excluded)
CELL_INDEX_OFFSET = 1          # local index 0 in the sliced array is cell 1, not cell 0

OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"


def load_vin(car_id: int, vin_file: str = VIN3_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain numpy array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def iter_evs(vin_file: str = VIN3_FILE):
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
# ALGORITHMS -- playground for deriving new variables / stats from raw columns
# ===========================================================================


def soc_diff(data: np.ndarray) -> np.ndarray:
    """(T, 109) matrix of cell_soc - mean_cell_soc for every cell (excl. cell 0) at every timestep.

    SOC is physically bounded to [0, 1]; raw vin_3 readings occasionally spike
    past that (sensor glitches, e.g. 2.25 seen in EV 113), which would make the
    diff exceed the mathematically possible [-1, 1] range. Clip to [0, 1] first
    so every downstream stat stays physically meaningful.
    """
    cell_soc = np.clip(data[:, CELL_SOC_COLS], 0.0, 1.0)  # (T, 109)
    mean_cell_soc = np.mean(cell_soc, axis=1)             # (T,)

    return cell_soc - mean_cell_soc[:, None]


def soc_outlier_stats(data: np.ndarray) -> tuple[float, int, float]:
    """Max |cell soc - mean cell soc| over all cells/time, the cell it occurs on,
    and the std of (cell soc - mean cell soc) for that same cell."""
    diff = soc_diff(data)  # (T, 109)
    abs_diff = np.abs(diff)

    t_idx, local_cell = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    max_abs_diff = float(abs_diff[t_idx, local_cell])
    diff_std_at_worst_cell = float(np.std(diff[:, local_cell]))
    worst_cell = int(local_cell) + CELL_INDEX_OFFSET

    return max_abs_diff, worst_cell, diff_std_at_worst_cell


if __name__ == "__main__":
    results = []

    for ev_id, data in iter_evs():
        max_abs_diff, worst_cell, diff_std = soc_outlier_stats(data)

        results.append({
            "ev_id": ev_id,
            "max_abs_soc_diff": max_abs_diff,
            "worst_cell": worst_cell,
            "soc_diff_std_at_worst_cell": diff_std,
        })

        print(f"EV {ev_id}: max_abs_soc_diff={max_abs_diff:.4f} (cell {worst_cell}), "
              f"soc_diff_std_at_worst_cell={diff_std:.4f}")

    df = pd.DataFrame(results)

    print("\nTop 10 outlier EVs by max |cell soc - mean cell soc|:")
    print(df.sort_values("max_abs_soc_diff", ascending=False)
            .head(10).to_string(index=False))

    print("\nTop 10 outlier EVs by std of (cell soc - mean cell soc) on their worst cell:")
    print(df.sort_values("soc_diff_std_at_worst_cell", ascending=False)
            .head(10).to_string(index=False))

    # SCATTER -- max_abs_soc_diff (x) vs soc_diff_std_at_worst_cell (y), faulty
    # EVs (ground-truth label, Labels.xls) highlighted in red.
    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    df = df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")

    faulty = df["Label"] == 1
    healthy = ~faulty

    OUT_DIR.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df.loc[healthy, "max_abs_soc_diff"], df.loc[healthy, "soc_diff_std_at_worst_cell"],
               s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax.scatter(df.loc[faulty, "max_abs_soc_diff"], df.loc[faulty, "soc_diff_std_at_worst_cell"],
               s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
    ax.set_xlabel("max |cell soc - mean cell soc|")
    ax.set_ylabel("std of (cell soc - mean cell soc) at worst cell")
    ax.set_title("EV scatter: max SoC deviation vs its std at the worst cell")
    ax.legend(fontsize=8)
    fig.tight_layout()

    out_path = OUT_DIR / "soc_max_diff_vs_std.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")

    plt.show()
