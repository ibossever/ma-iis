"""
Per-EV pack charge capacity, expressed as Ah needed for a full 0% -> 100%
SoC swing, estimated from stable constant-current charging windows in QAS
vin_2.pkl (pack SOC + pack current -- see QAS_data_format.md).

No coulombic efficiency / discharge-cycle calculation here (see capac.py /
cap3.py for that) -- this script only estimates how big each EV's pack is,
in Ah, and ranks the fleet by it.

--- Tried and reverted: a RELATIVE (self-normalized) version of this
feature, dividing each EV's mean ah_per_100soc by its own first qualifying
session, to remove the fleet's pack-size confound with the fault label
(see conversation) -- checked against real relevance screening
(fault_feature.py / features.py) and it did not clear the bar (point-
biserial r=-0.02, p=0.67, AUC=0.49, essentially chance level) and was also
badly outlier-prone (a few EVs' first qualifying session was itself a
near-degenerate/tiny value, producing ratios in the hundreds for the
mean-over-first-session division). Reverted back to the plain absolute
feature below. If revisited, a more robust starting baseline (e.g. an
early-window median rather than a single first session) would be worth
trying before concluding the idea itself doesn't work.

SIGN CONVENTION (verified empirically, see capac.py): current < 0 ->
charging, current > 0 -> discharging.

WHY "stable" windows: charging current, unlike driving current, is
regulated by the charger/BMS and sits essentially flat for long stretches.
Driving current is wavy -- it swings between acceleration draw (discharge)
and regen (brief charge dips) many times per minute. A bare current-sign
check ("current < 0") is not enough to isolate real charging: spot-checking
this dataset shows a handful of EVs (e.g. 10, 131) have long current < 0
stretches contaminated by sudden 20-40 A jumps mid-run, which a sign-only
segment would silently swallow as "charging". So a charging window here
must additionally have its current barely move step-to-step -- see
STABILITY_* below and find_stable_charging_windows.

ALGORITHM:
1. Per EV, load pack SOC + pack current, and slice off everything before
   normal_hour_h (extracted_data/pack_current_features.pkl, produced by
   current2.py) -- the recording's startup transient (current sensor
   pinned near -1000 A) is not real operating data.
2. find_stable_charging_windows: split the remaining recording into
   maximal runs where every sample is real charging current (more negative
   than -CHARGE_CURRENT_FLOOR_A, and not a later sensor-clip artifact --
   current2.py notes a few EVs clip back to the same ~-1000 A fault value
   mid-recording, not just at startup) AND every consecutive step stays
   within a stability tolerance (tighter for small currents, wider for
   large ones, see STABILITY_ABS_TOL_A / STABILITY_REL_TOL). Any
   disqualifying sample -- discharge, near-zero idle noise, a clip
   artifact, or too big a current jump -- ends the run right there.
3. Keep only windows whose SoC rises by >= MIN_SOC_INCREASE (20 pp) from
   first to last sample. All qualifying windows are kept for an EV, not
   just the best one -- an EV with several long stable charging sessions
   contributes several windows ("charging as much as you can").
4. Each window's ah_per_100soc = (sum(|current|) * dt_hours) / delta_soc --
   the Ah that would be needed for a full 0->100% swing if the pack always
   charged at this window's average rate.
5. Per EV: average ah_per_100soc across all its qualifying windows. EVs
   with zero qualifying windows are excluded from the ranking. This
   per-EV average is saved to FEATURES_OUT for reuse by other scripts.
6. Rank the fleet by this per-EV average; report the 10 lowest (smallest
   apparent pack capacity) and 10 highest (largest).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"
FEATURES_IN = Path(__file__).resolve().parents[2] / "extracted_data" / "pack_current_features.pkl"  # from current2.py

SOC_COL = 223      # pack SOC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], negative = charge, positive = discharge (verified empirically, see capac.py)

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

CHARGE_CURRENT_FLOOR_A = 1.0        # |current| must exceed this to count as real charging, not idle noise around 0
CURRENT_CLIP_THRESHOLD_A = -300.0   # a handful of EVs (e.g. 223, 318) clip back to the ~-1000 A sensor-fault value mid-recording (see current2.py) -- anything past this is not real current

STABILITY_ABS_TOL_A = 1.5   # max allowed step-to-step current change ...
STABILITY_REL_TOL = 0.05    # ... or this fraction of the larger sample's magnitude, whichever is bigger (scales the tolerance up for EVs that fast-charge at 50-80+ A)

MIN_SOC_INCREASE = 0.20  # a stable window must span >= 20 percentage points of SoC to qualify

FEATURES_OUT = Path(__file__).resolve().parents[2] / "extracted_data" / "whole_features" / "charge_capacity_per_100soc.pkl"
TOP_N = 10


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


def find_stable_charging_windows(current: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs (end exclusive) where every sample is real
    charging current (not idle noise, not a sensor-clip artifact) AND every
    consecutive step stays within the stability tolerance. A single
    disqualifying sample ends the run there -- see module docstring,
    ALGORITHM step 2."""
    is_charging = (current <= -CHARGE_CURRENT_FLOOR_A) & (current >= CURRENT_CLIP_THRESHOLD_A)

    step = np.abs(np.diff(current))
    tol = np.maximum(
        STABILITY_ABS_TOL_A,
        STABILITY_REL_TOL * np.maximum(np.abs(current[:-1]), np.abs(current[1:])),
    )
    connects = is_charging[:-1] & is_charging[1:] & (step <= tol)

    n = len(current)
    new_segment = np.ones(n, dtype=bool)
    new_segment[1:] = ~connects
    boundaries = np.flatnonzero(new_segment)
    boundaries = np.append(boundaries, n)
    return [(s, e) for s, e in zip(boundaries[:-1], boundaries[1:]) if is_charging[s]]


def window_ah_per_100soc(data: np.ndarray, start: int, end: int) -> float | None:
    """Ah that would be needed for a full 0->100% SoC swing, extrapolated
    from this window's own Coulomb count and SoC delta. None if the window
    doesn't reach MIN_SOC_INCREASE."""
    current = data[start:end, CURRENT_COL]
    soc = data[start:end, SOC_COL]

    delta_soc = float(soc[-1] - soc[0])
    if delta_soc < MIN_SOC_INCREASE:
        return None

    ah = float(np.sum(np.abs(current)) * SAMPLE_INTERVAL_S / 3600.0)
    return ah / delta_soc


def avg_ah_per_100soc_from_window(data: np.ndarray) -> dict | None:
    """Core of ev_avg_ah_per_100soc: find every qualifying stable charging
    window in `data` and average their ah_per_100soc. No start-of-operation
    slicing here -- safe to call directly on an already-sliced window
    (caller is responsible for anchoring/slicing). None if zero qualifying
    windows (e.g. the window is too short, or has too little SoC swing, to
    contain a full charging session -- see MIN_SOC_INCREASE)."""
    current = data[:, CURRENT_COL]

    values = []
    for start, end in find_stable_charging_windows(current):
        value = window_ah_per_100soc(data, start, end)
        if value is not None:
            values.append(value)

    if not values:
        return None

    return {
        "avg_ah_per_100soc": float(np.mean(values)),
        "n_windows": len(values),
    }


def ev_avg_ah_per_100soc(ev_id: int, data: np.ndarray, normal_hours: pd.Series) -> dict | None:
    """Slice off the startup transient (before normal_hour_h), then delegate
    to avg_ah_per_100soc_from_window. None if the EV has zero qualifying
    windows, or no normal_hour_h entry at all."""
    if ev_id not in normal_hours.index:
        return None

    start_idx = int(round(normal_hours.loc[ev_id] * 3600.0 / SAMPLE_INTERVAL_S))
    result = avg_ah_per_100soc_from_window(data[start_idx:])
    if result is None:
        return None

    return {"ev_id": ev_id, **result}


if __name__ == "__main__":
    normal_hours = pd.read_pickle(FEATURES_IN)["normal_hour_h"]

    rows = []
    n_total = 0
    for ev_id, data in iter_evs():
        n_total += 1
        result = ev_avg_ah_per_100soc(ev_id, data, normal_hours)
        if result is None:
            print(f"EV {ev_id}: no qualifying stable charging window (>= {MIN_SOC_INCREASE:.0%} SoC rise), skipping")
            continue

        rows.append(result)
        print(f"EV {ev_id}: avg_ah_per_100soc={result['avg_ah_per_100soc']:.2f} Ah "
              f"(from {result['n_windows']} stable window(s))")

    df = pd.DataFrame(rows).set_index("ev_id")
    print(f"\n{len(df)}/{n_total} EVs have >= 1 qualifying stable charging window")

    df[["avg_ah_per_100soc"]].to_pickle(FEATURES_OUT)
    print(f"Saved per-EV avg_ah_per_100soc to: {FEATURES_OUT}")

    print(f"\nLowest {TOP_N} EVs by avg Ah needed per 100% SoC (smallest apparent pack capacity):")
    print(df.sort_values("avg_ah_per_100soc", ascending=True).head(TOP_N).to_string())

    print(f"\nHighest {TOP_N} EVs by avg Ah needed per 100% SoC (largest apparent pack capacity):")
    print(df.sort_values("avg_ah_per_100soc", ascending=False).head(TOP_N).to_string())
