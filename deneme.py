"""
Find EVs where pack temperature rises even though pack current is ~0 (i.e.
no I^2R resistive heating source to explain it). Pack-level only -- no cell
voltage data used.

Loop DATA_ROOT for all EV ids, load each one's vin_2.pkl, flag every
timestep where this happens, and print which EVs it occurs in and exactly
when. Samples are 30 s apart.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"

TEMP_COL = 222     # pack temperature [degC]
CURRENT_COL = 224  # pack current [A], signed: positive = charge, negative = discharge

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

ZERO_CURRENT_THRESHOLD_A = 1.0  # |current| below this counts as "near-rest / no current"


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
# ALGORITHM -- step by step:
# 1. temp, current: the raw pack temperature and pack current columns.
# 2. delta_T = np.diff(temp): temperature change between each pair of
#    consecutive 30s samples, shape (T-1,).
# 3. near_zero_current: |current| is below ZERO_CURRENT_THRESHOLD_A at BOTH
#    ends of that interval (before and after the temperature change), so the
#    pack was at rest for the whole interval -- no measurable I^2R heating
#    source during that time.
# 4. rising: delta_T > 0, i.e. temperature actually increased over that
#    interval.
# 5. anomaly_mask = near_zero_current & rising: the physically odd case --
#    pack heating up despite (near) zero current flowing.
# ===========================================================================


def zero_current_temp_rise_events(data: np.ndarray) -> pd.DataFrame:
    """One row per timestep where pack temp rose while pack current stayed
    near zero across the whole interval."""
    temp = data[:, TEMP_COL]
    current = data[:, CURRENT_COL]

    delta_t = np.diff(temp)
    near_zero_current = (np.abs(current[:-1]) < ZERO_CURRENT_THRESHOLD_A) & \
                         (np.abs(current[1:]) < ZERO_CURRENT_THRESHOLD_A)
    rising = delta_t > 0

    anomaly_mask = near_zero_current & rising
    idx = np.flatnonzero(anomaly_mask)

    return pd.DataFrame({
        "timestep": idx,
        "hours": idx * SAMPLE_INTERVAL_S / 3600.0,
        "temp_before": temp[idx],
        "temp_after": temp[idx + 1],
        "delta_t": delta_t[idx],
        "current_before": current[idx],
        "current_after": current[idx + 1],
    })


if __name__ == "__main__":
    all_events = []

    for ev_id, data in iter_evs():
        events = zero_current_temp_rise_events(data)
        if events.empty:
            continue

        events.insert(0, "ev_id", ev_id)
        all_events.append(events)

        print(f"EV {ev_id}: {len(events)} zero-current temp-rise event(s)")
        print(events.to_string(index=False))
        print()

    if not all_events:
        print("No zero-current temperature-rise events found in any EV.")
    else:
        combined = pd.concat(all_events, ignore_index=True)
        summary = combined.groupby("ev_id").size().sort_values(ascending=False)
        print("Summary -- event count per EV:")
        print(summary.to_string())
