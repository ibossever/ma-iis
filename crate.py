"""
List the EVs with the highest charging C-rate in the QAS vin_2.pkl dataset,
and the EVs whose estimated capacity changes most between an early-life and
a late-life measurement. Pack-level only -- no individual cell voltages
used, only pack mean voltage and pack current.

vin_2.pkl has no rated/nominal capacity value anywhere in it, so capacity is
*estimated per EV* from the data itself. A naive global min-SoC-to-max-SoC
Coulomb count is unreliable because the drive has many small charge/
discharge cycles in between that pollute the current integral. Instead we
only trust a *single uninterrupted charging run*, and within it only a
sub-window that lies entirely inside the SoC 20%-80% band and spans at
least MIN_WINDOW_SOC_SPAN (20 percentage points) -- if a run doesn't reach
the full 20%-80% span, a shorter qualifying sub-window elsewhere in the
band (e.g. 30%-55%) is used instead of skipping the EV; only a run that
never manages 20 uninterrupted points within the band is rejected. See
ALGORITHM below. We do this twice per EV: once using the earliest such
qualifying window, once using the latest.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"

VOLTAGE_COLS = slice(2, 112)  # 110 series cell voltages [V] -- only ever averaged, never indexed individually
SOC_COL = 223      # SoC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], signed: positive = charge, negative = discharge

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

CHARGE_CURRENT_THRESHOLD_A = 1.0  # current above this counts as "charging" (vs. near-zero rest noise)
SOC_WINDOW_LOW = 0.20    # capacity window must lie entirely within [SOC_WINDOW_LOW, SOC_WINDOW_HIGH]...
SOC_WINDOW_HIGH = 0.80   # ...and span at least MIN_WINDOW_SOC_SPAN, all within ONE uninterrupted charge run
MIN_WINDOW_SOC_SPAN = 0.20  # minimum SoC span (20 percentage points) for a window to qualify


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
# 1. mean_pack_voltage(t) = mean of the 110 series cell voltages (pack-level
#    only, no single-cell values used anywhere below).
# 2. find_charging_runs(current): a "charging run" is a maximal contiguous
#    stretch of timesteps where current >= 0, i.e. never discharges. A single
#    low-but-nonnegative sample (sensor noise, CV taper, brief idle) does NOT
#    break the run -- only an actual discharge (current < 0) does. This is
#    what "without interruption" means: no discharge cycle sneaking in.
# 3. capacity_window_in_run(data, start, end), for each run: within that run
#    only, take every sample whose SoC lies in [SOC_WINDOW_LOW, SOC_WINDOW_HIGH]
#    (20%-80%) and use the widest available span between the first and last
#    such sample. A run that fully crosses 20%->80% gives the widest (best)
#    window; a run that only partially crosses the band (e.g. 30%-55%) still
#    qualifies as long as its span is >= MIN_WINDOW_SOC_SPAN (20 percentage
#    points) -- this is what lets an EV be used even when no single run spans
#    the full 20%-80% range. As a sanity check against idle/noise runs that
#    technically never discharge but aren't real charging, the window's mean
#    current must also exceed CHARGE_CURRENT_THRESHOLD_A.
# 4. window_capacity_ah(i1, i2): sum the pack current across the window and
#    convert A -> Ah (x SAMPLE_INTERVAL_S / 3600) to get the net charge that
#    flowed, then divide by the actual SoC change over the window
#    (soc[i2]-soc[i1], >= MIN_WINDOW_SOC_SPAN) to get capacity_ah.
# 5. Two measurements per EV: collect every qualifying window across the
#    whole recording, in chronological order. The first one is the
#    "early-life" capacity estimate, the last one is the "late-life"
#    estimate (they're the same window if only one qualifying run exists).
# 6. C-rate(t) = current(t) / capacity_ah, using the average of the two
#    capacity measurements as the best available estimate. Only charging
#    (current > 0) timesteps are considered; max_c_rate is the largest
#    C-rate among those for the whole recording.
# 7. capacity_diff = |capacity_late - capacity_early|: how much the two
#    measurements disagree, a proxy for capacity fade / data inconsistency.
# ===========================================================================


def mean_pack_voltage(data: np.ndarray) -> np.ndarray:
    """Mean of the 110 series cell voltages at each timestep, shape (T,)."""
    return np.mean(data[:, VOLTAGE_COLS], axis=1)


def find_charging_runs(current: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous [start, end) index ranges where current never goes
    negative (i.e. never discharges). A single low/near-zero-but-nonnegative
    sample (noise, CV taper, brief idle) does NOT break a run -- only an
    actual discharge does. Whether a run represents *real* charging is
    checked separately in capacity_window_in_run via the mean-current guard."""
    is_charging = current >= 0
    padded = np.concatenate(([False], is_charging, [False]))
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def capacity_window_in_run(data: np.ndarray, start: int, end: int) -> tuple[int, int] | None:
    """Widest available [i1, i2] sub-window inside one non-discharging run
    whose SoC values stay entirely within [SOC_WINDOW_LOW, SOC_WINDOW_HIGH].
    None if no such window exists, it's narrower than MIN_WINDOW_SOC_SPAN, or
    its average current doesn't clear CHARGE_CURRENT_THRESHOLD_A (i.e. it's
    just idle/noise drift, not a real charging session)."""
    soc_run = data[start:end, SOC_COL]

    in_band = np.flatnonzero((soc_run >= SOC_WINDOW_LOW) & (soc_run <= SOC_WINDOW_HIGH))
    if in_band.size == 0:
        return None
    a1, a2 = in_band[0], in_band[-1]

    if soc_run[a2] - soc_run[a1] < MIN_WINDOW_SOC_SPAN:
        return None

    i1, i2 = start + a1, start + a2
    if np.mean(data[i1:i2, CURRENT_COL]) <= CHARGE_CURRENT_THRESHOLD_A:
        return None

    return i1, i2


def window_capacity_ah(data: np.ndarray, i1: int, i2: int) -> float:
    """Coulomb-counting capacity estimate in Ah for one clean [i1, i2] window."""
    current = data[:, CURRENT_COL]
    soc = data[:, SOC_COL]

    charge_ah = np.sum(current[i1:i2]) * SAMPLE_INTERVAL_S / 3600.0
    delta_soc = soc[i2] - soc[i1]
    return charge_ah / delta_soc


def two_capacity_measurements(data: np.ndarray) -> tuple[float, float] | None:
    """(capacity_early, capacity_late) in Ah, from the first and last
    qualifying 20-80% charging window in the recording. None if no
    qualifying window exists at all."""
    current = data[:, CURRENT_COL]

    windows = []
    for start, end in find_charging_runs(current):
        window = capacity_window_in_run(data, start, end)
        if window is not None:
            windows.append(window)

    if not windows:
        return None

    windows.sort(key=lambda w: w[0])
    i1_early, i2_early = windows[0]
    i1_late, i2_late = windows[-1]

    capacity_early = window_capacity_ah(data, i1_early, i2_early)
    capacity_late = window_capacity_ah(data, i1_late, i2_late)
    return capacity_early, capacity_late


def max_charging_c_rate(data: np.ndarray, capacity_ah: float) -> tuple[float, int] | None:
    """Highest C-rate among charging (current > 0) timesteps, plus its index.
    None if this EV never charges."""
    current = data[:, CURRENT_COL]
    charging = current > 0
    if not np.any(charging):
        return None

    c_rate = current / capacity_ah
    charging_idx = np.flatnonzero(charging)
    best_idx = charging_idx[np.argmax(c_rate[charging_idx])]
    return float(c_rate[best_idx]), int(best_idx)


if __name__ == "__main__":
    results = []

    for ev_id, data in iter_evs():
        capacities = two_capacity_measurements(data)
        if capacities is None:
            print(f"EV {ev_id}: no clean 20-80% charging window found, skipping")
            continue

        capacity_early, capacity_late = capacities
        capacity_diff = abs(capacity_late - capacity_early)
        capacity_avg = (capacity_early + capacity_late) / 2.0

        outcome = max_charging_c_rate(data, capacity_avg)
        if outcome is None:
            print(f"EV {ev_id}: never charges, skipping")
            continue

        max_c_rate, idx = outcome
        voltage_at_max = float(mean_pack_voltage(data)[idx])

        results.append({
            "ev_id": ev_id,
            "max_c_rate": max_c_rate,
            "capacity_early_ah": capacity_early,
            "capacity_late_ah": capacity_late,
            "capacity_diff_ah": capacity_diff,
            "hours": idx * SAMPLE_INTERVAL_S / 3600.0,
            "mean_pack_voltage_at_max": voltage_at_max,
        })

        print(f"EV {ev_id}: max_c_rate={max_c_rate:.3f} C, "
              f"capacity_early={capacity_early:.2f} Ah, capacity_late={capacity_late:.2f} Ah, "
              f"diff={capacity_diff:.2f} Ah")

    df = pd.DataFrame(results)

    print("\nTop 10 EVs by max charging C-rate:")
    print(df.sort_values("max_c_rate", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 EVs by capacity difference (early vs. late measurement):")
    print(df.sort_values("capacity_diff_ah", ascending=False).head(10).to_string(index=False))
