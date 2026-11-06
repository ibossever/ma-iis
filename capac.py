"""
Coulombic efficiency for every EV in the QAS dataset, from vin_2.pkl only
(pack SOC + pack current -- see QAS_data_format.md).

SIGN CONVENTION (verified empirically): current < 0 -> charging, current > 0
-> discharging. Segment *kind* (charge vs discharge), however, is now
determined from the SOC trend, not the current sign (see step 1).

Per EV, in order:
0. Slice the recording to start at the stored normal-operation onset time
   (normal_hour_h, from extracted_data/pack_current_features.pkl, built by
   current2.py) -- this replaces trimming a leading stuck-sensor run with
   the onset current2.py already computed for that EV.
1. find_soc_segments(soc): split the recording into maximal contiguous runs
   of one SOC trend -- "charge" while SOC rises, "discharge" while SOC
   falls (a flat step inherits the most recently seen trend) -- a "cycle"
   here means one such run, not a matched charge+discharge pair. Every
   segment is kept, however small its SOC change.
2. cycle_data(...): per cycle, its Ah (Coulomb count: samples with
   |current| > CURRENT_THRESHOLD_A are dropped as implausible readings,
   the rest are summed with their actual sign -- e.g. a regen dip inside a
   'discharge' segment nets against that segment's discharge current
   instead of adding to it -- and the net is taken as a positive Ah) and
   delta_soc (%SOC covered), storing both for step 3.
3. ev_capacity(...): pool ALL kept charge cycles' Ah and delta_soc into
   total_charge_ah / total_charge_delta_soc_pct, and all kept discharge
   cycles' into total_discharge_ah / total_discharge_delta_soc_pct (each
   kept as its own variable and printed before step 4). Then divide each
   total Ah by its own total %SOC change, scaled to Ah per 100% SOC (one
   full cycle):
     charge_capacity_ah    = total_charge_ah    / total_charge_delta_soc_pct * 100
     discharge_capacity_ah = total_discharge_ah / total_discharge_delta_soc_pct * 100
4. coulombic_efficiency = discharge_capacity_ah / charge_capacity_ah.

Also prints, for every EV, max_discharge_A / max_charge_A straight from the
stored feature matrix (a simple ratio of the two stored extremes -- not
derived from this file's own segment/capacity calculation).

Runs this for every EV under DATA_ROOT, then lists the top 10 highest and
top 10 lowest coulombic efficiency.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
FEATURES_PATH = PROJECT_ROOT / "extracted_data" / "pack_current_features.pkl"  # built by current2.py
VIN_FILE = "vin_2.pkl"

SOC_COL = 223      # pack SOC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], signed: negative = charge, positive = discharge (verified empirically)

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

CURRENT_THRESHOLD_A = 300.5  # |current| above this is dropped from the Ah sum for that sample, not clipped


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


def slice_from_onset(data: np.ndarray, onset_hour: float) -> np.ndarray:
    """Step 0: drop everything before this EV's stored normal-operation
    onset time (see FEATURES_PATH), replacing ad-hoc stuck-sensor trimming
    with the onset current2.py already computed."""
    onset_idx = int(round(onset_hour * 3600.0 / SAMPLE_INTERVAL_S))
    return data[onset_idx:]


# ===========================================================================
# ALGORITHM
# ===========================================================================


def find_soc_segments(soc: np.ndarray) -> list[tuple[int, int, str]]:
    """Step 1: split into maximal contiguous runs of one SOC trend:
    [(start, end, kind), ...], end exclusive, kind 'charge' while SOC rises
    or 'discharge' while SOC falls. A flat step (no SOC change) inherits
    the most recently seen trend; any flat run before the first trend is
    treated as 'discharge'."""
    diffs = np.diff(soc)
    trend = np.where(diffs > 0, "charge", np.where(diffs < 0, "discharge", None))
    trend = pd.Series(trend).ffill().fillna("discharge").to_numpy()
    is_charge = np.append(trend, trend[-1] if len(trend) else "discharge") == "charge"

    change_points = np.flatnonzero(np.diff(is_charge)) + 1
    boundaries = [0, *change_points.tolist(), len(soc)]
    return [(start, end, "charge" if is_charge[start] else "discharge")
            for start, end in zip(boundaries[:-1], boundaries[1:])]


def cycle_data(data: np.ndarray, segments: list[tuple[int, int, str]]) -> list[dict]:
    """Step 2: Ah and delta_soc per segment -- every segment is kept, no
    matter how small its SOC change. Any sample with |current| >
    CURRENT_THRESHOLD_A is excluded from the sum (implausible reading, not
    real pack current). The remaining samples are summed with their actual
    sign -- e.g. a regen-braking dip inside a 'discharge' segment nets
    against that segment's discharge current instead of adding to it --
    then the net is taken as a positive Ah."""
    soc = data[:, SOC_COL]
    current = data[:, CURRENT_COL]
    dt_hours = SAMPLE_INTERVAL_S / 3600.0
    valid_current = np.abs(current) <= CURRENT_THRESHOLD_A

    cycles = []
    for start, end, kind in segments:
        delta_soc = abs(soc[end - 1] - soc[start])
        seg_current = current[start:end][valid_current[start:end]]
        ah = float(abs(np.sum(seg_current)) * dt_hours)
        cycles.append({"kind": kind, "ah": ah, "delta_soc": delta_soc})
    return cycles


def ev_capacity(cycles: list[dict]) -> dict | None:
    """Steps 3 & 4. None if there isn't at least one qualifying cycle of
    each kind.
      total_charge_ah / total_discharge_ah   -- sum of Ah over every kept
                          cycle of that kind (kept as its own variable).
      total_charge_delta_soc_pct / total_discharge_delta_soc_pct -- sum of
                          %SOC covered over every kept cycle of that kind
                          (kept as its own variable).
      charge_capacity_ah    = total_charge_ah    / total_charge_delta_soc_pct * 100
      discharge_capacity_ah = total_discharge_ah / total_discharge_delta_soc_pct * 100
                          -- i.e. total Ah divided by total %SOC change,
                          scaled up to Ah per 100% SOC (one full cycle).
      coulombic_efficiency = discharge_capacity_ah / charge_capacity_ah
    """
    charge = [c for c in cycles if c["kind"] == "charge"]
    discharge = [c for c in cycles if c["kind"] == "discharge"]
    if not charge or not discharge:
        return None

    total_charge_ah = sum(c["ah"] for c in charge)
    total_discharge_ah = sum(c["ah"] for c in discharge)
    total_charge_delta_soc_pct = sum(c["delta_soc"] for c in charge) * 100.0
    total_discharge_delta_soc_pct = sum(c["delta_soc"] for c in discharge) * 100.0

    charge_capacity_ah = total_charge_ah / total_charge_delta_soc_pct * 100.0
    discharge_capacity_ah = total_discharge_ah / total_discharge_delta_soc_pct * 100.0

    return {
        "total_charge_ah": total_charge_ah,
        "total_discharge_ah": total_discharge_ah,
        "total_charge_delta_soc_pct": total_charge_delta_soc_pct,
        "total_discharge_delta_soc_pct": total_discharge_delta_soc_pct,
        "charge_capacity_ah": charge_capacity_ah,
        "discharge_capacity_ah": discharge_capacity_ah,
        "coulombic_efficiency": discharge_capacity_ah * total_charge_delta_soc_pct / (charge_capacity_ah * total_discharge_delta_soc_pct),
    }


def analyze_ev(data: np.ndarray, onset_hour: float) -> dict | None:
    """Run steps 0-4 for one EV. None if it has no qualifying cycle of both
    kinds (can't compute a coulombic efficiency)."""
    data = slice_from_onset(data, onset_hour)
    segments = find_soc_segments(data[:, SOC_COL])
    cycles = cycle_data(data, segments)

    return ev_capacity(cycles)


if __name__ == "__main__":
    features_df = pd.read_pickle(FEATURES_PATH)

    # --- Output : coulombic efficiency, onset-sliced, SOC-trend segments ---
    results = []
    for ev_id, data in iter_evs():
        onset_hour = features_df.loc[ev_id, "normal_hour_h"]
        summary = analyze_ev(data, onset_hour)
        if summary is None:
            print(f"EV {ev_id}: no charge and discharge segment pair, skipping")
            continue

        results.append({"ev_id": ev_id, **summary})
        print(f"EV {ev_id}: "
              f"total_charge={summary['total_charge_ah']:.2f} Ah over {summary['total_charge_delta_soc_pct']:.1f}% SOC, "
              f"total_discharge={summary['total_discharge_ah']:.2f} Ah over {summary['total_discharge_delta_soc_pct']:.1f}% SOC "
              f"-> charge_capacity={summary['charge_capacity_ah']:.2f} Ah, "
              f"discharge_capacity={summary['discharge_capacity_ah']:.2f} Ah, "
              f"CE={summary['coulombic_efficiency']:.1%}")

    df = pd.DataFrame(results)

    print("\nTop 10 EVs by highest coulombic efficiency:")
    print(df.sort_values("coulombic_efficiency", ascending=False).head(10).to_string(index=False))

    print("\nTop 10 EVs by lowest coulombic efficiency:")
    print(df.sort_values("coulombic_efficiency", ascending=True).head(10).to_string(index=False))
