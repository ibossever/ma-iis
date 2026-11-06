# %%
"""
Windowed, full-fleet, 24h-only companion to sliding_window_fleet_features.py:
steps a WINDOW_HOURS=24 window through every EV's raw telemetry in
STRIDE_HOURS=12 increments, computing ONLY the subset of
filtered_features.pkl's 15 columns that are valid to evaluate on a 24h span
-- see kmeans24.py's module docstring for the full reasoning. This is a
genuinely SEPARATE, fresh extraction (own window sequence, own output file),
not a repackaging of anything sliding_window_fleet_features.py computes.

--- Why this is a separate script, not a mode of the 120h one --------------
sliding_window_fleet_features.py now evaluates ALL 15 filtered_features.pkl
columns over the full 120h window (FAST_SLICE_HOURS == WINDOW_HOURS there --
see its module docstring). Reusing any of its trailing-slice machinery for a
genuine 24h table would mean either (a) sub-slicing its 120h windows, which
does NOT reproduce a real 24h/12h-stride sequence (its window sequence only
emits when a FULL 120h fits, so the first 96h of every EV's recording and
any trailing sub-120h remainder are never represented -- see the long-form
explanation from the kmeans24.py discussion), or (b) computing a real 24h/
12h-stride sequence as its own pass. This script does (b): its own window
loop, starting at raw index 0, emitting whenever a full 24h fits -- so it
covers the ENTIRE recording (including the first 96h and any trailing
partial-120h-but-full-24h remainder that the 120h script drops).

--- Which 7 columns, and why the other 8 are skipped entirely --------------
Kept (all time-domain point statistics, genuinely meaningful on 24h):
  local_temp__min_val, local_temp__avg_amp, pack_temp__mean_pack_temp,
  voltage_features__ev_std_std/ev_std_max/max_dif_max,
  resistance__std_of_std_R.
Skipped entirely (not computed here at all -- not just computed-and-dropped):
  - current_fft__band1_mean_diff/band1_std_diff/band2_std_diff,
    soc_fft__band1_std_diff/band2_std_diff/band3_mean_diff,
    local_temp__mean_diff: all FFT-based, need a long span to resolve low
    frequencies -- meaningless on 24h. Skipping them also means Phase 1 of
    sliding_window_fleet_features.py (building the 3 fleet-wide FFT
    reference spectra, itself an expensive full-fleet raw pass) is not
    needed here at all.
  - charge_capacity_per_100soc__avg_ah_per_100soc: ratio-type, gated on a
    charging session spanning >= 20 SoC points inside its slice
    (capac2.py's MIN_SOC_INCREASE) -- a 24h slice makes that condition rare
    enough that the column would be mostly NaN, so it's not computed here
    (see kmeans24.py's module docstring for the same reasoning).

--- start-of-operation masking ----------------------------------------------
resistance__std_of_std_R is the only current-dependent feature kept here
(needs current for R = -dV/dI) -- same MIN_CURRENT_SPAN_HOURS /
normal_hour_h clipping convention as sliding_window_fleet_features.py: if
the window's post-start-of-operation span is under MIN_CURRENT_SPAN_HOURS,
it comes out NaN for that window while the other 6 features still compute
normally (they never touch the current channel).

Input:
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      read only to assert the 7 kept columns are still present in the
      current column list (fail loud if features.py's screening ever drops
      one of them).
  extracted_data/pack_current_features.pkl (current2.py) -- normal_hour_h.
  /media/ssd_2tb/jak-iis/<ev_id>/vin_1.pkl, vin_2.pkl -- raw QAS telemetry.
Output:
  extracted_data/early_warning/filtered_features_windows_24h.pkl -- one row
      per (ev_id, window position): ev_id, window_start, window_end (raw
      sample indices, window sequence anchored at 0), window_start_hours
      (elapsed hours relative to start-of-operation, can be negative), then
      the 7 kept features.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))  # sibling scripts in 00_data_analysis, imported as modules below

import analysis         # noqa: E402  (mean_pack_temp)
import resistance        # noqa: E402  (resistance_timeseries_from_window, summarize)
import before_plot        # noqa: E402  (voltage_outlier_features)
import plot_localtemp       # noqa: E402  (signal_freq_amp)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
SAMPLE_INTERVAL_S = 30.0  # must match every sibling script's own SAMPLE_INTERVAL_S

FILTERED_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # column presence check only
NORMAL_HOURS_IN = PROJECT_ROOT / "extracted_data" / "pack_current_features.pkl"  # from current2.py

OUT_DIR = PROJECT_ROOT / "extracted_data" / "early_warning"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "filtered_features_windows_24h.pkl"

WINDOW_HOURS = 24
STRIDE_HOURS = 12

# Minimum post-start-of-operation span resistance__std_of_std_R needs within
# its window to be computed at all -- same convention/value as
# sliding_window_fleet_features.py's MIN_CURRENT_SPAN_HOURS.
MIN_CURRENT_SPAN_HOURS = 6

WINDOW_SAMPLES = round(WINDOW_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
STRIDE_SAMPLES = round(STRIDE_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
MIN_CURRENT_SPAN_SAMPLES = round(MIN_CURRENT_SPAN_HOURS * 3600.0 / SAMPLE_INTERVAL_S)

KEPT_FEATURE_COLUMNS = [
    "local_temp__min_val", "local_temp__avg_amp", "pack_temp__mean_pack_temp",
    "voltage_features__ev_std_std", "voltage_features__ev_std_max", "voltage_features__max_dif_max",
    "resistance__std_of_std_R",
]


def iter_evs_raw():
    """Yield (ev_id, vin1, vin2) for every EV under DATA_ROOT with both raw
    files, each loaded ONCE as a plain numpy array (same reasoning as
    sliding_window_fleet_features.py's own iter_evs_raw)."""
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        if not ev_dir.is_dir():
            continue
        vin1_path = ev_dir / "vin_1.pkl"
        vin2_path = ev_dir / "vin_2.pkl"
        if not vin1_path.exists() or not vin2_path.exists():
            continue
        ev_id = int(ev_dir.name)
        vin1 = pd.read_pickle(vin1_path).cpu().numpy()
        vin2 = pd.read_pickle(vin2_path).cpu().numpy()
        yield ev_id, vin1, vin2


def extract_window_features_24h(
    vin1_window: np.ndarray, vin2_window: np.ndarray, window_start: int, window_end: int, anchor_idx: int,
) -> dict:
    """The 7 KEPT_FEATURE_COLUMNS for one 24h window. `vin1_window`/
    `vin2_window` are the FULL WINDOW_HOURS=24h slice, spanning absolute raw
    indices [window_start, window_end). `anchor_idx` (start-of-operation) is
    used only by resistance__std_of_std_R -- the other 6 never touch the
    current channel and use the window unmodified (same split as
    sliding_window_fleet_features.py's own current-dependent vs. not
    classification).
    """
    row: dict = {}

    local_temp = vin1_window[:, 0, plot_localtemp.MAX_TEMP_COL]
    row["local_temp__min_val"] = float(np.min(local_temp))
    _avg_freq, avg_amp = plot_localtemp.signal_freq_amp(local_temp)
    row["local_temp__avg_amp"] = avg_amp

    pack_temp = vin2_window[:, analysis.PACK_TEMP_COL]
    row["pack_temp__mean_pack_temp"] = analysis.mean_pack_temp(pack_temp)

    voltage_feats = before_plot.voltage_outlier_features(vin2_window)
    row["voltage_features__ev_std_std"] = voltage_feats["ev_std_std"]
    row["voltage_features__ev_std_max"] = voltage_feats["ev_std_max"]
    row["voltage_features__max_dif_max"] = voltage_feats["max_dif_max"]

    # resistance__std_of_std_R: current-dependent -- clip to start-of-operation
    # (see module docstring's "start-of-operation masking" section)
    valid_start = max(window_start, anchor_idx)
    if window_end - valid_start >= MIN_CURRENT_SPAN_SAMPLES:
        vin2_masked = vin2_window[valid_start - window_start:]
        mean_R, std_R, _soc = resistance.resistance_timeseries_from_window(vin2_masked)
        _mean_of_mean_R, _std_of_mean_R, _mean_of_std_R, std_of_std_R = resistance.summarize(mean_R, std_R)
        row["resistance__std_of_std_R"] = std_of_std_R
    else:
        row["resistance__std_of_std_R"] = np.nan

    return row


if __name__ == "__main__":
    # %% 0. sanity: the 7 kept columns must still exist in filtered_features.pkl
    current_columns = set(pd.read_pickle(FILTERED_FEATURES_IN).columns)
    missing = set(KEPT_FEATURE_COLUMNS) - current_columns
    if missing:
        raise RuntimeError(
            f"filtered_features.pkl no longer contains {sorted(missing)}, which this script expects to "
            "compute -- features.py's column set has drifted; update KEPT_FEATURE_COLUMNS above to match."
        )
    print(f"computing 24h-windowed versions of {len(KEPT_FEATURE_COLUMNS)} columns "
          f"(window={WINDOW_HOURS}h, stride={STRIDE_HOURS}h): {KEPT_FEATURE_COLUMNS}")

    # %% 1. per-EV, per-window feature extraction -- single pass, no FFT
    # reference-spectra phase needed (see module docstring)
    normal_hours = pd.read_pickle(NORMAL_HOURS_IN)["normal_hour_h"]

    rows = []
    windows_per_ev = {}
    skipped_evs = []  # (ev_id, reason)
    t_start = time.time()
    n_evs_processed = 0

    for ev_id, vin1, vin2 in iter_evs_raw():
        if ev_id not in normal_hours.index:
            skipped_evs.append((ev_id, "no normal_hour_h entry (no known start of operation)"))
            continue
        if vin1.shape[0] != vin2.shape[0]:
            skipped_evs.append((ev_id, f"vin_1.pkl/vin_2.pkl length mismatch ({vin1.shape[0]} vs {vin2.shape[0]} samples)"))
            continue

        n_samples = vin2.shape[0]
        anchor_idx = int(round(normal_hours.loc[ev_id] * 3600.0 / SAMPLE_INTERVAL_S))

        if n_samples < WINDOW_SAMPLES:
            windows_per_ev[ev_id] = 0
            total_h = n_samples * SAMPLE_INTERVAL_S / 3600.0
            skipped_evs.append((ev_id, f"only {total_h:.1f}h of recording total, need {WINDOW_HOURS}h for one window"))
            continue

        n_windows = 0
        pos = 0
        while pos + WINDOW_SAMPLES <= n_samples:
            window_end = pos + WINDOW_SAMPLES
            features = extract_window_features_24h(vin1[pos:window_end], vin2[pos:window_end], pos, window_end, anchor_idx)
            rows.append({
                "ev_id": ev_id,
                "window_start": pos,
                "window_end": window_end,
                "window_start_hours": (pos - anchor_idx) * SAMPLE_INTERVAL_S / 3600.0,
                **features,
            })
            n_windows += 1
            pos += STRIDE_SAMPLES

        windows_per_ev[ev_id] = n_windows
        trailing_unused_h = (n_samples - pos) * SAMPLE_INTERVAL_S / 3600.0
        if trailing_unused_h > 0:
            skipped_evs.append((ev_id, f"1 trailing partial window dropped ({trailing_unused_h:.1f}h remaining, < {WINDOW_HOURS}h)"))

        n_evs_processed += 1
        if n_evs_processed % 25 == 0:
            elapsed = time.time() - t_start
            print(f"  ...{n_evs_processed} EVs processed, {len(rows)} windows so far ({elapsed:.0f}s elapsed)")

    print(f"done: {n_evs_processed} EVs processed in {time.time() - t_start:.0f}s")

    # %% 2. assemble, save
    windows_df = pd.DataFrame(rows)
    n_rows = len(windows_df)
    print(f"\nassembled {n_rows} rows x {len(windows_df.columns)} columns")

    windows_df.to_pickle(OUT_PATH)
    actual_bytes = OUT_PATH.stat().st_size
    print(f"saved to {OUT_PATH} ({actual_bytes / 1e6:.1f} MB on disk)")

    # %% 3. summary
    print("\n--- Windows per EV ---")
    windows_per_ev_series = pd.Series(windows_per_ev, name="n_windows")
    print(f"{(windows_per_ev_series > 0).sum()}/{len(windows_per_ev_series)} EVs produced >= 1 window")
    print(f"total windows: {windows_per_ev_series.sum()}, "
          f"mean {windows_per_ev_series[windows_per_ev_series > 0].mean():.1f} windows/EV (among EVs with >= 1)")
    print(windows_per_ev_series.describe().to_string())

    print(f"\n--- Skipped / partial ({len(skipped_evs)} entries) ---")
    for ev_id, reason in skipped_evs:
        print(f"EV {ev_id}: {reason}")

    if n_rows:
        rows_with_nan = windows_df[KEPT_FEATURE_COLUMNS].isna().any(axis=1).sum()
        print(f"\n--- Data quality ---")
        print(f"{rows_with_nan}/{n_rows} rows ({rows_with_nan / n_rows:.1%}) have >= 1 NaN feature value")
        n_resistance_masked = windows_df["resistance__std_of_std_R"].isna().sum()
        print(f"{n_resistance_masked}/{n_rows} rows ({n_resistance_masked / n_rows:.1%}) have "
              f"resistance__std_of_std_R as NaN (< {MIN_CURRENT_SPAN_HOURS}h of post-start-of-operation current data)")
