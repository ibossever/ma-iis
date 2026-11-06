# %%
"""
Windowed, full-fleet version of filtered_features.pkl: steps a
WINDOW_HOURS window through every EV's raw telemetry (vin_1.pkl/vin_2.pkl)
in STRIDE_HOURS increments, and for each valid window position computes the
same 15 features filtered_features.pkl already screens down to -- just
windowed instead of whole-lifetime. Extra feature extraction ONLY: no model
fitting, no SHAP, no detection logic, and none of the whole_features/*.pkl
outputs are touched (this script imports the relevant functions from their
source scripts but never runs those scripts' own __main__ blocks).

--- Two internal slice lengths per window -----------------------------------
Per window, two views are derived from a SINGLE raw pull (see PIPELINE
below) -- never re-queried separately per feature:
  - the FULL WINDOW_HOURS span, for the 3 features that are FFT-based
    (need a long span to resolve low frequencies): current_fft__*,
    soc_fft__*, and local_temp__mean_diff. That last one is the one
    non-obvious case -- it lives in the "local_temp" whole_features file,
    not a "*_fft" one, but it's mechanistically an FFT pooled-spectrum
    diff (see plot_localtemp.py's fft_diff_features), so it's classified
    by MECHANISM here, not by filename, and gets the full window.
  - the trailing FAST_SLICE_HOURS of that same window, for every other
    feature (all time-domain: means, stds, maxima, peak/trough amplitude,
    coulomb counting).
FFT_SLICE_HOURS == WINDOW_HOURS below, so "the FFT slice" is just the whole
window, un-sliced further. FAST_SLICE_HOURS is ALSO set == WINDOW_HOURS
(see CONFIG) -- kmeans.py's windowed clustering wants every one of the 15
features evaluated over the same 120h span, not a trailing 24h sub-slice of
it, so there is no longer a real fast/full distinction in THIS script's
output: vin*_fast below degenerates to vin*_window, and fast_slice_start
degenerates to window_start. The two-views machinery is kept generic
(FAST_SLICE_HOURS is still a real, independently-settable constant) rather
than special-cased away, so this file stays a faithful reference for the
mechanism even though it's currently exercised with fast == full. The
dedicated trailing-24h extraction now lives in its own sibling script,
sliding_window_fleet_features_24h.py, which computes ONLY the 7 features
that survive a 24h span (see that script's docstring for which/why) as a
genuinely fresh 24h/12h-stride sequence, not a byproduct of this script.

--- FFT reference spectra ---------------------------------------------------
current_fft__*/soc_fft__*/local_temp__mean_diff are all "this window's
spectrum minus a pooled fleet-average reference spectrum" -- current_fft.py,
soc_fft.py and plot_localtemp.py each compute that reference internally
while running (whole-signal FFT, from index 0 for current/soc, from
start_hours+1h for local_temp -- exactly what each script does today) but
never save it anywhere; only the derived per-EV band-diff features get
persisted. So each reference is rebuilt ONCE here, upfront (Phase 1, one
full-fleet raw pass per reference), then frozen and reused as a fixed
diff target for every window of every EV in Phase 2 -- i.e. every window's
mean_diff/std_diff answers "how does this window's spectrum compare to
this signal's typical WHOLE-LIFETIME fleet-average shape", not to some
window-local baseline. This exactly reproduces what current_fft.py/
soc_fft.py/plot_localtemp.py already compute for the whole-lifetime
features (see refactor comments in those files), just evaluated on a
window instead of a whole signal. Note this means vin_2.pkl gets read an
extra 2x (once per reference: current, soc) beyond Phase 2's own read --
an accepted cost for reproducing the existing methodology exactly rather
than approximating it from windowed data (see conversation).

--- Windowing / anchoring ---------------------------------------------------
The window SEQUENCE (window_start/window_end -- the same boundaries every
feature in a row shares) starts at raw index 0 for every EV and steps
forward in STRIDE_HOURS increments; it does NOT skip ahead to
start-of-operation. A window is only emitted if the full WINDOW_HOURS span
fits within the EV's recorded data from that position -- never padded or
shortened. Because each EV's raw recording is one continuous, evenly-
sampled array (no internal gaps in the raw index -- see PIPELINE), the only
way a window gets skipped is running past the end of the recording, so per
EV this is really "how many full windows fit" plus, if any span remains,
exactly one un-emitted trailing partial window at the end -- both counts
are logged per EV.

--- start-of-operation only matters for the 3 pack-current-dependent
    features ------------------------------------------------------------
normal_hour_h (extracted_data/pack_current_features.pkl, same convention as
resistance.py/capac2.py/plot_localtemp.py) marks when the pack current
sensor leaves its invalid startup plateau (pinned near -1000 A). Only
features that actually read the current channel are corrupted by data
before that point:
  - current_fft__band1_mean_diff/band1_std_diff/band2_std_diff (FFT of
    pack current directly)
  - resistance__std_of_std_R (needs current for R = -dV/dI)
  - charge_capacity_per_100soc__avg_ah_per_100soc (needs current for
    coulomb counting)
soc_fft__*, local_temp__* (including local_temp__mean_diff, which is
FFT-based but never touches the current channel), pack_temp__mean_pack_temp
and voltage_features__* never read pack current and are NOT masked by
start-of-operation -- they use their window's full, unmodified span exactly
as before.

For the 3 current-dependent features, whichever slice they normally use
(the full window for current_fft, the trailing FAST_SLICE_HOURS for
resistance/charge_capacity) is additionally clipped to
[max(slice_start, anchor_idx), slice_end) before computing -- i.e. the
window boundaries stay aligned with every other feature, only the current-
reading portion of that specific window shrinks. If the clipped span is
under MIN_CURRENT_SPAN_HOURS (too little real current data to trust an FFT
or a resistance/capacity estimate from), or the window ends before
start-of-operation entirely, those 3 features come out NaN for that window
while the other 4 feature groups still compute normally from the same
window. In practice this only affects a window's very first few
occurrences per EV (normal_hour_h averages ~38h, max ~143h, against a 120h
window), not the whole timeline.

--- PIPELINE (per EV) --------------------------------------------------------
vin_1.pkl and vin_2.pkl are each loaded ONCE per EV (not once per window --
re-reading ~200-300MB raw tensors per EV from disk for every one of its
~dozens of windows would be pure waste) and kept in memory as plain numpy
arrays; every window for that EV is then just an in-memory slice of those
two arrays (the "single query/slice per window" from the task description,
read as "no redundant re-querying of raw data for the different feature
types within one window" -- satisfied via the two derived views below --
rather than "hit disk again for every window").

--- Validity guards ----------------------------------------------------------
charge_capacity_per_100soc__avg_ah_per_100soc is the one ratio-type feature
here (Ah / delta_soc). Its existing guard (capac2.py's MIN_SOC_INCREASE --
a window's charging session(s) must span >= 20 SoC percentage points to
count) already IS the "denominator range too small" guard; this script
just surfaces it as a NaN + a
charge_capacity_per_100soc__avg_ah_per_100soc_valid column instead of
silently averaging over zero qualifying sessions. None of the other 14
features are ratio-type, so no other feature gets a validity column.

Input:
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      loaded ONLY to read off its column list (which 15 features to
      compute windowed versions of); its actual values are not used.
  extracted_data/pack_current_features.pkl (current2.py) -- normal_hour_h,
      used as each EV's start-of-operation anchor.
  /media/ssd_2tb/jak-iis/<ev_id>/vin_1.pkl, vin_2.pkl -- raw QAS telemetry.
Output:
  extracted_data/early_warning/filtered_features_windows.pkl -- one row per
      (ev_id, window position): ev_id, window_start, window_end (raw sample
      indices, window sequence anchored at 0 -- see "Windowing / anchoring"
      below), window_start_hours (elapsed hours relative to
      start-of-operation -- can be NEGATIVE for early windows that start
      before it; kept relative rather than absolute so it's comparable
      across EVs regardless of each one's own normal_hour_h/recording
      length), then the 15 filtered features (plus the one _valid
      companion column).
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

import current_fft   # noqa: E402  (band-diff FFT features on pack current)
import soc_fft        # noqa: E402  (band-diff FFT features on pack SoC)
import analysis        # noqa: E402  (mean_pack_temp)
import resistance       # noqa: E402  (resistance_timeseries_from_window, summarize)
import capac2             # noqa: E402  (avg_ah_per_100soc_from_window)
import before_plot         # noqa: E402  (voltage_outlier_features)
import plot_localtemp        # noqa: E402  (signal_freq_amp, fft_diff_features, compute_fleet_pool_avg_spectrum)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
SAMPLE_INTERVAL_S = 30.0  # must match every sibling script's own SAMPLE_INTERVAL_S

FILTERED_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # column list only
NORMAL_HOURS_IN = PROJECT_ROOT / "extracted_data" / "pack_current_features.pkl"  # from current2.py

OUT_DIR = PROJECT_ROOT / "extracted_data" / "early_warning"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "filtered_features_windows.pkl"

WINDOW_HOURS = 120
STRIDE_HOURS = 12
FFT_SLICE_HOURS = 120    # FFT-band features use the full window
FAST_SLICE_HOURS = 120   # == WINDOW_HOURS: every other feature ALSO uses the full 120h window now (see
                          # module docstring's "Two internal slice lengths per window" -- was 24 previously;
                          # the dedicated trailing-24h table is now produced separately, by
                          # sliding_window_fleet_features_24h.py)

# Minimum post-start-of-operation span a current-dependent feature needs
# within its slice to be computed at all (see module docstring) -- below
# this, that feature is NaN for that window rather than computed from a
# too-short/unreliable sliver of real current data.
MIN_CURRENT_SPAN_HOURS = 6

WINDOW_SAMPLES = round(WINDOW_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
STRIDE_SAMPLES = round(STRIDE_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
FFT_SLICE_SAMPLES = round(FFT_SLICE_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
FAST_SLICE_SAMPLES = round(FAST_SLICE_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
MIN_CURRENT_SPAN_SAMPLES = round(MIN_CURRENT_SPAN_HOURS * 3600.0 / SAMPLE_INTERVAL_S)
assert FFT_SLICE_SAMPLES == WINDOW_SAMPLES, (
    "FFT_SLICE_HOURS must equal WINDOW_HOURS -- extract_window_features() below assumes the FFT "
    "slice IS the full window, not a further sub-slice of it."
)
assert FAST_SLICE_SAMPLES <= WINDOW_SAMPLES, "FAST_SLICE_HOURS must fit inside WINDOW_HOURS"


def iter_evs_raw():
    """Yield (ev_id, vin1, vin2) for every EV under DATA_ROOT with both raw
    files, each loaded ONCE as a plain numpy array (see module docstring's
    PIPELINE section for why loading is per-EV, not per-window)."""
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


def extract_window_features(
    vin1_window: np.ndarray, vin2_window: np.ndarray, window_start: int, window_end: int, anchor_idx: int,
    current_pool_avg: np.ndarray, soc_pool_avg: np.ndarray, local_temp_pool_avg: np.ndarray,
) -> dict:
    """All 15 filtered features for one window. `vin1_window`/`vin2_window`
    are the FULL WINDOW_HOURS slice (== the FFT slice, see module
    docstring), spanning absolute raw indices [window_start, window_end);
    the trailing FAST_SLICE_HOURS view is derived from them in-memory
    below, not re-pulled. `anchor_idx` is this EV's start-of-operation raw
    index -- used ONLY by the 3 pack-current-dependent features (see
    module docstring's "start-of-operation only matters for..." section);
    every other feature ignores it and uses its slice unmodified.
    """
    vin2_fast = vin2_window[-FAST_SLICE_SAMPLES:]
    vin1_fast = vin1_window[-FAST_SLICE_SAMPLES:]
    fast_slice_start = window_end - FAST_SLICE_SAMPLES

    row: dict = {}

    # --- current_fft: full WINDOW_HOURS/FFT_SLICE_HOURS span, clipped to
    # start-of-operation (current-dependent -- see module docstring) ------
    current_valid_start = max(window_start, anchor_idx)
    if window_end - current_valid_start >= MIN_CURRENT_SPAN_SAMPLES:
        current_full = vin2_window[current_valid_start - window_start:, current_fft.CURRENT_COL]
        freqs_per_h, spectrum = current_fft.fft_spectrum(current_full)
        current_spectrum = np.interp(current_fft.FFT_COMMON_FREQS, freqs_per_h, spectrum)
        current_bands = current_fft.band_diff_features(current_spectrum, current_pool_avg, current_fft.BAND_INTERVALS_PER_H)
        row["current_fft__band1_mean_diff"] = current_bands["band1_mean_diff"]
        row["current_fft__band1_std_diff"] = current_bands["band1_std_diff"]
        row["current_fft__band2_std_diff"] = current_bands["band2_std_diff"]
    else:
        row["current_fft__band1_mean_diff"] = np.nan
        row["current_fft__band1_std_diff"] = np.nan
        row["current_fft__band2_std_diff"] = np.nan

    # --- soc_fft / local_temp__mean_diff: FFT-based but NOT current-
    # dependent (see module docstring) -- full window, unmodified ---------
    soc_full = vin2_window[:, soc_fft.SOC_COL]
    freqs_per_h, spectrum = soc_fft.fft_spectrum(soc_full)
    soc_spectrum = np.interp(soc_fft.FFT_COMMON_FREQS, freqs_per_h, spectrum)
    soc_bands = soc_fft.band_diff_features(soc_spectrum, soc_pool_avg, soc_fft.BAND_INTERVALS_PER_H)
    row["soc_fft__band1_std_diff"] = soc_bands["band1_std_diff"]
    row["soc_fft__band2_std_diff"] = soc_bands["band2_std_diff"]
    row["soc_fft__band3_mean_diff"] = soc_bands["band3_mean_diff"]

    local_temp_full = vin1_window[:, 0, plot_localtemp.MAX_TEMP_COL]
    mean_diff, _std_diff = plot_localtemp.fft_diff_features(local_temp_full, local_temp_pool_avg)
    row["local_temp__mean_diff"] = mean_diff

    # --- local_temp__min_val/avg_amp, pack_temp, voltage_features: trailing
    # FAST_SLICE_HOURS, NOT current-dependent -- unmodified ----------------
    local_temp_fast = vin1_fast[:, 0, plot_localtemp.MAX_TEMP_COL]
    row["local_temp__min_val"] = float(np.min(local_temp_fast))
    _avg_freq, avg_amp = plot_localtemp.signal_freq_amp(local_temp_fast)
    row["local_temp__avg_amp"] = avg_amp

    pack_temp_fast = vin2_fast[:, analysis.PACK_TEMP_COL]
    row["pack_temp__mean_pack_temp"] = analysis.mean_pack_temp(pack_temp_fast)

    voltage_feats = before_plot.voltage_outlier_features(vin2_fast)
    row["voltage_features__ev_std_std"] = voltage_feats["ev_std_std"]
    row["voltage_features__ev_std_max"] = voltage_feats["ev_std_max"]
    row["voltage_features__max_dif_max"] = voltage_feats["max_dif_max"]

    # --- resistance / charge_capacity: trailing FAST_SLICE_HOURS, clipped
    # to start-of-operation (current-dependent -- see module docstring) ---
    fast_valid_start = max(fast_slice_start, anchor_idx)
    if window_end - fast_valid_start >= MIN_CURRENT_SPAN_SAMPLES:
        vin2_fast_masked = vin2_window[fast_valid_start - window_start:]

        mean_R, std_R, _soc = resistance.resistance_timeseries_from_window(vin2_fast_masked)
        _mean_of_mean_R, _std_of_mean_R, _mean_of_std_R, std_of_std_R = resistance.summarize(mean_R, std_R)
        row["resistance__std_of_std_R"] = std_of_std_R

        # ratio-type feature -- see module docstring's Validity guards section
        cap_result = capac2.avg_ah_per_100soc_from_window(vin2_fast_masked)
        if cap_result is None:
            row["charge_capacity_per_100soc__avg_ah_per_100soc"] = np.nan
            row["charge_capacity_per_100soc__avg_ah_per_100soc_valid"] = False
        else:
            row["charge_capacity_per_100soc__avg_ah_per_100soc"] = cap_result["avg_ah_per_100soc"]
            row["charge_capacity_per_100soc__avg_ah_per_100soc_valid"] = True
    else:
        row["resistance__std_of_std_R"] = np.nan
        row["charge_capacity_per_100soc__avg_ah_per_100soc"] = np.nan
        row["charge_capacity_per_100soc__avg_ah_per_100soc_valid"] = False

    return row



if __name__ == "__main__":
    # %% 0. sanity: confirm we compute exactly the columns filtered_features.pkl expects
    EXPECTED_FEATURE_COLUMNS = set(pd.read_pickle(FILTERED_FEATURES_IN).columns)
    COMPUTED_FEATURE_COLUMNS = {
        "current_fft__band1_mean_diff", "current_fft__band1_std_diff", "current_fft__band2_std_diff",
        "soc_fft__band1_std_diff", "soc_fft__band2_std_diff", "soc_fft__band3_mean_diff",
        "local_temp__mean_diff", "local_temp__min_val", "local_temp__avg_amp",
        "pack_temp__mean_pack_temp", "resistance__std_of_std_R",
        "voltage_features__ev_std_std", "voltage_features__ev_std_max", "voltage_features__max_dif_max",
        "charge_capacity_per_100soc__avg_ah_per_100soc",
    }
    if COMPUTED_FEATURE_COLUMNS != EXPECTED_FEATURE_COLUMNS:
        raise RuntimeError(
            "This script's COMPUTED_FEATURE_COLUMNS has drifted out of sync with filtered_features.pkl's "
            f"current column list.\n  only in filtered_features.pkl: {sorted(EXPECTED_FEATURE_COLUMNS - COMPUTED_FEATURE_COLUMNS)}\n"
            f"  only computed here:            {sorted(COMPUTED_FEATURE_COLUMNS - EXPECTED_FEATURE_COLUMNS)}\n"
            "Add/remove the corresponding extraction call(s) in extract_window_features() to match."
        )
    print(f"computing windowed versions of all {len(EXPECTED_FEATURE_COLUMNS)} filtered_features.pkl columns")

    # %% 1. Phase 1 -- rebuild the 3 fixed FFT reference spectra (one full-fleet
    # raw pass each, see module docstring)
    print(f"\nPhase 1/2: building fleet-wide FFT reference spectra "
          f"(window={WINDOW_HOURS}h, stride={STRIDE_HOURS}h, FFT slice={FFT_SLICE_HOURS}h, fast slice={FAST_SLICE_HOURS}h)...")

    t0 = time.time()
    current_pool_avg = current_fft.compute_fleet_pool_avg_spectrum()
    print(f"  current_fft pool-average spectrum built ({time.time() - t0:.0f}s)")

    t0 = time.time()
    soc_pool_avg = soc_fft.compute_fleet_pool_avg_spectrum()
    print(f"  soc_fft pool-average spectrum built ({time.time() - t0:.0f}s)")

    t0 = time.time()
    local_temp_pool_avg = plot_localtemp.compute_fleet_pool_avg_spectrum()
    print(f"  local_temp pool-average spectrum built ({time.time() - t0:.0f}s)")

    # %% 2. Phase 2 -- per-EV, per-window feature extraction
    print("\nPhase 2/2: sliding windows across the full fleet...")

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
        # Window sequence starts at raw index 0 -- NOT at start-of-operation --
        # so every feature group gets the same shared window boundaries; only
        # the 3 current-dependent features (inside extract_window_features)
        # additionally clip to start-of-operation. See module docstring.
        pos = 0
        while pos + WINDOW_SAMPLES <= n_samples:
            window_end = pos + WINDOW_SAMPLES
            features = extract_window_features(
                vin1[pos:window_end], vin2[pos:window_end], pos, window_end, anchor_idx,
                current_pool_avg, soc_pool_avg, local_temp_pool_avg,
            )
            rows.append({
                "ev_id": ev_id,
                "window_start": pos,
                "window_end": window_end,
                # relative to start-of-operation (can be negative for windows
                # that start before it) -- comparable across EVs regardless of
                # each one's own normal_hour_h/total recording length.
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

    print(f"Phase 2 done: {n_evs_processed} EVs processed in {time.time() - t_start:.0f}s")

    # %% 3. assemble, estimate size, save
    windows_df = pd.DataFrame(rows)

    n_rows = len(windows_df)
    n_cols = len(windows_df.columns)
    estimated_bytes = n_rows * n_cols * 8  # float64 upper bound; ev_id/window_start/end are ints of the same width, _valid is bool (smaller) -- this over-estimates slightly
    print(f"\nassembled {n_rows} rows x {n_cols} columns")
    print(f"estimated size before writing: ~{estimated_bytes / 1e6:.1f} MB (rough float64 upper bound)")

    windows_df.to_pickle(OUT_PATH)
    actual_bytes = OUT_PATH.stat().st_size
    print(f"saved to {OUT_PATH} ({actual_bytes / 1e6:.1f} MB on disk)")

    # %% 4. summary
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
        feature_cols = sorted(EXPECTED_FEATURE_COLUMNS)
        rows_with_nan = windows_df[feature_cols].isna().any(axis=1).sum()
        print(f"\n--- Data quality ---")
        print(f"{rows_with_nan}/{n_rows} rows ({rows_with_nan / n_rows:.1%}) have >= 1 NaN feature value")
        n_invalid_capacity = (~windows_df["charge_capacity_per_100soc__avg_ah_per_100soc_valid"]).sum()
        print(f"{n_invalid_capacity}/{n_rows} rows ({n_invalid_capacity / n_rows:.1%}) flagged invalid for "
              f"charge_capacity_per_100soc__avg_ah_per_100soc (no qualifying charging session in that window)")
        n_current_masked = windows_df["current_fft__band1_mean_diff"].isna().sum()
        n_resistance_masked = windows_df["resistance__std_of_std_R"].isna().sum()
        print(f"{n_current_masked}/{n_rows} rows ({n_current_masked / n_rows:.1%}) have current_fft__* as NaN "
              f"(< {MIN_CURRENT_SPAN_HOURS}h of post-start-of-operation current data in that window)")
        print(f"{n_resistance_masked}/{n_rows} rows ({n_resistance_masked / n_rows:.1%}) have "
              f"resistance__std_of_std_R as NaN (same reason, trailing {FAST_SLICE_HOURS}h slice)")
