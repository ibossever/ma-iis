# %%
"""
Per-cell internal resistance from the QAS vin_2.pkl dataset, using only pack
current and the raw per-cell voltages (no EKF columns): consecutive-sample
differencing cancels out the (roughly constant, over one 30 s step) OCV,
leaving the IR-drop-driven part of the voltage change:
  R_cell(t) = -delta_V_cell(t) / delta_I(t)

Sign: pack current is positive = discharge (QAS_data_format.md), so
V_terminal = OCV - I*R and R = -dV/dI for this to come out as a physical
(positive) resistance.

For each EV:
  - starts from the stored "normal operation" onset time (normal_hour_h in
    extracted_data/pack_current_features.pkl, produced by current2.py) --
    the initial ~-1000 A startup plateau is skipped entirely, same as
    current2.py.
  - a delta is masked out (set to NaN) if either endpoint's pack current is
    clipped/corrupted (current > 300.5 A or current < -300.5 A), or if
    delta_I is exactly 0 (undefined R).

At each remaining timestep t:
  mean_R(t) = mean across the 110 cells of R_cell(t), then masked to NaN if
              still <= 0 (a handful of timesteps come out non-positive even
              after averaging across cells -- small delta_I amplifying
              voltage noise into a spurious sign -- not a physically valid
              resistance)
  std_R(t)  = standard deviation across the 110 cells of R_cell(t)

Per EV, four summary features are computed from these two time series:
  mean_of_mean_R, std_of_mean_R, mean_of_std_R, std_of_std_R

The full mean_R(t)/std_R(t) time series (plus the aligned pack SoC(t), same
post-transition convention as resplot.py's soc[1:]) are also stored per EV,
for other scripts (e.g. cr7.py) to plot against SoC without recomputing the
resistance algorithm themselves.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
FEATURES_IN = Path(__file__).resolve().parents[2] / "extracted_data" / "pack_current_features.pkl"
OUT_PATH = Path(__file__).resolve().parents[2] / "extracted_data" / "resistance_features.pkl"
OUT_PATH_TIMESERIES = Path(__file__).resolve().parents[2] / "extracted_data" / "resistance_timeseries.pkl"
WHOLE_FEATURES_OUT = Path(__file__).resolve().parents[2] / "extracted_data" / "whole_features" / "resistance.pkl"

VOLTAGE_COLS = slice(2, 112)  # 110 series cell voltages [V]
SOC_COL = 223                 # pack SoC, stored as a 0-1 fraction
CURRENT_COL = 224             # pack current [A], signed: negative = charge, positive = discharge

SAMPLE_INTERVAL_S = 30.0
CURRENT_UPPER_THRESHOLD_A = 300.5   # mask a delta if either endpoint current exceeds this
CURRENT_LOWER_THRESHOLD_A = -300.5  # ...or falls below this
TOP_N = 10


def iter_evs():
    """Yield (car_id, data) for every EV under DATA_ROOT that has vin_2.pkl."""
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        pkl_path = ev_dir / "vin_2.pkl"
        if not ev_dir.is_dir() or not pkl_path.exists():
            continue
        tensor = pd.read_pickle(pkl_path)
        yield int(ev_dir.name), tensor.cpu().numpy()


def resistance_timeseries_from_window(data):
    """Core of resistance_timeseries: per-timestep mean and std (across
    cells) of cell resistance, plus the aligned pack SoC (%), computed
    directly from whatever `data` it's given -- no start-of-operation
    slicing here, so this is safe to call on an already-sliced window
    (caller is responsible for anchoring/slicing, e.g. a sliding-window
    driver anchoring at each EV's own start-of-operation)."""
    current = data[:, CURRENT_COL]
    voltage = data[:, VOLTAGE_COLS]
    soc = data[:, SOC_COL] * 100.0

    delta_i = np.diff(current)
    delta_v = np.diff(voltage, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_cell = -delta_v / delta_i[:, None]  # (T-1, 110); V = OCV - I*R -> R = -dV/dI

    endpoint_valid = (current > CURRENT_LOWER_THRESHOLD_A) & (current < CURRENT_UPPER_THRESHOLD_A)
    valid = endpoint_valid[:-1] & endpoint_valid[1:] & (delta_i != 0)
    r_cell[~valid, :] = np.nan

    with warnings.catch_warnings():
        # a timestep where every cell got filtered out above (all-NaN row) is
        # expected and handled downstream (summarize() uses nanmean/nanstd) -
        # just silence numpy's "empty slice" warning for it.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_R = np.nanmean(r_cell, axis=1)
        std_R = np.nanstd(r_cell, axis=1)

    mean_R[mean_R <= 0] = np.nan  # not a physically valid resistance (see module docstring)
    soc_aligned = soc[1:]  # post-transition SoC, same convention as resplot.py
    return mean_R, std_R, soc_aligned


def resistance_timeseries(data, normal_hour_h):
    """Algorithm: per-timestep mean and std (across cells) of cell resistance,
    from the EV's own normal-operation onset onward, plus the aligned pack
    SoC (%) at each of those timesteps. Whole-lifetime entry point used by
    the __main__ block below -- slices off the startup transient, then
    delegates to resistance_timeseries_from_window for the actual algorithm."""
    start_idx = int(round(normal_hour_h * 3600.0 / SAMPLE_INTERVAL_S))
    return resistance_timeseries_from_window(data[start_idx:])


def summarize(mean_R, std_R):
    return (
        float(np.nanmean(mean_R)),
        float(np.nanstd(mean_R)),
        float(np.nanmean(std_R)),
        float(np.nanstd(std_R)),
    )


# Everything below was previously bare top-level ("# %%" notebook-cell)
# code, which meant `import resistance` (e.g. from a sliding-window driver
# that only wants resistance_timeseries_from_window/summarize) silently
# re-ran this entire full-fleet extraction and overwrote OUT_PATH/
# WHOLE_FEATURES_OUT/OUT_PATH_TIMESERIES as a side effect. Wrapped in a
# standard __main__ guard instead -- same code, same "# %%" cell markers
# for interactive use, just no longer runs on import. Behavior when run
# directly (`python resistance.py`) is unchanged.
if __name__ == "__main__":
    # %%
    features_df = pd.read_pickle(FEATURES_IN)

    results = []
    timeseries_results = []
    for car_id, data in iter_evs():
        if car_id not in features_df.index:
            continue
        normal_hour_h = features_df.loc[car_id, "normal_hour_h"]
        mean_R, std_R, soc = resistance_timeseries(data, normal_hour_h)
        mean_of_mean_R, std_of_mean_R, mean_of_std_R, std_of_std_R = summarize(mean_R, std_R)
        results.append({
            "car_id": car_id,
            "mean_of_mean_R": mean_of_mean_R,
            "std_of_mean_R": std_of_mean_R,
            "mean_of_std_R": mean_of_std_R,
            "std_of_std_R": std_of_std_R,
        })
        timeseries_results.append({
            "car_id": car_id,
            "mean_R": mean_R,
            "std_R": std_R,
            "soc": soc,
        })

    resistance_df = pd.DataFrame(results).set_index("car_id")
    resistance_df.to_pickle(OUT_PATH)
    resistance_df.rename_axis("ev_id").to_pickle(WHOLE_FEATURES_OUT)

    resistance_timeseries_df = pd.DataFrame(timeseries_results).set_index("car_id")
    resistance_timeseries_df.to_pickle(OUT_PATH_TIMESERIES)

    # %%
    def print_top(title, column, ascending):
        ranked = resistance_df.sort_values(column, ascending=ascending).head(TOP_N)
        print(f"\n--- Top {TOP_N} EVs by {title} ---")
        print(ranked.to_string())


    FEATURE_LABELS = [
        ("mean_of_mean_R", "mean of mean cell resistance"),
        ("std_of_mean_R", "standard deviation of mean cell resistance"),
        ("mean_of_std_R", "mean of standard deviation of cell resistance across the pack"),
        ("std_of_std_R", "standard deviation of standard deviation of cell resistance across the pack"),
    ]

    for column, label in FEATURE_LABELS:
        print_top(f"highest {label}", column, ascending=False)
        print_top(f"lowest {label}", column, ascending=True)
