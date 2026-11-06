"""
Replace charge_capacity_per_100soc__avg_ah_per_100soc's VALUES (same
column name, same file, same shape, same _valid column) in
filtered_features_windows.pkl with a per-EV self-normalized fade ratio --
WINDOWED FEATURES ONLY -- AND add a second, new column alongside it (see
"SECOND FEATURE" below). Whole-session (filtered_features.pkl,
capac2.py, extracted_data/whole_features/*) is intentionally left
untouched -- a relative version of it was tried multiple ways and never
cleared real statistical screening (see conversation); windowed features
were never part of that screening, so there's no such gate here.

--- SECOND FEATURE: charge_capacity_per_100soc__anchor_avg_ratio --------
    anchor_avg_ratio = (early_life_capacity + late_life_capacity)
                        / (2 * this_window's_own_original_absolute_value)

  - early_life_capacity: the SAME per-EV baseline already used as the
    denominator for the first feature (median of the upper half of that
    EV's first <=EARLY_N_WINDOWS valid windows) -- reused as-is, not
    recomputed differently.
  - late_life_capacity (new): median of that EV's LAST <=EARLY_N_WINDOWS
    valid windows, chronologically. Plain median, NOT the upper-half trick
    -- that correction exists specifically because of a verified early-life
    BMS/estimator ramp-up bias (see ev 183 in conversation); there's no
    established artifact at the end of an EV's recorded life to correct
    for, so this deliberately isn't symmetric with early_life_capacity.
  - this window's own original absolute value: read from the pristine
    backup, same source as the first feature's numerator -- "the old
    feature value" is inherently per-window inside this table (there's no
    single whole-lifetime scalar living in filtered_features_windows.pkl;
    that concept only exists in the separate filtered_features.pkl, which
    this script does not touch).

IMPORTANT CAVEAT (see conversation): every windowed-track script
(detection.py, kmeans.py, wisek_window.py, clust.py, clust2.py,
sliding_windows.py, shap.py) selects WHICH columns to use from
filtered_features.pkl's (whole-session) column list, then reads values for
those names out of THIS file. filtered_features.pkl does not have
anchor_avg_ratio and is not being given it (whole session stays untouched
per instruction), so none of those scripts will pick anchor_avg_ratio up
automatically the way they do the first feature (same name as before) --
it is computed and stored here, correctly, but UNUSED by any current model
run unless/until it's also added to filtered_features.pkl.

SCOPE, AND WHY THIS STAYS SAFE (see conversation):
- Windowed features are NOT part of the correlation/fault-relevance
  screening chain (spearman_corr.py / pearson_corr.py / fault_feature.py /
  features.py) -- that chain only ever sees filtered_features.pkl
  (whole-session). This script does not touch that chain, or
  filtered_features.pkl, or extracted_data/whole_features/*, or
  capac2.py, or sliding_window_fleet_features.py, at all. The whole-session
  feature stays exactly as restored (absolute Ah) -- a relative version of
  it was tried twice and failed real statistical screening both times; this
  script does not reopen that question.
- Every downstream WINDOWED-track script (detection.py, kmeans.py,
  wisek_window.py, clust.py, clust2.py, sliding_windows.py, shap.py)
  selects WHICH columns to use from filtered_features.pkl's (whole-session,
  screened) column list, then reads the VALUES for those same column names
  out of filtered_features_windows.pkl. Since this script keeps the column
  name identical, none of those scripts, or their CAPACITY_COLUMN
  constants, need to change at all -- they pick up the new relative values
  exactly the way they already pick up any other column's values. This is
  the whole reason a rename was avoided this time (contrast with an
  earlier attempt that renamed the column and had to touch 7 files'
  CAPACITY_COLUMN constants -- reverted; see conversation).

BASELINE TECHNIQUE (identical to an earlier attempt at this exact problem,
already validated against this dataset's specific pitfall):
    baseline(ev) = median of the UPPER HALF of that EV's first (up to)
    EARLY_N_WINDOWS valid windows, chronologically (by window_start_hours).
A plain mean/median of the first N windows was checked against ev 183's
actual data and found to understate a real early-life RAMP (its first ~5-6
windows climb from ~52 to ~79 Ah/100soc before settling -- BMS/estimator
convergence, not physical fade; packs don't gain capacity in their first
days). Taking the upper half discards exactly those low-side ramp readings
without needing to know how many there are; taking the MEDIAN of that
upper half (not the single max) still guards against one spuriously-high
window. EVs with 1-4 valid windows still get a baseline -- the median
degenerates gracefully (1 value -> itself, 2 -> their max, etc.) -- rather
than being excluded: an earlier, stricter version of this that required
>= 5 windows excluded exactly 10 EVs, and every single one of them turned
out to be faulty, which is exactly the population most worth keeping.

EVs with ZERO windows at all (their total recording is shorter than one
120h window) simply have zero rows in filtered_features_windows.pkl to
begin with -- nothing to normalize for them, no special-case handling
needed here (contrast with the whole-session version, which needed an
explicit 1.0 fallback specifically because it had to produce some value
for every EV in filtered_features.pkl; that concern doesn't apply here).

VALIDITY: both features are valid EXACTLY when the window's own original
absolute value was (charge_capacity_per_100soc__avg_ah_per_100soc_valid is
left completely UNCHANGED and doubles as anchor_avg_ratio's validity flag
too -- no separate _valid column for it). If a window has its own valid
value, its EV by definition has at least one valid window (this one), so
both early_life_capacity and late_life_capacity can always be formed -- no
row that was valid before becomes invalid, and vice versa.

SAFETY:
- The original file is backed up ONCE, alongside itself, as
  filtered_features_windows.pkl.pre_windowed_relative_capacity.bak, before
  the first overwrite. Every run recomputes BOTH features from THAT
  pristine backup (never from the live file, which may already be this
  script's own prior output) -- reruns are idempotent, never compound.
- Before writing anything: shape (now +1 column for anchor_avg_ratio),
  every OTHER existing column's values, and the first feature's own values
  (must come out byte-identical to a fresh run of the original single-
  feature logic) are asserted unchanged/correct. The script raises (writes
  nothing) if any of these don't hold, rather than silently writing a
  corrupted file.

Input:  extracted_data/early_warning/filtered_features_windows.pkl
Output: same file, overwritten in place (backed up first, see above) --
    charge_capacity_per_100soc__avg_ah_per_100soc unchanged from the
    earlier run of this script, plus one new column,
    charge_capacity_per_100soc__anchor_avg_ratio.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_FILE = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows.pkl"
BACKUP_FILE = WINDOWS_FILE.with_name(WINDOWS_FILE.name + ".pre_windowed_relative_capacity.bak")

CAPACITY_COL = "charge_capacity_per_100soc__avg_ah_per_100soc"
CAPACITY_VALID_COL = f"{CAPACITY_COL}_valid"
ANCHOR_RATIO_COL = "charge_capacity_per_100soc__anchor_avg_ratio"

EARLY_N_WINDOWS = 10  # how many of an EV's earliest/latest valid windows feed early/late_life_capacity


def compute_ev_baseline(values_chronological: np.ndarray) -> float:
    """early_life_capacity: median of the upper half of the first (up to)
    EARLY_N_WINDOWS values -- see module docstring."""
    early = values_chronological[:EARLY_N_WINDOWS]
    n_early = len(early)
    top_k = -(-n_early // 2)  # ceil(n_early / 2), as a positive count
    upper_half = np.sort(early)[-top_k:]
    return float(np.median(upper_half))


def compute_ev_late_life_capacity(values_chronological: np.ndarray) -> float:
    """late_life_capacity: plain median of the LAST (up to) EARLY_N_WINDOWS
    values -- see module docstring for why this is NOT the same upper-half
    treatment as compute_ev_baseline."""
    late = values_chronological[-EARLY_N_WINDOWS:]
    return float(np.median(late))


def main() -> None:
    if not BACKUP_FILE.exists():
        shutil.copy2(WINDOWS_FILE, BACKUP_FILE)
        print(f"backed up original -> {BACKUP_FILE.name}")
    else:
        print(f"backup already exists, reading pristine original from it: {BACKUP_FILE.name}")

    original = pd.read_pickle(BACKUP_FILE)  # always the true original, never this script's own prior output

    relative = pd.Series(np.nan, index=original.index)
    anchor_ratio = pd.Series(np.nan, index=original.index)
    valid = original[original[CAPACITY_VALID_COL]]
    for ev_id, g in valid.groupby("ev_id"):
        g_sorted = g.sort_values("window_start_hours")
        values = g_sorted[CAPACITY_COL].to_numpy()

        early_life_capacity = compute_ev_baseline(values)
        late_life_capacity = compute_ev_late_life_capacity(values)

        relative.loc[g_sorted.index] = values / early_life_capacity
        anchor_ratio.loc[g_sorted.index] = (early_life_capacity + late_life_capacity) / (2 * values)

    df = original.copy()
    df[CAPACITY_COL] = relative
    df[ANCHOR_RATIO_COL] = anchor_ratio

    # --- safety checks ---
    assert df.shape[0] == original.shape[0], "row count changed -- aborting, nothing written"
    assert df.shape[1] == original.shape[1] + 1, "expected exactly one new column -- aborting, nothing written"
    unchanged_cols = [c for c in original.columns if c != CAPACITY_COL]
    assert df[unchanged_cols].equals(original[unchanged_cols]), "a non-target column changed -- aborting, nothing written"
    assert df[CAPACITY_COL].isna().equals(original[CAPACITY_COL].isna()), \
        "NaN pattern of the first feature changed -- aborting, nothing written"
    assert df[CAPACITY_VALID_COL].equals(original[CAPACITY_VALID_COL]), \
        "_valid column changed -- aborting, nothing written"
    assert df[ANCHOR_RATIO_COL].isna().equals(original[CAPACITY_COL].isna()), \
        "anchor_avg_ratio's NaN pattern doesn't match the shared validity flag -- aborting, nothing written"

    df.to_pickle(WINDOWS_FILE)
    n_valid = int(df[CAPACITY_VALID_COL].sum())
    print(f"wrote {WINDOWS_FILE} ({df.shape[1]} columns, was {original.shape[1]})")
    print(f"  {CAPACITY_COL}: unchanged from the earlier run ({n_valid}/{len(df)} valid rows)")
    print(df.loc[df[CAPACITY_VALID_COL], CAPACITY_COL].describe())
    print(f"  {ANCHOR_RATIO_COL}: new ({n_valid}/{len(df)} valid rows, same validity flag)")
    print(df.loc[df[CAPACITY_VALID_COL], ANCHOR_RATIO_COL].describe())


if __name__ == "__main__":
    main()
