"""
Shared helpers for the revision early-warning track (thesis_revision_plan.pdf,
tasks A6/A7/A10/A12): window loading + cleaning for BOTH window sources, the
ground-truth label, a pack-type proxy, and the vehicle-level CV folds. Imported
by select_k.py and ew_compare.py so both scripts see exactly the same windows.

--- WINDOW_SOURCE -------------------------------------------------------------
"120h": filtered_features_windows.pkl (sliding_window_fleet_features.py), the
    15 filtered_features.pkl columns. Cleaned exactly like detection.py's
    WINDOWED TRACK: capacity column forward-filled per EV, then any row still
    NaN dropped (-> 390 EVs).
"24h": filtered_features_windows_24h.pkl (sliding_window_fleet_features_24h.py),
    the 7 time-domain columns (same KEPT_FEATURE_COLUMNS as kmeans24.py). No
    capacity column, so only the NaN drop applies.

--- Pack type (proxy until A1's pack_type.csv exists) -------------------------
pack_type = "small" if the whole-session charge_capacity_per_100soc__avg_ah_per_100soc
< SMALL_PACK_MAX_AH (same 60 Ah cut as relative_charge_capacity.py), else
"large". Uses the whole-session average instead of A1's lifetime median of the
windowed values -- swap in pack_type.csv once A1 is done.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
REVISION_OUT = EXTRACTED_DATA / "revision"
REVISION_OUT.mkdir(exist_ok=True)

WHOLE_SESSION_FEATURES_IN = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"  # from features.py
WINDOW_FEATURES_IN = {
    "120h": EXTRACTED_DATA / "early_warning" / "filtered_features_windows.pkl",
    "24h": EXTRACTED_DATA / "early_warning" / "filtered_features_windows_24h.pkl",
}
FEATURES_24H = [  # == kmeans24.py's KEPT_FEATURE_COLUMNS
    "local_temp__min_val", "local_temp__avg_amp", "pack_temp__mean_pack_temp",
    "voltage_features__ev_std_std", "voltage_features__ev_std_max", "voltage_features__max_dif_max",
    "resistance__std_of_std_R",
]
CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is confirmed-faulty -- same ground truth everywhere
SMALL_PACK_MAX_AH = 60    # == relative_charge_capacity.py


def load_windows(window_source: str) -> tuple[pd.DataFrame, list[str]]:
    """Cleaned windows for one source, sorted by (ev_id, window_start), with
    is_faulty_ev and pack_type columns added. Returns (windows, feature_columns)."""
    whole_session = pd.read_pickle(WHOLE_SESSION_FEATURES_IN)
    feature_columns = list(whole_session.columns) if window_source == "120h" else FEATURES_24H

    windows = pd.read_pickle(WINDOW_FEATURES_IN[window_source])
    windows = windows.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
    if CAPACITY_COLUMN in feature_columns:
        windows[CAPACITY_COLUMN] = windows.groupby("ev_id")[CAPACITY_COLUMN].ffill()
    windows = windows.loc[windows[feature_columns].notna().all(axis=1)].reset_index(drop=True)

    windows["is_faulty_ev"] = windows["ev_id"] >= FAULT_EV_ID_CUTOFF
    pack_type = pd.Series(
        np.where(whole_session[CAPACITY_COLUMN] < SMALL_PACK_MAX_AH, "small", "large"), index=whole_session.index,
    )
    windows["pack_type"] = windows["ev_id"].map(pack_type)
    return windows, feature_columns


def ev_table(windows: pd.DataFrame) -> pd.DataFrame:
    """One row per EV: is_faulty_ev, pack_type, n_windows, last_window_start_hours."""
    return windows.groupby("ev_id").agg(
        is_faulty_ev=("is_faulty_ev", "first"),
        pack_type=("pack_type", "first"),
        n_windows=("window_start", "size"),
        last_window_start_hours=("window_start_hours", "max"),
    )


def vehicle_folds(evs: pd.DataFrame, n_splits: int, seed: int):
    """StratifiedKFold over VEHICLES, stratified on label x pack_type (A2).
    Yields (train_ev_ids, test_ev_ids)."""
    stratum = evs["is_faulty_ev"].map({True: "faulty", False: "healthy"}) + "_" + evs["pack_type"]
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(evs.index, stratum):
        yield evs.index[train_idx], evs.index[test_idx]
