# %%
"""
Adjusted Rand Index (ARI) between two WINDOWED K-means clusterings, run with
different SELECTION_MODE/K configurations -- reuses the same population-
selection / windowed-feature-prep machinery as wisek_window.py (see that
script's module docstring for the full rationale), just run twice and
compared instead of plotted.

Since the two clusterings can be fit on different EV populations (e.g. one
on "faulty" EVs only, the other on "all" EVs), ARI can only be computed on
the windows that are common to BOTH runs' populations -- each window is
identified by (ev_id, window_start). If one population is a subset of the
other (as with "faulty" vs "all"), this is exactly the smaller population's
windows.

For a start, this script compares:
  - Run A: k=16 K-means on windows of FAULTY EVs only (SELECTION_MODE="faulty")
  - Run B: k=9  K-means on windows of ALL EVs           (SELECTION_MODE="all")

Input: same as wisek_window.py --
  extracted_data/fault_detection/filtered_features.pkl
  extracted_data/early_warning/filtered_features_windows.pkl

Output:
  printed ARI (and cluster-size tables for both runs)
  figures/aris/ari_vs_k_<RUN_A selection_mode>_k<RUN_A k>_vs_<RUN_B selection_mode>_k<RUN_B k>.png --
      ARI vs. RUN_A's k, swept over +/-K_SWEEP_RADIUS around RUN_A["k"] (RUN_B held fixed at its
      configured k), chosen RUN_A["k"] marked -- same inertia/silhouette-style layout as
      wisek_window.py's plot_inertia_silhouette.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
WINDOW_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "aris"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONFIG -- the two windowed clustering runs being compared
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as wisek.py / wisek_window.py
CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"  # forward-filled per EV, same as wisek_window.py

RUN_A = {"selection_mode": "faulty", "k": 6}  # SELECTION_MODE, K -- same semantics as wisek_window.py
RUN_B = {"selection_mode": "all", "k": 3}

K_SWEEP_RADIUS = 5  # figure sweeps RUN_A's k over [RUN_A["k"] - this, RUN_A["k"] + this] (clipped to >=2),
                     # RUN_B held fixed at its configured k -- see plot_ari_vs_k_sweep

RANDOM_STATE = 0
N_INIT = 10  # K-means random restarts, explicit (not "auto") for reproducibility/reporting

FIGURE_DPI = 300
COLOR_ARI = RWTHColorScales.blue[0]  # same convention as wisek_window.py's silhouette line

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with wisek_window.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def select_population(features_all: pd.DataFrame, fault_label_all: pd.Series, mode: str) -> pd.Index:
    """Same semantics as wisek.py's/wisek_window.py's select_population,
    minus the "class" mode (not needed here -- add back if a future
    comparison needs it)."""
    if mode == "all":
        return features_all.index
    if mode == "faulty":
        return features_all.index[fault_label_all]
    raise ValueError(f"unknown selection_mode {mode!r} -- expected 'all' or 'faulty'")


def load_windows(feature_columns: list[str]) -> pd.DataFrame:
    """Load + clean the windowed features table, same steps as
    wisek_window.py step 2 (forward-fill CAPACITY_COLUMN per EV, drop rows
    with any remaining NaN feature value)."""
    window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
    missing_window_columns = set(feature_columns) - set(window_features_all.columns)
    if missing_window_columns:
        raise RuntimeError(
            f"filtered_features_windows.pkl is missing feature column(s) {sorted(missing_window_columns)} "
            "relative to the current filtered_features.pkl -- re-run sliding_window_fleet_features.py first."
        )

    window_features_all = window_features_all.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
    window_features_all[CAPACITY_COLUMN] = window_features_all.groupby("ev_id")[CAPACITY_COLUMN].ffill()
    window_valid_mask = window_features_all[feature_columns].notna().all(axis=1)
    n_dropped = int((~window_valid_mask).sum())
    window_features_all = window_features_all.loc[window_valid_mask].reset_index(drop=True)
    print(f"windowed track: dropped {n_dropped} window(s) with >=1 NaN feature value, "
          f"{len(window_features_all)} windows remain")
    return window_features_all


def run_windowed_kmeans(
    window_features_all: pd.DataFrame, feature_columns: list[str],
    selected_ev_index: pd.Index, k: int, run_name: str, verbose: bool = True,
) -> pd.DataFrame:
    """Restrict to the selected EVs' windows and K-means-cluster them (own
    StandardScaler fit fresh on this run's windows -- same convention as
    wisek_window.py, never shared across runs). Returns a DataFrame indexed
    by (ev_id, window_start) with a "cluster" column."""
    windows = window_features_all.loc[window_features_all["ev_id"].isin(selected_ev_index)].reset_index(drop=True)

    scaler = StandardScaler()
    windows_scaled = scaler.fit_transform(windows[feature_columns])

    kmeans_model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    windows["cluster"] = kmeans_model.fit_predict(windows_scaled)

    if verbose:
        cluster_sizes = windows["cluster"].value_counts().sort_index()
        print(f"\n--- {run_name}: cluster sizes (k={k}, {len(windows)} windows from "
              f"{windows['ev_id'].nunique()} EVs) ---")
        print(cluster_sizes.rename("n_windows").to_string())

    return windows.set_index(["ev_id", "window_start"])[["cluster"]].rename(columns={"cluster": run_name})


def sweep_ari_vs_run_a_k(
    window_features_all: pd.DataFrame, feature_columns: list[str],
    ev_index_a: pd.Index, clusters_b: pd.DataFrame, run_b_name: str,
) -> pd.DataFrame:
    """Re-run RUN_A's windowed K-means over k in [RUN_A["k"] - K_SWEEP_RADIUS,
    RUN_A["k"] + K_SWEEP_RADIUS] (clipped to >=2 and < RUN_A's window count),
    RUN_B's clustering held fixed (passed in already-fit), and compute ARI
    against RUN_B at each k. Returns a DataFrame with columns "k", "ari"."""
    n_windows_a = int(window_features_all["ev_id"].isin(ev_index_a).sum())
    k_values = [
        k for k in range(RUN_A["k"] - K_SWEEP_RADIUS, RUN_A["k"] + K_SWEEP_RADIUS + 1)
        if 2 <= k < n_windows_a
    ]

    rows = []
    for k in k_values:
        clusters_a_k = run_windowed_kmeans(
            window_features_all, feature_columns, ev_index_a, k, f"cluster_a_k{k}", verbose=False,
        )
        common_k = clusters_a_k.join(clusters_b, how="inner")
        ari_k = adjusted_rand_score(common_k[f"cluster_a_k{k}"], common_k[run_b_name])
        rows.append({"k": k, "ari": ari_k})

    sweep = pd.DataFrame(rows)
    print(f"\n--- ARI vs. RUN_A k sweep (k={k_values[0]}..{k_values[-1]}, "
          f"RUN_B fixed at selection_mode={RUN_B['selection_mode']!r}, k={RUN_B['k']}) ---")
    print(sweep.to_string(index=False))
    return sweep


def plot_ari_vs_k_sweep(sweep: pd.DataFrame, chosen_k: int, title: str, out_path: Path) -> None:
    """ARI vs. RUN_A's k, chosen_k marked with a vertical line -- same
    dual-purpose line-plot layout as wisek_window.py's
    plot_inertia_silhouette (single metric here, so a single y-axis)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sweep["k"], sweep["ari"], marker="o", color=COLOR_ARI, label="ARI")
    ax.set_xlabel("RUN_A k")
    ax.set_ylabel("Adjusted Rand Index (vs. RUN_B)")
    ax.set_xticks(sweep["k"])
    ax.axvline(chosen_k, color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"chosen k={chosen_k}")

    ax.legend(loc="best")
    ax.set_title(title)
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load whole-session features, select each run's EV population
features_all = pd.read_pickle(FEATURES_IN)
if features_all.isna().any().any():
    n_before = len(features_all)
    features_all = features_all.dropna()
    print(f"dropped {n_before - len(features_all)} EVs with missing whole-session feature values")

fault_label_all = pd.Series(features_all.index >= FAULT_EV_ID_CUTOFF, index=features_all.index, name="fault_label")
feature_columns = features_all.columns.tolist()

ev_index_a = select_population(features_all, fault_label_all, RUN_A["selection_mode"])
ev_index_b = select_population(features_all, fault_label_all, RUN_B["selection_mode"])
print(f"Run A (selection_mode={RUN_A['selection_mode']!r}, k={RUN_A['k']}): {len(ev_index_a)} EVs selected")
print(f"Run B (selection_mode={RUN_B['selection_mode']!r}, k={RUN_B['k']}): {len(ev_index_b)} EVs selected")

# %% 2. load windowed features (shared prep for both runs)
window_features_all = load_windows(feature_columns)

# %% 3. run each windowed K-means independently
run_a_name = f"cluster_{RUN_A['selection_mode']}_k{RUN_A['k']}"
run_b_name = f"cluster_{RUN_B['selection_mode']}_k{RUN_B['k']}"

clusters_a = run_windowed_kmeans(window_features_all, feature_columns, ev_index_a, RUN_A["k"], run_a_name)
clusters_b = run_windowed_kmeans(window_features_all, feature_columns, ev_index_b, RUN_B["k"], run_b_name)

# %% 4. ARI on the windows common to both runs' populations -- identified by
# (ev_id, window_start); if one population is a subset of the other (e.g.
# "faulty" vs "all"), this is exactly the smaller population's windows.
common = clusters_a.join(clusters_b, how="inner")
n_a, n_b, n_common = len(clusters_a), len(clusters_b), len(common)
print(f"\nRun A windows: {n_a}, Run B windows: {n_b}, common windows: {n_common}")
if n_common == 0:
    raise RuntimeError("no windows are common to both runs' populations -- ARI is undefined")
if n_common < min(n_a, n_b):
    print(f"WARNING: common windows ({n_common}) < smaller run's population ({min(n_a, n_b)}) -- "
          "populations aren't nested as expected, ARI is still computed on the overlap only")

ari = adjusted_rand_score(common[run_a_name], common[run_b_name])
print(f"\nAdjusted Rand Index ({run_a_name} vs. {run_b_name}, n={n_common} common windows): {ari:.4f}")

# %% 5. figure -- ARI vs. RUN_A's k, swept +/-K_SWEEP_RADIUS around its
# configured k, RUN_B held fixed at its own configured k (clusters_b reused
# as-is from step 3, never refit here)
sweep = sweep_ari_vs_run_a_k(window_features_all, feature_columns, ev_index_a, clusters_b, run_b_name)

fig_path = FIGURES_DIR / f"ari_vs_k_{run_a_name}_vs_{run_b_name}.png"
plot_ari_vs_k_sweep(
    sweep, chosen_k=RUN_A["k"],
    title=f"ARI vs. RUN_A k (selection_mode={RUN_A['selection_mode']!r}) -- "
          f"RUN_B fixed at selection_mode={RUN_B['selection_mode']!r}, k={RUN_B['k']}",
    out_path=fig_path,
)
print(f"saved {fig_path}")
