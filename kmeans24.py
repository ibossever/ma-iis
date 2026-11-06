# %%
"""
Windowed K-means clustering, restricted to features that are valid to
evaluate on a 24h span -- a companion to kmeans.py's WINDOW-SCALE
CLUSTERING section, not a replacement for it. Same architecture (fit once,
save fixed artifacts, never refit at scoring time), same k-selection
approach (elbow + silhouette sweep, k left unset until you've inspected the
diagnostics) -- the differences are the feature set AND the window source
(see below).

--- Why some features are excluded here -------------------------------------
filtered_features.pkl's 15 columns split into two groups by how much of a
window they actually need:
  - 8 columns are excluded from this 24h track: current_fft__*, soc_fft__*,
    local_temp__mean_diff (all FFT-based, need a long span to resolve low
    frequencies -- NOT meaningful on 24h of data) and
    charge_capacity_per_100soc__avg_ah_per_100soc (ratio-type, gated on a
    charging session spanning >= 20 SoC points inside its slice -- a 24h
    slice makes that condition rare enough that the column would be mostly
    NaN). sliding_window_fleet_features_24h.py, this script's data source,
    never computes these 8 at all (see its module docstring).
  - The remaining 7 columns (local_temp__min_val, local_temp__avg_amp,
    pack_temp__mean_pack_temp, voltage_features__ev_std_std/ev_std_max/
    max_dif_max, resistance__std_of_std_R) are all time-domain point
    statistics, genuinely meaningful on 24h -- these are what this script
    clusters on.

--- Where the 24h data comes from --------------------------------------------
This script reads extracted_data/early_warning/filtered_features_windows_24h.pkl,
produced by sliding_window_fleet_features_24h.py -- a genuinely separate,
fresh 24h/12h-stride extraction pass over raw telemetry (own window
sequence, anchored at raw index 0 same as the 120h track), NOT a
repackaging of kmeans.py's windowed input. Earlier this script read a
subset of columns out of the shared filtered_features_windows.pkl, where
the "24h" values were actually the trailing-24h slice of a 120h window
(offset ~96h into it, and missing both the first 96h of every EV's
recording and any trailing sub-120h remainder) -- that limitation is gone
now that sliding_window_fleet_features_24h.py computes a real, complete
24h/12h-stride series. See that script's module docstring for the full
reasoning on which 7 columns it computes and why the other 8 are skipped
entirely (not just computed-and-dropped).

Input:
  extracted_data/early_warning/filtered_features_windows_24h.pkl (from
      sliding_window_fleet_features_24h.py) -- one row per (ev_id, window),
      the 7 columns that are valid to evaluate on a 24h span.
Output (diagnostics -- always produced, filenames prefixed "24h_" so they
are never confused with kmeans.py's own windowed figures):
  figures/24h_kmeans_elbow.png
  figures/24h_kmeans_silhouette_scores.png
  figures/24h_kmeans_silhouette_samples_k345.png
Output (only once FINAL_K_24H is set below):
  extracted_data/early_warning/kmeans24_centroids.pkl -- {feature_columns,
      scaler (fitted StandardScaler, faulty windows only), centroids
      (DataFrame, k rows x feature_columns, standardized space),
      healthy_reference (Series, standardized space)}.
  extracted_data/early_warning/kmeans24_unrestricted_diagnostic.pkl --
      OPTIONAL: exploratory unrestricted (healthy+faulty, unfiltered)
      K-means at the same k, supplementary thesis discussion only -- NOT
      used anywhere in the early-warning system itself.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG
# ===========================================================================
FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py -- column presence check only
WINDOW_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows_24h.pkl"  # from sliding_window_fleet_features_24h.py

OUT_DIR = PROJECT_ROOT / "extracted_data" / "early_warning"
CENTROIDS_OUT = OUT_DIR / "kmeans24_centroids.pkl"
UNRESTRICTED_DIAGNOSTIC_OUT = OUT_DIR / "kmeans24_unrestricted_diagnostic.pkl"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as kmeans.py / detection.py

RANDOM_STATE = 0
N_INIT = 10

K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_KS = [3, 4, 5]

# Left as None so the script stops after diagnostics -- same convention as
# kmeans.py's FINAL_K / FINAL_K_WINDOWED.
FINAL_K_24H = 2

FIGURE_DPI = 300

COLOR_FAULTY = RWTHColorScales.red[0]
CLUSTER_COLORS = RWTHColorScales.cycle

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)

# The 7 columns filtered_features_windows_24h.pkl actually contains --
# sliding_window_fleet_features_24h.py never computes the other 8 at all
# (see module docstring), so there's no drop-set to apply here anymore.
KEPT_FEATURE_COLUMNS = [
    "local_temp__min_val", "local_temp__avg_amp", "pack_temp__mean_pack_temp",
    "voltage_features__ev_std_std", "voltage_features__ev_std_max", "voltage_features__max_dif_max",
    "resistance__std_of_std_R",
]


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with kmeans.py / before_kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def plot_elbow_24h(diagnostics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["inertia"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow plot (faulty EV windows, 24h feature set)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_vs_k_24h(diagnostics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("mean silhouette score")
    ax.set_title("K-means silhouette score vs. k (faulty EV windows, 24h feature set)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_samples_grid_24h(X: np.ndarray, fitted_models: dict, ks: list, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(ks), figsize=(6.5 * len(ks), 6), squeeze=False)
    axes = axes[0]
    for ax, k in zip(axes, ks):
        labels = fitted_models[k].labels_
        sample_silhouette_values = silhouette_samples(X, labels)
        mean_score = sample_silhouette_values.mean()
        y_lower = 10
        for cluster in range(k):
            cluster_values = np.sort(sample_silhouette_values[labels == cluster])
            y_upper = y_lower + len(cluster_values)
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, facecolor=color, edgecolor=color, alpha=0.85)
            ax.text(-0.05, y_lower + 0.5 * len(cluster_values), str(cluster))
            y_lower = y_upper + 10
        ax.axvline(mean_score, color="black", linestyle="--", linewidth=1, label=f"mean = {mean_score:.3f}")
        ax.set_title(f"k = {k}")
        ax.set_xlabel("silhouette coefficient")
        ax.set_yticks([])
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("points per cluster (sorted by silhouette value)")
    fig.suptitle("Per-window silhouette plots (faulty EV windows, 24h feature set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 0. sanity: the 7 kept columns must still exist in filtered_features.pkl
# (fails loud rather than silently drifting if features.py's column set
# changes) and must be exactly what filtered_features_windows_24h.pkl
# actually contains (fails loud if the two scripts' column lists diverge).
current_filtered_columns = set(pd.read_pickle(FEATURES_IN).columns)
missing_expected = set(KEPT_FEATURE_COLUMNS) - current_filtered_columns
if missing_expected:
    raise RuntimeError(
        f"filtered_features.pkl no longer contains {sorted(missing_expected)}, which this script expects -- "
        "features.py's column set has drifted; update KEPT_FEATURE_COLUMNS above to match."
    )
feature_columns_24h = KEPT_FEATURE_COLUMNS
print(f"24h track: using {len(feature_columns_24h)} columns: {feature_columns_24h}")

# %% 1. load windowed features, split into confirmed-faulty / healthy by the
# same ev_id >= FAULT_EV_ID_CUTOFF convention as kmeans.py, drop any window
# with a NaN in one of the 7 kept columns (same "drop rather than impute"
# convention used throughout)
window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
missing_window_columns = set(feature_columns_24h) - set(window_features_all.columns)
if missing_window_columns:
    raise RuntimeError(
        f"filtered_features_windows_24h.pkl is missing feature column(s) {sorted(missing_window_columns)} -- "
        "re-run sliding_window_fleet_features_24h.py first."
    )

window_valid_mask = window_features_all[feature_columns_24h].notna().all(axis=1)
n_windows_dropped = int((~window_valid_mask).sum())
window_features_all = window_features_all.loc[window_valid_mask].reset_index(drop=True)
print(f"\n24h track: dropped {n_windows_dropped} window(s) with >=1 NaN in the 7 kept columns, "
      f"{len(window_features_all)} windows remain")

window_fault_label = window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF
faulty_windows = window_features_all.loc[window_fault_label].reset_index(drop=True)
healthy_windows = window_features_all.loc[~window_fault_label].reset_index(drop=True)
print(f"24h track: {len(faulty_windows)} windows from {faulty_windows['ev_id'].nunique()} confirmed-faulty "
      f"EVs, {len(healthy_windows)} windows from {healthy_windows['ev_id'].nunique()} healthy EVs")

# %% 2. fit a NEW StandardScaler on the stacked faulty-window matrix
# (7-dimensional here, not 15 -- a separate scaler from both kmeans.py's
# whole-session and windowed scalers), sweep k=2..8
scaler_24h = StandardScaler()
faulty_windows_scaled = scaler_24h.fit_transform(faulty_windows[feature_columns_24h])

fitted_models_24h: dict[int, KMeans] = {}
diagnostics_24h_rows = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(faulty_windows_scaled)
    fitted_models_24h[k] = model
    diagnostics_24h_rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": silhouette_score(faulty_windows_scaled, labels),
    })

diagnostics_24h = pd.DataFrame(diagnostics_24h_rows)
print("\n--- K-means diagnostics (faulty EV windows, 24h feature set) ---")
print(diagnostics_24h.to_string(index=False))

plot_elbow_24h(diagnostics_24h, FIGURES_DIR / "24h_kmeans_elbow.png")
plot_silhouette_vs_k_24h(diagnostics_24h, FIGURES_DIR / "24h_kmeans_silhouette_scores.png")
plot_silhouette_samples_grid_24h(
    faulty_windows_scaled, fitted_models_24h, SILHOUETTE_SAMPLE_KS,
    FIGURES_DIR / "24h_kmeans_silhouette_samples_k345.png",
)

print("\n24h diagnostics done. Set FINAL_K_24H near the top of this script "
      "and rerun to fit the final clustering and save kmeans24_centroids.pkl.")

# %% 3. final clustering -- only runs once FINAL_K_24H is set
if FINAL_K_24H is not None:
    final_model_24h = fitted_models_24h.get(FINAL_K_24H) or KMeans(
        n_clusters=FINAL_K_24H, random_state=RANDOM_STATE, n_init=N_INIT,
    ).fit(faulty_windows_scaled)

    cluster_label_24h = pd.Series(final_model_24h.labels_, name="cluster")
    cluster_sizes_24h = cluster_label_24h.value_counts().sort_index()
    print(f"\n--- Cluster sizes (k={FINAL_K_24H}, 24h feature set) ---")
    print(cluster_sizes_24h.rename("n_windows").to_string())

    # cluster characterization: per-cluster mean/std of every kept feature,
    # plus the overall faulty-window-population mean -- same convention as
    # kmeans.py's whole-session and windowed characterization tables.
    characterization_24h = pd.DataFrame(index=feature_columns_24h)
    for cluster in sorted(cluster_label_24h.unique()):
        cluster_windows = faulty_windows.loc[(cluster_label_24h == cluster).to_numpy(), feature_columns_24h]
        characterization_24h[f"cluster_{cluster}_mean"] = cluster_windows.mean()
        characterization_24h[f"cluster_{cluster}_std"] = cluster_windows.std()
    characterization_24h["overall_faulty_mean"] = faulty_windows[feature_columns_24h].mean()

    print(f"\n--- Cluster characterization (k={FINAL_K_24H}, 24h feature set) ---")
    print(characterization_24h.to_string(float_format=lambda v: f"{v:.3f}"))

    windows_per_faulty_ev_24h = faulty_windows.groupby("ev_id").size().rename("n_windows")
    print(f"\n--- Windows per confirmed-faulty EV (n={len(windows_per_faulty_ev_24h)} EVs, "
          f"{windows_per_faulty_ev_24h.sum()} points total) ---")
    print(windows_per_faulty_ev_24h.to_string())

    # healthy reference: SAME scaler as the faulty fit above (not refit)
    healthy_windows_scaled = scaler_24h.transform(healthy_windows[feature_columns_24h])
    healthy_reference_24h = pd.Series(healthy_windows_scaled.mean(axis=0), index=feature_columns_24h)

    centroids_24h = pd.DataFrame(
        final_model_24h.cluster_centers_, index=range(FINAL_K_24H), columns=feature_columns_24h,
    )
    centroids_24h.index.name = "cluster"

    pd.to_pickle({
        "feature_columns": feature_columns_24h,
        "scaler": scaler_24h,
        "centroids": centroids_24h,
        "healthy_reference": healthy_reference_24h,
    }, CENTROIDS_OUT)
    print(f"\nsaved 24h centroids + healthy reference + scaler to {CENTROIDS_OUT}")

    # %% 4. [OPTIONAL / diagnostic only, NOT part of the early-warning
    # system] exploratory unrestricted K-means on ALL windows at the same k
    # -- own scaler, fit on the full (healthy+faulty) population, same
    # reasoning as kmeans.py's windowed track.
    scaler_unrestricted_24h = StandardScaler()
    all_windows_scaled_24h = scaler_unrestricted_24h.fit_transform(window_features_all[feature_columns_24h])
    model_unrestricted_24h = KMeans(n_clusters=FINAL_K_24H, random_state=RANDOM_STATE, n_init=N_INIT)
    unrestricted_labels_24h = model_unrestricted_24h.fit_predict(all_windows_scaled_24h)

    unrestricted_table_24h = window_features_all[["ev_id"]].copy()
    unrestricted_table_24h["cluster"] = unrestricted_labels_24h
    unrestricted_table_24h["is_confirmed_faulty"] = window_fault_label.to_numpy()

    composition_24h = (
        unrestricted_table_24h.groupby("cluster")["is_confirmed_faulty"]
        .agg(pct_faulty="mean", n_windows="size")
    )
    composition_24h["pct_faulty"] *= 100
    composition_24h["pct_healthy"] = 100 - composition_24h["pct_faulty"]

    print(f"\n--- Unrestricted K-means (k={FINAL_K_24H}, ALL windows, 24h feature set, diagnostic only) ---")
    print("cluster composition (% healthy vs. % confirmed-faulty, by window):")
    print(composition_24h[["n_windows", "pct_healthy", "pct_faulty"]].to_string(float_format=lambda v: f"{v:.1f}"))

    pd.to_pickle({
        "feature_columns": feature_columns_24h,
        "scaler": scaler_unrestricted_24h,
        "cluster_centers": model_unrestricted_24h.cluster_centers_,
        "labels_table": unrestricted_table_24h,
        "composition": composition_24h,
    }, UNRESTRICTED_DIAGNOSTIC_OUT)
    print(f"saved unrestricted 24h diagnostic to {UNRESTRICTED_DIAGNOSTIC_OUT}")
