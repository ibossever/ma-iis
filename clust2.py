# %%
"""
Same visualizations as clust.py, but treating every (ev_id, window) row as
its OWN point to cluster -- the WINDOW-SCALE approach (one point per
window, like kmeans.py's WINDOW-SCALE CLUSTERING section), NOT the
EV-lifetime-vector approach clust.py uses (one point per EV, built by
concatenating a fixed-length sequence of that EV's windows). Copied from
clust.py as a starting point; CLUSTERING_OPTION and N are gone entirely --
there's no "how many windows per EV, padded or dropped" question when each
window clusters independently, so every valid window from every eligible
EV is used, unrestricted.

--- What changes when clustering is per-window instead of per-EV ---------
K-Means here operates directly on window-level feature vectors -- no
per-EV concatenation, no padding, no truncation. Consequently figures
2/2b's embedding is ALSO computed at window granularity, fresh, directly
from the same standardized window-level matrix that was clustered (PCA and
UMAP, same UMAP hyperparameters as before_kmeans.py) -- NOT reused from
extracted_data/vork.csv, which is a per-EV embedding (one point per EV,
computed once from each EV's whole-lifetime features) and therefore cannot
place individual windows at all. Every point drawn in figures 2/2b here is
a genuine (ev_id, window) row, colored by that window's own cluster --
never an EV-level summary. When POPULATION="faulty", healthy EVs' windows
are additionally projected through the SAME fitted PCA/UMAP (fit only on
the faulty windows that were actually clustered) via .transform(), and
drawn as a separate flat-colored reference trace, so they sit in the same
coordinate system for visual context without having influenced the fit.

--- WINDOW_SOURCE -----------------------------------------------------------
"120h": filtered_features_windows.pkl (sliding_window_fleet_features.py),
    all 15 filtered_features.pkl columns, each evaluated over a full 120h
    window (see that script's module docstring).
"24h": filtered_features_windows_24h.pkl
    (sliding_window_fleet_features_24h.py), the 7 columns that are valid to
    evaluate on a 24h span (FFT-based and the capacity ratio feature are
    not computed at all in that source -- see its own module docstring).
"whole_session": NO windowing at all -- filtered_features.pkl (features.py)
    loaded directly, one row per EV, the same 15 columns as "120h" but each
    a single whole-session value. Each EV's whole-session feature vector is
    already exactly one row, so it's used as its own point with no sliding
    window, no per-EV concatenation, and no N/CLUSTERING_OPTION choice
    needed (there's nothing to pad/truncate/drop). Every "window"-level
    concept below (one point per window, ANOVA "by window", etc.) reduces
    to one point per EV; the same plotting code runs unchanged.

Input:
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      column list only for WINDOW_SOURCE="120h"; loaded IN FULL (one row
      per EV, whole-session feature values) for WINDOW_SOURCE=
      "whole_session".
  extracted_data/early_warning/filtered_features_windows.pkl or
      filtered_features_windows_24h.pkl, per WINDOW_SOURCE above (not read
      at all for WINDOW_SOURCE="whole_session").
Output (under figures/clust2/, filenames tagged with the chosen config --
SEPARATE folder from clust.py's figures/clusters/ so the two tracks' output
is never confused):
  clust2_<tag>_inertia_silhouette.png -- inertia + silhouette vs. k, one
      plot (dual y-axis), chosen FINAL_K marked.
  clust2_<tag>_pca_umap.png -- PCA and UMAP scatter, one point per WINDOW,
      colored by that window's own cluster (embedding fit fresh on the
      window-level matrix -- see above, NOT vork.csv).
  clust2_<tag>_3d_pca_umap.html -- same idea, interactive Plotly, two
      stacked rotatable 3D scenes, one point per window colored by cluster.
  clust2_<tag>_top_features.png -- one subplot per feature, the 6 features
      that vary the most across THIS run's clusters (highest one-way ANOVA
      F-statistic among the cluster groups, computed directly on WINDOW-
      level values), each a violin plot per cluster plus the full fleet's
      faulty-window-population and healthy-window-population distributions.
  clust2_<tag>_global_top_features.png -- same violin treatment, but for a
      FIXED set of up to 12 features (4 cols x 3 rows) selected once,
      independent of POPULATION/FINAL_K: highest variance after min-max
      normalizing each feature to [0, 1] across every window of this
      WINDOW_SOURCE (see cell 1b's comment for why min-max, not z-scoring).
      This feature list never changes for a given WINDOW_SOURCE, no matter
      which POPULATION/FINAL_K you rerun with -- only the violins' cluster
      data does.
where <tag> = f"{WINDOW_SOURCE}_{POPULATION}_k{FINAL_K}".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
from sklearn.decomposition import PCA

# umap-learn is not a core dependency of this repo yet -- if this import
# fails, run: pip install umap-learn (same note as before_kmeans.py)
import umap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
FEATURES_IN = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"  # from features.py -- column list only for "120h"/"24h", full whole-session data for WINDOW_SOURCE="whole_session"
WINDOW_FEATURES_120H_IN = EXTRACTED_DATA / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
WINDOW_FEATURES_24H_IN = EXTRACTED_DATA / "early_warning" / "filtered_features_windows_24h.pkl"  # from sliding_window_fleet_features_24h.py
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "clust2"  # SEPARATE from clust.py's figures/clusters/
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONFIG -- pick exactly one configuration to visualize, then rerun
# ===========================================================================
WINDOW_SOURCE = "whole_session"      # "120h" (15 features), "24h" (7 features), or "whole_session" (15 features, no windowing -- see module docstring)
POPULATION = "faulty"       # "faulty" (confirmed-faulty EVs' windows only) or "all" (every EV's windows, unsupervised)
FINAL_K = 2                 # number of clusters to fit and visualize

FAULT_EV_ID_CUTOFF = 335    # EV number >= this is faulty -- same ground truth as kmeans.py / detection.py
RANDOM_STATE = 0
N_INIT = 10
K_RANGE = range(2, 13)       # k values swept for the inertia/silhouette plot

# UMAP hyperparameters for figures 2/2b's fresh window-level embedding --
# same values/meaning as before_kmeans.py's own UMAP_N_NEIGHBORS/UMAP_MIN_DIST
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

FIGURE_DPI = 300
COLOR_FAULTY = RWTHColorScales.red[0]
COLOR_HEALTHY = RWTHColorScales.blue[0]
CLUSTER_COLORS = RWTHColorScales.cycle

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with kmeans.py / before_kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def _to_rgb_string(color: tuple) -> str:
    """RWTHColorScales colors are (r, g, b) float tuples in [0, 1]
    (matplotlib's format); Plotly instead wants an 'rgb(r, g, b)' string
    with 0-255 ints -- same conversion as clust.py / before_kmeans.py's
    plot_3d_faulty_only_interactive, duplicated here rather than imported.
    """
    r, g, b = (round(c * 255) for c in color)
    return f"rgb({r}, {g}, {b})"


CLUSTER_COLORS_RGB = [_to_rgb_string(c) for c in CLUSTER_COLORS]  # Plotly-compatible version of CLUSTER_COLORS
COLOR_HEALTHY_RGB = _to_rgb_string(COLOR_HEALTHY)  # Plotly-compatible version of COLOR_HEALTHY


def plot_3d_clusters_interactive(
    pca_coords: np.ndarray, umap_coords: np.ndarray, cluster_labels: np.ndarray, hover_labels, title: str, out_path: Path,
    healthy_pca_coords: np.ndarray | None = None, healthy_umap_coords: np.ndarray | None = None,
    healthy_hover_labels=None,
) -> None:
    """Single interactive Plotly HTML, two STACKED rotatable 3D scatter
    scenes (PCA PC1-3 on top, UMAP UMAP1-3 below), ONE POINT PER WINDOW
    colored by that window's own cluster. Duplicated from clust.py, adapted
    for window-level points: `hover_labels` is a precomputed string per row
    (ev_id + window info -- built by the caller, since a raw ev_id alone is
    ambiguous once an EV has many points). Traces are paired across the two
    scenes via legendgroup, so toggling a cluster in the legend hides/shows
    it in both scenes at once.

    healthy_pca_coords/healthy_umap_coords/healthy_hover_labels (all
    optional, all-or-nothing): when POPULATION="faulty" only faulty EVs'
    windows were clustered, so healthy EVs' windows would otherwise be
    entirely absent from this plot -- passing their (already-transformed,
    same fitted PCA/UMAP) coordinates here draws them too, as a separate
    "healthy (reference, not clustered)" trace in a distinct flat color.
    Left None when POPULATION="all".
    """
    fig = make_subplots(
        rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "scene"}]],
        subplot_titles=("PCA (PC1-3)", "UMAP (UMAP1-3)"),
        vertical_spacing=0.05,
    )
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        color = CLUSTER_COLORS_RGB[cluster % len(CLUSTER_COLORS_RGB)]
        hover_text = list(np.asarray(hover_labels)[mask])
        marker = dict(size=3, color=color, line=dict(width=0.3, color="white"))

        fig.add_trace(
            go.Scatter3d(
                x=pca_coords[mask, 0], y=pca_coords[mask, 1], z=pca_coords[mask, 2],
                mode="markers", marker=marker, text=hover_text, hoverinfo="text",
                name=f"cluster {cluster}", legendgroup=f"cluster {cluster}", showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=umap_coords[mask, 0], y=umap_coords[mask, 1], z=umap_coords[mask, 2],
                mode="markers", marker=marker, text=hover_text, hoverinfo="text",
                name=f"cluster {cluster}", legendgroup=f"cluster {cluster}", showlegend=False,
            ),
            row=2, col=1,
        )

    if healthy_pca_coords is not None:
        healthy_hover_text = list(healthy_hover_labels)
        healthy_marker = dict(size=2, color=COLOR_HEALTHY_RGB, opacity=0.4, line=dict(width=0.3, color="white"))

        fig.add_trace(
            go.Scatter3d(
                x=healthy_pca_coords[:, 0], y=healthy_pca_coords[:, 1], z=healthy_pca_coords[:, 2],
                mode="markers", marker=healthy_marker, text=healthy_hover_text, hoverinfo="text",
                name="healthy (reference, not clustered)", legendgroup="healthy", showlegend=True,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter3d(
                x=healthy_umap_coords[:, 0], y=healthy_umap_coords[:, 1], z=healthy_umap_coords[:, 2],
                mode="markers", marker=healthy_marker, text=healthy_hover_text, hoverinfo="text",
                name="healthy (reference, not clustered)", legendgroup="healthy", showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend_title="cluster",
        height=1500,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.write_html(out_path)


# %% 0. resolve WINDOW_SOURCE -> input file + feature columns
if WINDOW_SOURCE == "120h":
    WINDOW_FEATURES_IN = WINDOW_FEATURES_120H_IN
    feature_columns = pd.read_pickle(FEATURES_IN).columns.tolist()
elif WINDOW_SOURCE == "24h":
    WINDOW_FEATURES_IN = WINDOW_FEATURES_24H_IN
    feature_columns = [  # same KEPT_FEATURE_COLUMNS as kmeans24.py / sliding_window_fleet_features_24h.py
        "local_temp__min_val", "local_temp__avg_amp", "pack_temp__mean_pack_temp",
        "voltage_features__ev_std_std", "voltage_features__ev_std_max", "voltage_features__max_dif_max",
        "resistance__std_of_std_R",
    ]
elif WINDOW_SOURCE == "whole_session":
    WINDOW_FEATURES_IN = None  # not read at all -- FEATURES_IN is loaded in full instead, see cell 1
    feature_columns = pd.read_pickle(FEATURES_IN).columns.tolist()
else:
    raise ValueError(f"WINDOW_SOURCE must be '120h', '24h', or 'whole_session', got {WINDOW_SOURCE!r}")

if POPULATION not in ("faulty", "all"):
    raise ValueError(f"POPULATION must be 'faulty' or 'all', got {POPULATION!r}")

IS_WHOLE_SESSION = WINDOW_SOURCE == "whole_session"
POINT_UNIT = "session" if IS_WHOLE_SESSION else "window"  # what one point/row represents, for labels/prints below

CONFIG_TAG = f"{WINDOW_SOURCE}_{POPULATION}_k{FINAL_K}"
print(f"clust2.py config ({POINT_UNIT}-as-point): WINDOW_SOURCE={WINDOW_SOURCE} ({len(feature_columns)} features), "
      f"POPULATION={POPULATION}, FINAL_K={FINAL_K}")

# %% 1. load + clean windowed features -- charge_capacity forward-fill (only
# relevant when it's in feature_columns, i.e. WINDOW_SOURCE="120h"; the 24h
# source never computes that column at all), then drop any window still
# NaN in a kept column (same convention as kmeans.py's W1 / clust.py)
if IS_WHOLE_SESSION:
    # no windowing at all -- FEATURES_IN is one row per EV already (ev_id in
    # the index); promote it to a column and add placeholder
    # window_start/window_start_hours columns (always 0) purely so the rest
    # of this script, written for one-row-per-window tables, needs no
    # further special-casing downstream.
    window_features_all = pd.read_pickle(FEATURES_IN)
    window_features_all.index.name = "ev_id"
    window_features_all = window_features_all.reset_index()
    window_features_all["window_start"] = 0
    window_features_all["window_start_hours"] = 0.0
else:
    window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
    missing_cols = set(feature_columns) - set(window_features_all.columns)
    if missing_cols:
        raise RuntimeError(f"{WINDOW_FEATURES_IN} is missing feature column(s) {sorted(missing_cols)}")

CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"
window_features_all = window_features_all.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
if CAPACITY_COLUMN in feature_columns:
    window_features_all[CAPACITY_COLUMN] = window_features_all.groupby("ev_id")[CAPACITY_COLUMN].ffill()

valid_mask = window_features_all[feature_columns].notna().all(axis=1)
n_dropped = int((~valid_mask).sum())
window_features_all = window_features_all.loc[valid_mask].reset_index(drop=True)
print(f"dropped {n_dropped} {POINT_UNIT}(s) with >=1 NaN feature value, {len(window_features_all)} {POINT_UNIT}s remain")

# %% 1b. GLOBALLY most-varying features -- fixed per WINDOW_SOURCE, computed
# ONCE from the full cleaned window table (every EV, every window) BEFORE
# the POPULATION filter below, so it does NOT depend on POPULATION or
# FINAL_K -- rerunning with a different population/k never changes this
# list, only WINDOW_SOURCE can (different feature set entirely). "Most
# varying" = highest variance AFTER min-max normalizing each feature to
# [0, 1] across all windows first -- min-max normalizing first means a
# large-magnitude feature (e.g. pack_temp in degrees C, spanning ~15-30)
# can't dominate a small-magnitude one (e.g. a ratio-type feature near 0)
# just by unit choice; variance is then compared on a common, unit-free
# scale instead of raw units. (Plain z-scoring was NOT used here -- it
# forces every feature's variance to exactly 1 by construction, which would
# make "most varying" meaningless; min-max preserves each feature's actual
# relative spread while still removing the unit/scale dependence.)
_global_min = window_features_all[feature_columns].min()
_global_max = window_features_all[feature_columns].max()
_global_range = (_global_max - _global_min).replace(0, np.nan)
_normalized_global = (window_features_all[feature_columns] - _global_min) / _global_range
global_variability = _normalized_global.var().dropna().sort_values(ascending=False)
GLOBAL_TOP_N = min(12, len(global_variability))
global_top_features = global_variability.head(GLOBAL_TOP_N).index.tolist()
print(f"\n--- top-{GLOBAL_TOP_N} GLOBALLY most-varying features (min-max-normalized variance, all "
      f"{WINDOW_SOURCE} {POINT_UNIT}s, independent of POPULATION/FINAL_K) ---")
print(global_variability.loc[global_top_features].rename("normalized_variance").to_string())

# %% 2. select the chosen population's windows -- EVERY valid window is used
# directly as its own point, no per-EV concatenation/padding/truncation
population_windows = (
    window_features_all.loc[window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF].reset_index(drop=True)
    if POPULATION == "faulty" else window_features_all
)
if len(population_windows) < 2:
    raise RuntimeError(f"only {len(population_windows)} {POINT_UNIT}(s) available for POPULATION={POPULATION!r}")

X = population_windows[feature_columns].to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"clustering {len(X)} (ev_id, {POINT_UNIT}) points from {population_windows['ev_id'].nunique()} EVs, "
      f"{X.shape[1]} dimensions ({len(feature_columns)} features, one {POINT_UNIT} = one point)")

# %% 3. k=2..8 sweep (for figure 1) + committed fit at FINAL_K
usable_ks = [k for k in K_RANGE if 2 <= k < len(X)]
if FINAL_K not in usable_ks:
    raise RuntimeError(f"FINAL_K={FINAL_K} not valid for {len(X)} points (usable k: {usable_ks})")

sweep_rows = []
fitted_models: dict[int, KMeans] = {}
for k in usable_ks:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT).fit(X_scaled)
    fitted_models[k] = model
    sweep_rows.append({"k": k, "inertia": model.inertia_, "silhouette_score": silhouette_score(X_scaled, model.labels_)})
diagnostics = pd.DataFrame(sweep_rows)
print("\n--- k sweep ---")
print(diagnostics.to_string(index=False))

final_model = fitted_models[FINAL_K]
population_windows = population_windows.copy()
population_windows["cluster"] = final_model.labels_
print(f"\n--- cluster sizes (k={FINAL_K}, by {POINT_UNIT.upper()}) ---")
print(population_windows["cluster"].value_counts().sort_index().rename(f"n_{POINT_UNIT}s").to_string())

# %% 4. figure 1 -- inertia + silhouette, single plot (dual y-axis), chosen
# FINAL_K marked with a vertical line
fig, ax1 = plt.subplots(figsize=(8, 5.5))
ax1.plot(diagnostics["k"], diagnostics["inertia"], marker="o", color=COLOR_FAULTY, label="inertia")
ax1.set_xlabel("k")
ax1.set_ylabel("inertia (within-cluster sum of squares)", color=COLOR_FAULTY)
ax1.tick_params(axis="y", labelcolor=COLOR_FAULTY)
ax1.set_xticks(diagnostics["k"])
ax1.axvline(FINAL_K, color="black", linestyle="--", linewidth=1, alpha=0.6)

ax2 = ax1.twinx()
ax2.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="s", color=COLOR_HEALTHY, label="silhouette")
ax2.set_ylabel("mean silhouette score", color=COLOR_HEALTHY)
ax2.tick_params(axis="y", labelcolor=COLOR_HEALTHY)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.set_title(f"Inertia & silhouette vs. k (chosen k={FINAL_K} marked) -- {CONFIG_TAG}, {POINT_UNIT}-as-point")
_style_scatter_axes(ax1)
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust2_{CONFIG_TAG}_inertia_silhouette.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 5. fresh PCA/UMAP embedding, fit DIRECTLY on the window-level
# standardized matrix that was actually clustered (X_scaled) -- one point
# per (ev_id, window), never per-EV (see module docstring for why this
# replaces vork.csv's per-EV embedding entirely for this script).
pca = PCA(n_components=3, random_state=RANDOM_STATE)
pca_coords = pca.fit_transform(X_scaled)

reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=3, random_state=RANDOM_STATE)
umap_coords = reducer.fit_transform(X_scaled)

plot_clusters = population_windows["cluster"].to_numpy()
hover_labels = (
    [f"ev_id {ev} (whole session)" for ev in population_windows["ev_id"]]
    if IS_WHOLE_SESSION else
    [f"ev_id {ev}, t={h:.0f}h" for ev, h in zip(population_windows["ev_id"], population_windows["window_start_hours"])]
)

# when POPULATION="faulty", also project healthy EVs' windows through the
# SAME fitted scaler/PCA/UMAP (never refit) so they sit in the same
# coordinate system for the reference trace below; left None for "all"
# since healthy windows already appear there via their own cluster
healthy_pca_coords = healthy_umap_coords = healthy_hover_labels = None
if POPULATION == "faulty":
    healthy_windows = window_features_all.loc[window_features_all["ev_id"] < FAULT_EV_ID_CUTOFF].reset_index(drop=True)
    healthy_X_scaled = scaler.transform(healthy_windows[feature_columns].to_numpy())
    healthy_pca_coords = pca.transform(healthy_X_scaled)
    healthy_umap_coords = reducer.transform(healthy_X_scaled)
    healthy_hover_labels = (
        [f"ev_id {ev} (healthy, whole session)" for ev in healthy_windows["ev_id"]]
        if IS_WHOLE_SESSION else
        [
            f"ev_id {ev}, t={h:.0f}h (healthy)"
            for ev, h in zip(healthy_windows["ev_id"], healthy_windows["window_start_hours"])
        ]
    )

# %% 5a. figure 2 -- static PCA/UMAP scatter, one point per WINDOW, colored
# by that window's own cluster
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, coords, xlabel, ylabel, title in [
    (axes[0], pca_coords, "PC1", "PC2", "PCA"),
    (axes[1], umap_coords, "UMAP-1", "UMAP-2", "UMAP"),
]:
    for cluster in sorted(np.unique(plot_clusters)):
        mask = plot_clusters == cluster
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]], s=18, edgecolor="white", linewidth=0.3,
            alpha=0.75, label=f"cluster {cluster}", zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(loc="best", fontsize=8, title="cluster")
fig.suptitle(f"(ev_id, {POINT_UNIT}) points in a fresh {POINT_UNIT}-level PCA/UMAP embedding, colored by cluster -- {CONFIG_TAG}")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust2_{CONFIG_TAG}_pca_umap.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 5b. figure 2's interactive 3D companion -- same points/clusters as
# above (PC3/UMAP-3 added back in). When POPULATION="faulty" the clustering
# itself never saw healthy EVs' windows, so they're added as a separate
# reference trace for context; when POPULATION="all" they already appear
# via their own cluster, so nothing extra is added.
healthy_3d_kwargs = {}
if POPULATION == "faulty":
    healthy_3d_kwargs = dict(
        healthy_pca_coords=healthy_pca_coords,
        healthy_umap_coords=healthy_umap_coords,
        healthy_hover_labels=healthy_hover_labels,
    )

plot_3d_clusters_interactive(
    pca_coords, umap_coords, plot_clusters, hover_labels,
    title=f"(ev_id, {POINT_UNIT}) points in a fresh {POINT_UNIT}-level 3D PCA/UMAP embedding, colored by cluster -- {CONFIG_TAG}",
    out_path=FIGURES_DIR / f"clust2_{CONFIG_TAG}_3d_pca_umap.html",
    **healthy_3d_kwargs,
)

# %% 6. figure 3 -- one subplot PER FEATURE, the 6 features that vary the
# MOST across clusters, each drawn as a violin per cluster plus the faulty-
# window-population and healthy-window-population references. Unlike
# clust.py, everything here is already at WINDOW granularity -- no per-EV
# averaging needed, since windows are exactly what was clustered.
#
# --- "varies the most across clusters": one-way ANOVA F-statistic --------
# Same method as clust.py: for each feature, scipy.stats.f_oneway compares
# the CLUSTER groups' window-level values -- F = (variance BETWEEN cluster
# means) / (variance WITHIN clusters). High F = that feature actually
# separates the clusters, not just looks different by eye/scale. Top 6 kept.
all_faulty_windows = window_features_all.loc[window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF, feature_columns]
all_healthy_windows = window_features_all.loc[window_features_all["ev_id"] < FAULT_EV_ID_CUTOFF, feature_columns]

clusters = sorted(population_windows["cluster"].unique())
cluster_groups = {c: population_windows.loc[population_windows["cluster"] == c, feature_columns] for c in clusters}

f_scores = {}
for feature in feature_columns:
    groups = [cluster_groups[c][feature].to_numpy() for c in clusters]
    f_stat, _p_value = f_oneway(*groups)
    f_scores[feature] = f_stat
top6_features = pd.Series(f_scores).dropna().sort_values(ascending=False).head(6).index.tolist()
if not top6_features:
    raise RuntimeError(
        "every feature's ANOVA F-statistic came out NaN (likely all clusters are constant/singleton "
        "for every feature) -- can't pick top-varying features for this configuration"
    )
print(f"\n--- top-6 most cluster-varying features (one-way ANOVA F-statistic, by {POINT_UNIT}) ---")
print(pd.Series(f_scores).loc[top6_features].rename("F_statistic").to_string())

group_labels = [f"cl. {c}" for c in clusters] + ["faulty\npop.", "healthy\npop."]
group_colors = {**{f"cl. {c}": CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters},
                 "faulty\npop.": COLOR_FAULTY, "healthy\npop.": COLOR_HEALTHY}

ncols = min(3, len(top6_features))
nrows = int(np.ceil(len(top6_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
axes_flat = axes.flatten()

for ax, feature in zip(axes_flat, top6_features):
    long_rows = []
    for c in clusters:
        long_rows += [{"group": f"cl. {c}", "value": v} for v in cluster_groups[c][feature]]
    long_rows += [{"group": "faulty\npop.", "value": v} for v in all_faulty_windows[feature]]
    long_rows += [{"group": "healthy\npop.", "value": v} for v in all_healthy_windows[feature]]
    long_df = pd.DataFrame(long_rows)

    sns.violinplot(
        data=long_df, x="group", y="value", hue="group", order=group_labels, hue_order=group_labels,
        palette=group_colors, cut=0, linewidth=1.2, linecolor="black", legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(feature, fontsize=8)
    ax.set_title(f"{feature}  (F={f_scores[feature]:.2f})", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _style_scatter_axes(ax)

for ax in axes_flat[len(top6_features):]:
    ax.axis("off")

fig.suptitle(f"Top-6 most cluster-varying features (one-way ANOVA F, by {POINT_UNIT}) -- cluster vs. "
             f"faulty/healthy {POINT_UNIT} population -- {CONFIG_TAG}")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust2_{CONFIG_TAG}_top_features.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 7. figure 4 -- the GLOBAL_TOP_N (up to 12) GLOBALLY most-varying
# features computed in cell 1b (fixed per WINDOW_SOURCE, independent of
# POPULATION/FINAL_K -- see that cell's comment), same violin treatment as
# figure 3 (per-cluster + faulty-/healthy-window-population distributions),
# laid out 4 columns x 3 rows. Only the FEATURE SELECTION is global/fixed;
# the violins themselves still reflect THIS run's actual clustering.
ncols4, nrows4 = 4, 3
fig, axes = plt.subplots(nrows4, ncols4, figsize=(5.5 * ncols4, 4.5 * nrows4), squeeze=False)
axes_flat = axes.flatten()

for ax, feature in zip(axes_flat, global_top_features):
    long_rows = []
    for c in clusters:
        long_rows += [{"group": f"cl. {c}", "value": v} for v in cluster_groups[c][feature]]
    long_rows += [{"group": "faulty\npop.", "value": v} for v in all_faulty_windows[feature]]
    long_rows += [{"group": "healthy\npop.", "value": v} for v in all_healthy_windows[feature]]
    long_df = pd.DataFrame(long_rows)

    sns.violinplot(
        data=long_df, x="group", y="value", hue="group", order=group_labels, hue_order=group_labels,
        palette=group_colors, cut=0, linewidth=1.2, linecolor="black", legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(feature, fontsize=8)
    ax.set_title(f"{feature}  (norm.var={global_variability[feature]:.3f})", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _style_scatter_axes(ax)

for ax in axes_flat[len(global_top_features):]:
    ax.axis("off")

fig.suptitle(f"Top-{GLOBAL_TOP_N} GLOBALLY most-varying features (fixed for this WINDOW_SOURCE, "
             f"independent of POPULATION/FINAL_K) -- cluster vs. faulty/healthy {POINT_UNIT} population -- {CONFIG_TAG}")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust2_{CONFIG_TAG}_global_top_features.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

print(f"\nsaved 4 figures + 1 interactive 3D HTML to {FIGURES_DIR} (tag: {CONFIG_TAG})")
