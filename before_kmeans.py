# %%
"""
Multivariate exploration of the faulty-vs-healthy feature space, as
preparation for K-means clustering of fault subtypes.

This does NOT run K-means yet. It answers the question that has to come
first: does the faulty group show any visually apparent sub-structure at
all, and how much of the feature space's variance can 2D projections
realistically capture? PCA and UMAP are two different lenses on the same
question -- PCA is linear and its axes are directly interpretable via
loadings, UMAP is non-linear and better at revealing local cluster
structure but its axes have no direct feature interpretation.

Input: extracted_data/fault_detection/filtered_features.pkl -- one row per
EV (index = ev_id), one column per scalar feature. This is already the
final, fault-relevant feature set selected upstream (see features.py, which
filters on the relevance criteria computed in fault_feature.py) -- no
further feature selection happens here. Note this set may still contain
mutually redundant features -- see spearman_corr.py, which reports but does
not remove them.
Ground truth fault label: EV number >= 335 is faulty, <= 334 is healthy --
same convention used throughout 01_fault_detection/ (see detection.py).

Output:
  extracted_data/vork.csv -- ev_id, pca_1, pca_2, pca_3, umap_1, umap_2,
      umap_3, fault_label for every EV, to be reused as-is by the K-means
      script.
  experiments/01_fault_detection/figures/pca_scatter_by_fault.png
  experiments/01_fault_detection/figures/pca_scatter_faulty_only.png
  experiments/01_fault_detection/figures/pca_scatter_faulty_only_3d.png       -- static (PNG), faulty EVs only
  experiments/01_fault_detection/figures/pca_scatter_faulty_only_3d.html      -- rotatable (Plotly), ALL EVs:
      healthy in a flat reference color, faulty colored by a self-contained
      k=WHOLE_SESSION_CLUSTER_K clustering (see that constant's comment --
      NOT this script's actual K-means step, which deliberately doesn't
      exist yet) -- unlike the static PNG above, which stays faulty-only
      and single-colored.
  experiments/01_fault_detection/figures/pca_loadings.png
  experiments/01_fault_detection/figures/umap_scatter_by_fault.png
  experiments/01_fault_detection/figures/umap_scatter_faulty_only.png
  experiments/01_fault_detection/figures/umap_scatter_faulty_only_3d.png      -- static (PNG), faulty EVs only
  experiments/01_fault_detection/figures/umap_scatter_faulty_only_3d.html     -- rotatable (Plotly), same
      healthy-reference / faulty-by-cluster coloring as the PCA HTML above
  experiments/01_fault_detection/figures/whole_session_faulty_k<K>_top_features.png -- the 6 whole-session
      features that vary most (one-way ANOVA F) across the faulty-only
      k=WHOLE_SESSION_CLUSTER_K clustering, each a violin per cluster plus
      faulty-/healthy-population references (same idea as clust.py/
      clust2.py's own top-features figures, at whole-session granularity).
  experiments/01_fault_detection/figures/whole_session_all_k<K>_3d_pca_umap.html -- rotatable (Plotly),
      a SEPARATE population=all (every EV, no fault-label split)
      whole-session k=WHOLE_SESSION_CLUSTER_K clustering, colored purely by
      cluster (mirrors clust.py/clust2.py's population="all" 3D HTML).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the "3d" projection used below)
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import f_oneway

# umap-learn is not a core dependency of this repo yet -- if this import
# fails, run: pip install umap-learn
import umap

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../before_kmeans.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
OUT_CSV = PROJECT_ROOT / "extracted_data" / "vork.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change/tune later
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as detection.py / fault_feature.py

RANDOM_STATE = 0  # shared seed for PCA and UMAP, for reproducible embeddings across reruns

TOP_N_LOADINGS = 8  # top contributing original features shown per PCA component

# UMAP hyperparameters -- exposed here so they're easy to sweep later without
# digging through the function body.
UMAP_N_NEIGHBORS = 15  # local neighborhood size: higher = more global structure, lower = more local detail
UMAP_MIN_DIST = 0.1    # minimum distance between points in the embedding: higher = more spread out/uniform

FIGURE_DPI = 300  # export resolution for every figure -- 300 stays legible when zoomed in

COLOR_HEALTHY = RWTHColorScales.blue[0]
COLOR_FAULTY = RWTHColorScales.red[0]
CLUSTER_COLORS = RWTHColorScales.cycle  # same qualitative palette as kmeans.py/clust.py's cluster plots

# k used ONLY to color the faulty EVs in the two interactive 3D HTML plots
# below (figures 2c-3d/3c) by cluster instead of a single flat "faulty"
# color -- a self-contained fit purely for that visualization, NOT this
# script's actual K-means step (there isn't one; see module docstring,
# this file deliberately runs BEFORE clustering). Mirrors kmeans.py's
# whole-session track (faulty EVs only, its own StandardScaler) so the
# coloring is directly comparable to kmeans_pca_scatter_by_cluster.png /
# kmeans_umap_scatter_by_cluster.png once that script's FINAL_K is set.
WHOLE_SESSION_CLUSTER_K = 4

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _to_rgb_string(color: tuple) -> str:
    """RWTHColorScales colors are (r, g, b) float tuples in [0, 1]
    (matplotlib's format); Plotly instead wants an 'rgb(r, g, b)' string
    with 0-255 ints, so convert once here rather than hardcoding a second
    copy of the palette.
    """
    r, g, b = (round(c * 255) for c in color)
    return f"rgb({r}, {g}, {b})"


COLOR_HEALTHY_RGB = _to_rgb_string(COLOR_HEALTHY)  # Plotly-compatible version of COLOR_HEALTHY, for the interactive 3D plots
COLOR_FAULTY_RGB = _to_rgb_string(COLOR_FAULTY)  # Plotly-compatible version of COLOR_FAULTY, for the interactive 3D plots
CLUSTER_COLORS_RGB = [_to_rgb_string(c) for c in CLUSTER_COLORS]  # Plotly-compatible version of CLUSTER_COLORS


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with the relevance-analysis
    scatter plots (fault_feature.py's r-vs-AUC plots).
    """
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def plot_2d_by_fault(
    coords: np.ndarray, fault_label: pd.Series, xlabel: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """2D scatter of `coords` (n_samples x 2), colored by fault label, with
    a healthy/faulty legend. Shared style for both the PCA and UMAP
    "colored by fault" plots so the two are visually comparable.
    """
    faulty_mask = fault_label.to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[~faulty_mask, 0], coords[~faulty_mask, 1], c=[COLOR_HEALTHY],
               s=30, edgecolor="white", linewidth=0.4, alpha=0.85, label="healthy", zorder=2)
    ax.scatter(coords[faulty_mask, 0], coords[faulty_mask, 1], c=[COLOR_FAULTY],
               s=30, edgecolor="white", linewidth=0.4, alpha=0.85, label="faulty", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_2d_faulty_only(coords_faulty: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
    """2D scatter of only the faulty samples' coordinates, single color
    (no cluster labels exist yet -- that's what K-means is for). Purely a
    visual check for apparent sub-structure before clustering.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords_faulty[:, 0], coords_faulty[:, 1], c=[COLOR_FAULTY],
               s=35, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_3d_faulty_only(
    coords_faulty: np.ndarray, xlabel: str, ylabel: str, zlabel: str, title: str, out_path: Path,
) -> None:
    """3D scatter of only the faulty samples' first-three-PC coordinates,
    single color, same rationale as plot_2d_faulty_only -- no cluster
    labels exist yet, this is purely a sub-structure sanity check, now
    with the PC3 axis added back in.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(coords_faulty[:, 0], coords_faulty[:, 1], coords_faulty[:, 2], c=[COLOR_FAULTY],
               s=35, edgecolor="white", linewidth=0.4, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_3d_by_fault_interactive(
    coords: np.ndarray, fault_label: pd.Series, xlabel: str, ylabel: str, zlabel: str, title: str, out_path: Path,
    faulty_cluster_label: pd.Series | None = None,
) -> None:
    """Plotly HTML 3D scatter of ALL EVs (not faulty-only), that you can
    drag to rotate/zoom in a browser. The static PNG (plot_3d_faulty_only)
    is fine for the thesis document and deliberately stays faulty-only
    there (this is a pre-clustering sub-structure check, healthy EVs would
    just clutter it); this interactive version instead also shows the
    healthy population (always its own flat COLOR_HEALTHY, as a reference)
    for visual context on where the faulty EVs sit relative to it -- a
    fixed camera angle on a static plot can hide or fake sub-structure that
    only shows up from certain viewing angles, and having the healthy
    reference alongside it here makes "is this faulty-only sub-structure or
    just where healthy EVs also live" a one-drag check instead of a guess.
    Hovering a point shows its ev_id and fault status.

    faulty_cluster_label (optional): a per-EV cluster assignment (index =
    ev_id, values = cluster id) for the faulty subset ONLY, from the
    self-contained WHOLE_SESSION_CLUSTER_K fit above -- NOT this script's
    actual K-means step, which deliberately doesn't exist yet (see module
    docstring). When given, faulty points are colored per cluster (same
    CLUSTER_COLORS palette as kmeans.py) instead of a single flat "faulty"
    color; when omitted, faulty points fall back to that flat color.
    """
    faulty_mask = fault_label.to_numpy(dtype=bool)
    ev_ids = fault_label.index

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coords[~faulty_mask, 0], y=coords[~faulty_mask, 1], z=coords[~faulty_mask, 2],
        mode="markers",
        marker=dict(size=4, color=COLOR_HEALTHY_RGB, opacity=0.6, line=dict(width=0.5, color="white")),
        text=[f"ev_id {ev_id} (healthy)" for ev_id in np.asarray(ev_ids)[~faulty_mask]],
        hoverinfo="text",
        name="healthy",
    ))

    if faulty_cluster_label is not None:
        cluster_by_ev = faulty_cluster_label.reindex(ev_ids)  # NaN for healthy EVs, by construction
        for cluster in sorted(faulty_cluster_label.unique()):
            cluster_mask = (cluster_by_ev == cluster).to_numpy()
            color = CLUSTER_COLORS_RGB[int(cluster) % len(CLUSTER_COLORS_RGB)]
            fig.add_trace(go.Scatter3d(
                x=coords[cluster_mask, 0], y=coords[cluster_mask, 1], z=coords[cluster_mask, 2],
                mode="markers",
                marker=dict(size=5, color=color, opacity=0.9, line=dict(width=0.5, color="white")),
                text=[f"ev_id {ev_id} (faulty, cluster {int(cluster)})" for ev_id in np.asarray(ev_ids)[cluster_mask]],
                hoverinfo="text",
                name=f"faulty cluster {int(cluster)}",
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=coords[faulty_mask, 0], y=coords[faulty_mask, 1], z=coords[faulty_mask, 2],
            mode="markers",
            marker=dict(size=5, color=COLOR_FAULTY_RGB, opacity=0.9, line=dict(width=0.5, color="white")),
            text=[f"ev_id {ev_id} (faulty)" for ev_id in np.asarray(ev_ids)[faulty_mask]],
            hoverinfo="text",
            name="faulty",
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=xlabel, yaxis_title=ylabel, zaxis_title=zlabel),
        legend_title="group",
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_path)


def plot_3d_all_population_clusters_interactive(
    pca_coords: np.ndarray, umap_coords: np.ndarray, cluster_labels: np.ndarray, ev_ids, title: str, out_path: Path,
) -> None:
    """Single interactive Plotly HTML, two STACKED rotatable 3D scatter
    scenes (PCA PC1-3 on top, UMAP UMAP1-3 below), ONE POINT PER EV colored
    by a population=all whole-session cluster (see WHOLE_SESSION_CLUSTER_K)
    -- no fault-label split here (unlike plot_3d_by_fault_interactive
    above): every EV, healthy or faulty, is already clustered and colored
    purely by its assigned cluster, same convention as clust.py/clust2.py's
    population="all" 3D HTML (which likewise adds no separate healthy
    trace, since healthy EVs already appear via their own cluster).
    """
    fig = make_subplots(
        rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "scene"}]],
        subplot_titles=("PCA (PC1-3)", "UMAP (UMAP1-3)"),
        vertical_spacing=0.05,
    )
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        color = CLUSTER_COLORS_RGB[int(cluster) % len(CLUSTER_COLORS_RGB)]
        hover_text = [f"ev_id {ev_id}" for ev_id in np.asarray(ev_ids)[mask]]
        marker = dict(size=4, color=color, line=dict(width=0.5, color="white"))

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

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        scene2=dict(xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title="UMAP-3"),
        legend_title="cluster",
        height=1500,
        margin=dict(l=0, r=0, b=0, t=60),
    )
    fig.write_html(out_path)


def plot_pca_loadings(pca: PCA, feature_names: pd.Index, top_n: int, out_path: Path) -> None:
    """Horizontal bar plots of the top_n original features by |loading|,
    one subplot each for PC1 and PC2 -- a rough guide to what each axis
    represents. Loadings are components_[i], the coefficients of the
    original (standardized) features in that component's direction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * top_n + 2))

    for ax, component_idx in zip(axes, [0, 1]):
        loadings = pd.Series(pca.components_[component_idx], index=feature_names)
        top_loadings = loadings.reindex(loadings.abs().sort_values(ascending=False).index).head(top_n)
        top_loadings = top_loadings.iloc[::-1]  # largest at the top of the barh

        colors = [COLOR_FAULTY if v < 0 else COLOR_HEALTHY for v in top_loadings]
        ax.barh(top_loadings.index, top_loadings.values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("loading")
        ax.set_title(f"PC{component_idx + 1} top {top_n} loadings")

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load data, select features (filtered_features.pkl already IS the
# final relevant/non-redundant feature set -- see module docstring), standardize
features = pd.read_pickle(FEATURES_IN)

# filtered_features.pkl is produced clean by features.py, but guard anyway
# since it's a separate, still-evolving upstream script.
if features.isna().any().any():
    n_before = len(features)
    features = features.dropna()
    print(f"dropped {n_before - len(features)} EVs with missing feature values")

fault_label = pd.Series(features.index >= FAULT_EV_ID_CUTOFF, index=features.index, name="fault_label")
print(f"{fault_label.sum()}/{len(fault_label)} EVs labeled faulty (EV id >= {FAULT_EV_ID_CUTOFF})")
print(f"using {features.shape[1]} relevant, non-redundant features: {features.columns.tolist()}")

# %% 1a. GLOBALLY most-varying features -- fixed for this feature set,
# computed ONCE from every EV's whole-session values, independent of
# WHOLE_SESSION_CLUSTER_K or which clustering (faulty-only vs. all) is
# used below -- same method as clust2.py's own global-feature-selection
# cell: "most varying" = highest variance AFTER min-max normalizing each
# feature to [0, 1] across all EVs first, so a large-magnitude feature
# (e.g. pack_temp in degrees C) can't dominate a small-magnitude one (e.g.
# a ratio-type feature near 0) just by unit choice -- comparable on a
# common, unit-free scale instead. (Plain z-scoring was NOT used -- it
# forces every feature's variance to exactly 1 by construction, which
# would make "most varying" meaningless.)
_global_min = features.min()
_global_max = features.max()
_global_range = (_global_max - _global_min).replace(0, np.nan)
_normalized_global = (features - _global_min) / _global_range
global_variability = _normalized_global.var().dropna().sort_values(ascending=False)
GLOBAL_TOP_N = min(12, len(global_variability))
global_top_features = global_variability.head(GLOBAL_TOP_N).index.tolist()
print(f"\n--- top-{GLOBAL_TOP_N} GLOBALLY most-varying whole-session features (min-max-normalized "
      "variance, all EVs, independent of WHOLE_SESSION_CLUSTER_K) ---")
print(global_variability.loc[global_top_features].rename("normalized_variance").to_string())

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# %% 1b. k=WHOLE_SESSION_CLUSTER_K clustering of the faulty EVs, ONLY to
# color the interactive 3D HTML plots below (figures 2c-3d/3c) -- see
# WHOLE_SESSION_CLUSTER_K's own comment above for why this is a
# self-contained fit and not "the" K-means step (there isn't one in this
# script). Own StandardScaler fit on the faulty subset only, same
# convention as kmeans.py's whole-session track (its scaler never sees
# healthy EVs either).
faulty_scaler = StandardScaler()
faulty_features_scaled = faulty_scaler.fit_transform(features.loc[fault_label])
whole_session_cluster_model = KMeans(n_clusters=WHOLE_SESSION_CLUSTER_K, random_state=RANDOM_STATE, n_init=10)
faulty_cluster_label = pd.Series(
    whole_session_cluster_model.fit_predict(faulty_features_scaled),
    index=features.index[fault_label.to_numpy(dtype=bool)], name="cluster",
)
print(f"\n(for 3D HTML coloring only) k={WHOLE_SESSION_CLUSTER_K} cluster sizes among faulty EVs:")
print(faulty_cluster_label.value_counts().sort_index().rename("n_evs").to_string())

# %% 2. PCA
pca = PCA(random_state=RANDOM_STATE)  # n_components=None -> all components, so we can inspect the full spectrum
pca_all = pca.fit_transform(features_scaled)
pca_coords = pca_all[:, :3]  # first three components: PC1/PC2 for the 2D plots/CSV, PC3 added for the 3D check

# 2a. explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
print("\n--- PCA explained variance ratio ---")
for i, (ratio, cum) in enumerate(zip(pca.explained_variance_ratio_, cumulative_variance), start=1):
    print(f"PC{i}: {ratio:.1%}  (cumulative: {cum:.1%})")
print(f"first 2 components capture {cumulative_variance[1]:.1%} of total variance")
print(f"first 3 components capture {cumulative_variance[2]:.1%} of total variance")

# 2b. scatter colored by fault label
plot_2d_by_fault(
    pca_coords, fault_label,
    xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
    ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
    title="PCA of relevant features, by fault label",
    out_path=FIGURES_DIR / "pca_scatter_by_fault.png",
)

# 2c. faulty-only scatter -- visual pre-clustering sub-structure check
faulty_mask = fault_label.to_numpy(dtype=bool)
plot_2d_faulty_only(
    pca_coords[faulty_mask],
    xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
    ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
    title="PCA of faulty EVs only (no cluster labels yet)",
    out_path=FIGURES_DIR / "pca_scatter_faulty_only.png",
)

# 2c-3d. same faulty-only check, with PC3 added -- a 3D view can reveal
# sub-structure that's flattened away when only PC1/PC2 are plotted.
plot_3d_faulty_only(
    pca_coords[faulty_mask],
    xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
    ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
    zlabel=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} var.)",
    title="PCA of faulty EVs only (no cluster labels yet)",
    out_path=FIGURES_DIR / "pca_scatter_faulty_only_3d.png",
)
plot_3d_by_fault_interactive(
    pca_coords, fault_label,
    xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)",
    ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)",
    zlabel=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} var.)",
    title=f"PCA of all EVs -- healthy vs. faulty (faulty colored by k={WHOLE_SESSION_CLUSTER_K} cluster)",
    out_path=FIGURES_DIR / "pca_scatter_faulty_only_3d.html",
    faulty_cluster_label=faulty_cluster_label,
)

# 2d. PC1/PC2 loadings -- what each axis roughly represents
print("\n--- PC1 top loadings ---")
print(pd.Series(pca.components_[0], index=features.columns)
      .reindex(pd.Series(pca.components_[0], index=features.columns).abs().sort_values(ascending=False).index)
      .head(TOP_N_LOADINGS).to_string())
print("\n--- PC2 top loadings ---")
print(pd.Series(pca.components_[1], index=features.columns)
      .reindex(pd.Series(pca.components_[1], index=features.columns).abs().sort_values(ascending=False).index)
      .head(TOP_N_LOADINGS).to_string())
plot_pca_loadings(pca, features.columns, TOP_N_LOADINGS, FIGURES_DIR / "pca_loadings.png")

# %% 3. UMAP
# Fit once with 3 components (same approach as PCA above): UMAP_1/UMAP_2 for
# the 2D plots/CSV, UMAP_3 added on for the 3D check. Note this means the
# saved umap_1/umap_2 columns come from a 3-component embedding, not a
# separately-fit 2-component one -- UMAP's embedding depends on how many
# components it's asked for, so refitting with n_components=2 would give
# slightly different UMAP-1/UMAP-2 coordinates than the ones plotted here.
reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=3, random_state=RANDOM_STATE)
umap_coords = reducer.fit_transform(features_scaled)

# 3a. scatter colored by fault label, same style as the PCA plot
plot_2d_by_fault(
    umap_coords, fault_label,
    xlabel="UMAP-1", ylabel="UMAP-2",
    title=f"UMAP of relevant features, by fault label (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})",
    out_path=FIGURES_DIR / "umap_scatter_by_fault.png",
)

# 3b. faulty-only scatter
plot_2d_faulty_only(
    umap_coords[faulty_mask],
    xlabel="UMAP-1", ylabel="UMAP-2",
    title="UMAP of faulty EVs only (no cluster labels yet)",
    out_path=FIGURES_DIR / "umap_scatter_faulty_only.png",
)

# 3c. same faulty-only check in 3D, static + rotatable, mirroring the PCA 3D plots above
plot_3d_faulty_only(
    umap_coords[faulty_mask],
    xlabel="UMAP-1", ylabel="UMAP-2", zlabel="UMAP-3",
    title="UMAP of faulty EVs only (no cluster labels yet)",
    out_path=FIGURES_DIR / "umap_scatter_faulty_only_3d.png",
)
plot_3d_by_fault_interactive(
    umap_coords, fault_label,
    xlabel="UMAP-1", ylabel="UMAP-2", zlabel="UMAP-3",
    title=f"UMAP of all EVs -- healthy vs. faulty (faulty colored by k={WHOLE_SESSION_CLUSTER_K} cluster)",
    out_path=FIGURES_DIR / "umap_scatter_faulty_only_3d.html",
    faulty_cluster_label=faulty_cluster_label,
)

# %% 4. save PCA (PC1-3) + UMAP (UMAP1-3) coordinates for reuse in the K-means script
embeddings = pd.DataFrame({
    "pca_1": pca_coords[:, 0],
    "pca_2": pca_coords[:, 1],
    "pca_3": pca_coords[:, 2],
    "umap_1": umap_coords[:, 0],
    "umap_2": umap_coords[:, 1],
    "umap_3": umap_coords[:, 2],
    "fault_label": fault_label,
}, index=features.index)
embeddings.to_csv(OUT_CSV)
print(f"\nsaved PCA + UMAP embeddings for {len(embeddings)} EVs to {OUT_CSV}")

# ===========================================================================
# %% WHOLE-SESSION CLUSTER VISUALIZATION -- additive, purely visual, using
# the self-contained WHOLE_SESSION_CLUSTER_K clustering(s) from step 1b
# (and a NEW population=all one below) -- NOT this script's actual
# K-means step (there isn't one; see module docstring). Same idea as
# clust.py/clust2.py's own top-features and 3D-cluster figures, but at
# WHOLE-SESSION (one row per EV) granularity instead of windowed.
# ===========================================================================

# %% 5. "top features" violin plot for the faulty-only k=WHOLE_SESSION_CLUSTER_K
# clustering from step 1b -- one subplot per feature, the 6 features that
# vary the MOST across clusters (highest one-way ANOVA F-statistic among
# the cluster groups -- same method as clust.py/clust2.py's own top-features
# figure: F = variance BETWEEN cluster means / variance WITHIN clusters, so
# a high F means that feature actually separates the clusters rather than
# just looking different by eye/scale), each drawn as a violin (full
# per-EV distribution) per cluster plus the whole faulty-population and
# healthy-population reference distributions.
faulty_features = features.loc[fault_label.to_numpy(dtype=bool)]
healthy_features = features.loc[~fault_label.to_numpy(dtype=bool)]

whole_session_clusters = sorted(faulty_cluster_label.unique())
whole_session_cluster_groups = {
    c: faulty_features.loc[faulty_cluster_label.index[faulty_cluster_label == c]] for c in whole_session_clusters
}

whole_session_f_scores = {}
for feature in features.columns:
    groups = [whole_session_cluster_groups[c][feature].to_numpy() for c in whole_session_clusters]
    f_stat, _p_value = f_oneway(*groups)
    whole_session_f_scores[feature] = f_stat
whole_session_top6_features = (
    pd.Series(whole_session_f_scores).dropna().sort_values(ascending=False).head(6).index.tolist()
)
if not whole_session_top6_features:
    raise RuntimeError(
        "every feature's ANOVA F-statistic came out NaN (likely all clusters are constant/singleton "
        "for every feature) -- can't pick top-varying features for the whole-session clustering"
    )
print(f"\n--- whole-session top-6 most cluster-varying features (one-way ANOVA F-statistic, "
      f"k={WHOLE_SESSION_CLUSTER_K}, faulty EVs only) ---")
print(pd.Series(whole_session_f_scores).loc[whole_session_top6_features].rename("F_statistic").to_string())

whole_session_group_labels = [f"cl. {c}" for c in whole_session_clusters] + ["faulty\npop.", "healthy\npop."]
whole_session_group_colors = {
    **{f"cl. {c}": CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in whole_session_clusters},
    "faulty\npop.": COLOR_FAULTY, "healthy\npop.": COLOR_HEALTHY,
}

ncols = min(3, len(whole_session_top6_features))
nrows = int(np.ceil(len(whole_session_top6_features) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
axes_flat = axes.flatten()

for ax, feature in zip(axes_flat, whole_session_top6_features):
    long_rows = []
    for c in whole_session_clusters:
        long_rows += [{"group": f"cl. {c}", "value": v} for v in whole_session_cluster_groups[c][feature]]
    long_rows += [{"group": "faulty\npop.", "value": v} for v in faulty_features[feature]]
    long_rows += [{"group": "healthy\npop.", "value": v} for v in healthy_features[feature]]
    long_df = pd.DataFrame(long_rows)

    sns.violinplot(
        data=long_df, x="group", y="value", hue="group", order=whole_session_group_labels,
        hue_order=whole_session_group_labels, palette=whole_session_group_colors,
        cut=0, linewidth=1.2, linecolor="black", legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(feature, fontsize=8)
    ax.set_title(f"{feature}  (F={whole_session_f_scores[feature]:.2f})", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _style_scatter_axes(ax)

for ax in axes_flat[len(whole_session_top6_features):]:
    ax.axis("off")

fig.suptitle(f"Top-6 most cluster-varying whole-session features (one-way ANOVA F) -- cluster vs. "
             f"faulty/healthy population (k={WHOLE_SESSION_CLUSTER_K}, faulty EVs only)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"whole_session_faulty_k{WHOLE_SESSION_CLUSTER_K}_top_features.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"saved whole-session top-features plot to "
      f"{FIGURES_DIR / f'whole_session_faulty_k{WHOLE_SESSION_CLUSTER_K}_top_features.png'}")

# %% 5b. the GLOBAL_TOP_N (up to 12) GLOBALLY most-varying whole-session
# features computed in step 1a (fixed for this feature set, independent of
# WHOLE_SESSION_CLUSTER_K -- see that cell's comment), same violin
# treatment as the figure above (per-cluster + faulty-/healthy-population
# distributions), laid out 4 columns x 3 rows -- same idea as clust2.py's
# own clust2_<tag>_global_top_features.png. Only the FEATURE SELECTION is
# global/fixed; the violins themselves still reflect THIS clustering
# (faulty-only, k=WHOLE_SESSION_CLUSTER_K).
ncols4, nrows4 = 4, 3
fig, axes = plt.subplots(nrows4, ncols4, figsize=(5.5 * ncols4, 4.5 * nrows4), squeeze=False)
axes_flat = axes.flatten()

for ax, feature in zip(axes_flat, global_top_features):
    long_rows = []
    for c in whole_session_clusters:
        long_rows += [{"group": f"cl. {c}", "value": v} for v in whole_session_cluster_groups[c][feature]]
    long_rows += [{"group": "faulty\npop.", "value": v} for v in faulty_features[feature]]
    long_rows += [{"group": "healthy\npop.", "value": v} for v in healthy_features[feature]]
    long_df = pd.DataFrame(long_rows)

    sns.violinplot(
        data=long_df, x="group", y="value", hue="group", order=whole_session_group_labels,
        hue_order=whole_session_group_labels, palette=whole_session_group_colors,
        cut=0, linewidth=1.2, linecolor="black", legend=False, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(feature, fontsize=8)
    ax.set_title(f"{feature}  (norm.var={global_variability[feature]:.3f})", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _style_scatter_axes(ax)

for ax in axes_flat[len(global_top_features):]:
    ax.axis("off")

fig.suptitle(f"Top-{GLOBAL_TOP_N} GLOBALLY most-varying whole-session features (fixed, independent of "
             f"WHOLE_SESSION_CLUSTER_K) -- cluster vs. faulty/healthy population (k={WHOLE_SESSION_CLUSTER_K}, "
             "faulty EVs only)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"whole_session_faulty_k{WHOLE_SESSION_CLUSTER_K}_global_top_features.png",
            dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)
print(f"saved whole-session global top-{GLOBAL_TOP_N}-features plot to "
      f"{FIGURES_DIR / f'whole_session_faulty_k{WHOLE_SESSION_CLUSTER_K}_global_top_features.png'}")

# %% 6. population=all whole-session clustering (k=WHOLE_SESSION_CLUSTER_K) +
# its interactive 3D HTML -- a SECOND, separate self-contained fit (own
# clustering, but the SAME features_scaled/pca_coords/umap_coords already
# computed above for ALL EVs -- that fit was never faulty-only, so no
# refitting is needed here, just a fresh k-means on top of it). No
# fault-label split in the resulting plot -- every EV appears purely by
# its own assigned cluster, mirroring clust.py/clust2.py's population="all"
# convention.
all_population_cluster_model = KMeans(n_clusters=WHOLE_SESSION_CLUSTER_K, random_state=RANDOM_STATE, n_init=10)
all_population_cluster_label = pd.Series(
    all_population_cluster_model.fit_predict(features_scaled), index=features.index, name="cluster",
)
print(f"\n(for the population=all 3D HTML only) k={WHOLE_SESSION_CLUSTER_K} whole-session cluster sizes "
      "among ALL EVs:")
print(all_population_cluster_label.value_counts().sort_index().rename("n_evs").to_string())

plot_3d_all_population_clusters_interactive(
    pca_coords, umap_coords, all_population_cluster_label.to_numpy(), features.index,
    title=f"All EVs (population=all, no fault-label split) in 3D PCA/UMAP embedding, "
          f"colored by whole-session k={WHOLE_SESSION_CLUSTER_K} cluster",
    out_path=FIGURES_DIR / f"whole_session_all_k{WHOLE_SESSION_CLUSTER_K}_3d_pca_umap.html",
)
print(f"saved whole-session population=all 3D HTML to "
      f"{FIGURES_DIR / f'whole_session_all_k{WHOLE_SESSION_CLUSTER_K}_3d_pca_umap.html'}")
