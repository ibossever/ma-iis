# %%
"""
Visualize ONE specific EV-lifetime clustering configuration, windowed
features only (either the 120h or the 24h feature source -- see
WINDOW_SOURCE below). Companion to kmeans.py's EV-LIFETIME VECTOR
CLUSTERING section: same Option 1 (truncate-and-drop) / Option 2
(truncate-and-pad) mechanics, duplicated here rather than imported --
kmeans.py is a top-level script with its own side effects (figures, a full
sweep across every N/Option/population), not an importable library, same
convention as centroid.py/sliding_windows.py's own duplicated
assign_nearest_centroid.

Unlike kmeans.py's section 3 (which sweeps every N x Option x population
combination for comparison, prints only, no figures), this script commits
to and visualizes exactly ONE configuration -- WINDOW_SOURCE,
CLUSTERING_OPTION, POPULATION, N, FINAL_K below. Change any of them and
rerun to visualize a different case; nothing here is auto-selected.

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
    a single whole-session value. Each EV already has exactly one feature
    vector, so it's used as its own point directly -- CLUSTERING_OPTION and
    N are IGNORED for this source (there's nothing to concatenate, pad, or
    truncate: no per-EV sequence of windows exists to build one from).
    Figures 2/2b (PCA/UMAP, reused from vork.csv) and figure 4 (confusion
    bars) are unaffected either way, since both are already keyed by ev_id.

--- POPULATION ---------------------------------------------------------------
"faulty": confirmed-faulty EVs only (ev_id >= FAULT_EV_ID_CUTOFF).
"all": every EV.
"class": same class-imbalance selection as wisek.py -- every confirmed-
    faulty EV + every FP EV (per the cached whole-fleet IsolationForest,
    shared with wisek.py -- see select_population) + CLASS_N_RANDOM_HEALTHY
    more healthy EVs sampled at random.

Input:
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      column list for WINDOW_SOURCE="120h"; also loaded in full (all EVs'
      whole-session feature values) to drive POPULATION selection and the
      cached whole-fleet IsolationForest (see IF_CACHE_OUT below), AND, for
      WINDOW_SOURCE="whole_session", as the clustering input itself.
  extracted_data/early_warning/filtered_features_windows.pkl or
      filtered_features_windows_24h.pkl, per WINDOW_SOURCE above (not read
      at all for WINDOW_SOURCE="whole_session").
  extracted_data/vork.csv (before_kmeans.py) -- PCA/UMAP coords per ev_id,
      reused as-is for figure 2, NOT recomputed here.
  extracted_data/early_warning/wisek_isolation_forest_whole_session.pkl --
      cached whole-fleet IsolationForest results, shared with wisek.py /
      wisek_window.py (created here too if it doesn't exist yet -- same
      fit-once/cache-forever contract, see wisek.py).
Output (under figures/clusters/, filenames tagged with the chosen config):
  clust_<tag>_inertia_silhouette.png -- inertia + silhouette vs. k, one
      plot (dual y-axis), chosen FINAL_K marked.
  clust_<tag>_pca_umap.png -- PCA and UMAP scatter, colored by cluster.
  clust_<tag>_3d_pca_umap.html -- single interactive Plotly page, two
      STACKED rotatable 3D scatter scenes (PCA PC1-3 on top, UMAP UMAP1-3
      below), points colored by cluster. Same embeddings as the PNG above
      (reused from vork.csv) and the same rotatable-HTML idea as
      before_kmeans.py's pca/umap_scatter_faulty_only_3d.html, but combined
      into one file covering both embeddings, and colored by cluster
      instead of a single flat "faulty" color -- the goal being to inspect
      how the clusters actually sit relative to each other in 3D, for
      whichever configuration is currently selected above. When
      POPULATION="faulty" or "class", healthy EVs not part of that
      clustering (all of them for "faulty"; whichever ones
      select_population didn't pull in for "class") are additionally drawn
      as a separate "healthy (reference)" trace for context; when
      POPULATION="all" nothing extra is added, since every EV already
      appears there via its own assigned cluster.
  clust_<tag>_top_features.png -- one subplot per feature, the 6 features
      that vary the most across clusters (highest one-way ANOVA F-statistic
      among the cluster groups -- see figure 3's own code comment), each a
      violin plot (full per-EV distribution, not just mean/std) per cluster
      plus the full fleet's faulty-population and healthy-population
      distributions for the same feature.
  clust_<tag>_confusion_bars.png -- stacked bar chart, one bar per cluster,
      stacked by TN/TP/FP/FN count -- each EV in this clustering compared
      against the SAME cached whole-fleet IsolationForest used by wisek.py
      (extracted_data/early_warning/wisek_isolation_forest_whole_session.pkl,
      fit once on ALL EVs' whole-session features, contamination = the
      actual labeled-faulty fraction -- see wisek.py's module docstring for
      the full rationale; loaded here as-is, never refit). Same color
      convention as wisek.py's own cluster_confusion_bars figure
      (TN=green, TP=blue, FP=orange, FN=red). Purely additive -- doesn't
      change any of this script's existing clustering/other figures.
where <tag> = f"{WINDOW_SOURCE}_opt{CLUSTERING_OPTION}_{POPULATION}_N{N}_k{FINAL_K}", except for
WINDOW_SOURCE="whole_session" where <tag> = f"{WINDOW_SOURCE}_{POPULATION}_k{FINAL_K}" (opt/N dropped
since they're ignored for that source).
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
FEATURES_IN = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"  # from features.py -- column list only
WINDOW_FEATURES_120H_IN = EXTRACTED_DATA / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
WINDOW_FEATURES_24H_IN = EXTRACTED_DATA / "early_warning" / "filtered_features_windows_24h.pkl"  # from sliding_window_fleet_features_24h.py
EMBEDDINGS_IN = EXTRACTED_DATA / "vork.csv"  # PCA/UMAP coords, from before_kmeans.py
IF_CACHE_OUT = EXTRACTED_DATA / "early_warning" / "wisek_isolation_forest_whole_session.pkl"  # SAME cache as wisek.py/wisek_window.py -- shared, never refit twice

FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "clusters"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONFIG -- pick exactly one configuration to visualize, then rerun
# ===========================================================================
WINDOW_SOURCE = "120h"      # "120h" (15 features), "24h" (7 features), or "whole_session" (15 features, no windowing -- see module docstring)
CLUSTERING_OPTION = 2       # 1 = truncate-and-drop, 2 = truncate-and-pad -- see kmeans.py's EV-LIFETIME VECTOR CLUSTERING section; IGNORED when WINDOW_SOURCE="whole_session"
POPULATION = "faulty"          # "faulty" (confirmed-faulty EVs only), "all" (every EV), or "class" (same
                             # class-imbalance selection as wisek.py: every faulty EV + every FP EV +
                             # CLASS_N_RANDOM_HEALTHY random healthy EVs -- see select_population)
CLASS_N_RANDOM_HEALTHY = 10  # only used when POPULATION == "class" -- see select_population
N = 15                     # number of windows per EV; IGNORED when WINDOW_SOURCE="whole_session"
FINAL_K = 6          # number of clusters to fit and visualize

FAULT_EV_ID_CUTOFF = 335    # EV number >= this is faulty -- same ground truth as kmeans.py / detection.py
RANDOM_STATE = 0
N_INIT = 10
K_RANGE = range(2, 13)       # k values swept for the inertia/silhouette plot
N_ESTIMATORS = 100          # IsolationForest trees, only used if IF_CACHE_OUT doesn't exist yet (see wisek.py)

FIGURE_DPI = 300
COLOR_FAULTY = RWTHColorScales.red[0]
COLOR_HEALTHY = RWTHColorScales.blue[0]
CLUSTER_COLORS = RWTHColorScales.cycle
CONFUSION_COLORS = {  # same convention as wisek.py's cluster_confusion_bars figure
    "TN": RWTHColorScales.green[0],
    "TP": RWTHColorScales.blue[0],
    "FP": RWTHColorScales.orange[0],
    "FN": RWTHColorScales.red[0],
}
CONFUSION_ORDER = ["TN", "TP", "FP", "FN"]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with kmeans.py / before_kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def confusion_labels(predicted_fault: np.ndarray, actual_fault: np.ndarray) -> np.ndarray:
    """Identical to wisek.py's confusion_labels -- see there."""
    labels = np.full(len(predicted_fault), "", dtype=object)
    labels[predicted_fault & actual_fault] = "TP"
    labels[~predicted_fault & ~actual_fault] = "TN"
    labels[predicted_fault & ~actual_fault] = "FP"
    labels[~predicted_fault & actual_fault] = "FN"
    return labels


def get_or_fit_isolation_forest(features_all: pd.DataFrame, fault_label_all: pd.Series) -> pd.DataFrame:
    """Identical to wisek.py's get_or_fit_isolation_forest, pointed at the
    SAME IF_CACHE_OUT path -- shared with wisek.py/wisek_window.py, loaded
    as-is if it already exists rather than refit here."""
    if IF_CACHE_OUT.exists():
        cached = pd.read_pickle(IF_CACHE_OUT)
        print(f"loaded cached whole-fleet IsolationForest results from {IF_CACHE_OUT} "
              f"(shared with wisek.py; delete this file and rerun to recompute)")
        return cached["results"]

    contamination_ratio = float(fault_label_all.mean())
    if_model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    if_model.fit(features_all)
    anomaly_score = -if_model.score_samples(features_all)
    threshold = np.quantile(anomaly_score, 1 - contamination_ratio)
    predicted_fault = anomaly_score > threshold

    results = pd.DataFrame({
        "anomaly_score": anomaly_score,
        "predicted_fault": predicted_fault,
        "confusion": confusion_labels(predicted_fault, fault_label_all.to_numpy()),
    }, index=features_all.index)

    pd.to_pickle({
        "model": if_model,
        "feature_columns": features_all.columns.tolist(),
        "contamination_ratio": contamination_ratio,
        "anomaly_score_threshold": threshold,
        "results": results,
    }, IF_CACHE_OUT)
    print(f"fit whole-fleet IsolationForest on {len(features_all)} EVs "
          f"(contamination_ratio={contamination_ratio:.1%}, threshold={threshold:.3f}), "
          f"cached to {IF_CACHE_OUT}")
    return results


def select_population(
    features_all: pd.DataFrame, fault_label_all: pd.Series, if_results_all: pd.DataFrame, mode: str,
) -> pd.Index:
    """Identical semantics to wisek.py's select_population, duplicated here
    (not imported) -- same sibling-scripts convention as everything else
    this file duplicates from kmeans.py/wisek.py. "class" always keeps
    every confirmed-faulty EV and every FP EV (per the cached whole-fleet
    IsolationForest results), regardless of CLASS_N_RANDOM_HEALTHY, plus
    CLASS_N_RANDOM_HEALTHY more healthy EVs sampled at random (seeded by
    RANDOM_STATE) from whichever healthy EVs are NOT already FP -- see
    wisek.py's module docstring for the full rationale.
    """
    if mode == "all":
        return features_all.index
    if mode == "faulty":
        return features_all.index[fault_label_all]
    if mode == "class":
        faulty_ids = set(features_all.index[fault_label_all])
        fp_ids = set(if_results_all.index[if_results_all["confusion"] == "FP"])

        healthy_pool = [
            ev_id for ev_id in features_all.index
            if not fault_label_all.loc[ev_id] and ev_id not in fp_ids
        ]
        rng = np.random.RandomState(RANDOM_STATE)
        n_random = min(CLASS_N_RANDOM_HEALTHY, len(healthy_pool))
        random_healthy_ids = (
            {int(ev_id) for ev_id in rng.choice(healthy_pool, size=n_random, replace=False)} if n_random > 0 else set()
        )

        print(f"class selection: {len(faulty_ids)} faulty + {len(fp_ids)} FP healthy "
              f"+ {len(random_healthy_ids)} randomly sampled healthy EVs "
              f"(random_state={RANDOM_STATE}): {sorted(random_healthy_ids)}")

        return pd.Index(sorted(faulty_ids | fp_ids | random_healthy_ids))
    raise ValueError(f"unknown POPULATION {mode!r} -- expected 'all', 'faulty', or 'class'")


def plot_cluster_confusion_bars(cluster_confusion: pd.DataFrame, title: str, out_path: Path) -> None:
    """Stacked bar chart: one bar per cluster, height = cluster size (# EVs),
    stacked by TN/TP/FP/FN count -- same style as wisek.py's own
    cluster_confusion_bars figure."""
    counts = cluster_confusion.groupby(["cluster", "confusion"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CONFUSION_ORDER, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(counts) + 2.5), 6))
    bottom = np.zeros(len(counts))
    for label in CONFUSION_ORDER:
        values = counts[label].to_numpy()
        ax.bar(counts.index.astype(str), values, bottom=bottom, color=CONFUSION_COLORS[label],
               edgecolor="white", linewidth=0.6, label=label, zorder=2)
        bottom += values

    ax.set_xlabel("cluster")
    ax.set_ylabel("# EVs")
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(title="IsolationForest vs. ground truth", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _to_rgb_string(color: tuple) -> str:
    """RWTHColorScales colors are (r, g, b) float tuples in [0, 1]
    (matplotlib's format); Plotly instead wants an 'rgb(r, g, b)' string
    with 0-255 ints -- same conversion as before_kmeans.py's
    plot_3d_faulty_only_interactive, duplicated here rather than imported
    (this script doesn't import anything else from before_kmeans.py either).
    """
    r, g, b = (round(c * 255) for c in color)
    return f"rgb({r}, {g}, {b})"


CLUSTER_COLORS_RGB = [_to_rgb_string(c) for c in CLUSTER_COLORS]  # Plotly-compatible version of CLUSTER_COLORS
COLOR_HEALTHY_RGB = _to_rgb_string(COLOR_HEALTHY)  # Plotly-compatible version of COLOR_HEALTHY


def plot_3d_clusters_interactive(
    pca_coords: np.ndarray, umap_coords: np.ndarray, cluster_labels: np.ndarray, ev_ids, title: str, out_path: Path,
    healthy_pca_coords: np.ndarray | None = None, healthy_umap_coords: np.ndarray | None = None,
    healthy_ev_ids=None,
) -> None:
    """Single interactive Plotly HTML, two STACKED rotatable 3D scatter
    scenes (PCA PC1-3 on top, UMAP UMAP1-3 below), points colored by
    cluster -- same embeddings as figure 2's static PCA/UMAP scatter
    (reused as-is from vork.csv), and the same "drag to rotate" idea as
    before_kmeans.py's plot_3d_faulty_only_interactive, but combined into
    ONE file covering both embeddings, and colored by cluster instead of a
    single flat "faulty" color -- so how the clusters actually sit relative
    to each other in 3D can be inspected directly, for whichever
    configuration is currently selected. Hovering a point shows its ev_id.
    Traces are paired across the two scenes via legendgroup, so toggling a
    cluster in the legend hides/shows it in both scenes at once.

    healthy_pca_coords/healthy_umap_coords/healthy_ev_ids (all optional,
    all-or-nothing): when POPULATION="faulty" only faulty EVs were
    clustered, so healthy EVs would otherwise be entirely absent from this
    plot -- passing their coordinates here draws them too, as a separate
    "healthy (reference, not clustered)" trace in a distinct flat color, for
    visual context on where the faulty clusters sit relative to the healthy
    population. Left None when POPULATION="all", since every healthy EV
    already appears there via its own assigned cluster color.
    """
    fig = make_subplots(
        rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "scene"}]],
        subplot_titles=("PCA (PC1-3)", "UMAP (UMAP1-3)"),
        vertical_spacing=0.05,
    )
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        color = CLUSTER_COLORS_RGB[cluster % len(CLUSTER_COLORS_RGB)]
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

    if healthy_pca_coords is not None:
        healthy_hover_text = [f"ev_id {ev_id} (healthy)" for ev_id in np.asarray(healthy_ev_ids)]
        healthy_marker = dict(size=3, color=COLOR_HEALTHY_RGB, opacity=0.5, line=dict(width=0.5, color="white"))

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


def build_lifetime_matrix_truncated(window_table: pd.DataFrame, feature_columns: list[str], N: int):
    """Option 1: EVs that actually HAVE every window_index 0..N-1 (checked
    by set containment, not just a window COUNT >= N -- a handful of EVs
    have a LEADING gap, e.g. window_index 0-2 masked by very-late
    start-of-operation and dropped by the NaN guard below, so window_index
    isn't guaranteed contiguous from 0 for every EV). Each row = concat of
    that EV's window_index 0..N-1 feature vectors (window-major,
    feature-minor -- see column_labels). Duplicated verbatim from
    kmeans.py's EV-LIFETIME VECTOR CLUSTERING section.
    Returns (ev_ids, X [n_evs, N*n_features], column_labels).
    """
    indexed = window_table.set_index(["ev_id", "window_index"])[feature_columns]
    available_indices = window_table.groupby("ev_id")["window_index"].apply(set)
    required = set(range(N))
    eligible_evs = sorted(ev_id for ev_id, idxs in available_indices.items() if required.issubset(idxs))

    rows = [np.concatenate([indexed.loc[(ev_id, wi)].to_numpy() for wi in range(N)]) for ev_id in eligible_evs]
    X = np.vstack(rows) if rows else np.empty((0, N * len(feature_columns)))
    column_labels = [f"{f}__w{wi}" for wi in range(N) for f in feature_columns]
    return np.array(eligible_evs), X, column_labels


def build_lifetime_matrix_padded(window_table: pd.DataFrame, feature_columns: list[str], N: int):
    """Option 2: EVERY EV with >= 1 window in range(N), padded by repeating
    the last OBSERVED (real) window's feature vector wherever window_index
    wi is missing for that EV -- also transparently covers LEADING gaps (a
    handful of EVs miss window_index 0..2): if wi=0 itself is missing, the
    fallback below carries the EV's own earliest available real window
    BACKWARD to fill the leading slot(s). One extra n_real_windows_used
    dimension is appended per EV. Duplicated verbatim from kmeans.py's
    EV-LIFETIME VECTOR CLUSTERING section.
    Returns (ev_ids, X [n_evs, N*n_features + 1], column_labels, n_real_windows_used).
    """
    indexed = window_table.set_index(["ev_id", "window_index"])[feature_columns]
    available_indices = window_table.groupby("ev_id")["window_index"].apply(set)
    all_evs = sorted(available_indices.index)

    rows = []
    n_real_used = []
    kept_evs = []
    for ev_id in all_evs:
        idxs = available_indices.loc[ev_id]
        if not idxs & set(range(N)):
            continue  # no real window at all within range(N) for this EV -- nothing to build or pad from, skip it
        last_real = None
        n_real = 0
        vecs = []
        for wi in range(N):
            if wi in idxs:
                last_real = indexed.loc[(ev_id, wi)].to_numpy()
                n_real += 1
            vecs.append(last_real)
        if vecs[0] is None:
            first_real = next(v for v in vecs if v is not None)
            vecs = [first_real if v is None else v for v in vecs]
        rows.append(np.concatenate(vecs))
        n_real_used.append(n_real)
        kept_evs.append(ev_id)

    X_features = np.vstack(rows) if rows else np.empty((0, N * len(feature_columns)))
    n_real_used = np.array(n_real_used, dtype=float)
    X = np.column_stack([X_features, n_real_used]) if rows else np.empty((0, N * len(feature_columns) + 1))
    column_labels = [f"{f}__w{wi}" for wi in range(N) for f in feature_columns] + ["n_real_windows_used"]
    return np.array(kept_evs), X, column_labels, n_real_used


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

IS_WHOLE_SESSION = WINDOW_SOURCE == "whole_session"
POINT_UNIT = "session" if IS_WHOLE_SESSION else "window"  # what one point/row represents, for labels/prints below

if not IS_WHOLE_SESSION and CLUSTERING_OPTION not in (1, 2):
    raise ValueError(f"CLUSTERING_OPTION must be 1 or 2, got {CLUSTERING_OPTION!r}")
if POPULATION not in ("faulty", "all", "class"):
    raise ValueError(f"POPULATION must be 'faulty', 'all', or 'class', got {POPULATION!r}")

if IS_WHOLE_SESSION:
    CONFIG_TAG = f"{WINDOW_SOURCE}_{POPULATION}_k{FINAL_K}"
    print(f"clust.py config: WINDOW_SOURCE={WINDOW_SOURCE} ({len(feature_columns)} features, no windowing), "
          f"POPULATION={POPULATION}, FINAL_K={FINAL_K}")
else:
    CONFIG_TAG = f"{WINDOW_SOURCE}_opt{CLUSTERING_OPTION}_{POPULATION}_N{N}_k{FINAL_K}"
    print(f"clust.py config: WINDOW_SOURCE={WINDOW_SOURCE} ({len(feature_columns)} features), "
          f"CLUSTERING_OPTION={CLUSTERING_OPTION}, POPULATION={POPULATION}, N={N}, FINAL_K={FINAL_K}")

# %% 1. load + clean windowed features -- charge_capacity forward-fill (only
# relevant when it's in feature_columns, i.e. WINDOW_SOURCE="120h"; the 24h
# source never computes that column at all), then drop any window still
# NaN in a kept column (same convention as kmeans.py's W1)
if IS_WHOLE_SESSION:
    # no windowing at all -- FEATURES_IN is one row per EV already (ev_id in
    # the index); promote it to a column and add a placeholder window_start
    # column (always 0) purely so the rest of this script, written for
    # one-row-per-window tables, needs no further special-casing downstream.
    window_features_all = pd.read_pickle(FEATURES_IN)
    window_features_all.index.name = "ev_id"
    window_features_all = window_features_all.reset_index()
    window_features_all["window_start"] = 0
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

if IS_WHOLE_SESSION:
    window_features_all["window_index"] = 0  # each EV has exactly one (whole-session) "window"
else:
    # window_index, same raw-recording-position basis as kmeans.py's EV-lifetime
    # section (window_start // stride, t=0 at raw index 0 regardless of
    # start-of-operation)
    _window_stride_samples = int(
        window_features_all.loc[window_features_all["window_start"] > 0, "window_start"].min()
    )
    window_features_all["window_index"] = (window_features_all["window_start"] // _window_stride_samples).astype(int)

# %% 2. build the chosen population's lifetime matrix. Loads the SAME
# cached whole-fleet IsolationForest as wisek.py (never refit here if it
# already exists) -- needed for POPULATION="class" (its FP EVs), and reused
# as-is by figure 4 further below rather than reloaded a second time.
whole_session_features_all = pd.read_pickle(FEATURES_IN)
if whole_session_features_all.isna().any().any():
    whole_session_features_all = whole_session_features_all.dropna()
whole_session_fault_label_all = pd.Series(
    whole_session_features_all.index >= FAULT_EV_ID_CUTOFF, index=whole_session_features_all.index,
)
if_results_all = get_or_fit_isolation_forest(whole_session_features_all, whole_session_fault_label_all)

selected_ev_ids = select_population(whole_session_features_all, whole_session_fault_label_all, if_results_all, POPULATION)
population_table = window_features_all.loc[window_features_all["ev_id"].isin(selected_ev_ids)]

if IS_WHOLE_SESSION:
    # each EV already has exactly one row -- use it directly as its own
    # point, no concatenation/padding/truncation needed at all
    ev_ids = population_table["ev_id"].to_numpy()
    X = population_table[feature_columns].to_numpy()
    column_labels = feature_columns
elif CLUSTERING_OPTION == 1:
    ev_ids, X, column_labels = build_lifetime_matrix_truncated(population_table, feature_columns, N)
else:
    ev_ids, X, column_labels, n_real_used = build_lifetime_matrix_padded(population_table, feature_columns, N)

if len(ev_ids) < 2:
    if IS_WHOLE_SESSION:
        raise RuntimeError(f"only {len(ev_ids)} EV(s) available for POPULATION={POPULATION!r}")
    raise RuntimeError(
        f"only {len(ev_ids)} EV(s) available for POPULATION={POPULATION!r}, N={N} -- pick a smaller N "
        "or CLUSTERING_OPTION=2 (pads short EVs instead of dropping them)"
    )

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if IS_WHOLE_SESSION:
    print(f"clustering {len(ev_ids)} EVs, {X.shape[1]} dimensions ({len(feature_columns)} whole-session "
          "features, no windowing)")
else:
    print(f"clustering {len(ev_ids)} EVs, {X.shape[1]} dimensions ({len(feature_columns)} features x {N} windows"
          f"{' + n_real_windows_used' if CLUSTERING_OPTION == 2 else ''})")

# %% 3. k=2..8 sweep (for figure 1) + committed fit at FINAL_K
usable_ks = [k for k in K_RANGE if 2 <= k < len(ev_ids)]
if FINAL_K not in usable_ks:
    raise RuntimeError(f"FINAL_K={FINAL_K} not valid for {len(ev_ids)} EVs (usable k: {usable_ks})")

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
cluster_label = pd.Series(final_model.labels_, index=ev_ids, name="cluster")
print(f"\n--- cluster sizes (k={FINAL_K}) ---")
print(cluster_label.value_counts().sort_index().rename("n_evs").to_string())

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
ax1.set_title(f"Inertia & silhouette vs. k (chosen k={FINAL_K} marked) -- {CONFIG_TAG}")
_style_scatter_axes(ax1)
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust_{CONFIG_TAG}_inertia_silhouette.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 5. figure 2 -- PCA/UMAP scatter, colored by cluster (embeddings reused
# as-is from vork.csv, same convention as kmeans.py/centroid.py -- NOT
# recomputed here)
embeddings = pd.read_csv(EMBEDDINGS_IN, index_col="ev_id")
missing_embeddings = sorted(set(ev_ids) - set(embeddings.index))
if missing_embeddings:
    print(f"warning: {len(missing_embeddings)} EV(s) in this clustering have no PCA/UMAP embedding in "
          f"vork.csv, excluded from figure 2: {missing_embeddings}")
plot_ev_ids = [e for e in ev_ids if e in embeddings.index]
plot_embeddings = embeddings.loc[plot_ev_ids]
plot_clusters = cluster_label.loc[plot_ev_ids].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, x_col, y_col, xlabel, ylabel, title in [
    (axes[0], "pca_1", "pca_2", "PC1", "PC2", "PCA"),
    (axes[1], "umap_1", "umap_2", "UMAP-1", "UMAP-2", "UMAP"),
]:
    for cluster in sorted(np.unique(plot_clusters)):
        mask = plot_clusters == cluster
        ax.scatter(
            plot_embeddings.loc[mask, x_col], plot_embeddings.loc[mask, y_col],
            c=[CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]], s=40, edgecolor="white", linewidth=0.4,
            alpha=0.85, label=f"cluster {cluster}", zorder=2,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(loc="best", fontsize=8, title="cluster")
fig.suptitle(f"EV clusters in PCA/UMAP embedding -- {CONFIG_TAG}")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust_{CONFIG_TAG}_pca_umap.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 5b. figure 2's interactive 3D companion -- same points/clusters as
# above (PC3/UMAP-3 added back in), one rotatable HTML with both embeddings
# stacked, colored by cluster. When POPULATION="faulty" or "class", some/all
# healthy EVs were never part of the clustering, so any healthy EV NOT in
# ev_ids is added here as a separate reference trace for visual context
# (for "faulty" that's every healthy EV, same as before; for "class" it's
# just the ones select_population didn't pull in); when POPULATION="all"
# every EV is already in plot_embeddings/plot_clusters via its own assigned
# cluster, so nothing extra is added.
healthy_3d_kwargs = {}
if POPULATION in ("faulty", "class"):
    healthy_embeddings = embeddings.loc[(embeddings.index < FAULT_EV_ID_CUTOFF) & (~embeddings.index.isin(ev_ids))]
    if len(healthy_embeddings) > 0:
        healthy_3d_kwargs = dict(
            healthy_pca_coords=healthy_embeddings[["pca_1", "pca_2", "pca_3"]].to_numpy(),
            healthy_umap_coords=healthy_embeddings[["umap_1", "umap_2", "umap_3"]].to_numpy(),
            healthy_ev_ids=healthy_embeddings.index,
        )

plot_3d_clusters_interactive(
    plot_embeddings[["pca_1", "pca_2", "pca_3"]].to_numpy(),
    plot_embeddings[["umap_1", "umap_2", "umap_3"]].to_numpy(),
    plot_clusters, plot_embeddings.index,
    title=f"EV clusters in 3D PCA/UMAP embedding -- {CONFIG_TAG}",
    out_path=FIGURES_DIR / f"clust_{CONFIG_TAG}_3d_pca_umap.html",
    **healthy_3d_kwargs,
)

# %% 6. figure 3 -- one subplot PER FEATURE, the 6 features that vary the
# MOST across clusters, each drawn as a violin (full per-EV distribution,
# not just mean +/- std -- "2D", same style as fault_feature_distributions.png)
# for every cluster plus the faulty-population and healthy-population
# references. Computed on each EV's own per-EV mean across its REAL
# (non-padded) windows in range(N), so Option 2's repeated padding doesn't
# distort a feature's apparent importance or spread. The faulty-/healthy-
# population references are drawn from the full fleet's faulty/healthy
# subsets (every such EV with >= 1 real window in range(N)) -- a fixed,
# stable baseline regardless of which POPULATION was actually clustered.
#
# --- "varies the most across clusters": one-way ANOVA F-statistic --------
# For each feature, scipy.stats.f_oneway compares the CLUSTER groups'
# per-EV values: F = (variance BETWEEN cluster means) / (variance WITHIN
# clusters). A high F means the clusters' means are far apart relative to
# their own internal spread -- i.e. that feature actually separates the
# clusters, as opposed to a feature with similar means everywhere (low F,
# doesn't distinguish clusters) or one with huge within-cluster noise (also
# low F, unreliable even if means differ). This is the standard measure for
# "how different is X across groups" and avoids picking features that only
# LOOK different by eye because of scale. The 6 highest-F features are kept.
real_windows_in_range = (
    window_features_all if IS_WHOLE_SESSION else window_features_all[window_features_all["window_index"] < N]
)
ev_feature_means = real_windows_in_range.groupby("ev_id")[feature_columns].mean()

cluster_ev_means = ev_feature_means.loc[ev_feature_means.index.isin(ev_ids)].copy()
cluster_ev_means["cluster"] = cluster_label.reindex(cluster_ev_means.index).to_numpy()

all_faulty_ev_ids = window_features_all.loc[window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF, "ev_id"].unique()
all_healthy_ev_ids = window_features_all.loc[window_features_all["ev_id"] < FAULT_EV_ID_CUTOFF, "ev_id"].unique()
faulty_pop = ev_feature_means.loc[ev_feature_means.index.isin(all_faulty_ev_ids)]
healthy_pop = ev_feature_means.loc[ev_feature_means.index.isin(all_healthy_ev_ids)]

clusters = sorted(cluster_ev_means["cluster"].unique())
cluster_groups = {c: cluster_ev_means.loc[cluster_ev_means["cluster"] == c, feature_columns] for c in clusters}

f_scores = {}
for feature in feature_columns:
    groups = [cluster_groups[c][feature].to_numpy() for c in clusters]
    # scipy handles singleton groups fine (a cluster with 1 EV just
    # contributes 0 to within-group variance); only truly degenerate case
    # is a feature that's constant across every group (0/0 -> nan), which
    # f_oneway already returns as nan itself rather than raising
    f_stat, _p_value = f_oneway(*groups)
    f_scores[feature] = f_stat
top6_features = pd.Series(f_scores).dropna().sort_values(ascending=False).head(6).index.tolist()
if not top6_features:
    raise RuntimeError(
        "every feature's ANOVA F-statistic came out NaN (likely all clusters are constant/singleton "
        "for every feature) -- can't pick top-varying features for this configuration"
    )
print(f"\n--- top-6 most cluster-varying features (one-way ANOVA F-statistic) ---")
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
    long_rows += [{"group": "faulty\npop.", "value": v} for v in faulty_pop[feature]]
    long_rows += [{"group": "healthy\npop.", "value": v} for v in healthy_pop[feature]]
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

fig.suptitle(f"Top-6 most cluster-varying features (one-way ANOVA F) -- cluster vs. "
             f"faulty/healthy population -- {CONFIG_TAG}")
fig.tight_layout()
fig.savefig(FIGURES_DIR / f"clust_{CONFIG_TAG}_top_features.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(fig)

# %% 7. figure 4 -- cluster composition vs. IsolationForest detection
# (TN/TP/FP/FN), same idea as wisek.py's cluster_confusion_bars figure but
# for THIS script's own EV-lifetime clustering. Reuses if_results_all from
# step 2 above (same cached whole-fleet IsolationForest as wisek.py, loaded
# once, not reloaded here) -- purely additive, doesn't touch anything else
# in this script.
missing_if_results = sorted(set(ev_ids) - set(if_results_all.index))
if missing_if_results:
    print(f"warning: {len(missing_if_results)} EV(s) in this clustering have no cached IsolationForest "
          f"result, excluded from figure 4: {missing_if_results}")
confusion_ev_ids = [e for e in ev_ids if e in if_results_all.index]

cluster_confusion = pd.DataFrame({
    "cluster": cluster_label.loc[confusion_ev_ids].to_numpy(),
    "confusion": if_results_all.loc[confusion_ev_ids, "confusion"].to_numpy(),
}, index=confusion_ev_ids)

print(f"\n--- Per-cluster TN/TP/FP/FN composition vs. cached IsolationForest (k={FINAL_K}) ---")
print(cluster_confusion.groupby(["cluster", "confusion"]).size().unstack(fill_value=0)
      .reindex(columns=CONFUSION_ORDER, fill_value=0).to_string())

plot_cluster_confusion_bars(
    cluster_confusion,
    title=f"Cluster composition vs. IsolationForest detection -- {CONFIG_TAG}",
    out_path=FIGURES_DIR / f"clust_{CONFIG_TAG}_confusion_bars.png",
)

print(f"\nsaved 4 figures + 1 interactive 3D HTML to {FIGURES_DIR} (tag: {CONFIG_TAG})")
