# %%
"""
K-means clustering of WINDOWED EV features, with the SAME class-imbalance
population-selection machinery as wisek.py (whole-session track), adapted
for windows. See wisek.py's module docstring for the full rationale on
SELECTION_MODE/CLASS_N_RANDOM_HEALTHY/the fixed whole-fleet IsolationForest
-- this script reuses those EV-level mechanics unchanged, just projected
down onto windows.

--- Why the figures had to change for windows -----------------------------
wisek.py's TP/TN/FP/FN figures rely on ground truth EXISTING per sample
(EV number >= FAULT_EV_ID_CUTOFF is a real, known label for every EV). At
window granularity there is no such ground truth -- we don't know which
of a faulty EV's individual windows actually show the fault vs. which are
still pre-onset/normal-looking (this is exactly the ambiguity
sliding_windows.py's windowed track has to work around elsewhere). So
instead of per-window TP/TN/FP/FN:
  - Every window is tagged with its OWN EV's whole-session confusion label
    (TP/TN/FP/FN, from the cached whole-fleet IsolationForest in wisek.py --
    see get_or_fit_isolation_forest below, same cache file, never refit
    here) -- i.e. "this window belongs to an EV IsolationForest got wrong
    (FP or FN)", not "this window itself is wrong".
  - Figure 1 (cluster_confusion_bars): per cluster, how many windows come
    from FN EVs, how many from FP EVs, and how many from everything else
    (TP/TN EVs) -- stacked bar, same color convention as wisek.py (FN=red,
    FP=orange), "other" in neutral gray.
  - Figure 2 (pca_umap_by_cluster): PCA/UMAP of windows (computed fresh
    here -- no precomputed windowed embedding file exists, unlike vork.csv
    for whole-session), colored by cluster; windows belonging to an FN EV
    drawn FILLED+bold, windows belonging to an FP EV drawn UNFILLED+bold
    (same fill convention wisek.py uses to tell FN/FP apart).
  - Figure 3 (pca_umap_single_ev_drift, NEW -- no whole-session equivalent):
    same PCA/UMAP map as figure 2, but as plain background context (small,
    faint, uncolored by cluster) with ONE EV's own windows (DRIFT_EV_ID)
    highlighted on top, colored by window_start_hours (a sequential
    colormap) and connected in chronological order, so that EV's drift
    through feature space over its lifetime is directly visible as a path
    across the map.

Population selection (SELECTION_MODE, CLASS_N_RANDOM_HEALTHY): identical
semantics to wisek.py, applied at the EV level -- selecting an EV pulls in
ALL of its windows (not a per-window sub-selection).

Input:
  extracted_data/fault_detection/filtered_features.pkl -- whole-session
      features, only used here for EV-level SELECTION_MODE / the cached
      whole-fleet IsolationForest (same as wisek.py).
  extracted_data/early_warning/filtered_features_windows.pkl (from
      sliding_window_fleet_features.py) -- one row per (ev_id, window).
  extracted_data/early_warning/wisek_isolation_forest_whole_session.pkl --
      cached whole-fleet IsolationForest results (from wisek.py's
      get_or_fit_isolation_forest; created here too if it doesn't exist
      yet -- same fit-once/cache-forever contract, see wisek.py).

Output:
  figures/wisek_windowed/cluster_confusion_bars_<selection>_k<K>.png
  figures/wisek_windowed/pca_umap_by_cluster_<selection>_k<K>.png
  figures/wisek_windowed/pca_umap_single_ev_drift_<selection>_k<K>_ev<DRIFT_EV_ID>.png
  figures/wisek_windowed/inertia_silhouette_<selection>_k<K>.png -- inertia +
      silhouette vs. k over K_DIAGNOSTIC_RANGE (dual y-axis, same style as
      clust2.py's own inertia/silhouette figure), chosen K marked.
  figures/wisek_windowed/top_features_<selection>_k<K>.png -- one subplot per
      feature, the top-6 features that vary the most across this run's
      clusters (one-way ANOVA F-statistic, by window), violin per cluster
      plus full-fleet faulty/healthy window-population references -- same
      method/layout as clust2.py's own top_features figure.
  figures/wisek_windowed/all_features_<selection>_k<K>.png -- same violin
      treatment as above, but for EVERY feature (not just the top-6) --
      analogous to clust2.py's global_top_features figure, except the
      feature set here is ALL features rather than a fixed top-12; grid and
      per-subplot width scale with feature count / K.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

# umap-learn is not a core dependency of this repo yet -- if this import
# fails, run: pip install umap-learn
import umap

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/02_.../wisek_window.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
WINDOW_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
IF_CACHE_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "wisek_isolation_forest_whole_session.pkl"  # SAME cache as wisek.py -- shared, never refit twice
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "wisek_windowed"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as wisek.py / kmeans.py / detection.py
CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"  # forward-filled per EV, see W1-style guard below

SELECTION_MODE = "all"  # "all" | "faulty" | "class" -- same semantics as wisek.py, applied at EV level
CLASS_N_RANDOM_HEALTHY = 17  # only used when SELECTION_MODE == "class" -- see wisek.py's select_population

K = 9 # number of K-means clusters (on WINDOWS, not EVs) -- set by hand, no sweep in this script
K_DIAGNOSTIC_RANGE = range(2, 20)  # inertia/silhouette sweep, printed ONLY when SELECTION_MODE is "class" or "all" -- see step 4a

DRIFT_EV_ID = 390  # figure 3: which single EV's windows to trace across the PCA/UMAP map over time.
                    # Must be part of the population SELECTION_MODE produces (else figure 3 is skipped
                    # with a warning -- see module docstring: "whole map" means this run's own population).

# NOTE: no CONTAMINATION_RATIO knob here either -- see wisek.py; the cached
# whole-fleet IsolationForest (IF_CACHE_OUT) is shared with wisek.py and
# never refit by this script if it already exists.

RANDOM_STATE = 0
N_INIT = 10        # K-means random restarts, explicit (not "auto") for reproducibility/reporting
N_ESTIMATORS = 100  # IsolationForest trees, only used if IF_CACHE_OUT doesn't exist yet (see wisek.py)

# UMAP hyperparameters, same defaults as before_kmeans.py
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

FIGURE_DPI = 300

CLUSTER_COLORS = RWTHColorScales.cycle  # qualitative palette for cluster identity
COLOR_FAULTY = RWTHColorScales.red[0]   # inertia line, same convention as clust2.py
COLOR_HEALTHY = RWTHColorScales.blue[0]  # silhouette line, same convention as clust2.py
CONFUSION_COLORS = {  # same convention as wisek.py: FP/FN = wrong (orange/red), "other" = neutral gray
    "other": "lightgray",
    "FP": RWTHColorScales.orange[0],
    "FN": RWTHColorScales.red[0],
}
CONFUSION_ORDER = ["other", "FN", "FP"]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with wisek.py/kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def select_population(
    features_all: pd.DataFrame, fault_label_all: pd.Series, if_results_all: pd.DataFrame, mode: str,
) -> pd.Index:
    """Identical to wisek.py's select_population -- duplicated here (not
    imported) since these are sibling standalone "# %%" scripts, same
    convention as kmeans.py/detection.py's windowed tracks duplicating
    whole-session helpers rather than importing them. See wisek.py's own
    docstring for the full rationale.
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
    raise ValueError(f"unknown SELECTION_MODE {mode!r} -- expected 'all', 'faulty', or 'class'")


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
    SAME IF_CACHE_OUT path -- if wisek.py has already been run, this just
    loads its cached results; only fits (and caches) its own if neither
    script has produced IF_CACHE_OUT yet. Never fit twice."""
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


def plot_inertia_silhouette(diagnostics: pd.DataFrame, chosen_k: int, title: str, out_path: Path) -> None:
    """Inertia + silhouette vs. k, single plot (dual y-axis), chosen_k marked
    with a vertical line -- same layout/convention as clust2.py's own
    inertia/silhouette figure."""
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(diagnostics["k"], diagnostics["inertia"], marker="o", color=COLOR_FAULTY, label="inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("inertia (within-cluster sum of squares)", color=COLOR_FAULTY)
    ax1.tick_params(axis="y", labelcolor=COLOR_FAULTY)
    ax1.set_xticks(diagnostics["k"])
    ax1.axvline(chosen_k, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="s", color=COLOR_HEALTHY, label="silhouette")
    ax2.set_ylabel("mean silhouette score", color=COLOR_HEALTHY)
    ax2.tick_params(axis="y", labelcolor=COLOR_HEALTHY)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title(title)
    _style_scatter_axes(ax1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _violin_groups(windows: pd.DataFrame, window_features_all: pd.DataFrame, feature_columns: list[str]) -> tuple:
    """Shared setup for the per-feature violin figures below: per-cluster
    window groups (from THIS run's selected population) plus the full
    fleet's faulty-window-population / healthy-window-population reference
    groups, and the label/color scheme tying them together."""
    all_faulty_windows = window_features_all.loc[window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF, feature_columns]
    all_healthy_windows = window_features_all.loc[window_features_all["ev_id"] < FAULT_EV_ID_CUTOFF, feature_columns]

    clusters = sorted(windows["cluster"].unique())
    cluster_groups = {c: windows.loc[windows["cluster"] == c, feature_columns] for c in clusters}

    group_labels = [f"cl. {c}" for c in clusters] + ["faulty\npop.", "healthy\npop."]
    group_colors = {**{f"cl. {c}": CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in clusters},
                     "faulty\npop.": COLOR_FAULTY, "healthy\npop.": COLOR_HEALTHY}
    return clusters, cluster_groups, all_faulty_windows, all_healthy_windows, group_labels, group_colors


def _draw_feature_violin(
    ax: plt.Axes, feature: str, clusters: list, cluster_groups: dict,
    all_faulty_windows: pd.DataFrame, all_healthy_windows: pd.DataFrame, group_labels: list, group_colors: dict,
) -> None:
    """Draw one feature's violin-per-group subplot (clusters + faulty/healthy
    population references) onto `ax`. Shared by plot_top_features and
    plot_all_features."""
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
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    _style_scatter_axes(ax)


def plot_top_features(
    windows: pd.DataFrame, window_features_all: pd.DataFrame, feature_columns: list[str], title: str, out_path: Path,
) -> None:
    """One subplot per feature, the top-6 features that vary the most across
    THIS run's clusters (highest one-way ANOVA F-statistic among the cluster
    groups, computed on WINDOW-level values), each a violin plot per cluster
    plus the full fleet's faulty-window-population and healthy-window-
    population reference distributions -- same method/layout as clust2.py's
    figure 3, adapted to this script's ev_id-based fault definition."""
    clusters, cluster_groups, all_faulty_windows, all_healthy_windows, group_labels, group_colors = (
        _violin_groups(windows, window_features_all, feature_columns)
    )

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
    print("\n--- top-6 most cluster-varying features (one-way ANOVA F-statistic, by window) ---")
    print(pd.Series(f_scores).loc[top6_features].rename("F_statistic").to_string())

    ncols = min(3, len(top6_features))
    nrows = int(np.ceil(len(top6_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, feature in zip(axes_flat, top6_features):
        _draw_feature_violin(ax, feature, clusters, cluster_groups, all_faulty_windows, all_healthy_windows,
                              group_labels, group_colors)
        ax.set_title(f"{feature}  (F={f_scores[feature]:.2f})", fontsize=9)

    for ax in axes_flat[len(top6_features):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_all_features(
    windows: pd.DataFrame, window_features_all: pd.DataFrame, feature_columns: list[str], title: str, out_path: Path,
) -> None:
    """Same violin treatment as plot_top_features, but for EVERY feature
    (not just the top-6 by ANOVA F-statistic) -- one subplot per feature,
    cluster groups plus full-fleet faulty/healthy window-population
    references. Grid is sized to the number of features (fixed 4 columns,
    as many rows as needed), and each subplot is WIDENED as the number of
    violin groups (clusters + 2 population references) grows, so x-tick
    labels stay legible instead of overlapping when K is large."""
    clusters, cluster_groups, all_faulty_windows, all_healthy_windows, group_labels, group_colors = (
        _violin_groups(windows, window_features_all, feature_columns)
    )

    n_features = len(feature_columns)
    ncols = min(4, n_features)
    nrows = int(np.ceil(n_features / ncols))
    n_groups = len(group_labels)  # clusters + faulty/healthy population references
    subplot_width = 5.5 * max(1.0, n_groups / 5)  # widen once violin groups start crowding the x-axis
    fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_width * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, feature in zip(axes_flat, feature_columns):
        _draw_feature_violin(ax, feature, clusters, cluster_groups, all_faulty_windows, all_healthy_windows,
                              group_labels, group_colors)
        ax.set_title(feature, fontsize=9)

    for ax in axes_flat[n_features:]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_confusion_bars(windows: pd.DataFrame, title: str, out_path: Path) -> None:
    """Stacked bar chart: one bar per cluster, height = # windows in that
    cluster, stacked by whether each window's OWN EV is an FN EV, an FP EV,
    or neither ("other" -- from a TP or TN EV). See module docstring for
    why this replaces wisek.py's per-sample TP/TN/FP/FN bars at window
    granularity."""
    counts = windows.groupby(["cluster", "ev_confusion_group"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=CONFUSION_ORDER, fill_value=0).sort_index()

    fig, ax = plt.subplots(figsize=(max(6.0, 1.3 * len(counts) + 2.5), 6))
    bottom = np.zeros(len(counts))
    for label in CONFUSION_ORDER:
        values = counts[label].to_numpy()
        ax.bar(counts.index.astype(str), values, bottom=bottom, color=CONFUSION_COLORS[label],
               edgecolor="white", linewidth=0.6, label=("other (TP/TN EV)" if label == "other" else f"{label} EV"),
               zorder=2)
        bottom += values

    ax.set_xlabel("cluster")
    ax.set_ylabel("# windows")
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(title="window's own EV, per wisek.py", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_embeddings_by_cluster(coords: dict, windows: pd.DataFrame, title: str, out_path: Path) -> None:
    """PCA + UMAP scatter subplots of WINDOWS, colored by cluster. Windows
    belonging to an FN EV are drawn larger/filled/black-outlined; windows
    belonging to an FP EV are drawn larger/UNFILLED (hollow, cluster-colored
    outline) -- same fill convention wisek.py uses to distinguish FN from
    FP at a glance. `coords` is {"PCA": (n,2) array, "UMAP": (n,2) array},
    row-aligned with `windows`."""
    clusters = windows["cluster"].to_numpy()
    ev_group = windows["ev_confusion_group"].to_numpy()
    cluster_ids = sorted(windows["cluster"].unique())

    panels = [("PCA", "PC1", "PC2"), ("UMAP", "UMAP-1", "UMAP-2")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, (name, xlabel, ylabel) in zip(axes, panels):
        xy = coords[name]

        for cluster in cluster_ids:
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            base_mask = (clusters == cluster) & (ev_group == "other")
            ax.scatter(xy[base_mask, 0], xy[base_mask, 1], c=[color], s=25, alpha=0.35,
                       edgecolor="none", zorder=2)

        for cluster in cluster_ids:
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]

            fn_mask = (clusters == cluster) & (ev_group == "FN")
            ax.scatter(xy[fn_mask, 0], xy[fn_mask, 1], c=[color], s=70, alpha=1.0,
                       edgecolor="black", linewidth=1.0, zorder=3)

            fp_mask = (clusters == cluster) & (ev_group == "FP")
            ax.scatter(xy[fp_mask, 0], xy[fp_mask, 1], facecolor="none", s=70,
                       edgecolor=[color], linewidth=1.5, zorder=3)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        _style_scatter_axes(ax)

    cluster_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
               markeredgecolor="none", markersize=8, label=f"cluster {c}")
        for c in cluster_ids
    ]
    fn_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="dimgray",
                        markeredgecolor="black", markeredgewidth=1.2, markersize=9,
                        label="window's EV is FN (filled)")
    fp_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                        markeredgecolor="dimgray", markeredgewidth=1.6, markersize=9,
                        label="window's EV is FP (unfilled)")
    axes[1].legend(handles=cluster_handles + [fn_handle, fp_handle], loc="upper left",
                   bbox_to_anchor=(1.02, 1.0), fontsize=8, title="legend", borderaxespad=0.0)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_single_ev_drift(coords: dict, windows: pd.DataFrame, drift_ev_id: int, title: str, out_path: Path) -> None:
    """PCA + UMAP subplots: the whole selected-population map as faint gray
    background context, with ONE EV's (drift_ev_id) own windows highlighted
    on top, colored by window_start_hours (sequential colormap) and
    connected in chronological order -- so that EV's drift through feature
    space over its lifetime reads as a path across the map. Triangle marks
    its first window, square its last."""
    ev_mask = (windows["ev_id"] == drift_ev_id).to_numpy()
    ev_windows = windows.loc[ev_mask].sort_values("window_start_hours")
    order = ev_windows.index.to_numpy()
    hours = ev_windows["window_start_hours"].to_numpy()

    panels = [("PCA", "PC1", "PC2"), ("UMAP", "UMAP-1", "UMAP-2")]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    scatter_for_colorbar = None

    for ax, (name, xlabel, ylabel) in zip(axes, panels):
        xy_all = coords[name]
        ax.scatter(xy_all[:, 0], xy_all[:, 1], c="lightgray", s=15, alpha=0.4, edgecolor="none", zorder=1)

        xy_ev = xy_all[windows.index.get_indexer(order)]
        ax.plot(xy_ev[:, 0], xy_ev[:, 1], color="black", linewidth=0.8, alpha=0.5, zorder=2)
        scatter_for_colorbar = ax.scatter(xy_ev[:, 0], xy_ev[:, 1], c=hours, cmap="viridis", s=70,
                                           edgecolor="black", linewidth=0.6, zorder=3)
        ax.scatter(xy_ev[0, 0], xy_ev[0, 1], marker="^", s=160, facecolor="none",
                   edgecolor="black", linewidth=1.8, zorder=4, label="first window")
        ax.scatter(xy_ev[-1, 0], xy_ev[-1, 1], marker="s", s=160, facecolor="none",
                   edgecolor="black", linewidth=1.8, zorder=4, label="last window")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        _style_scatter_axes(ax)
        ax.legend(loc="best", fontsize=8)

    fig.colorbar(scatter_for_colorbar, ax=axes, label="window_start_hours", shrink=0.85)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load whole-session features + cached whole-fleet IsolationForest
# (shared with wisek.py -- see get_or_fit_isolation_forest), select EV
# population per SELECTION_MODE (identical semantics to wisek.py)
features_all = pd.read_pickle(FEATURES_IN)
if features_all.isna().any().any():
    n_before = len(features_all)
    features_all = features_all.dropna()
    print(f"dropped {n_before - len(features_all)} EVs with missing whole-session feature values")

fault_label_all = pd.Series(features_all.index >= FAULT_EV_ID_CUTOFF, index=features_all.index, name="fault_label")
if_results_all = get_or_fit_isolation_forest(features_all, fault_label_all)

selected_ev_index = select_population(features_all, fault_label_all, if_results_all, SELECTION_MODE)
print(f"SELECTION_MODE={SELECTION_MODE!r}: {len(selected_ev_index)} EVs selected "
      f"({int(fault_label_all.loc[selected_ev_index].sum())} faulty, "
      f"{int((~fault_label_all.loc[selected_ev_index]).sum())} healthy, of {len(features_all)} total)")

# %% 2. load windowed features, forward-fill CAPACITY_COLUMN per EV, drop
# any remaining NaN rows -- same convention as kmeans.py's/detection.py's
# windowed tracks
window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
missing_window_columns = set(features_all.columns) - set(window_features_all.columns)
if missing_window_columns:
    raise RuntimeError(
        f"filtered_features_windows.pkl is missing feature column(s) {sorted(missing_window_columns)} "
        "relative to the current filtered_features.pkl -- re-run sliding_window_fleet_features.py first."
    )
feature_columns = features_all.columns.tolist()

window_features_all = window_features_all.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
window_features_all[CAPACITY_COLUMN] = window_features_all.groupby("ev_id")[CAPACITY_COLUMN].ffill()
window_valid_mask = window_features_all[feature_columns].notna().all(axis=1)
n_windows_dropped = int((~window_valid_mask).sum())
window_features_all = window_features_all.loc[window_valid_mask].reset_index(drop=True)
print(f"\nwindowed track: dropped {n_windows_dropped} window(s) with >=1 NaN feature value, "
      f"{len(window_features_all)} windows remain")

# %% 3. restrict to windows of the selected EV population, tag each window
# with its OWN EV's whole-session confusion label (TP/TN/FP/FN) and a
# coarser ev_confusion_group ("FN" / "FP" / "other") used by the figures
windows = window_features_all.loc[window_features_all["ev_id"].isin(selected_ev_index)].reset_index(drop=True)
windows["ev_confusion"] = if_results_all.loc[windows["ev_id"], "confusion"].to_numpy()
windows["ev_confusion_group"] = np.where(
    windows["ev_confusion"] == "FN", "FN", np.where(windows["ev_confusion"] == "FP", "FP", "other"),
)
print(f"windowed track: {len(windows)} windows from {windows['ev_id'].nunique()} of the "
      f"{len(selected_ev_index)} selected EVs "
      f"({int((windows['ev_confusion_group'] == 'FN').sum())} from FN EVs, "
      f"{int((windows['ev_confusion_group'] == 'FP').sum())} from FP EVs)")

# %% 4. K-means on standardized windowed features (own StandardScaler, fit
# fresh on this run's selected windows -- same convention as wisek.py's
# whole-session K-means, never cached/shared across runs unlike the
# IsolationForest above)
scaler = StandardScaler()
windows_scaled = scaler.fit_transform(windows[feature_columns])

# %% 4a. diagnostics sweep -- inertia + silhouette score over
# K_DIAGNOSTIC_RANGE, always computed (feeds figure 4, the inertia/
# silhouette plot, below) but only PRINTED as a table when SELECTION_MODE is
# "class" or "all" (not "faulty" -- that population is small enough that a
# full k sweep is cheap either way, but the printout is specifically meant
# to inform the two larger, imbalance-relevant populations). Same k
# range/metrics as kmeans.py's own elbow/silhouette diagnostics and
# clust2.py's figure 1.
diagnostics_ks = [k_diag for k_diag in K_DIAGNOSTIC_RANGE if 2 <= k_diag < len(windows)]
diagnostics_rows = []
for k_diag in diagnostics_ks:
    diag_model = KMeans(n_clusters=k_diag, random_state=RANDOM_STATE, n_init=N_INIT)
    diag_labels = diag_model.fit_predict(windows_scaled)
    diagnostics_rows.append({
        "k": k_diag,
        "inertia": diag_model.inertia_,
        "silhouette_score": silhouette_score(windows_scaled, diag_labels),
    })
diagnostics = pd.DataFrame(diagnostics_rows)

if SELECTION_MODE in ("class", "all"):
    print(f"\n--- K-means diagnostics sweep (windows, k={K_DIAGNOSTIC_RANGE.start}..{K_DIAGNOSTIC_RANGE.stop - 1}, "
          f"SELECTION_MODE={SELECTION_MODE!r}) ---")
    print(diagnostics.to_string(index=False))

kmeans_model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
windows["cluster"] = kmeans_model.fit_predict(windows_scaled)

if 2 <= K < len(windows):
    print(f"K-means (k={K}) silhouette score: {silhouette_score(windows_scaled, windows['cluster']):.3f}")

cluster_sizes = windows["cluster"].value_counts().sort_index()
print(f"\n--- Cluster sizes (k={K}, windows) ---")
print(cluster_sizes.rename("n_windows").to_string())

print(f"\n--- Per-cluster FN/FP/other window composition (k={K}) ---")
print(windows.groupby(["cluster", "ev_confusion_group"]).size().unstack(fill_value=0)
      .reindex(columns=CONFUSION_ORDER, fill_value=0).to_string())

# %% 5. PCA + UMAP embeddings of the windows -- computed fresh here (no
# precomputed windowed embedding file exists, unlike vork.csv for
# whole-session), same hyperparameters as before_kmeans.py
pca_coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(windows_scaled)
umap_coords = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, random_state=RANDOM_STATE,
).fit_transform(windows_scaled)
coords = {"PCA": pca_coords, "UMAP": umap_coords}

# %% 6. figures
run_tag = f"{SELECTION_MODE}_k{K}"
title_suffix = f"selection={SELECTION_MODE}, k={K}, n={len(windows)} windows from {windows['ev_id'].nunique()} EVs"

plot_inertia_silhouette(
    diagnostics, chosen_k=K,
    title=f"Inertia & silhouette vs. k (chosen k={K} marked) -- windowed, {title_suffix}",
    out_path=FIGURES_DIR / f"inertia_silhouette_{run_tag}.png",
)
print(f"\nsaved {FIGURES_DIR / f'inertia_silhouette_{run_tag}.png'}")

plot_cluster_confusion_bars(
    windows,
    title=f"Windowed cluster composition, FN/FP-EV windows marked ({title_suffix})",
    out_path=FIGURES_DIR / f"cluster_confusion_bars_{run_tag}.png",
)
print(f"saved {FIGURES_DIR / f'cluster_confusion_bars_{run_tag}.png'}")

plot_embeddings_by_cluster(
    coords, windows,
    title=f"Windowed PCA / UMAP by cluster, FN/FP-EV windows bolded ({title_suffix})",
    out_path=FIGURES_DIR / f"pca_umap_by_cluster_{run_tag}.png",
)
print(f"saved {FIGURES_DIR / f'pca_umap_by_cluster_{run_tag}.png'}")

if DRIFT_EV_ID in windows["ev_id"].to_numpy():
    plot_single_ev_drift(
        coords, windows, DRIFT_EV_ID,
        title=f"EV {DRIFT_EV_ID} window drift across PCA / UMAP map ({title_suffix})",
        out_path=FIGURES_DIR / f"pca_umap_single_ev_drift_{run_tag}_ev{DRIFT_EV_ID}.png",
    )
    print(f"saved {FIGURES_DIR / f'pca_umap_single_ev_drift_{run_tag}_ev{DRIFT_EV_ID}.png'}")
else:
    print(f"\nWARNING: DRIFT_EV_ID={DRIFT_EV_ID} has no windows in this run's population "
          f"(SELECTION_MODE={SELECTION_MODE!r}) -- skipped figure 3. Pick an EV that's part of "
          "this selection, or change SELECTION_MODE.")

plot_top_features(
    windows, window_features_all, feature_columns,
    title=f"Top-6 most cluster-varying features (one-way ANOVA F, by window) -- cluster vs. "
          f"faulty/healthy window population -- windowed, {title_suffix}",
    out_path=FIGURES_DIR / f"top_features_{run_tag}.png",
)
print(f"saved {FIGURES_DIR / f'top_features_{run_tag}.png'}")

plot_all_features(
    windows, window_features_all, feature_columns,
    title=f"All features by cluster (one-way ANOVA-style violin, by window) -- cluster vs. "
          f"faulty/healthy window population -- windowed, {title_suffix}",
    out_path=FIGURES_DIR / f"all_features_{run_tag}.png",
)
print(f"saved {FIGURES_DIR / f'all_features_{run_tag}.png'}")
