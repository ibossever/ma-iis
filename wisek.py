# %%
"""
K-means clustering of whole-session EV features, with per-cluster
IsolationForest fault-detection agreement (TP/TN/FP/FN vs. ground truth) --
built to probe how class imbalance (the healthy:faulty ratio fed into
clustering/detection) affects cluster composition and detection agreement.

Only whole-session features are used here (extracted_data/fault_detection/
filtered_features.pkl, one row per EV) -- windowed features are out of scope
for this script; see kmeans.py / detection.py for the windowed tracks.

Population selection (SELECTION_MODE, set below):
  "all"    -- every EV.
  "faulty" -- confirmed-faulty EVs only (ev_id >= FAULT_EV_ID_CUTOFF).
  "class"  -- every confirmed-faulty EV (always included, regardless of
              anything else, so the faulty population is never subsampled
              away) PLUS every healthy EV the cached whole-fleet
              IsolationForest got wrong (its FP EVs -- see
              get_or_fit_isolation_forest) PLUS CLASS_N_RANDOM_HEALTHY more
              healthy EVs sampled at random (RANDOM_STATE-seeded, so
              reproducible run to run) from whichever healthy EVs are NOT
              already FP -- to test how clustering/detection behave as the
              healthy:faulty ratio shrinks, with the known-hard (FP) healthy
              EVs always present and a user-controlled number of "easy"
              healthy EVs mixed in alongside them.

IsolationForest fault detection is DELIBERATELY NOT re-fit per selection --
it is fit exactly ONCE, on ALL EVs' whole-session features, with
contamination = the actual labeled-faulty fraction (fault_label_all.mean(),
~0.15 -- NOT a hand-set knob), and the resulting per-EV anomaly scores/
predictions/TN-TP-FP-FN labels are cached to IF_CACHE_OUT (see
get_or_fit_isolation_forest). Every run of this script (any SELECTION_MODE,
any K) loads that same cached result and just subsets it down to whichever
population is selected -- so changing SELECTION_MODE/K only changes what
K-means clusters, never what IsolationForest was trained on or how it
scored any given EV. Delete IF_CACHE_OUT by hand if it ever needs to be
recomputed (e.g. filtered_features.pkl changes upstream).

For the selected population:
  1. K-means (k = K below, single fixed value -- no elbow/silhouette sweep;
     this script is for probing a specific k, not for picking one) on
     standardized whole-session features.
  2. The cached whole-fleet IsolationForest TN/TP/FP/FN labels (see above),
     subset down to the selected population -- not recomputed.

Ground truth fault label: EV number >= 335 is faulty, <= 334 is healthy --
same convention used throughout 01_fault_detection/ and 02_early_warning/.

Input:
  extracted_data/fault_detection/filtered_features.pkl -- one row per EV
      (index = ev_id), one column per fault-relevant scalar feature.
  extracted_data/vork.csv -- PCA (pca_1-3) and UMAP (umap_1-3) coordinates
      per EV (all 393 EVs), from before_kmeans.py. Reused as-is (NOT
      recomputed here), subset down to the selected population, same
      convention as kmeans.py.

Output:
  extracted_data/early_warning/wisek_isolation_forest_whole_session.pkl --
      cached whole-fleet IsolationForest fit + per-EV results (see above).
      Produced once, on the first run; reused (not recomputed) afterwards.
  figures/wisek/cluster_confusion_bars_<selection>_k<K>.png -- stacked bar
      chart, one bar per cluster, stacked by TN/TP/FP/FN count.
  figures/wisek/pca_umap_by_cluster_<selection>_k<K>.png -- PCA + UMAP
      scatter subplots, colored by cluster, with FP/FN EVs drawn bolder.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/02_.../wisek.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
EMBEDDINGS_IN = PROJECT_ROOT / "extracted_data" / "vork.csv"  # PCA/UMAP coords, from before_kmeans.py
IF_CACHE_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "wisek_isolation_forest_whole_session.pkl"
FIGURES_DIR = Path(__file__).resolve().parent / "figures" / "wisek"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as kmeans.py / detection.py

SELECTION_MODE = "faulty"  # "all" | "faulty" | "class" -- see module docstring
CLASS_N_RANDOM_HEALTHY = 17  # only used when SELECTION_MODE == "class": # of healthy EVs randomly added
                              # on top of the (always-included) FP EVs -- see select_population

K = 6  # number of K-means clusters -- set by hand, no sweep in this script

# NOTE: there is deliberately no CONTAMINATION_RATIO knob here -- the
# IsolationForest is fit once on ALL EVs with contamination = the actual
# labeled-faulty fraction, cached, and reused unchanged by every
# SELECTION_MODE/K combination. See get_or_fit_isolation_forest / module
# docstring.

RANDOM_STATE = 0
N_INIT = 10        # K-means random restarts, explicit (not "auto") for reproducibility/reporting
N_ESTIMATORS = 100  # IsolationForest trees, same as detection.py

FIGURE_DPI = 300

CLUSTER_COLORS = RWTHColorScales.cycle  # qualitative palette for cluster identity
CONFUSION_COLORS = {  # TN/TP = correct (green/blue), FP/FN = wrong (orange/red, FN reddest -- missed fault)
    "TN": RWTHColorScales.green[0],
    "TP": RWTHColorScales.blue[0],
    "FP": RWTHColorScales.orange[0],
    "FN": RWTHColorScales.red[0],
}
CONFUSION_ORDER = ["TN", "TP", "FP", "FN"]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def select_population(
    features_all: pd.DataFrame, fault_label_all: pd.Series, if_results_all: pd.DataFrame, mode: str,
) -> pd.Index:
    """Return the ev_id index to cluster/detect on, per SELECTION_MODE (see
    module docstring). "class" always keeps every confirmed-faulty EV and
    every FP EV (per the cached whole-fleet IsolationForest results),
    regardless of CLASS_N_RANDOM_HEALTHY, so neither is ever itself
    subsampled away -- only the extra "easy" healthy EVs mixed in on top are
    randomly varied, to test class-imbalance sensitivity.
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
    """Elementwise TN/TP/FP/FN label, vectorized version of the TP/TN/FP/FN
    tally in detection.py's PLAYGROUND section."""
    labels = np.full(len(predicted_fault), "", dtype=object)
    labels[predicted_fault & actual_fault] = "TP"
    labels[~predicted_fault & ~actual_fault] = "TN"
    labels[predicted_fault & ~actual_fault] = "FP"
    labels[~predicted_fault & actual_fault] = "FN"
    return labels


def get_or_fit_isolation_forest(features_all: pd.DataFrame, fault_label_all: pd.Series) -> pd.DataFrame:
    """Fit-once, cache-forever whole-fleet IsolationForest (see module
    docstring): if IF_CACHE_OUT already exists, its saved per-EV results are
    loaded and returned as-is -- NOT recomputed, even if K/SELECTION_MODE
    change from run to run. Only the first-ever run (or a run after the
    cache file has been deleted by hand) actually fits the model.

    Fit on ALL EVs' (unscaled) whole-session features -- same
    detect_faults() convention as detection.py -- with contamination set to
    the actual labeled-faulty fraction (fault_label_all.mean(), ~0.15) as
    the top-quantile anomaly-score cutoff, rather than a hand-picked knob.

    Returns a DataFrame indexed by ev_id (all EVs in features_all), columns
    anomaly_score / predicted_fault / confusion.
    """
    if IF_CACHE_OUT.exists():
        cached = pd.read_pickle(IF_CACHE_OUT)
        print(f"loaded cached whole-fleet IsolationForest results from {IF_CACHE_OUT} "
              f"(delete this file and rerun to recompute)")
        return cached["results"]

    contamination_ratio = float(fault_label_all.mean())
    if_model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    if_model.fit(features_all)
    anomaly_score = -if_model.score_samples(features_all)  # score_samples: higher = more normal, so negate
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


def plot_cluster_confusion_bars(results: pd.DataFrame, title: str, out_path: Path) -> None:
    """Stacked bar chart: one bar per cluster, height = cluster size, stacked
    by TN/TP/FP/FN count -- shows both how big each cluster is and how well
    IsolationForest agrees with ground truth within it."""
    counts = results.groupby(["cluster", "confusion"]).size().unstack(fill_value=0)
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


def plot_embeddings_by_cluster(
    embeddings: pd.DataFrame, results: pd.DataFrame, excluded_embeddings: pd.DataFrame,
    title: str, out_path: Path,
) -> None:
    """PCA + UMAP scatter subplots, colored by cluster. FP/FN EVs (where
    IsolationForest disagrees with ground truth) are drawn larger, fully
    opaque, and outlined in black so they stand out against the faded
    TP/TN points -- same cluster color, just bolder/brighter. FN and FP are
    additionally distinguished from each other by fill: FN (missed fault)
    is drawn FILLED (solid cluster color), FP (false alarm) is drawn
    UNFILLED (hollow -- cluster-colored outline only, white/transparent
    inside). EVs NOT part of this run's SELECTION_MODE population
    (excluded_embeddings) are drawn underneath everything else, tiny and
    faint gray, so their overall distribution is still visible but they
    never compete visually with the clustered/detected EVs."""
    clusters = results["cluster"].to_numpy()
    confusion = results["confusion"].to_numpy()
    cluster_ids = sorted(results["cluster"].unique())

    panels = [
        ("pca_1", "pca_2", "PC1", "PC2", "PCA"),
        ("umap_1", "umap_2", "UMAP-1", "UMAP-2", "UMAP"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    for ax, (xcol, ycol, xlabel, ylabel, name) in zip(axes, panels):
        coords = embeddings.loc[results.index, [xcol, ycol]].to_numpy()

        if len(excluded_embeddings) > 0:
            excluded_coords = excluded_embeddings[[xcol, ycol]].to_numpy()
            ax.scatter(excluded_coords[:, 0], excluded_coords[:, 1], c="lightgray", s=12, alpha=0.5,
                       edgecolor="none", zorder=1)

        for cluster in cluster_ids:
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
            base_mask = (clusters == cluster) & ~np.isin(confusion, ["FP", "FN"])
            ax.scatter(coords[base_mask, 0], coords[base_mask, 1], c=[color], s=35, alpha=0.4,
                       edgecolor="none", zorder=2)

        for cluster in cluster_ids:
            color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]

            fn_mask = (clusters == cluster) & (confusion == "FN")
            ax.scatter(coords[fn_mask, 0], coords[fn_mask, 1], c=[color], s=120, alpha=1.0,
                       edgecolor="black", linewidth=1.2, zorder=3)

            fp_mask = (clusters == cluster) & (confusion == "FP")
            ax.scatter(coords[fp_mask, 0], coords[fp_mask, 1], facecolor="none", s=120,
                       edgecolor=[color], linewidth=1.8, zorder=3)

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
                        markeredgecolor="black", markeredgewidth=1.4, markersize=10,
                        label="FN -- missed fault (filled)")
    fp_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                        markeredgecolor="dimgray", markeredgewidth=1.8, markersize=10,
                        label="FP -- false alarm (unfilled)")
    handles = cluster_handles + [fn_handle, fp_handle]
    if len(excluded_embeddings) > 0:
        excluded_handle = Line2D([0], [0], marker="o", linestyle="", markerfacecolor="lightgray",
                                  markeredgecolor="none", markersize=6, alpha=0.7,
                                  label="not in this run's population")
        handles.append(excluded_handle)
    axes[1].legend(handles=handles, loc="best", fontsize=8, title="legend")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load whole-session features, select population per SELECTION_MODE
features_all = pd.read_pickle(FEATURES_IN)

if features_all.isna().any().any():
    n_before = len(features_all)
    features_all = features_all.dropna()
    print(f"dropped {n_before - len(features_all)} EVs with missing feature values")

fault_label_all = pd.Series(features_all.index >= FAULT_EV_ID_CUTOFF, index=features_all.index, name="fault_label")

# %% 1b. whole-fleet IsolationForest -- fit once (or loaded from cache),
# never refit per selection/K (see module docstring + get_or_fit_isolation_forest)
if_results_all = get_or_fit_isolation_forest(features_all, fault_label_all)

selected_index = select_population(features_all, fault_label_all, if_results_all, SELECTION_MODE)
features = features_all.loc[selected_index]
actual_fault = fault_label_all.loc[selected_index]
if_results = if_results_all.loc[selected_index]

print(f"SELECTION_MODE={SELECTION_MODE!r}: {len(features)} EVs selected "
      f"({int(actual_fault.sum())} faulty, {int((~actual_fault).sum())} healthy, "
      f"of {len(features_all)} total)")

# %% 2. K-means on standardized whole-session features (same scaling
# convention as kmeans.py -- K-means is distance-based, so features on
# larger raw scales shouldn't dominate just by unit choice)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans_model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
cluster_label = pd.Series(kmeans_model.fit_predict(features_scaled), index=features.index, name="cluster")

if 2 <= K < len(features):
    print(f"K-means (k={K}) silhouette score: {silhouette_score(features_scaled, cluster_label):.3f}")

cluster_sizes = cluster_label.value_counts().sort_index()
print(f"\n--- Cluster sizes (k={K}) ---")
print(cluster_sizes.rename("n_evs").to_string())

# %% 3. combine into one results table: cluster + ground truth + the
# cached whole-fleet IsolationForest prediction/confusion label, subset to
# the selected population (NOT recomputed -- see get_or_fit_isolation_forest)
predicted_fault = if_results["predicted_fault"]
print(f"\nIsolationForest (cached, whole-fleet fit): {int(predicted_fault.sum())}/{len(predicted_fault)} "
      "EVs in the selected population flagged as faulty")

results = pd.DataFrame({
    "cluster": cluster_label,
    "actual_fault": actual_fault,
    "predicted_fault": predicted_fault,
    "confusion": if_results["confusion"],
})

confusion_totals = results["confusion"].value_counts().reindex(CONFUSION_ORDER, fill_value=0)
TP, TN, FP, FN = (int(confusion_totals[label]) for label in ["TP", "TN", "FP", "FN"])
precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

print(f"\n--- IsolationForest vs. ground truth, over selected population ---")
print(f"TP: {TP}  TN: {TN}  FP: {FP}  FN: {FN}")
print(f"Precision: {precision:.1%}   Recall: {recall:.1%}   F1: {f1:.1%}")

print(f"\n--- Per-cluster TN/TP/FP/FN composition (k={K}) ---")
print(results.groupby(["cluster", "confusion"]).size().unstack(fill_value=0)
      .reindex(columns=CONFUSION_ORDER, fill_value=0).to_string())

# %% 5. figures
run_tag = f"{SELECTION_MODE}_k{K}"
title_suffix = f"selection={SELECTION_MODE}, k={K}, n={len(features)} EVs"

plot_cluster_confusion_bars(
    results,
    title=f"Cluster composition vs. IsolationForest detection ({title_suffix})",
    out_path=FIGURES_DIR / f"cluster_confusion_bars_{run_tag}.png",
)
print(f"\nsaved {FIGURES_DIR / f'cluster_confusion_bars_{run_tag}.png'}")

embeddings_all = pd.read_csv(EMBEDDINGS_IN, index_col="ev_id")
embeddings = embeddings_all.loc[features.index]
excluded_index = features_all.index.difference(selected_index)
excluded_embeddings = embeddings_all.loc[excluded_index.intersection(embeddings_all.index)]

plot_embeddings_by_cluster(
    embeddings, results, excluded_embeddings,
    title=f"PCA / UMAP by cluster, FP & FN bolded ({title_suffix})",
    out_path=FIGURES_DIR / f"pca_umap_by_cluster_{run_tag}.png",
)
print(f"saved {FIGURES_DIR / f'pca_umap_by_cluster_{run_tag}.png'}")
