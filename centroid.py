# %%
"""
Nearest-centroid fault-subtype diagnosis.

Continues branch 2 of the pipeline (kmeans.py -> this script): kmeans.py
found that faulty EVs split into k fault subtypes (clusters); this script
assigns EVERY EV detection.py flags as faulty (is_fault == True in
fault_scores.pkl) to its nearest subtype centroid, whether or not that EV
was part of kmeans.py's original clustered population.

--- Reproducing kmeans.py's centroids exactly, not approximately ----------
kmeans.py clusters in STANDARDIZED space (StandardScaler fit on the faulty
subset) but saves RAW feature values to extracted_data/kmeans.csv. To place
a NEW point in that same space:
  1. refit a StandardScaler on kmeans.csv's own raw feature columns (the
     exact population kmeans.py standardized against) -- reproduces
     kmeans.py's scaler bit-for-bit, since it's the same data + same
     sklearn default (ddof=0) transform.
  2. take the per-cluster mean of kmeans.csv's rows in that standardized
     space -- this reproduces KMeans.cluster_centers_ from kmeans.py
     exactly (same data, same scaling, and a cluster's centroid IS the
     within-cluster mean by construction), without needing kmeans.py to
     have pickled its fitted model.
  3. standardize each EV being diagnosed with the SAME scaler from step 1,
     then assign to the nearest of the centroids from step 2 (Euclidean
     distance).

--- Why this script REQUIRES kmeans.csv and filtered_features.pkl to agree
on their feature columns -------------------------------------------------
features.py's screening changes as upstream feature work continues (see its
own docstring), and kmeans.csv freezes whatever feature set was current the
last time kmeans.py was run with FINAL_K set. If those two have drifted
apart, standardizing a "new" EV's filtered_features.pkl row against a
scaler fit on kmeans.csv's (possibly different) columns would silently
compare EVs in an ill-defined, partially-mismatched feature space. Rather
than quietly falling back to whatever columns happen to overlap, this
script ASSERTS the column sets are identical and raises a clear error
telling you to re-run kmeans.py (with FINAL_K set) against the current
filtered_features.pkl first. This was a deliberate choice: kmeans.csv on
disk may be stale relative to filtered_features.pkl at any given time, and
staying loud about that beats a silently-degraded diagnosis.

--- Soft assignment / "confidence" -----------------------------------------
Besides the single nearest cluster, every diagnosed EV also gets a full
distribution over all k clusters via softmax(-distance / SOFTMAX_TEMPERATURE).
This is a DISTANCE-based heuristic, not a calibrated probability from a
generative model (it ignores per-cluster covariance/density -- a Gaussian
Mixture Model posterior would account for that, at the cost of needing
enough points per cluster to estimate a covariance; with cluster sizes as
small as ~6 here, that's a real cost, which is why this script uses the
simpler distance-based heuristic instead and names it "confidence" rather
than "probability"). Lower SOFTMAX_TEMPERATURE sharpens the distribution
toward a hard assignment; higher flattens it toward uniform. An even
simpler alternative (not used here) is plain inverse-distance weighting
(w_j = 1/d_j, normalized to sum to 1) -- monotonically similar to softmax
but without a temperature knob to pick; softmax is used here since
sliding_windows.py reuses this exact mechanism per-window and benefits from
the temperature knob to tune how sharply windowed probabilities react to
small distance changes.

Input:
  extracted_data/kmeans.csv (kmeans.py) -- faulty EVs (ground-truth id>=335)
      with their cluster label and the raw feature values they were
      clustered on.
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      current feature matrix for ALL EVs.
  extracted_data/fault_detection/fault_scores.pkl (detection.py) -- is_fault
      flag, used to pick which EVs to diagnose.
  extracted_data/vork.csv (before_kmeans.py) -- PCA/UMAP coordinates, reused
      as-is for the diagnosis scatter plot.
Output:
  extracted_data/early_warning/cluster_centroids.pkl -- {feature_columns,
      scaler_mean, scaler_scale, centroids (k x n_features, standardized
      space), cluster_ids} -- reused by sliding_windows.py so it doesn't
      need to re-derive centroids from kmeans.csv itself.
  extracted_data/early_warning/centroid_diagnosis.csv -- one row per
      flagged EV: assigned_cluster, distance_to_assigned,
      confidence_assigned, prob_cluster_{j} for every cluster,
      already_clustered (was this EV part of kmeans.csv's original
      population), kmeans_cluster (its original label, if already_clustered),
      agrees_with_kmeans (only meaningful when already_clustered).
  experiments/02_early_warning/figures/centroid_diagnosis_pca.png
  experiments/02_early_warning/figures/centroid_diagnosis_umap.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG
# ===========================================================================
EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
KMEANS_IN = EXTRACTED_DATA / "kmeans.csv"  # from kmeans.py -- MUST be current, see module docstring
FEATURES_IN = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"  # from features.py
FAULT_SCORES_IN = EXTRACTED_DATA / "fault_detection" / "fault_scores.pkl"  # from detection.py
EMBEDDINGS_IN = EXTRACTED_DATA / "vork.csv"  # PCA/UMAP coords, from before_kmeans.py

OUT_DIR = EXTRACTED_DATA / "early_warning"
OUT_DIR.mkdir(exist_ok=True)
CENTROIDS_OUT = OUT_DIR / "cluster_centroids.pkl"
DIAGNOSIS_OUT = OUT_DIR / "centroid_diagnosis.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

SOFTMAX_TEMPERATURE = 1.0  # see module docstring -- lower = sharper/more confident assignments

FIGURE_DPI = 300
CLUSTER_COLORS = RWTHColorScales.cycle

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def fit_reference_scaler_and_centroids(
    kmeans_table: pd.DataFrame, feature_columns: list[str],
) -> tuple[StandardScaler, pd.DataFrame]:
    """Refit the StandardScaler kmeans.py implicitly used (same data, same
    default ddof=0 transform) and compute each cluster's centroid as the
    per-cluster mean in that standardized space -- see module docstring for
    why this reproduces kmeans.py's KMeans.cluster_centers_ exactly.
    Returns (scaler, centroids) where centroids is indexed by cluster id,
    columns = feature_columns, in standardized units.
    """
    raw = kmeans_table[feature_columns]
    scaler = StandardScaler().fit(raw)
    standardized = pd.DataFrame(scaler.transform(raw), index=raw.index, columns=feature_columns)
    standardized["cluster"] = kmeans_table["cluster"].to_numpy()
    centroids = standardized.groupby("cluster")[feature_columns].mean()
    return scaler, centroids


def assign_nearest_centroid(
    standardized_query: np.ndarray, centroids: np.ndarray, temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euclidean distance from every query row to every centroid, the
    nearest cluster per row, and a softmax(-distance / temperature)
    confidence distribution over all clusters per row.
    Returns (distances [n_query, k], assigned_cluster [n_query], probs [n_query, k]).
    """
    distances = np.linalg.norm(standardized_query[:, None, :] - centroids[None, :, :], axis=2)
    assigned_cluster = distances.argmin(axis=1)

    scaled_neg_dist = -distances / temperature
    scaled_neg_dist -= scaled_neg_dist.max(axis=1, keepdims=True)  # numerical stability, doesn't change softmax output
    exp_scores = np.exp(scaled_neg_dist)
    probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    return distances, assigned_cluster, probs


def plot_diagnosis_embedding(
    coords: np.ndarray, assigned_cluster: np.ndarray, already_clustered: np.ndarray,
    xlabel: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """2D scatter colored by assigned cluster, with already-clustered EVs
    (part of kmeans.py's original population) shown as filled circles and
    newly diagnosed EVs (detection.py flags outside that population) shown
    as open triangles -- same color encoding as kmeans.py's cluster plots,
    extended with a marker distinction for "known" vs. "new" diagnosis.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(np.unique(assigned_cluster)):
        color = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        cluster_mask = assigned_cluster == cluster

        known_mask = cluster_mask & already_clustered
        ax.scatter(coords[known_mask, 0], coords[known_mask, 1], c=[color], marker="o",
                   s=45, edgecolor="white", linewidth=0.4, alpha=0.9, label=f"cluster {cluster} (known)", zorder=2)

        new_mask = cluster_mask & ~already_clustered
        ax.scatter(coords[new_mask, 0], coords[new_mask, 1], c="none", marker="^",
                   s=60, edgecolor=color, linewidth=1.4, alpha=0.9, label=f"cluster {cluster} (new)", zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    ax.legend(loc="best", fontsize=8, title="assigned cluster", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load inputs, assert kmeans.csv and filtered_features.pkl agree on
# their feature columns (see module docstring -- this is intentionally a
# hard failure, not a silent fallback to the intersection)
kmeans_table = pd.read_csv(KMEANS_IN, index_col="ev_id")
features = pd.read_pickle(FEATURES_IN)
if features.isna().any().any():
    n_before = len(features)
    features = features.dropna()
    print(f"dropped {n_before - len(features)} EVs with missing feature values")

kmeans_feature_columns = set(kmeans_table.columns) - {"cluster"}
current_feature_columns = set(features.columns)
if kmeans_feature_columns != current_feature_columns:
    only_in_kmeans = sorted(kmeans_feature_columns - current_feature_columns)
    only_in_current = sorted(current_feature_columns - kmeans_feature_columns)
    raise RuntimeError(
        "kmeans.csv's feature columns don't match filtered_features.pkl's current columns -- "
        "kmeans.csv is stale relative to today's features.py output.\n"
        f"  only in kmeans.csv (stale):        {only_in_kmeans}\n"
        f"  only in filtered_features.pkl now: {only_in_current}\n"
        "Re-run kmeans.py (with FINAL_K set) against the current filtered_features.pkl, "
        "which will overwrite kmeans.csv with a matching feature set, then re-run this script."
    )
feature_columns = features.columns.tolist()  # canonical column order from here on
print(f"kmeans.csv and filtered_features.pkl agree on {len(feature_columns)} feature columns")

fault_scores = pd.read_pickle(FAULT_SCORES_IN).loc[features.index]
flagged_ev_ids = fault_scores.loc[fault_scores["is_fault"]].index

# %% 2. reproduce kmeans.py's centroids exactly (see module docstring)
scaler, centroids = fit_reference_scaler_and_centroids(kmeans_table, feature_columns)
k = len(centroids)
print(f"reproduced {k} cluster centroids from kmeans.csv ({len(kmeans_table)} EVs used for clustering)")

# %% 3. standardize every flagged EV with the SAME scaler, assign nearest centroid + confidence
query_features = features.loc[flagged_ev_ids, feature_columns]
standardized_query = scaler.transform(query_features)
distances, assigned_cluster, probs = assign_nearest_centroid(
    standardized_query, centroids.to_numpy(), SOFTMAX_TEMPERATURE,
)

# %% 4. assemble diagnosis table, marking which EVs were already clustered
already_clustered = query_features.index.isin(kmeans_table.index)
diagnosis = pd.DataFrame({
    "anomaly_score": fault_scores.loc[flagged_ev_ids, "anomaly_score"],
    "assigned_cluster": assigned_cluster,
    "distance_to_assigned": distances[np.arange(len(distances)), assigned_cluster],
    "confidence_assigned": probs[np.arange(len(probs)), assigned_cluster],
    "already_clustered": already_clustered,
}, index=query_features.index)

for cluster in centroids.index:
    diagnosis[f"prob_cluster_{cluster}"] = probs[:, cluster]

diagnosis["kmeans_cluster"] = kmeans_table["cluster"].reindex(query_features.index)
diagnosis["agrees_with_kmeans"] = np.where(
    already_clustered, diagnosis["assigned_cluster"] == diagnosis["kmeans_cluster"], np.nan,
)

diagnosis = diagnosis.sort_values("anomaly_score", ascending=False)
diagnosis.to_csv(DIAGNOSIS_OUT)
print(f"\nsaved nearest-centroid diagnosis for {len(diagnosis)} flagged EVs to {DIAGNOSIS_OUT}")

# %% 5. save the scaler + centroids for reuse by sliding_windows.py
pd.to_pickle({
    "feature_columns": feature_columns,
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "centroids": centroids.to_numpy(),
    "cluster_ids": centroids.index.to_numpy(),
    "softmax_temperature": SOFTMAX_TEMPERATURE,
}, CENTROIDS_OUT)
print(f"saved scaler + centroids to {CENTROIDS_OUT}")

# %% 6. sanity check -- for EVs that WERE part of kmeans.py's original
# clustering, does nearest-centroid assignment recover their own known
# cluster label? Not guaranteed to be 100% (a point can end up nearer a
# different cluster's mean than its own, especially near a cluster
# boundary), but should be high if the reproduction in step 2 is correct.
known_subset = diagnosis[diagnosis["already_clustered"]]
agreement_rate = known_subset["agrees_with_kmeans"].mean()
print(f"\n--- Sanity check: nearest-centroid vs. kmeans.py's own label ---")
print(f"{known_subset['agrees_with_kmeans'].sum():.0f}/{len(known_subset)} "
      f"({agreement_rate:.1%}) of the already-clustered flagged EVs are assigned back to their own cluster")

n_new = len(diagnosis) - len(known_subset)
print(f"\n{n_new} newly diagnosed EV(s) (flagged by detection.py, not part of kmeans.py's original population)")
print(diagnosis.loc[~diagnosis["already_clustered"], ["anomaly_score", "assigned_cluster", "confidence_assigned"]].to_string())

# %% 7. diagnosis scatter plots, reusing before_kmeans.py's saved embedding
embeddings = pd.read_csv(EMBEDDINGS_IN, index_col="ev_id").loc[diagnosis.index]
already_clustered_arr = diagnosis["already_clustered"].to_numpy()
assigned_cluster_arr = diagnosis["assigned_cluster"].to_numpy()

plot_diagnosis_embedding(
    embeddings[["pca_1", "pca_2"]].to_numpy(), assigned_cluster_arr, already_clustered_arr,
    xlabel="PC1", ylabel="PC2",
    title="Nearest-centroid diagnosis of flagged EVs (PCA)\nfilled = part of kmeans.py's population, open = newly diagnosed",
    out_path=FIGURES_DIR / "centroid_diagnosis_pca.png",
)
plot_diagnosis_embedding(
    embeddings[["umap_1", "umap_2"]].to_numpy(), assigned_cluster_arr, already_clustered_arr,
    xlabel="UMAP-1", ylabel="UMAP-2",
    title="Nearest-centroid diagnosis of flagged EVs (UMAP)\nfilled = part of kmeans.py's population, open = newly diagnosed",
    out_path=FIGURES_DIR / "centroid_diagnosis_umap.png",
)
print(f"\nsaved diagnosis scatter plots to {FIGURES_DIR}")


# ===========================================================================
# WINDOW-SCALE NEAREST-CENTROID -- separate, additive track ================
# ===========================================================================
# This section only ever computes distances to the fixed, pre-fit centroids
# saved by kmeans.py's windowed track (kmeans_centroids_windowed.pkl). It
# must NEVER call .fit() on any clustering object, import KMeans, or refit
# the scaler -- all fitting for the windowed track happens exclusively in
# kmeans.py. Kept structurally separate from the whole-session logic above,
# which it does not read from, modify, or depend on.

WINDOWED_CENTROIDS_IN = EXTRACTED_DATA / "early_warning" / "kmeans_centroids_windowed.pkl"  # from kmeans.py's windowed track

# Loaded once, at import time, not per-call (sliding_windows.py's windowed
# counterpart will call assign_nearest_centroid_windowed() in a loop). Guarded
# with try/except rather than a bare read: this file's whole-session logic
# above must keep working even before kmeans.py's windowed track has been run
# once (i.e. before kmeans_centroids_windowed.pkl exists) -- the artifact is
# only required once assign_nearest_centroid_windowed() is actually called.
try:
    _windowed_artifact = pd.read_pickle(WINDOWED_CENTROIDS_IN)
except FileNotFoundError:
    _windowed_artifact = None
    print(f"\nnote: {WINDOWED_CENTROIDS_IN} not found yet -- run kmeans.py's windowed track "
          "(set FINAL_K_WINDOWED) before calling assign_nearest_centroid_windowed()")

if _windowed_artifact is not None:
    _windowed_feature_columns = _windowed_artifact["feature_columns"]
    _windowed_scaler = _windowed_artifact["scaler"]
    _windowed_centroids = _windowed_artifact["centroids"]  # DataFrame, index=cluster id, columns=feature_columns
    _windowed_centroids_matrix = _windowed_centroids.to_numpy()
    _windowed_cluster_ids = _windowed_centroids.index.to_numpy()
    _windowed_healthy_reference = _windowed_artifact["healthy_reference"].to_numpy()  # standardized space


def assign_nearest_centroid_windowed(x_raw: np.ndarray) -> dict:
    """One window's raw (unscaled) filtered feature vector, in the pinned
    _windowed_feature_columns order from kmeans_centroids_windowed.pkl.
    Transforms with the loaded scaler (never refit) and returns distances
    to every windowed fault-subtype centroid plus the windowed healthy
    reference vector -- pure distance computation, no fitting.
    """
    if _windowed_artifact is None:
        raise RuntimeError(
            f"{WINDOWED_CENTROIDS_IN} not found -- run kmeans.py's windowed track (set "
            "FINAL_K_WINDOWED) to produce it before calling this function."
        )

    x_scaled = _windowed_scaler.transform(np.asarray(x_raw, dtype=float).reshape(1, -1))[0]
    distances_to_clusters = np.linalg.norm(_windowed_centroids_matrix - x_scaled[None, :], axis=1)
    nearest_idx = int(distances_to_clusters.argmin())
    distance_to_healthy = float(np.linalg.norm(x_scaled - _windowed_healthy_reference))

    return {
        "nearest_cluster": int(_windowed_cluster_ids[nearest_idx]),
        "distance_to_nearest_cluster": float(distances_to_clusters[nearest_idx]),
        "distance_to_healthy": distance_to_healthy,
        "distances_to_all_clusters": {
            int(cluster_id): float(dist) for cluster_id, dist in zip(_windowed_cluster_ids, distances_to_clusters)
        },
    }


# ===========================================================================
# WINDOW-SCALE NEAREST-CENTROID, UNSUPERVISED VARIANT -- separate, additive =
# ===========================================================================
# Same pure-distance-computation contract as assign_nearest_centroid_windowed
# above, but against kmeans.py's W4 artifact (kmeans_windowed_unrestricted_
# diagnostic.pkl): K-means fit on ALL windows (healthy + faulty, unfiltered
# by label), not just confirmed-faulty ones. kmeans.py's own docstring flags
# that artifact as diagnostic-only / "NOT used anywhere in the early-warning
# system itself" -- sliding_windows.py's CLUSTER_SOURCE = "unsupervised" is
# what actually exercises it as an alternative early-warning signal, so
# treat its precision/recall as unconfirmed relative to the "faulty_only"
# track above until validated. Must NEVER call .fit() on anything here --
# all fitting happens exclusively in kmeans.py.

WINDOWED_UNSUPERVISED_CENTROIDS_IN = EXTRACTED_DATA / "early_warning" / "kmeans_windowed_unrestricted_diagnostic.pkl"  # from kmeans.py's windowed track (W4)

try:
    _windowed_unsupervised_artifact = pd.read_pickle(WINDOWED_UNSUPERVISED_CENTROIDS_IN)
except FileNotFoundError:
    _windowed_unsupervised_artifact = None
    print(f"\nnote: {WINDOWED_UNSUPERVISED_CENTROIDS_IN} not found yet -- run kmeans.py's windowed track "
          "(set FINAL_K_WINDOWED) before calling assign_nearest_centroid_windowed_unsupervised()")

if _windowed_unsupervised_artifact is not None:
    _windowed_unsupervised_feature_columns = _windowed_unsupervised_artifact["feature_columns"]
    _windowed_unsupervised_scaler = _windowed_unsupervised_artifact["scaler"]
    _windowed_unsupervised_centers = _windowed_unsupervised_artifact["cluster_centers"]  # k x n_features, standardized space
    _windowed_unsupervised_cluster_ids = np.arange(len(_windowed_unsupervised_centers))


def assign_nearest_centroid_windowed_unsupervised(x_raw: np.ndarray) -> dict:
    """Same mechanism as assign_nearest_centroid_windowed above, but against
    the UNRESTRICTED windowed K-means fit (kmeans_windowed_unrestricted_
    diagnostic.pkl) instead of the label-restricted one -- see the module
    banner just above for why this is a separate, less-validated signal.

    x_raw: one window's raw (unscaled) feature vector, in
        _windowed_unsupervised_feature_columns order (same order as
        _windowed_feature_columns above -- both ultimately come from
        filtered_features.pkl's column order, checked identical when this
        was written, but this function uses its own artifact's saved order
        rather than assuming the two agree).

    Returns {"nearest_cluster", "distance_to_nearest_cluster",
    "distances_to_all_clusters"} -- no "distance_to_healthy" key, since the
    W4 artifact saves no healthy-reference vector (that's only computed for
    the label-restricted track above).
    """
    if _windowed_unsupervised_artifact is None:
        raise RuntimeError(
            f"{WINDOWED_UNSUPERVISED_CENTROIDS_IN} not found -- run kmeans.py's windowed track (set "
            "FINAL_K_WINDOWED) to produce it before calling this function."
        )

    x_scaled = _windowed_unsupervised_scaler.transform(np.asarray(x_raw, dtype=float).reshape(1, -1))[0]
    distances_to_clusters = np.linalg.norm(_windowed_unsupervised_centers - x_scaled[None, :], axis=1)
    nearest_idx = int(distances_to_clusters.argmin())

    return {
        "nearest_cluster": int(_windowed_unsupervised_cluster_ids[nearest_idx]),
        "distance_to_nearest_cluster": float(distances_to_clusters[nearest_idx]),
        "distances_to_all_clusters": {
            int(cluster_id): float(dist) for cluster_id, dist in zip(_windowed_unsupervised_cluster_ids, distances_to_clusters)
        },
    }
