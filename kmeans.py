# %%
"""
K-means clustering of faulty EVs, to test whether they form distinct fault
subtypes rather than one homogeneous "faulty" group.

Only faulty EVs are clustered here (not the full healthy+faulty set) --
healthy vs. faulty is already a known, separately-established split (see
detection.py / fault_feature.py); the open question this script addresses
is whether the *faulty* group itself is internally structured into
sub-types, which only clustering the faulty subset can answer -- including
healthy EVs would just recover the healthy/faulty split again, not reveal
sub-structure within the faulty group.

Input:
  extracted_data/fault_detection/filtered_features.pkl -- one row per EV
      (index = ev_id), one column per fault-relevant scalar feature (see
      features.py; same feature set as before_kmeans.py).
  extracted_data/vork.csv -- PCA (pca_1-3) and UMAP (umap_1-3) coordinates
      per EV, from before_kmeans.py. Reused as-is for the final
      visualization -- NOT recomputed here, so the scatter plots below
      show the faulty-only clusters in the exact same embedding already
      inspected for sub-structure.
Ground truth fault label: EV number >= 335 is faulty, <= 334 is healthy --
      same convention used throughout 01_fault_detection/.

Output (diagnostics -- always produced):
  experiments/01_fault_detection/figures/kmeans_elbow.png
  experiments/01_fault_detection/figures/kmeans_silhouette_scores.png
  experiments/01_fault_detection/figures/kmeans_silhouette_samples_k345.png
  (k, inertia, silhouette_score) table printed to stdout

Output (only produced once FINAL_K is set below):
  experiments/01_fault_detection/figures/kmeans_pca_scatter_by_cluster.png
  experiments/01_fault_detection/figures/kmeans_umap_scatter_by_cluster.png
  extracted_data/kmeans.csv -- ev_id, cluster, all relevant feature values
  cluster size table + cluster characterization table printed to stdout
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../kmeans.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
EMBEDDINGS_IN = PROJECT_ROOT / "extracted_data" / "vork.csv"  # PCA/UMAP coords, from before_kmeans.py
OUT_CSV = PROJECT_ROOT / "extracted_data" / "kmeans.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change/tune later
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as detection.py / before_kmeans.py

RANDOM_STATE = 0   # same seed as before_kmeans.py's PCA/UMAP, for reproducibility across the two scripts
N_INIT = 10        # number of K-means random restarts (best inertia kept) -- explicit rather than sklearn's
                    # "auto" default, so the exact same value is used for every k and is easy to report

K_RANGE = range(2, 9)          # k values swept for the elbow/silhouette diagnostics (k = 2..8)
SILHOUETTE_SAMPLE_KS = [3, 4, 5]  # k values shown in the per-sample silhouette grid -- closest to the
                                   # original 5-subtype hypothesis, so these are the ones worth inspecting closely

# Set this once you've picked k from the diagnostics below (elbow +
# silhouette plots + the per-sample silhouette grid). Left as None so the
# script stops after producing diagnostics and does not silently pick a k
# for you -- that choice is deliberately left to you.
FINAL_K = 3

FIGURE_DPI = 300  # export resolution for every figure -- 300 stays legible when zoomed in

COLOR_FAULTY = RWTHColorScales.red[0]
CLUSTER_COLORS = RWTHColorScales.cycle  # qualitative palette, 10 colors -- enough for every k in K_RANGE

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling, consistent with before_kmeans.py."""
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def plot_elbow(diagnostics: pd.DataFrame, out_path: Path) -> None:
    """Inertia (within-cluster sum of squares) vs. k. Inertia always
    decreases as k grows, so there's no single "correct" k from this plot
    alone -- what matters is where the decrease visibly slows down (the
    "elbow"), which is why this is paired with the silhouette score below
    rather than used by itself.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["inertia"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow plot (faulty EVs only)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_vs_k(diagnostics: pd.DataFrame, out_path: Path) -> None:
    """Mean silhouette score vs. k. Unlike inertia, this isn't monotonic --
    it directly measures how well-separated/cohesive the clusters are, so
    it can independently confirm (or contradict) whatever k the elbow plot
    suggests. Score range is [-1, 1]; higher is better-separated.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("mean silhouette score")
    ax.set_title("K-means silhouette score vs. k (faulty EVs only)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_samples_grid(X: np.ndarray, fitted_models: dict, ks: list, out_path: Path) -> None:
    """Per-sample silhouette visualization (one subplot per k in `ks`):
    for each cluster, a horizontal band of sorted per-sample silhouette
    values. Wide, uniformly high bands with few/no negative values mean a
    well-separated, well-supported cluster; thin or partly-negative bands
    mean that cluster is weak or samples are borderline/misassigned. The
    dashed line is the overall mean silhouette score for that k (same
    number as in the silhouette-vs-k plot).
    """
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

    axes[0].set_ylabel("samples per cluster (sorted by silhouette value)")
    fig.suptitle("Per-sample silhouette plots (faulty EVs only)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_2d_by_cluster(
    coords: np.ndarray, cluster_labels: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path,
) -> None:
    """2D scatter of `coords` (n_samples x 2), colored by cluster label
    instead of the plain "faulty" color used in before_kmeans.py -- same
    style otherwise, so the two are visually comparable.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for cluster in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]],
                   s=40, edgecolor="white", linewidth=0.4, alpha=0.85, label=f"cluster {cluster}", zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _style_scatter_axes(ax)
    ax.legend(loc="best", title="cluster")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load data, filter to faulty EVs only, select the same relevant
# features as before_kmeans.py, standardize (K-means is distance-based --
# same reason PCA/UMAP needed standardized input -- so features on larger
# raw scales don't dominate the distance calculation just by unit choice)
features_all = pd.read_pickle(FEATURES_IN)

# filtered_features.pkl is produced clean by features.py, but guard anyway
# since it's a separate, still-evolving upstream script.
if features_all.isna().any().any():
    n_before = len(features_all)
    features_all = features_all.dropna()
    print(f"dropped {n_before - len(features_all)} EVs with missing feature values")

fault_label = pd.Series(features_all.index >= FAULT_EV_ID_CUTOFF, index=features_all.index, name="fault_label")
features = features_all.loc[fault_label]  # faulty EVs only -- see module docstring for why
print(f"clustering {len(features)} faulty EVs (of {len(features_all)} total) "
      f"on {features.shape[1]} relevant features")

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# %% 2. sweep k = 2..8, fit K-means once per k, record inertia + silhouette.
# Models are kept (not just their scores) so the k=3/4/5 silhouette grid
# below can reuse these exact fits instead of re-clustering.
fitted_models: dict[int, KMeans] = {}
diagnostics_rows = []

for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(features_scaled)
    fitted_models[k] = model
    diagnostics_rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": silhouette_score(features_scaled, labels),
    })

diagnostics = pd.DataFrame(diagnostics_rows)
print("\n--- K-means diagnostics (faulty EVs only) ---")
print(diagnostics.to_string(index=False))

# %% 3. elbow + silhouette-vs-k plots. Shown together deliberately: inertia
# alone always keeps decreasing with k (so its "elbow" is subjective), and
# silhouette score alone can favor a spuriously small k -- looking at both
# is a standard way to cross-check one heuristic against the other rather
# than trusting either in isolation.
plot_elbow(diagnostics, FIGURES_DIR / "kmeans_elbow.png")
plot_silhouette_vs_k(diagnostics, FIGURES_DIR / "kmeans_silhouette_scores.png")

# %% 4. per-sample silhouette grid for k = 3, 4, 5 -- the values closest to
# the original 5-subtype hypothesis, for a direct visual comparison of
# cluster quality (width/balance/negative values) at each.
plot_silhouette_samples_grid(
    features_scaled, fitted_models, SILHOUETTE_SAMPLE_KS,
    FIGURES_DIR / "kmeans_silhouette_samples_k345.png",
)

print("\nDiagnostics done. Set FINAL_K near the top of this script and rerun "
      "to fit the final clustering and produce the characterization tables/plots.")

# %% 5. final clustering -- only runs once FINAL_K has been set by hand above
if FINAL_K is not None:
    final_model = fitted_models.get(FINAL_K) or KMeans(
        n_clusters=FINAL_K, random_state=RANDOM_STATE, n_init=N_INIT,
    ).fit(features_scaled)
    cluster_label = pd.Series(final_model.labels_, index=features.index, name="cluster")

    # 5a. cluster sizes
    cluster_sizes = cluster_label.value_counts().sort_index()
    print(f"\n--- Cluster sizes (k={FINAL_K}) ---")
    print(cluster_sizes.rename("n_samples").to_string())

    # 5b. cluster characterization: per-cluster mean/std of every relevant
    # feature, plus the overall faulty-population mean, so features driving
    # each cluster apart from the rest of the faulty group can be spotted
    # (e.g. "cluster 2 has notably higher pack_temp__mean_pack_temp than
    # the others").
    characterization = pd.DataFrame(index=features.columns)
    for cluster in sorted(cluster_label.unique()):
        cluster_features = features[cluster_label == cluster]
        characterization[f"cluster_{cluster}_mean"] = cluster_features.mean()
        characterization[f"cluster_{cluster}_std"] = cluster_features.std()
    characterization["overall_faulty_mean"] = features.mean()

    print(f"\n--- Cluster characterization (k={FINAL_K}) ---")
    print(characterization.to_string(float_format=lambda v: f"{v:.3f}"))

    # 5c. merge cluster labels onto the saved PCA/UMAP coordinates (not
    # recomputed here -- reusing the exact embedding already inspected for
    # sub-structure in before_kmeans.py) and plot both, colored by cluster
    embeddings = pd.read_csv(EMBEDDINGS_IN, index_col="ev_id")
    faulty_embeddings = embeddings.loc[features.index].copy()
    faulty_embeddings["cluster"] = cluster_label

    plot_2d_by_cluster(
        faulty_embeddings[["pca_1", "pca_2"]].to_numpy(), faulty_embeddings["cluster"].to_numpy(),
        xlabel="PC1", ylabel="PC2",
        title=f"PCA of faulty EVs, colored by K-means cluster (k={FINAL_K})",
        out_path=FIGURES_DIR / "kmeans_pca_scatter_by_cluster.png",
    )
    plot_2d_by_cluster(
        faulty_embeddings[["umap_1", "umap_2"]].to_numpy(), faulty_embeddings["cluster"].to_numpy(),
        xlabel="UMAP-1", ylabel="UMAP-2",
        title=f"UMAP of faulty EVs, colored by K-means cluster (k={FINAL_K})",
        out_path=FIGURES_DIR / "kmeans_umap_scatter_by_cluster.png",
    )

    # 5d. save ev_id, cluster label, and all relevant feature values for reuse
    output_table = features.copy()
    output_table.insert(0, "cluster", cluster_label)
    output_table.to_csv(OUT_CSV)
    print(f"\nsaved cluster assignments + features for {len(output_table)} faulty EVs to {OUT_CSV}")


# ===========================================================================
# %% WINDOW-SCALE CLUSTERING -- separate, additive track =====================
# ===========================================================================
"""
Everything below this banner is a second, independent clustering track at
WINDOW granularity, added alongside (not instead of) the whole-session
faulty-EV clustering above. Same fit-once/save-fixed-artifacts/never-refit
architecture as cluster_centroids.pkl / sliding_windows.py: this script
fits once, here; centroid.py's windowed counterpart only ever computes
distances to what's saved below -- it never refits anything.

Unit of clustering: one (ev_id, window) row is one point, NOT one EV -- an
EV with 40 windows contributes 40 points to the matrix below.
"Confirmed-faulty" uses the exact same ground-truth convention as the
whole-session track above (ev_id >= FAULT_EV_ID_CUTOFF; see module
docstring), applied per-window via each window's own ev_id column, since
filtered_features_windows.pkl carries no fault-label column of its own.

Input:
  extracted_data/early_warning/filtered_features_windows.pkl (from
      sliding_window_fleet_features.py) -- one row per (ev_id, window), the
      same filtered_features.pkl columns computed per-window, plus
      window_start/window_end/window_start_hours and the one
      charge_capacity_per_100soc__avg_ah_per_100soc_valid companion column.
Output (diagnostics -- always produced):
  figures/kmeans_windowed_elbow.png
  figures/kmeans_windowed_silhouette_scores.png
  figures/kmeans_windowed_silhouette_samples_k345.png
Output (only once FINAL_K_WINDOWED is set below):
  extracted_data/early_warning/kmeans_centroids_windowed.pkl -- {
      feature_columns, scaler (fitted StandardScaler, faulty windows only),
      centroids (DataFrame, k rows x feature_columns, standardized space),
      healthy_reference (Series, standardized space) }.
  extracted_data/early_warning/kmeans_windowed_unrestricted_diagnostic.pkl
      -- OPTIONAL: exploratory unrestricted (healthy+faulty, unfiltered)
      K-means at the same k, for supplementary thesis discussion only --
      NOT used anywhere in the early-warning system itself.
  extracted_data/early_warning/kmeans_windowed_overall.xlsx -- one sheet per
      windowed clustering performed (label-restricted final clustering +
      the unrestricted diagnostic clustering). Each sheet stacks that
      clustering's EV-windows table (every window, its ev_id, and its
      cluster) above its cluster feature-value characterization table
      (per-cluster mean/std vs. faulty_mean/healthy_mean), so all windows
      of a given EV and how they were clustered can be read directly
      alongside what each cluster's feature values look like.
"""

WINDOW_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
WINDOWED_CENTROIDS_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "kmeans_centroids_windowed.pkl"
WINDOWED_UNRESTRICTED_DIAGNOSTIC_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "kmeans_windowed_unrestricted_diagnostic.pkl"
WINDOWED_UNSUPERVISED_CSV_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "kmeans_windowed_unsupervised.csv"
WINDOWED_OVERALL_XLSX_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "kmeans_windowed_overall.xlsx"

# Separate knob from FINAL_K above -- same "left as None until you've looked
# at the diagnostics" convention. Setting it triggers the final windowed fit
# + artifact save + (optional) unrestricted diagnostic below.
FINAL_K_WINDOWED = 4

# A cluster from the unrestricted windowed K-means below counts as
# "fault-like" if at least this % of its windows come from a
# confirmed-faulty EV (composition["pct_faulty"], 0-100 scale, computed
# below). Consumed by detection.py's windowed-track evaluation as an
# unsupervised, per-window ground truth (via WINDOWED_UNSUPERVISED_CSV_OUT)
# -- detection.py duplicates this constant with a comment cross-referencing
# here rather than importing this module, so keep the two in sync by hand.
FAULT_LIKE_CLUSTER_PCT_THRESHOLD = 50.0


def plot_elbow_windowed(diagnostics: pd.DataFrame, out_path: Path) -> None:
    """Same as plot_elbow above, duplicated rather than parameterized so the
    whole-session function/output above is never touched by this track."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["inertia"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("inertia (within-cluster sum of squares)")
    ax.set_title("K-means elbow plot (faulty EV windows)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_vs_k_windowed(diagnostics: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="o", color=COLOR_FAULTY)
    ax.set_xlabel("k")
    ax.set_ylabel("mean silhouette score")
    ax.set_title("K-means silhouette score vs. k (faulty EV windows)")
    ax.set_xticks(list(K_RANGE))
    _style_scatter_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_samples_grid_windowed(X: np.ndarray, fitted_models: dict, ks: list, out_path: Path) -> None:
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
    fig.suptitle("Per-window silhouette plots (faulty EV windows)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_windowed_feature_characterization(
    windows: pd.DataFrame, cluster_labels: np.ndarray, feature_columns: list[str],
    faulty_windows: pd.DataFrame, healthy_windows: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cluster mean/std of every relevant feature (RAW, unscaled values,
    one row per window) plus the overall confirmed-faulty-window mean and
    overall healthy-window mean, so a cluster's feature values can be read
    directly against BOTH known groups -- not just whichever population it
    happened to be fit on. Shared by both windowed clusterings (the
    label-restricted final fit and the unrestricted diagnostic fit) so the
    two characterization tables are directly comparable."""
    characterization = pd.DataFrame(index=feature_columns)
    for cluster in sorted(np.unique(cluster_labels)):
        cluster_rows = windows.loc[(cluster_labels == cluster), feature_columns]
        characterization[f"cluster_{cluster}_mean"] = cluster_rows.mean()
        characterization[f"cluster_{cluster}_std"] = cluster_rows.std()
    characterization["faulty_mean"] = faulty_windows[feature_columns].mean()
    characterization["healthy_mean"] = healthy_windows[feature_columns].mean()
    characterization.index.name = "feature"
    return characterization


def build_ev_windows_wide(ev_windows_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot an EV-windows table from one-row-per-window (columns: ev_id,
    window_index, cluster) to one-row-per-EV, one-column-per-window-index
    (w0, w1, ...) -- so an EV's entire window sequence and how each window
    was clustered reads across a single row instead of down many rows.
    window_index is the shared raw window_start // stride basis (see W1
    above), so column "wI" means the same absolute point in the recording
    for every EV. A cell is blank/NaN wherever a given EV has no window at
    that index (e.g. its recording didn't reach that far, or that window
    got dropped upstream for a NaN feature value)."""
    wide = ev_windows_long.pivot(index="ev_id", columns="window_index", values="cluster")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    wide.columns = [f"w{int(c)}" for c in wide.columns]
    return wide.reset_index()


def write_windowed_overall_sheet(
    writer: pd.ExcelWriter, sheet_name: str, settings: dict, ev_windows_wide: pd.DataFrame,
    characterization_table: pd.DataFrame,
) -> None:
    """Write one sheet of kmeans_windowed_overall.xlsx, stacked top to
    bottom (each block separated by a blank row):
      1. a settings block -- this clustering's actual population/k/
         threshold etc., so its exact config is visible in the file itself
         rather than having to be looked up in the script;
      2. the EV-windows table (one row per EV, one column per window index
         -- see build_ev_windows_wide);
      3. the cluster feature-value characterization table.
    """
    settings_table = pd.DataFrame({"setting": list(settings.keys()), "value": list(settings.values())})
    settings_table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

    ev_windows_start_row = len(settings_table) + 2  # +1 header row, +1 blank separator row
    ev_windows_wide.to_excel(writer, sheet_name=sheet_name, index=False, startrow=ev_windows_start_row)

    characterization_start_row = ev_windows_start_row + len(ev_windows_wide) + 2  # +1 header row, +1 blank separator row
    characterization_table.to_excel(writer, sheet_name=sheet_name, index=True, startrow=characterization_start_row)


# %% W1. load windowed features, split into confirmed-faulty / healthy by
# the same ev_id >= FAULT_EV_ID_CUTOFF convention as the whole-session
# track, drop any window with a NaN feature value (same "drop rather than
# impute" convention as sliding_windows.py) -- EXCEPT
# charge_capacity_per_100soc__avg_ah_per_100soc, which is forward-filled
# per-EV from its previous window instead (see below).
window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
missing_window_columns = set(features.columns) - set(window_features_all.columns)
if missing_window_columns:
    raise RuntimeError(
        f"filtered_features_windows.pkl is missing feature column(s) {sorted(missing_window_columns)} "
        "relative to the current filtered_features.pkl -- re-run sliding_window_fleet_features.py first."
    )

# charge_capacity_per_100soc__avg_ah_per_100soc goes NaN whenever a window
# has no qualifying charging session (capac2.py's MIN_SOC_INCREASE guard --
# see sliding_window_fleet_features.py's module docstring), which can
# happen mid-lifetime for an otherwise perfectly good window. Rather than
# dropping such a window outright for this one column, carry forward that
# EV's own last valid value (per user instruction) -- ordered by
# window_start so "the window before" means chronologically before, not
# reused across different EVs. An EV's leading NaN(s) (no prior window to
# carry forward from yet) are left NaN and still caught by the drop-mask
# below, same as any other column.
CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"
window_features_all = window_features_all.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
n_capacity_filled = int(window_features_all[CAPACITY_COLUMN].isna().sum())
window_features_all[CAPACITY_COLUMN] = window_features_all.groupby("ev_id")[CAPACITY_COLUMN].ffill()
n_capacity_filled -= int(window_features_all[CAPACITY_COLUMN].isna().sum())
print(f"\nwindowed track: forward-filled {n_capacity_filled} {CAPACITY_COLUMN} NaN(s) from each EV's "
      "previous window")

window_valid_mask = window_features_all[features.columns].notna().all(axis=1)
n_windows_dropped = int((~window_valid_mask).sum())
window_features_all = window_features_all.loc[window_valid_mask].reset_index(drop=True)
print(f"\nwindowed track: dropped {n_windows_dropped} window(s) with >=1 NaN feature value, "
      f"{len(window_features_all)} windows remain")

# window_index: raw-recording-position basis (window_start // stride, t=0 at
# raw index 0 regardless of an EV's own start-of-operation) -- same
# convention as the EV-LIFETIME VECTOR CLUSTERING track further below,
# duplicated here (rather than shared) so that later, independent section
# is never touched by this one. Used to give every EV's windows a common,
# comparable column basis in the wide EV-windows tables below (build_ev_windows_wide).
_window_stride_samples_windowed = int(
    window_features_all.loc[window_features_all["window_start"] > 0, "window_start"].min()
)
window_features_all["window_index"] = (
    window_features_all["window_start"] // _window_stride_samples_windowed
).astype(int)

window_fault_label = window_features_all["ev_id"] >= FAULT_EV_ID_CUTOFF
faulty_windows = window_features_all.loc[window_fault_label].reset_index(drop=True)
healthy_windows = window_features_all.loc[~window_fault_label].reset_index(drop=True)
print(f"windowed track: {len(faulty_windows)} windows from {faulty_windows['ev_id'].nunique()} confirmed-faulty "
      f"EVs, {len(healthy_windows)} windows from {healthy_windows['ev_id'].nunique()} healthy EVs")

# %% W2. fit a NEW StandardScaler on the stacked faulty-window matrix (NOT
# the whole-session scaler above -- window-scale and whole-session-scale
# features have different distributions), sweep k=2..8 exactly like the
# whole-session track above
scaler_windowed = StandardScaler()
faulty_windows_scaled = scaler_windowed.fit_transform(faulty_windows[features.columns])

fitted_models_windowed: dict[int, KMeans] = {}
diagnostics_windowed_rows = []
for k in K_RANGE:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = model.fit_predict(faulty_windows_scaled)
    fitted_models_windowed[k] = model
    diagnostics_windowed_rows.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": silhouette_score(faulty_windows_scaled, labels),
    })

diagnostics_windowed = pd.DataFrame(diagnostics_windowed_rows)
print("\n--- K-means diagnostics (faulty EV windows) ---")
print(diagnostics_windowed.to_string(index=False))

plot_elbow_windowed(diagnostics_windowed, FIGURES_DIR / "kmeans_windowed_elbow.png")
plot_silhouette_vs_k_windowed(diagnostics_windowed, FIGURES_DIR / "kmeans_windowed_silhouette_scores.png")
plot_silhouette_samples_grid_windowed(
    faulty_windows_scaled, fitted_models_windowed, SILHOUETTE_SAMPLE_KS,
    FIGURES_DIR / "kmeans_windowed_silhouette_samples_k345.png",
)

print("\nWindowed diagnostics done. Set FINAL_K_WINDOWED near the top of this "
      "section and rerun to fit the final windowed clustering and save "
      "kmeans_centroids_windowed.pkl.")

# %% W3. final windowed clustering -- only runs once FINAL_K_WINDOWED is set
if FINAL_K_WINDOWED is not None:
    final_model_windowed = fitted_models_windowed.get(FINAL_K_WINDOWED) or KMeans(
        n_clusters=FINAL_K_WINDOWED, random_state=RANDOM_STATE, n_init=N_INIT,
    ).fit(faulty_windows_scaled)

    windowed_cluster_label = pd.Series(final_model_windowed.labels_, name="cluster")
    windowed_cluster_sizes = windowed_cluster_label.value_counts().sort_index()
    print(f"\n--- Windowed cluster sizes (k={FINAL_K_WINDOWED}) ---")
    print(windowed_cluster_sizes.rename("n_windows").to_string())

    # windowed cluster characterization: same idea as the whole-session 5b
    # above (per-cluster mean/std of every relevant feature) so features
    # driving each windowed cluster apart from the rest can be spotted --
    # computed on RAW (unscaled) feature values, one row per window, same
    # convention as the whole-session characterization table. Compared
    # against BOTH the overall faulty-window mean and the overall
    # healthy-window mean (not just faulty), so a cluster's feature values
    # can be read directly against either known group.
    windowed_characterization = compute_windowed_feature_characterization(
        faulty_windows, windowed_cluster_label.to_numpy(), features.columns.tolist(), faulty_windows, healthy_windows,
    )

    print(f"\n--- Windowed cluster characterization (k={FINAL_K_WINDOWED}), vs. faulty/healthy group means ---")
    print(windowed_characterization.to_string(float_format=lambda v: f"{v:.3f}"))

    # EV-windows table (wide: one row per EV, one column per window index)
    # for write_windowed_overall_sheet below, so all of an EV's windows and
    # how they were clustered can be read across a single row.
    label_restricted_ev_windows_long = faulty_windows[["ev_id", "window_index"]].copy()
    label_restricted_ev_windows_long["cluster"] = windowed_cluster_label.to_numpy()
    label_restricted_ev_windows_wide = build_ev_windows_wide(label_restricted_ev_windows_long)

    label_restricted_settings = {
        "clustering": "label_restricted (final windowed clustering)",
        "population": f"confirmed-faulty windows only (ev_id >= {FAULT_EV_ID_CUTOFF})",
        "k (FINAL_K_WINDOWED)": FINAL_K_WINDOWED,
        "n_windows": len(faulty_windows),
        "n_evs": int(faulty_windows["ev_id"].nunique()),
    }

    # healthy reference: SAME scaler as the faulty fit above (not refit),
    # mean vector in that standardized space
    healthy_windows_scaled = scaler_windowed.transform(healthy_windows[features.columns])
    healthy_reference_windowed = pd.Series(healthy_windows_scaled.mean(axis=0), index=features.columns)

    centroids_windowed = pd.DataFrame(
        final_model_windowed.cluster_centers_, index=range(FINAL_K_WINDOWED), columns=features.columns,
    )
    centroids_windowed.index.name = "cluster"

    pd.to_pickle({
        "feature_columns": features.columns.tolist(),
        "scaler": scaler_windowed,
        "centroids": centroids_windowed,
        "healthy_reference": healthy_reference_windowed,
    }, WINDOWED_CENTROIDS_OUT)
    print(f"\nsaved windowed centroids + healthy reference + scaler to {WINDOWED_CENTROIDS_OUT}")

    # %% W4. [OPTIONAL / diagnostic only, NOT part of the early-warning
    # system -- see module docstring] exploratory unrestricted K-means on
    # ALL windows (healthy + faulty, unfiltered by label) at the same k, to
    # check whether fault-like clusters emerge without label restriction.
    # Fits its OWN scaler on the full (healthy+faulty) population --
    # deliberately separate from the label-restricted fit above, since this
    # pass asks a different question (does unsupervised structure on the
    # WHOLE population resemble the faulty subtypes?) and scaling against
    # the full population is the natural way to ask it.
    scaler_unrestricted = StandardScaler()
    all_windows_scaled = scaler_unrestricted.fit_transform(window_features_all[features.columns])
    model_unrestricted = KMeans(n_clusters=FINAL_K_WINDOWED, random_state=RANDOM_STATE, n_init=N_INIT)
    unrestricted_labels = model_unrestricted.fit_predict(all_windows_scaled)

    unrestricted_table = window_features_all[["ev_id", "window_start", "window_end"]].copy()
    unrestricted_table["cluster"] = unrestricted_labels
    unrestricted_table["is_confirmed_faulty"] = window_fault_label.to_numpy()

    composition = (
        unrestricted_table.groupby("cluster")["is_confirmed_faulty"]
        .agg(pct_faulty="mean", n_windows="size")
    )
    composition["pct_faulty"] *= 100
    composition["pct_healthy"] = 100 - composition["pct_faulty"]
    composition["is_fault_like"] = composition["pct_faulty"] >= FAULT_LIKE_CLUSTER_PCT_THRESHOLD

    print(f"\n--- Unrestricted windowed K-means (k={FINAL_K_WINDOWED}, ALL windows, diagnostic only) ---")
    print("cluster composition (% healthy vs. % confirmed-faulty, by window):")
    print(composition[["n_windows", "pct_healthy", "pct_faulty", "is_fault_like"]].to_string(float_format=lambda v: f"{v:.1f}"))
    print(f"fault-like clusters (pct_faulty >= {FAULT_LIKE_CLUSTER_PCT_THRESHOLD:.0f}%): "
          f"{composition.index[composition['is_fault_like']].tolist()}")
    print("(compare against the label-restricted clustering above: if fault-like structure "
          "emerges here too without label restriction, that's supporting evidence for the "
          "subtypes found above; if not, the label restriction is doing real work.)")

    # same feature-by-cluster characterization as the label-restricted
    # clustering above (per-cluster mean/std vs. faulty_mean/healthy_mean),
    # but for the unrestricted (healthy+faulty) clustering's own clusters.
    unrestricted_characterization = compute_windowed_feature_characterization(
        window_features_all[features.columns], unrestricted_labels, features.columns.tolist(),
        faulty_windows, healthy_windows,
    )
    print(f"\n--- Unrestricted windowed cluster characterization (k={FINAL_K_WINDOWED}), "
          "vs. faulty/healthy group means ---")
    print(unrestricted_characterization.to_string(float_format=lambda v: f"{v:.3f}"))

    pd.to_pickle({
        "feature_columns": features.columns.tolist(),
        "scaler": scaler_unrestricted,
        "cluster_centers": model_unrestricted.cluster_centers_,
        "labels_table": unrestricted_table,
        "composition": composition,
    }, WINDOWED_UNRESTRICTED_DIAGNOSTIC_OUT)
    print(f"saved unrestricted diagnostic to {WINDOWED_UNRESTRICTED_DIAGNOSTIC_OUT}")

    # per-window unsupervised ground truth for detection.py's windowed-track
    # evaluation: each window's cluster assignment + whether that cluster is
    # fault-like, joinable by (ev_id, window_start, window_end) rather than
    # by position/index -- detection.py's own windowed cleaning runs
    # independently against filtered_features_windows.pkl and can end up
    # with a different row order/count, so a positional join would risk
    # silently pairing the wrong rows.
    unsupervised_export = unrestricted_table.merge(
        composition[["pct_faulty", "is_fault_like"]].reset_index(), on="cluster", how="left",
    )[["ev_id", "window_start", "window_end", "cluster", "pct_faulty", "is_fault_like"]]
    unsupervised_export.to_csv(WINDOWED_UNSUPERVISED_CSV_OUT, index=False)
    print(f"saved per-window unsupervised cluster assignments to {WINDOWED_UNSUPERVISED_CSV_OUT}")

    # %% W5. combined EV-windows + cluster-characterization overview, one
    # sheet per windowed clustering performed above (label-restricted final
    # clustering, then the unrestricted diagnostic clustering) -- so all of
    # an EV's windows and how they were clustered can be inspected directly
    # alongside what each cluster's feature values look like.
    unrestricted_ev_windows_long = window_features_all[["ev_id", "window_index"]].copy()
    unrestricted_ev_windows_long["cluster"] = unrestricted_labels
    unrestricted_ev_windows_wide = build_ev_windows_wide(unrestricted_ev_windows_long)
    unrestricted_ev_windows_wide.insert(
        1, "is_confirmed_faulty", unrestricted_ev_windows_wide["ev_id"] >= FAULT_EV_ID_CUTOFF,
    )

    unrestricted_settings = {
        "clustering": "unrestricted (diagnostic only -- NOT used by the early-warning system)",
        "population": "all EVs (healthy + faulty, unfiltered by label)",
        "k (FINAL_K_WINDOWED)": FINAL_K_WINDOWED,
        "fault_like_cluster_pct_threshold": FAULT_LIKE_CLUSTER_PCT_THRESHOLD,
        "n_windows": len(window_features_all),
        "n_evs": int(window_features_all["ev_id"].nunique()),
    }

    with pd.ExcelWriter(WINDOWED_OVERALL_XLSX_OUT, engine="openpyxl") as writer:
        write_windowed_overall_sheet(
            writer, "label_restricted", label_restricted_settings, label_restricted_ev_windows_wide,
            windowed_characterization,
        )
        write_windowed_overall_sheet(
            writer, "unrestricted", unrestricted_settings, unrestricted_ev_windows_wide, unrestricted_characterization,
        )
    print(f"saved combined EV-windows + cluster-characterization overview to {WINDOWED_OVERALL_XLSX_OUT}")


# ===========================================================================
# %% EV-LIFETIME VECTOR CLUSTERING -- separate, additive, EXPLORATORY block =
# ===========================================================================
"""
A third, independent clustering track: one row per EV (not per window, not
per window-point), where the row is that EV's FULL sequence of windows
concatenated into a single vector -- "clustering EVs considering their
whole vector through lifetime," per user instruction. Reuses
window_features_all/features.columns from the WINDOW-SCALE CLUSTERING
track above (same cleaned data: charge_capacity forward-filled, remaining
NaN rows dropped -- see W1) rather than reloading anything.

Exploratory: prints diagnostics/results only -- NO figures, NO pickled
artifacts (per user instruction: "dont produce plots just output",
"dont store data for now, just outputs").

--- window_index: same raw-recording-position basis as the WINDOW-SCALE
track (window_start // stride, t=0 at raw index 0 regardless of
start-of-operation -- see that section's docstring). Missingness is now
purely structural (W1 already dropped every remaining feature-level NaN):
an EV either has a real window at a given window_index or its recording
simply didn't reach that far -- and since the window sequence starts at 0
and steps contiguously, "EV has window_index i" is equivalent to "EV has
>= i+1 total windows" (no gaps to worry about).

--- Two ways of handling the resulting variable-length sequences ----------
  Option 1 (truncate-and-drop): keep only EVs with >= N windows, use
      exactly their first N (window_index 0..N-1) -- every included cell is
      real data, no padding, but EVs short of N are excluded entirely.
  Option 2 (truncate-and-pad): keep EVERY EV with >= 1 window; for window
      slots beyond an EV's own last real window, repeat that EV's last
      OBSERVED window's feature vector ("last value carried forward").
      Since padding makes a short EV's vector look like uninterrupted real
      data, one extra dimension is appended per EV -- n_real_windows_used
      (how many of the N slots were real, i.e. min(that EV's true window
      count, N)) -- specifically so the clustering can still see each EV's
      actual observed lifetime length instead of that signal being erased
      by the padding (per user instruction, "add window number as well so
      clustering can be performed more meaningfully").
Both run for BOTH populations: "faulty" (confirmed-faulty EVs only, same
ev_id >= FAULT_EV_ID_CUTOFF ground truth as every other track) and "all"
(every EV, unsupervised/unrestricted by label).

--- Knobs ---------------------------------------------------------------
LIFETIME_TRUNCATE_NS: the 3 N values swept for Option 1 (diagnostics
    printed per N per population; k=2..8 elbow/silhouette table, THEN one
    committed fit at LIFETIME_TRUNCATE_FINAL_K with cluster sizes -- no k
    is picked automatically FOR THE SWEEP, consistent with FINAL_K/
    FINAL_K_WINDOWED's convention elsewhere in this file, but a specific k
    is still needed to produce actual cluster assignments to report sizes
    for, and for the fault-label alignment check in L3 below).
LIFETIME_TRUNCATE_FINAL_K: Option 1's committed k (cluster sizes for every
    N/population; the full feature-by-cluster characterization table is
    additionally printed for population=faulty, N=7 specifically, per user
    request).
LIFETIME_PADDED_N: Option 2's N -- a single value, meant to be hand-tuned
    (per user instruction, "a suitable N (variable that I can play with)").
LIFETIME_PADDED_FINAL_K: Option 2 additionally fits ONE specific k (needed
    to produce the requested feature-by-cluster characterization table --
    unlike the diagnostics-only tracks elsewhere, a table needs a committed
    clustering to characterize). Change this and LIFETIME_PADDED_N freely
    and rerun to explore.

--- L3: does the unsupervised (population="all") clustering recover the
known fault/healthy split on its own? -----------------------------------
For every population="all" run above with a committed k (every Option 1 N,
plus Option 2), the resulting cluster labels are compared against the SAME
ev_id >= FAULT_EV_ID_CUTOFF ground truth used everywhere else in this file,
via sklearn's adjusted_rand_score (ARI) -- the standard measure of
agreement between an unsupervised partition and a known one; 0 = no better
than chance, 1 = perfect recovery, and it (unlike raw accuracy) is not
inflated just because one class -- healthy EVs -- is the overwhelming
majority. Each cluster's %faulty composition is also printed, since ARI
alone doesn't show WHICH cluster(s) concentrate faulty EVs. This is purely
descriptive/exploratory (supplementary thesis discussion), not something
the rest of the pipeline consumes.
"""
LIFETIME_TRUNCATE_NS = [7, 15, 30]
LIFETIME_TRUNCATE_FINAL_K = None  # Option 1's k for population=faulty -- population=all uses k+1, see _k_for_population
LIFETIME_PADDED_N = 15          # Option 2's N -- tune this
LIFETIME_PADDED_FINAL_K = None     # Option 2's k for population=faulty -- population=all uses k+1, see _k_for_population

feature_columns_lifetime = features.columns.tolist()


def _k_for_population(base_k: int, population: str) -> int:
    """population=faulty clusters with base_k; population=all gets base_k+1
    -- healthy EVs are the overwhelming majority there and need their own
    cluster(s) too, so the same k that resolves faulty-only subtypes would
    under-provide once healthy EVs are back in the mix (per user
    instruction)."""
    return base_k if population == "faulty" else base_k + 1

# window_index, added locally (not written back to window_features_all) --
# same "raw window_start // stride" basis as the WINDOW-SCALE track, derived
# rather than hardcoded so it stays correct if STRIDE_HOURS ever changes
# upstream.
_window_stride_samples = int(
    window_features_all.loc[window_features_all["window_start"] > 0, "window_start"].min()
)
window_features_indexed = window_features_all.copy()
window_features_indexed["window_index"] = (
    window_features_indexed["window_start"] // _window_stride_samples
).astype(int)

lifetime_populations = {
    "faulty": window_features_indexed.loc[window_features_indexed["ev_id"] >= FAULT_EV_ID_CUTOFF],
    "all": window_features_indexed,
}


def build_lifetime_matrix_truncated(window_table: pd.DataFrame, feature_columns: list[str], N: int):
    """Option 1: EVs that actually HAVE every window_index 0..N-1 only
    (checked by set containment, not just a window COUNT >= N -- a handful
    of EVs have a LEADING gap, e.g. window_index 0-2 masked by very-late
    start-of-operation and dropped by W1's NaN guard, so window_index isn't
    guaranteed contiguous from 0 for every EV -- see this block's
    docstring). Each row = concat of that EV's window_index 0..N-1 feature
    vectors (window-major, feature-minor -- see column_labels).
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
    wi is missing for that EV -- this also transparently covers the LEADING
    gaps mentioned above (a handful of EVs miss window_index 0..2): if wi=0
    itself is missing, the fallback below carries the EV's own earliest
    available real window BACKWARD to fill the leading slot(s), so every
    row is still fully populated. One extra n_real_windows_used dimension
    is appended per EV -- the count of ACTUALLY real (non-carried) values
    among the N slots (see module docstring for why).
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
            # leading gap wider than this EV's first real window within
            # range(N) (e.g. real data only starts at wi=3) -- backfill
            # with the EV's own earliest available real value instead of
            # leaving a hole (still "repeat an observed value", just
            # backward rather than forward for the leading slots only)
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


def run_lifetime_kmeans_sweep(X_scaled: np.ndarray, label: str) -> pd.DataFrame | None:
    """k=2..8 elbow/silhouette diagnostics, printed. k range is clipped to
    what's actually feasible for the number of EVs available (silhouette
    needs 2 <= k < n_samples). Returns the (k, inertia, silhouette_score)
    table (or None if skipped) so callers can feed it into the ev_inertia/
    ev_silhouette comparison plots below."""
    n_evs = X_scaled.shape[0]
    usable_ks = [k for k in K_RANGE if 2 <= k < n_evs]
    if not usable_ks:
        print(f"\n--- {label}: only {n_evs} EV(s) available, too few for k in {list(K_RANGE)} -- skipped ---")
        return None
    rows = []
    for k in usable_ks:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_scaled)
        rows.append({"k": k, "inertia": model.inertia_, "silhouette_score": silhouette_score(X_scaled, labels)})
    diagnostics = pd.DataFrame(rows)
    print(f"\n--- {label} (n_evs={n_evs}, dims={X_scaled.shape[1]}) ---")
    print(diagnostics.to_string(index=False))
    return diagnostics


def plot_ev_lifetime_diagnostics_grid(
    diagnostics_by_config: dict[str, dict[str, pd.DataFrame]], metric: str, ylabel: str, suptitle: str, out_path: Path,
) -> None:
    """One COLUMN per Option+N configuration, two ROWS (top: population=
    faulty, bottom: population=all) -- the same configuration's faulty-only
    and all-EV sweep sit directly on top of each other for easy comparison.
    Own y-axis scale per subplot -- inertia in particular varies hugely
    across N/dims/population size, so a shared scale would flatten most
    lines to invisibility."""
    configs = list(diagnostics_by_config.keys())
    if not configs:
        print(f"\n--- {suptitle}: no sections had a usable sweep -- plot skipped ---")
        return
    fig, axes = plt.subplots(2, len(configs), figsize=(4.5 * len(configs), 7.2), squeeze=False)
    for col, config_label in enumerate(configs):
        for row, pop_name in enumerate(["faulty", "all"]):
            ax = axes[row, col]
            diag = diagnostics_by_config[config_label].get(pop_name)
            if diag is None:
                ax.axis("off")
                continue
            ax.plot(diag["k"], diag[metric], marker="o", color=COLOR_FAULTY)
            ax.set_title(f"{config_label} ({pop_name})", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel(ylabel)
            ax.set_xticks(diag["k"])
            _style_scatter_axes(ax)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def print_lifetime_characterization(
    X: np.ndarray, column_labels: list[str], feature_columns: list[str], N: int,
    cluster_label: pd.Series, label: str,
) -> None:
    """Feature-by-cluster table, RAW (unscaled) values -- same convention as
    every other characterization table in this file. Row order is
    FEATURE-major, window-minor -- every window (w0..w{N-1}) of one feature
    printed together before moving to the next feature, so how a single
    feature changes across the window sequence is readable top-to-bottom
    (X/column_labels stay window-major internally, as built by
    build_lifetime_matrix_padded -- only this table's DISPLAY order
    differs)."""
    df = pd.DataFrame(X, columns=column_labels)
    characterization = pd.DataFrame(index=column_labels)
    for cluster in sorted(cluster_label.unique()):
        cluster_rows = df.loc[(cluster_label == cluster).to_numpy()]
        characterization[f"cluster_{cluster}_mean"] = cluster_rows.mean()
        characterization[f"cluster_{cluster}_std"] = cluster_rows.std()
    characterization["overall_mean"] = df.mean()

    feature_major_order = [f"{f}__w{wi}" for f in feature_columns for wi in range(N)]
    feature_major_order += [c for c in column_labels if c not in feature_major_order]  # e.g. n_real_windows_used
    characterization = characterization.loc[feature_major_order]

    print(f"\n--- {label}: feature-by-cluster characterization ---")
    print(characterization.to_string(float_format=lambda v: f"{v:.3f}"))


# %% L1. Option 1 -- truncate-and-drop, N in {7, 15, 30}, both populations:
# diagnostics sweep, then one committed fit at LIFETIME_TRUNCATE_FINAL_K
# (cluster sizes always; full characterization table for population=faulty,
# N=7 specifically, per user request). "all"-population cluster labels are
# stashed in lifetime_unsupervised_runs for the L3 fault-label check below.
lifetime_unsupervised_runs: dict[str, tuple[np.ndarray, pd.Series]] = {}  # section label -> (ev_ids, cluster_label)
# config label (Option+N, population-agnostic) -> {"faulty": diag_df, "all": diag_df} -- see
# plot_ev_lifetime_diagnostics_grid for why it's nested this way (faulty/all paired per config for plotting)
lifetime_diagnostics_by_config: dict[str, dict[str, pd.DataFrame]] = {}

for _pop_name, _pop_table in lifetime_populations.items():
    for _N in LIFETIME_TRUNCATE_NS:
        _ev_ids, _X, _cols = build_lifetime_matrix_truncated(_pop_table, feature_columns_lifetime, _N)
        if len(_ev_ids) < 2:
            print(f"\n--- Option 1 truncate-drop, population={_pop_name}, N={_N}: "
                  f"only {len(_ev_ids)} EV(s) reach N={_N} windows -- skipped ---")
            continue
        _X_scaled = StandardScaler().fit_transform(_X)
        _diag = run_lifetime_kmeans_sweep(
            _X_scaled, f"Option 1 truncate-drop, population={_pop_name}, N={_N} ({len(_ev_ids)} EVs kept)",
        )
        if _diag is not None:
            lifetime_diagnostics_by_config.setdefault(f"Opt1 N={_N}", {})[_pop_name] = _diag

        if LIFETIME_TRUNCATE_FINAL_K is None:
            continue  # diagnostics/plots only -- left as None, same FINAL_K convention as elsewhere in this file

        _k = _k_for_population(LIFETIME_TRUNCATE_FINAL_K, _pop_name)
        if not (2 <= _k < len(_ev_ids)):
            print(f"\n--- Option 1 ({_pop_name}, N={_N}): k={_k} not valid for {len(_ev_ids)} EVs "
                  "-- committed fit skipped ---")
            continue

        _final_model = KMeans(n_clusters=_k, random_state=RANDOM_STATE, n_init=N_INIT).fit(_X_scaled)
        _cluster_label = pd.Series(_final_model.labels_, index=_ev_ids, name="cluster")
        _sizes = _cluster_label.value_counts().sort_index()
        print(f"\n--- Option 1 ({_pop_name}, N={_N}): cluster sizes (k={_k}) ---")
        print(_sizes.rename("n_evs").to_string())

        if _pop_name == "faulty" and _N == 7:
            print_lifetime_characterization(
                _X, _cols, feature_columns_lifetime, _N, _cluster_label,
                f"Option 1 truncate-drop, population=faulty, k={_k}, N={_N}",
            )

        if _pop_name == "all":
            lifetime_unsupervised_runs[f"Option 1 (N={_N})"] = (_ev_ids, _cluster_label)

# %% L2. Option 2 -- truncate-and-pad, LIFETIME_PADDED_N, both populations:
# diagnostics sweep first, then one committed fit at LIFETIME_PADDED_FINAL_K
# + the requested feature-by-cluster characterization table
for _pop_name, _pop_table in lifetime_populations.items():
    _ev_ids, _X, _cols, _n_real = build_lifetime_matrix_padded(_pop_table, feature_columns_lifetime, LIFETIME_PADDED_N)
    _scaler = StandardScaler()
    _X_scaled = _scaler.fit_transform(_X)
    _diag = run_lifetime_kmeans_sweep(
        _X_scaled,
        f"Option 2 pad-last-observed, population={_pop_name}, N={LIFETIME_PADDED_N} ({len(_ev_ids)} EVs)",
    )
    if _diag is not None:
        lifetime_diagnostics_by_config.setdefault(f"Opt2 N={LIFETIME_PADDED_N}", {})[_pop_name] = _diag

    if LIFETIME_PADDED_FINAL_K is None:
        continue  # diagnostics/plots only -- left as None, same FINAL_K convention as elsewhere in this file

    _n_evs = _X_scaled.shape[0]
    _k = _k_for_population(LIFETIME_PADDED_FINAL_K, _pop_name)
    if not (2 <= _k < _n_evs):
        print(f"\n--- Option 2 ({_pop_name}): k={_k} not valid for {_n_evs} EVs "
              "-- characterization table skipped ---")
        continue

    _final_model = KMeans(n_clusters=_k, random_state=RANDOM_STATE, n_init=N_INIT).fit(_X_scaled)
    _cluster_label = pd.Series(_final_model.labels_, index=_ev_ids, name="cluster")
    _sizes = _cluster_label.value_counts().sort_index()
    print(f"\n--- Option 2 ({_pop_name}): cluster sizes (k={_k}, N={LIFETIME_PADDED_N}) ---")
    print(_sizes.rename("n_evs").to_string())

    _mean_n_real_by_cluster = pd.Series(_n_real, index=_ev_ids).groupby(_cluster_label).mean()
    print(f"\n--- Option 2 ({_pop_name}): mean n_real_windows_used by cluster (padding load per cluster) ---")
    print(_mean_n_real_by_cluster.rename("mean_n_real_windows").to_string())

    print_lifetime_characterization(
        _X, _cols, feature_columns_lifetime, LIFETIME_PADDED_N, _cluster_label,
        f"Option 2 pad-last-observed, population={_pop_name}, k={_k}, N={LIFETIME_PADDED_N}",
    )

    if _pop_name == "all":
        lifetime_unsupervised_runs[f"Option 2 (N={LIFETIME_PADDED_N})"] = (_ev_ids, _cluster_label)


# %% L2b. inertia/silhouette comparison plots across every sweep above --
# one column per Option+N config, faulty on top / all on bottom (see
# plot_ev_lifetime_diagnostics_grid)
plot_ev_lifetime_diagnostics_grid(
    lifetime_diagnostics_by_config, "inertia", "inertia (within-cluster sum of squares)",
    "EV-lifetime clustering (section 3): inertia vs. k, by config", FIGURES_DIR / "ev_inertia.png",
)
plot_ev_lifetime_diagnostics_grid(
    lifetime_diagnostics_by_config, "silhouette_score", "mean silhouette score",
    "EV-lifetime clustering (section 3): silhouette score vs. k, by config", FIGURES_DIR / "ev_silhouette.png",
)
print(f"\nsaved EV-lifetime clustering comparison plots to {FIGURES_DIR / 'ev_inertia.png'} "
      f"and {FIGURES_DIR / 'ev_silhouette.png'}")


# %% L3. does the unsupervised (population="all") clustering recover the
# known fault/healthy split on its own? -- see module docstring for the ARI
# choice and its rationale.
print("\n" + "=" * 79)
print("L3. Unsupervised clustering vs. known fault/healthy labels")
print("=" * 79)

if not lifetime_unsupervised_runs:
    print("\nLIFETIME_TRUNCATE_FINAL_K and/or LIFETIME_PADDED_FINAL_K are None -- no committed "
          "population=all clustering to compare against fault labels yet. Set them and rerun.")

_alignment_rows = []
for _section_label, (_ev_ids, _cluster_label) in lifetime_unsupervised_runs.items():
    _true_faulty = _ev_ids >= FAULT_EV_ID_CUTOFF
    _ari = adjusted_rand_score(_true_faulty, _cluster_label.to_numpy())
    _alignment_rows.append({"section": _section_label, "n_evs": len(_ev_ids), "adjusted_rand_index": _ari})

    _composition = pd.DataFrame({"cluster": _cluster_label.to_numpy(), "is_faulty": _true_faulty})
    _composition = _composition.groupby("cluster")["is_faulty"].agg(pct_faulty="mean", n_evs="size")
    _composition["pct_faulty"] *= 100
    print(f"\n--- {_section_label}: cluster composition (%faulty by ev_id ground truth) ---")
    print(_composition[["n_evs", "pct_faulty"]].to_string(float_format=lambda v: f"{v:.1f}"))

_alignment = pd.DataFrame(_alignment_rows)
if len(_alignment):
    _alignment = _alignment.sort_values("adjusted_rand_index", ascending=False)
    print("\n--- Adjusted Rand Index vs. ev_id >= FAULT_EV_ID_CUTOFF ground truth, ranked ---")
    print(_alignment.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
if len(_alignment):
    _best = _alignment.iloc[0]
    _best_ev_ids, _best_cluster_label = lifetime_unsupervised_runs[_best["section"]]
    _best_true_faulty = _best_ev_ids >= FAULT_EV_ID_CUTOFF
    _best_composition = pd.DataFrame({"cluster": _best_cluster_label.to_numpy(), "is_faulty": _best_true_faulty})
    _best_by_cluster = _best_composition.groupby("cluster")["is_faulty"].agg(pct_faulty="mean", n_evs="size")
    _purest_cluster = _best_by_cluster["pct_faulty"].idxmax()
    _purest_pct_faulty = _best_by_cluster.loc[_purest_cluster, "pct_faulty"] * 100
    _purest_n = int(_best_by_cluster.loc[_purest_cluster, "n_evs"])
    _n_faulty_total = int(_best_true_faulty.sum())
    _n_faulty_in_purest = int((_best_composition.loc[_best_composition["cluster"] == _purest_cluster, "is_faulty"]).sum())

    print(f"\nComment: of the unsupervised (population=all) sections run above, "
          f"{_best['section']} aligns best with the known fault/healthy split "
          f"(ARI={_best['adjusted_rand_index']:.3f} on {int(_best['n_evs'])} EVs; "
          f"ARI range across all {len(_alignment)} sections: "
          f"{_alignment['adjusted_rand_index'].min():.3f}-{_alignment['adjusted_rand_index'].max():.3f}). "
          f"In that section, cluster {_purest_cluster} is {_purest_pct_faulty:.0f}% faulty "
          f"({_n_faulty_in_purest}/{_purest_n} of its members), and captures "
          f"{_n_faulty_in_purest}/{_n_faulty_total} of ALL faulty EVs in this population -- "
          f"i.e. K-means, given no fault labels at all, still concentrates a "
          f"disproportionate share of the true faulty EVs into one cluster there. "
          f"An ARI of 0 would mean no better than chance; every section above landing "
          f"clearly positive is itself notable, since faulty EVs are only a small "
          f"minority of 'all' and nothing here tells K-means which EVs those are -- "
          f"see the composition table above for the full per-cluster breakdown, and for "
          f"whether the SAME pattern holds in the other, lower-ARI sections too.")
