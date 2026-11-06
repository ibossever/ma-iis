# %%
"""
Pearson correlation between per-EV scalar features.

`feature_matrix` has one row per EV (index = ev_id) and one column per
scalar numeric feature. Correlation is computed with
feature_matrix.corr(method='pearson') -- linear correlation, unlike the
rank-based Spearman correlation in spearman_corr.py.

feature_matrix is built automatically from every table under
extracted_data/whole_features/ -- drop a new per-EV feature table in there
and it joins feature_matrix on the next run, no code change needed.

Alongside the correlation matrix, scipy.stats.pearsonr gives the p-value
for each pair (null hypothesis: the two features are not linearly
related) -- computed pairwise, same NaN-dropping as pandas' .corr(), so a
gap in one feature doesn't discard EVs from pairs that don't involve it.

Like spearman_corr.py (which also no longer touches filtered_features.pkl),
this script does not touch filtered_features.pkl -- it only computes and
plots the Pearson correlation of feature_matrix, and reports (without
discarding anything) groups of features that are mutually redundant by
Pearson correlation -- see find_redundant_feature_groups below. features.py
combines these groups with spearman_corr.py's equivalent as the last step
of its filtering.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../pearson_corr.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
CORROLATION_DIR = EXTRACTED_DATA / "corrolation"
CORROLATION_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Every per-EV scalar feature table under extracted_data/whole_features/,
# picked up automatically in filename order -- adding/removing a table there
# changes feature_matrix on the next run, no code change needed here.
# Columns get prefixed with the source's stem so same-named columns from
# different sources (e.g. two "mean" columns) can't collide in feature_matrix.
FEATURE_SOURCES = [f"whole_features/{p.name}" for p in sorted((EXTRACTED_DATA / "whole_features").glob("*.pkl"))]

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def build_feature_matrix(sources: list[str]) -> pd.DataFrame:
    """Outer-join every per-EV scalar feature table in `sources` (each
    indexed by EV id) into a single DataFrame: one row per EV, one column
    per scalar feature."""
    tables = []
    for name in sources:
        df = pd.read_pickle(EXTRACTED_DATA / name)
        df.index.name = "ev_id"
        tables.append(df.add_prefix(f"{Path(name).stem}__"))
    return pd.concat(tables, axis=1, join="outer")


def drop_constant_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with zero variance. Pearson correlation is undefined
    (NaN) for a constant column."""
    return feature_matrix.loc[:, feature_matrix.nunique(dropna=True) > 1]


def pearson_correlation(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation between every pair of columns in
    `feature_matrix`, which must have EVs as rows and scalar numeric
    features as columns."""
    return feature_matrix.corr(method="pearson")


def pearson_pvalues(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """p-value of the Pearson correlation (scipy.stats.pearsonr) for
    every pair of columns in `feature_matrix`. Each pair's NaNs are dropped
    independently, matching the pairwise behaviour of pearson_correlation
    above."""
    cols = feature_matrix.columns
    pvalues = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, col_i in enumerate(cols):
        for col_j in cols[i:]:
            # .to_numpy() (rather than selecting [col_i, col_j] as columns)
            # so a self-pair (col_i == col_j) doesn't collapse into a single
            # duplicate-labelled column.
            pair = feature_matrix.loc[:, [col_i, col_j]].dropna().to_numpy()
            _, p = pearsonr(pair[:, 0], pair[:, 1])
            pvalues.loc[col_i, col_j] = pvalues.loc[col_j, col_i] = p
    return pvalues


def find_redundant_feature_groups(
    corr: pd.DataFrame, corr_threshold: float = 0.8,
) -> list[list[str]]:
    """Report groups of mutually redundant features -- does NOT drop or pick
    a representative for any of them, just identifies them for a human (or,
    combined with spearman_corr.py's equivalent, features.py's final
    dedup step) to act on.

    Hierarchically cluster columns on the distance 1 - |r| and cut the
    dendrogram at 1 - corr_threshold, so a group is a set of features that
    are all within corr_threshold Pearson correlation of each other
    (directly, or chained through the rest of the group) -- same method as
    spearman_corr.py's find_redundant_feature_groups, just fed the linear
    (Pearson) correlation matrix instead of the rank-based (Spearman) one.
    Kept as two independent angles on redundancy since a pair can be
    strongly monotonic but only weakly linear (or vice versa) -- see
    features.py, which only dedupes a group when spearman_corr.py and this
    function agree.

    Only groups with more than one member are returned -- a singleton
    "group" just means that feature isn't redundant with anything.
    """
    distance = (1 - corr.abs()).to_numpy(copy=True)
    np.fill_diagonal(distance, 0)  # exactly 0 by definition; avoids float noise
    condensed = squareform(distance, checks=False)
    clustering = linkage(condensed, method="average")
    cluster_ids = fcluster(clustering, t=1 - corr_threshold, criterion="distance")

    clusters = pd.Series(cluster_ids, index=corr.columns)
    groups = [sorted(members) for members in clusters.groupby(clusters).groups.values()]
    return [group for group in groups if len(group) > 1]


def _plot_heatmap(values: pd.DataFrame, out_path: Path, *, vmin: float, vmax: float,
                   cmap: str, colorbar_label: str, title: str) -> None:
    n = len(values.columns)
    fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 3))
    im = ax.imshow(values.values, vmin=vmin, vmax=vmax, cmap=cmap)

    ax.set_xticks(range(n))
    ax.set_xticklabels(values.columns, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(values.columns)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{values.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: Path) -> None:
    _plot_heatmap(
        corr, out_path, vmin=-1, vmax=1, cmap="RdBu_r",
        colorbar_label="Pearson correlation",
        title="Pearson correlation between per-EV features",
    )


def plot_pvalue_heatmap(pvalues: pd.DataFrame, out_path: Path) -> None:
    # Reversed sequential map: small (significant) p-values stand out dark,
    # p close to 1 fades to white -- unlike correlation, p-values aren't
    # signed, so a diverging colormap would be misleading here.
    _plot_heatmap(
        pvalues, out_path, vmin=0, vmax=1, cmap="Reds_r",
        colorbar_label="p-value",
        title="Pearson correlation p-values between per-EV features",
    )


# %%
feature_matrix = build_feature_matrix(FEATURE_SOURCES)
feature_matrix = drop_constant_features(feature_matrix)

# %%
corr_matrix = pearson_correlation(feature_matrix)
corr_matrix.to_pickle(CORROLATION_DIR / "feature_matrix_pearson_corr.pkl")
plot_correlation_heatmap(corr_matrix, FIGURES_DIR / "pearson_correlation.png")

# %%
pvalue_matrix = pearson_pvalues(feature_matrix)
pvalue_matrix.to_pickle(CORROLATION_DIR / "feature_matrix_pearson_pval.pkl")
plot_pvalue_heatmap(pvalue_matrix, FIGURES_DIR / "pearson_pvalues.png")

# %%
# Report-only: this does NOT touch filtered_features.pkl, and no feature is
# discarded here -- see find_redundant_feature_groups' docstring. Same
# corr_threshold as spearman_corr.py's REDUNDANCY_CORR_THRESHOLD, so the two
# reports (and features.py's combined dedup step, which requires both to
# agree) are comparing groups found at the same cutoff.
REDUNDANCY_CORR_THRESHOLD = 0.8
redundant_groups = find_redundant_feature_groups(corr_matrix, REDUNDANCY_CORR_THRESHOLD)

print(f"{len(redundant_groups)} redundant feature groups found (|r| >= {REDUNDANCY_CORR_THRESHOLD}):")
print("# TODO: pick one representative per group below, then reflect the choice in features.py")
for group in redundant_groups:
    print(f"# redundant group ({len(group)} features, keep one): {group}")

# ===========================================================================
# Feature vs. fault label -- same Pearson correlation, but each column of
# feature_matrix against the binary fault label rather than against every
# other column (see pearson_correlation/pearson_pvalues above for the
# feature-feature version). Ground truth: EV number >= FAULT_EV_ID_CUTOFF is
# faulty, below it is healthy -- same convention as detection.py and
# fault_feature.py.
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as detection.py


def pearson_fault_correlation(feature_matrix: pd.DataFrame, fault_label: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Pearson correlation (and its p-value) of every column in
    `feature_matrix` against the binary `fault_label`, one pair at a time
    with its own NaN-dropping -- same pairwise treatment as pearson_pvalues
    above."""
    corr, pval = {}, {}
    fault_numeric = fault_label.astype(float)
    for col in feature_matrix.columns:
        pair = pd.concat([feature_matrix[col], fault_numeric], axis=1).dropna().to_numpy()
        r, p = pearsonr(pair[:, 0], pair[:, 1])
        corr[col] = r
        pval[col] = p
    return pd.Series(corr, name="pearson_r"), pd.Series(pval, name="pearson_p")


def plot_fault_correlation_values(values: pd.Series, out_path: Path, *, vmin: float, vmax: float,
                                   cmap: str, colorbar_label: str, title: str,
                                   fmt: str = "{:.2f}") -> None:
    """Single-column heatmap of one value per feature (sorted ascending),
    with the numeric value written into each cell -- same style as
    _plot_heatmap above, just one column wide since there's only ever one
    fault label to compare against."""
    ordered = values.sort_values()
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(ordered) + 2))
    im = ax.imshow(ordered.values.reshape(-1, 1), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")

    ax.set_xticks([])
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered.index)

    for i, value in enumerate(ordered.values):
        ax.text(0, i, fmt.format(value), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# %%
fault_label = pd.Series(feature_matrix.index >= FAULT_EV_ID_CUTOFF, index=feature_matrix.index, name="fault_label")
print(f"{fault_label.sum()}/{len(fault_label)} EVs labeled faulty (EV id >= {FAULT_EV_ID_CUTOFF})")

fault_corr, fault_pval = pearson_fault_correlation(feature_matrix, fault_label)
fault_corr.to_pickle(CORROLATION_DIR / "feature_fault_pearson_corr.pkl")
fault_pval.to_pickle(CORROLATION_DIR / "feature_fault_pearson_pval.pkl")

# %%
plot_fault_correlation_values(
    fault_corr, FIGURES_DIR / "pearson_fault_correlation.png",
    vmin=-1, vmax=1, cmap="RdBu_r",
    colorbar_label="Pearson correlation r",
    title="Pearson correlation: features vs. fault label",
)
plot_fault_correlation_values(
    fault_pval, FIGURES_DIR / "pearson_fault_pvalues.png",
    vmin=0, vmax=1, cmap="Reds_r",
    colorbar_label="p-value",
    title="Pearson correlation p-values: features vs. fault label",
    fmt="{:.1e}",
)
