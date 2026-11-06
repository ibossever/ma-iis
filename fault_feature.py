# %%
"""
Feature-vs-fault-label relevance analysis.

Follow-up to the feature-feature redundancy analysis in spearman_corr.py /
pearson_corr.py (which only look at how features relate to *each other*).
This script instead asks, for every feature in feature_matrix, how well it
relates to the *fault label itself* -- using four complementary angles per
feature (point-biserial correlation, Mann-Whitney U, Cohen's d, AUC-ROC) so
that a feature ranking doesn't depend on picking just one statistic.

Its output (feature_fault_analysis.csv/.pkl) is the input features.py uses
to build filtered_features.pkl -- see features.py's module docstring for how
the columns computed here get combined into a keep/drop decision.

Input:  extracted_data/corrolation/feature_matrix.pkl (full, pre-dedup
        feature set, from spearman_corr.py) -- one row per EV (index = ev_id),
        one column per scalar feature.
Ground truth fault label: EV number >= 335 is faulty, <= 334 is healthy --
        same convention used in detection.py's manual ground-truth check.

Output:
  extracted_data/feature_fault_analysis.pkl   -- summary table (see below)
  extracted_data/feature_fault_analysis.csv   -- same, as CSV
  extracted_data/feature_fault_analysis.html  -- same, pandas-styled
  experiments/01_fault_detection/figures/fault_point_biserial_bar.png
  experiments/01_fault_detection/figures/fault_feature_distributions.png
  experiments/01_fault_detection/figures/fault_r_vs_auc_raw.png       -- raw r vs. raw AUC, no thresholds
  experiments/01_fault_detection/figures/fault_r_vs_auc.png           -- |r| vs. adjusted AUC, thresholds shaded
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, mannwhitneyu
from sklearn.metrics import roc_auc_score

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../fault_feature.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
FEATURE_MATRIX_IN = EXTRACTED_DATA / "corrolation" / "feature_matrix.pkl"  # from spearman_corr.py
OUT_PKL = EXTRACTED_DATA / "feature_fault_analysis.pkl"
OUT_CSV = EXTRACTED_DATA / "feature_fault_analysis.csv"
OUT_HTML = EXTRACTED_DATA / "feature_fault_analysis.html"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change/tune later
# ===========================================================================
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as detection.py

ALPHA = 0.05                    # significance threshold for both p-values
COHENS_D_THRESHOLD = 0.5        # |cohens_d| above this counts toward "large_effect"
POINT_BISERIAL_R_THRESHOLD = 0.2  # |point_biserial_r| above this counts toward "large_effect"
RANK_DISAGREEMENT_THRESHOLD = 5   # |rank_by_r - rank_by_auc| above this is flagged as disagreeing

TOP_N_DISTRIBUTION_PLOTS = 8    # most relevant features shown in the box/violin grid

# "Meaningful" thresholds for the r-vs-AUC scatter plot's shaded quadrant --
# tune freely, these are just reasonable screening defaults.
# R_THRESHOLD: |r| = 0.2 is the conventional (Cohen) small-to-moderate
# correlation cutoff, and matches POINT_BISERIAL_R_THRESHOLD above so the
# scatter plot's shaded zone lines up with the large_effect flag.
R_THRESHOLD = 0.2
# AUC_THRESHOLD: on adjusted_auc (see below), 0.5 = chance and 1.0 = perfect
# separation; 0.7 is the standard rule-of-thumb cutoff for "acceptable"
# single-feature discrimination in diagnostic screening (0.7-0.8 acceptable,
# 0.8-0.9 excellent).
AUC_THRESHOLD = 0.7

FIGURE_DPI = 300  # export resolution for every figure -- 300 stays legible when zoomed in

# significant / not-significant colors for the bar chart, same RWTH palette
# family as the correlation heatmaps (which use RdBu_r)
COLOR_SIGNIFICANT = RWTHColorScales.blue[0]
COLOR_NOT_SIGNIFICANT = RWTHColorScales.blue[3]
COLOR_HEALTHY = RWTHColorScales.blue[0]
COLOR_FAULTY = RWTHColorScales.red[0]
# threshold-based (not significance-based) coloring on the adjusted
# r-vs-AUC scatter -- same blue pair as COLOR_SIGNIFICANT/COLOR_NOT_SIGNIFICANT,
# just driven by a different criterion (meets_both_thresholds)
COLOR_MEETS_THRESHOLD = RWTHColorScales.blue[0]
COLOR_BELOW_THRESHOLD = RWTHColorScales.blue[3]

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def _style_scatter_axes(ax: plt.Axes) -> None:
    """Shared grid/spine styling for the r-vs-AUC scatter plots: light
    dotted gridlines behind the data, no top/right border clutter.
    """
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Cohen's d effect size between two independent samples, using the
    pooled (variance-weighted) standard deviation:

        d = (mean_a - mean_b) / pooled_std
        pooled_std = sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    Positive d means group_a has the larger mean. Called below as
    cohens_d(faulty, healthy), so positive d means the feature is elevated
    in faulty EVs -- same sign convention as point_biserial_r.
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = group_a.var(ddof=1), group_b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    return (group_a.mean() - group_b.mean()) / pooled_std


def compute_feature_fault_stats(feature_matrix: pd.DataFrame, fault_label: pd.Series) -> pd.DataFrame:
    """For every column in `feature_matrix`, compute its relationship to the
    binary `fault_label` (same index) via four independent angles and return
    one row per feature. See module docstring for why each statistic is
    included.
    """
    faulty_mask = fault_label.astype(bool)
    rows = []
    for feature in feature_matrix.columns:
        values = feature_matrix[feature]
        faulty_vals = values[faulty_mask].to_numpy()
        healthy_vals = values[~faulty_mask].to_numpy()

        r, p_r = pointbiserialr(faulty_mask.to_numpy(), values.to_numpy())
        _, p_u = mannwhitneyu(faulty_vals, healthy_vals, alternative="two-sided")
        d = cohens_d(faulty_vals, healthy_vals)
        auc = roc_auc_score(faulty_mask.to_numpy(), values.to_numpy())

        rows.append({
            "feature": feature,
            "point_biserial_r": r,
            "point_biserial_p": p_r,
            "mannwhitney_p": p_u,
            "cohens_d": d,
            "auc_roc": auc,
        })

    summary = pd.DataFrame(rows)
    summary["significant"] = (summary["point_biserial_p"] < ALPHA) & (summary["mannwhitney_p"] < ALPHA)
    summary["large_effect"] = (summary["cohens_d"].abs() > COHENS_D_THRESHOLD) & \
                               (summary["point_biserial_r"].abs() > POINT_BISERIAL_R_THRESHOLD)
    return summary


def rank_and_flag_disagreement(summary: pd.DataFrame) -> pd.DataFrame:
    """Sort `summary` by |point_biserial_r| descending (primary ranking),
    attach a secondary ranking by AUC-ROC separation strength, and flag
    features where the two rankings drift more than
    RANK_DISAGREEMENT_THRESHOLD positions apart -- i.e. a feature that
    "sounds" relevant by one statistic but not the other.

    The secondary ranking uses |auc_roc - 0.5| rather than raw auc_roc.
    Raw AUC is direction-sensitive (a perfectly *negatively* correlated
    feature scores auc_roc ~= 0, not ~= 1), so ranking on it directly would
    flag every negatively-correlated feature as "disagreeing" with r purely
    because of sign, not because it actually separates the classes any
    differently. |auc_roc - 0.5| measures separation strength regardless of
    direction -- the same information as the Mann-Whitney U statistic,
    since AUC = U / (n_faulty * n_healthy) -- so a real disagreement here
    means the feature's *rank* by separation strength doesn't track its
    rank by linear correlation (e.g. a threshold-like or non-monotonic
    relationship that point-biserial r underrates).
    """
    summary = summary.sort_values("point_biserial_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    summary["rank_by_r"] = summary.index + 1
    summary["rank_by_auc"] = summary["auc_roc"].sub(0.5).abs().rank(ascending=False, method="min").astype(int)
    summary["rank_diff"] = (summary["rank_by_r"] - summary["rank_by_auc"]).abs()
    summary["rankings_disagree"] = summary["rank_diff"] > RANK_DISAGREEMENT_THRESHOLD
    return summary


def plot_point_biserial_bar(summary: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of point-biserial r per feature, sorted by
    signed r (so positive/negative bars read as a left/right diverging
    tornado plot) and color-coded by statistical significance.
    """
    ordered = summary.sort_values("point_biserial_r", ascending=True)
    colors = ordered["significant"].map({True: COLOR_SIGNIFICANT, False: COLOR_NOT_SIGNIFICANT})

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(ordered) + 2))
    ax.barh(ordered["feature"], ordered["point_biserial_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Point-biserial correlation r (feature vs. fault label)")
    ax.set_title("Feature-fault point-biserial correlation")

    # manual legend since bar color encodes significance, not a hue column
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_SIGNIFICANT, label=f"significant (p < {ALPHA})"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_NOT_SIGNIFICANT, label="not significant"),
    ]
    ax.legend(handles=handles, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_top_feature_distributions(
    feature_matrix: pd.DataFrame, fault_label: pd.Series, summary: pd.DataFrame,
    out_path: Path, top_n: int,
) -> None:
    """Violin plot (value distribution, faulty vs. healthy) for the top_n
    features by |point_biserial_r| -- all features at once (33 here) would
    be unreadable in a single figure, and the top-N by relevance are the
    ones worth inspecting visually anyway.
    """
    top_features = summary.reindex(summary["point_biserial_r"].abs().sort_values(ascending=False).index) \
                           .head(top_n)["feature"].tolist()

    n_cols = 4
    n_rows = -(-len(top_features) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    group_labels = fault_label.map({False: "healthy", True: "faulty"}).rename("group")
    for ax, feature in zip(axes.flat, top_features):
        plot_df = pd.DataFrame({"group": group_labels, "value": feature_matrix[feature]})
        sns.violinplot(
            data=plot_df, x="group", y="value", order=["healthy", "faulty"],
            hue="group", palette={"healthy": COLOR_HEALTHY, "faulty": COLOR_FAULTY},
            legend=False, ax=ax, cut=0,
        )
        ax.set_title(feature, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # hide any unused trailing subplot axes when top_n isn't a multiple of n_cols
    for ax in axes.flat[len(top_features):]:
        ax.set_visible(False)

    fig.suptitle(f"Top {top_n} features by |point-biserial r|: value distribution by fault label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_r_vs_auc_scatter_raw(summary: pd.DataFrame, out_path: Path) -> None:
    """Original (pre-adjustment) scatter: raw point_biserial_r vs. raw
    auc_roc, each on its natural range, no threshold guides. Kept
    side-by-side with the adjusted version below purely for comparison --
    this is the view where a negatively-correlated feature (auc_roc ~= 0)
    visually reads as "not discriminative", even though it separates the
    classes just as well as a positively-correlated one, only inverted.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = summary["significant"].map({True: COLOR_SIGNIFICANT, False: COLOR_NOT_SIGNIFICANT})

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)

    # zero/chance-level reference lines (r = 0: no correlation; AUC = 0.5:
    # chance-level classifier) -- not the R_THRESHOLD/AUC_THRESHOLD screening
    # cutoffs (those only appear on the adjusted plot below).
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    ax.scatter(summary["point_biserial_r"], summary["auc_roc"], c=colors,
               s=42, edgecolor="white", linewidth=0.5, zorder=2)
    for _, row in summary.iterrows():
        ax.annotate(row["feature"], (row["point_biserial_r"], row["auc_roc"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points", zorder=3)

    ax.set_xlabel("Point-biserial correlation r")
    ax.set_ylabel("AUC-ROC (single feature as classifier, raw)")
    ax.set_title("Point-biserial r vs. raw AUC-ROC (before direction adjustment)")
    _style_scatter_axes(ax)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_SIGNIFICANT, label=f"significant (p < {ALPHA})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_NOT_SIGNIFICANT, label="not significant"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_r_vs_auc_scatter_adjusted(summary: pd.DataFrame, out_path: Path) -> None:
    """Direction-adjusted scatter: both axes rescaled to represent distance
    from their own no-information baseline -- adjusted_r = |r| (baseline 0)
    on x, adjusted_auc = max(auc_roc, 1 - auc_roc) (baseline 0.5) on y --
    so a feature's position reflects discriminative strength alone, with
    sign no longer conflated with magnitude (compare against the raw
    version above, where sign visually distorted "how discriminative" a
    negatively-correlated feature looked).

    Point color now reflects R_THRESHOLD/AUC_THRESHOLD ("meets both
    thresholds" vs. not) instead of statistical significance, so coloring
    agrees with the shaded acceptable zone instead of encoding an unrelated
    criterion.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = summary["meets_both_thresholds"].map({True: COLOR_MEETS_THRESHOLD, False: COLOR_BELOW_THRESHOLD})

    # explicit, padded limits computed up front so the threshold shading
    # below can be placed exactly in axes-fraction coordinates
    x_min, x_max = -0.02, 1.02
    y_min, y_max = 0.48, 1.02
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # shade the "strong candidate" zone: adjusted_r >= R_THRESHOLD AND
    # adjusted_auc >= AUC_THRESHOLD -- just the top-right corner now, since
    # both axes are already direction-agnostic (no mirrored left side left
    # to shade) -- drawn first (zorder=0) so points/labels sit on top of it.
    auc_frac = (AUC_THRESHOLD - y_min) / (y_max - y_min)
    ax.axvspan(R_THRESHOLD, x_max, ymin=auc_frac, ymax=1, color=COLOR_MEETS_THRESHOLD, alpha=0.12, zorder=0)

    # threshold guides -- thinner than the shaded zone's border needs, since
    # the shading itself already carries most of the visual weight here
    ax.axvline(R_THRESHOLD, color="black", linewidth=0.7, linestyle="--", zorder=1)
    ax.axhline(AUC_THRESHOLD, color="black", linewidth=0.7, linestyle="--", zorder=1)

    ax.scatter(summary["adjusted_r"], summary["adjusted_auc"], c=colors,
               s=42, edgecolor="white", linewidth=0.5, zorder=2)
    for _, row in summary.iterrows():
        ax.annotate(row["feature"], (row["adjusted_r"], row["adjusted_auc"]),
                    fontsize=6.5, xytext=(3, 3), textcoords="offset points", zorder=3)

    ax.set_xlabel("Adjusted |r| = |point-biserial r|  (distance from 0, direction-agnostic)")
    ax.set_ylabel("Adjusted AUC-ROC = max(AUC, 1 - AUC)  (distance from 0.5, direction-agnostic)")
    ax.set_title("Discriminative strength: |point-biserial r| vs. AUC-ROC")
    _style_scatter_axes(ax)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_MEETS_THRESHOLD,
                   label=f"meets both thresholds (|r| ≥ {R_THRESHOLD}, AUC ≥ {AUC_THRESHOLD})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_BELOW_THRESHOLD,
                   label="below at least one threshold"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def export_summary_table(summary: pd.DataFrame, csv_path: Path, html_path: Path) -> None:
    """Write the summary table as plain CSV plus a pandas-styled HTML
    table: signed-value columns get the same RdBu_r diverging color scheme
    as the Pearson/Spearman heatmaps, p-values in scientific notation, and
    significant/large-effect rows highlighted for quick scanning.
    """
    summary.to_csv(csv_path, index=False)

    styled = (
        summary.style
        .background_gradient(subset=["point_biserial_r"], cmap="RdBu_r", vmin=-1, vmax=1)
        .background_gradient(subset=["cohens_d"], cmap="RdBu_r", vmin=-2, vmax=2)
        .background_gradient(subset=["auc_roc"], cmap="RdBu_r", vmin=0, vmax=1)
        .format({
            "point_biserial_r": "{:.3f}", "cohens_d": "{:.3f}", "auc_roc": "{:.3f}",
            "point_biserial_p": "{:.2e}", "mannwhitney_p": "{:.2e}",
        })
        .apply(lambda col: ["font-weight: bold" if v else "" for v in summary["significant"]], subset=["feature"])
        .set_caption("Feature-vs-fault-label relevance (bold feature = significant in both tests)")
    )
    styled.to_html(html_path)


# %%
feature_matrix = pd.read_pickle(FEATURE_MATRIX_IN)

# feature_matrix.pkl is produced clean by spearman_corr.py, but guard anyway
# since it's a separate, still-evolving upstream script.
if feature_matrix.isna().any().any():
    n_before = len(feature_matrix)
    feature_matrix = feature_matrix.dropna()
    print(f"dropped {n_before - len(feature_matrix)} EVs with missing feature values")

fault_label = pd.Series(feature_matrix.index >= FAULT_EV_ID_CUTOFF, index=feature_matrix.index, name="fault_label")
print(f"{fault_label.sum()}/{len(fault_label)} EVs labeled faulty (EV id >= {FAULT_EV_ID_CUTOFF})")

# %% 1. per-feature statistics
summary = compute_feature_fault_stats(feature_matrix, fault_label)

# %% 2+3. flags, then sort by |point_biserial_r| and attach the AUC-ROC secondary ranking
summary = rank_and_flag_disagreement(summary)

# adjusted_auc = max(auc_roc, 1 - auc_roc): direction-agnostic discriminative
# strength on [0.5, 1], used for the r-vs-AUC scatter plot below. Kept as its
# own column (auc_roc itself is left untouched) so a feature's distance from
# 0.5 reflects how well it separates the classes regardless of which
# direction (high or low values -> fault) that separation runs.
summary["adjusted_auc"] = np.maximum(summary["auc_roc"], 1 - summary["auc_roc"])

# adjusted_r = |r|: the analogous "distance from the no-information
# baseline" treatment for r, whose baseline is 0 (no correlation), not 0.5.
# r's domain is [-1, 1], not [0, 1] like auc_roc, so max(r, 1 - r) isn't the
# right formula here -- |r| is the actual analogue.
summary["adjusted_r"] = summary["point_biserial_r"].abs()

# feature falls in the scatter plot's shaded "strong candidate" zone --
# used both for the shading itself and (on the adjusted plot) for point
# color, so the two agree instead of encoding different criteria.
summary["meets_both_thresholds"] = (summary["adjusted_r"] >= R_THRESHOLD) & (summary["adjusted_auc"] >= AUC_THRESHOLD)

summary.to_pickle(OUT_PKL)

n_significant = summary["significant"].sum()
n_large_effect = summary["large_effect"].sum()
n_disagree = summary["rankings_disagree"].sum()
print(f"{n_significant}/{len(summary)} features significant in both tests (p < {ALPHA})")
print(f"{n_large_effect}/{len(summary)} features flagged large_effect "
      f"(|d| > {COHENS_D_THRESHOLD} and |r| > {POINT_BISERIAL_R_THRESHOLD})")
print(f"{n_disagree}/{len(summary)} features where r-rank and AUC-rank "
      f"disagree by more than {RANK_DISAGREEMENT_THRESHOLD} positions")
if n_disagree:
    print(summary.loc[summary["rankings_disagree"],
                       ["feature", "point_biserial_r", "rank_by_r", "auc_roc", "rank_by_auc"]].to_string(index=False))

# %% 4. plots
plot_point_biserial_bar(summary, FIGURES_DIR / "fault_point_biserial_bar.png")
plot_top_feature_distributions(feature_matrix, fault_label, summary,
                                FIGURES_DIR / "fault_feature_distributions.png", TOP_N_DISTRIBUTION_PLOTS)
plot_r_vs_auc_scatter_raw(summary, FIGURES_DIR / "fault_r_vs_auc_raw.png")
plot_r_vs_auc_scatter_adjusted(summary, FIGURES_DIR / "fault_r_vs_auc.png")

# %% 5. export table
export_summary_table(summary, OUT_CSV, OUT_HTML)
