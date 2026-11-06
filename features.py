# %%
"""
Build filtered_features.pkl: subset feature_matrix.pkl down to the features
that clear the fault-relevance screening in feature_fault_analysis.csv, then
collapse any features that are still redundant with each other within that
kept set.

Step 1 (relevance): every screening decision (which p-value counts as
significant, what effect size is "large", what AUC-ROC is "acceptable"
discrimination) is already made in fault_feature.py and baked into
feature_fault_analysis.csv's boolean columns -- this script only combines
those columns into one keep/drop decision per feature (see KEEP below) and
subsets feature_matrix accordingly.

Step 2 (redundancy, last step): relevance alone says nothing about whether
two *kept* features carry the same information. spearman_corr.py and
pearson_corr.py each independently report groups of mutually redundant
features (rank-based and linear correlation respectively) without acting on
them. Here, restricted to the features that survived step 1, a group is
only deduped down to one representative when spearman_corr.py's and
pearson_corr.py's reports *agree* that every pair in the group is
redundant -- the representative is the group member with the highest
combined relevance/robustness score (see SELECTION_SCORE below). Every
merge is printed as it happens.

Nothing here is a hand-picked list, so rerunning this script after its
upstream inputs change reproduces filtered_features.pkl deterministically,
with no manual step.

Input:
  extracted_data/corrolation/feature_matrix.pkl (from spearman_corr.py) --
      one row per EV (index = ev_id), one column per scalar feature.
  extracted_data/feature_fault_analysis.csv (from fault_feature.py) -- one
      row per feature, with the "significant" / "large_effect" /
      "meets_both_thresholds" flags combined below, plus the "adjusted_r" /
      "adjusted_auc" / "rank_diff" columns used to score step 2.
  extracted_data/corrolation/feature_matrix_spearman_corr.pkl (from
      spearman_corr.py) and feature_matrix_pearson_corr.pkl (from
      pearson_corr.py) -- full feature-feature correlation matrices, used
      only in step 2, and only for the columns that survived step 1.
Output:
  extracted_data/fault_detection/filtered_features.pkl -- feature_matrix
      subset to the features kept after both steps, in feature_matrix's
      original column order. Overwritten in place on every run.

See the bottom of this file for a summary of how the filtering works.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
FEATURE_MATRIX_IN = EXTRACTED_DATA / "corrolation" / "feature_matrix.pkl"  # from spearman_corr.py
FAULT_ANALYSIS_IN = EXTRACTED_DATA / "feature_fault_analysis.csv"  # from fault_feature.py
SPEARMAN_CORR_IN = EXTRACTED_DATA / "corrolation" / "feature_matrix_spearman_corr.pkl"  # from spearman_corr.py
PEARSON_CORR_IN = EXTRACTED_DATA / "corrolation" / "feature_matrix_pearson_corr.pkl"  # from pearson_corr.py
FILTERED_FEATURES_OUT = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"
FILTERED_FEATURES_OUT.parent.mkdir(exist_ok=True)

# ===========================================================================
# CONFIG -- step 1: which of feature_fault_analysis.csv's screening flags a
# feature must pass to be kept. Each flag already bakes in specific
# thresholds, set and explained in fault_feature.py's own CONFIG section
# (ALPHA, COHENS_D_THRESHOLD, POINT_BISERIAL_R_THRESHOLD, R_THRESHOLD,
# AUC_THRESHOLD) -- see the summary at the bottom of this file for how they
# combine into the three flags used here.
# ===========================================================================
REQUIRE_SIGNIFICANT = True             # both point_biserial_p and mannwhitney_p < ALPHA (0.05)
REQUIRE_LARGE_EFFECT = True            # |cohens_d| > 0.5 AND |point_biserial_r| > 0.2
REQUIRE_MEETS_BOTH_THRESHOLDS = True   # |point_biserial_r| >= 0.2 AND adjusted AUC-ROC >= 0.7

# ===========================================================================
# CONFIG -- step 2 (last step): redundancy dedup within the kept set. Same
# corr_threshold as spearman_corr.py's / pearson_corr.py's own reports, so
# "agree" below means agreeing at the same cutoff, not a looser/stricter one.
# ===========================================================================
DEDUPE_REDUNDANT_FEATURES = True   # set False to skip step 2 entirely
REDUNDANCY_CORR_THRESHOLD = 0.8    # |correlation| at/above this counts as redundant, in EACH of spearman/pearson


def find_redundant_groups_both_agree(
    features: list[str], corr_a: pd.DataFrame, corr_b: pd.DataFrame, threshold: float,
) -> list[list[str]]:
    """Union-find over `features`: two features are merged into the same
    redundant group only when BOTH corr_a and corr_b place |correlation| at
    or above `threshold` for that pair. Chains merge transitively (A-B and
    B-C both agree-redundant puts all three in one group, regardless of
    whether A-C itself clears the threshold), same as the hierarchical
    clustering in spearman_corr.py / pearson_corr.py's own
    find_redundant_feature_groups -- this just additionally requires
    agreement between the two correlation measures before a pair merges,
    since a pair that is e.g. strongly monotonic but only weakly linear (or
    vice versa) shouldn't be collapsed on one test's say-so alone.
    """
    parent = {f: f for f in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if abs(corr_a.loc[f1, f2]) >= threshold and abs(corr_b.loc[f1, f2]) >= threshold:
                union(f1, f2)

    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(find(f), []).append(f)
    return [sorted(members) for members in groups.values() if len(members) > 1]


# %%
feature_matrix = pd.read_pickle(FEATURE_MATRIX_IN)
fault_analysis = pd.read_csv(FAULT_ANALYSIS_IN).set_index("feature")

# feature_fault_analysis.csv is a separate, upstream-generated file -- guard
# against it drifting out of sync with feature_matrix.pkl (e.g. a new
# feature table dropped into whole_features/ since fault_feature.py last
# ran) rather than silently mis-scoring or crashing on a missing flag.
unscored = feature_matrix.columns.difference(fault_analysis.index)
if len(unscored):
    print(f"{len(unscored)} feature(s) in feature_matrix.pkl have no row in "
          f"feature_fault_analysis.csv, excluded until it is rerun: {list(unscored)}")

# %% step 1: relevance screening
keep = pd.Series(True, index=fault_analysis.index)
if REQUIRE_SIGNIFICANT:
    keep &= fault_analysis["significant"]
if REQUIRE_LARGE_EFFECT:
    keep &= fault_analysis["large_effect"]
if REQUIRE_MEETS_BOTH_THRESHOLDS:
    keep &= fault_analysis["meets_both_thresholds"]

kept_features = [f for f in feature_matrix.columns if keep.get(f, False)]
n_total = feature_matrix.shape[1]
print(f"step 1 (relevance): kept {len(kept_features)}/{n_total} features")
print(f"  dropped: {[f for f in feature_matrix.columns if f not in kept_features]}")

# %% step 2 (last step): redundancy dedup within the kept set -- only when
# spearman_corr.py and pearson_corr.py agree a pair is redundant
if DEDUPE_REDUNDANT_FEATURES and len(kept_features) > 1:
    spearman_corr = pd.read_pickle(SPEARMAN_CORR_IN).loc[kept_features, kept_features]
    pearson_corr = pd.read_pickle(PEARSON_CORR_IN).loc[kept_features, kept_features]

    redundant_groups = find_redundant_groups_both_agree(
        kept_features, spearman_corr, pearson_corr, REDUNDANCY_CORR_THRESHOLD,
    )

    if redundant_groups:
        # relevance: how strongly a feature relates to the fault label,
        # direction-agnostic -- adjusted_r (= |point_biserial_r|) is already
        # in [0, 1]; adjusted_auc is in [0.5, 1] and rescaled the same way
        # ((adjusted_auc - 0.5) * 2) so the two average on a common [0, 1]
        # scale instead of adjusted_auc dominating by starting at 0.5.
        relevance = (fault_analysis["adjusted_r"] + (fault_analysis["adjusted_auc"] - 0.5) * 2) / 2
        # robustness: how well the two independent rankings from
        # fault_feature.py (by |r|, by AUC-separation) agree on this
        # feature -- rank_diff is |rank_by_r - rank_by_auc|, so a feature
        # whose apparent relevance doesn't depend on which single statistic
        # ranks it is treated as more robust than one where the two tests
        # disagree.
        max_rank_diff = fault_analysis["rank_diff"].max() or 1
        robustness = 1 - fault_analysis["rank_diff"] / max_rank_diff
        selection_score = (relevance + robustness) / 2

        print(f"step 2 (redundancy dedup): {len(redundant_groups)} group(s) where spearman_corr.py and "
              f"pearson_corr.py both agree |correlation| >= {REDUNDANCY_CORR_THRESHOLD}")
        removed_by_dedup: list[str] = []
        for group in redundant_groups:
            winner = selection_score.loc[group].idxmax()
            losers = [f for f in group if f != winner]
            removed_by_dedup.extend(losers)
            scores = {f: round(float(selection_score[f]), 3) for f in group}
            # comment-style report: which features were deduped from which
            # redundant set, and which one was selected as representative
            print(f"  # redundant set {group} (scores={scores}) "
                  f"-> selected {winner!r}, deduped {losers}")

        kept_features = [f for f in kept_features if f not in removed_by_dedup]
        print(f"  deduped {len(removed_by_dedup)} feature(s), {len(kept_features)} remain")
    else:
        print(f"step 2 (redundancy dedup): no group where spearman_corr.py and pearson_corr.py both agree "
              f"|correlation| >= {REDUNDANCY_CORR_THRESHOLD} -- nothing to dedupe")

# %%
filtered_features = feature_matrix.loc[:, kept_features]
filtered_features.to_pickle(FILTERED_FEATURES_OUT)

n_kept = filtered_features.shape[1]
print(f"kept {n_kept}/{n_total} features in filtered_features.pkl")
print(f"final features: {filtered_features.columns.tolist()}")

# ===========================================================================
# SUMMARY -- how filtered_features.pkl is chosen and why
# ===========================================================================
# STEP 1 -- relevance. A feature from feature_matrix.pkl survives this step
# only if ALL THREE of the flags below (computed once, in fault_feature.py,
# against the EV-id fault label convention used throughout
# 01_fault_detection/) are True:
#
# 1. significant       point_biserial_p < 0.05 AND mannwhitney_p < 0.05
#                       Two different tests of "this feature's distribution
#                       actually differs between healthy and faulty EVs, not
#                       by chance": point-biserial assumes a roughly linear
#                       relationship to the binary label, Mann-Whitney U
#                       makes no distributional assumption at all. Requiring
#                       both guards against either test's blind spot (e.g.
#                       point-biserial missing a non-monotonic split, or
#                       Mann-Whitney flagging a difference driven purely by
#                       a handful of outliers).
#
# 2. large_effect       |cohens_d| > 0.5  AND  |point_biserial_r| > 0.2
#                       Significance alone isn't enough: with ~500 EVs, a
#                       trivially small difference can still be "significant"
#                       (p < 0.05 is about sample size, not size of effect).
#                       Cohen's d (standardized mean difference) and |r|
#                       (correlation strength) each have to clear their own
#                       conventional small-to-moderate cutoff, so a feature
#                       also has to separate the two groups by a
#                       *meaningful* amount, not just a detectable one.
#
# 3. meets_both_thresholds   |point_biserial_r| >= 0.2  AND  adjusted AUC-ROC >= 0.7
#                       A third, independent check that the feature is
#                       usable as a single-feature classifier on its own:
#                       AUC-ROC (direction-adjusted, so a feature that's
#                       elevated in *healthy* EVs isn't penalized for its
#                       sign) has to clear 0.7, the standard rule-of-thumb
#                       floor for "acceptable" diagnostic discrimination.
#                       This catches cases (2) can miss -- e.g. a
#                       threshold-like or non-monotonic relationship where
#                       Cohen's d / point-biserial r understate how well the
#                       feature actually separates the classes.
#
# Requiring all three (rather than any one, or a weighted score) means a
# feature only survives step 1 if it looks relevant from every angle tested
# -- deliberately conservative, since feeding IsolationForest (used
# downstream in detection.py) too many weak/noisy dimensions dilutes the few
# genuinely informative ones (curse of dimensionality, with only ~500 EVs to
# work with). Each of the three booleans, and the numeric thresholds behind
# them, can be toggled independently via the CONFIG flags above.
#
# STEP 2 -- redundancy dedup (last step). Passing step 1 says a feature is
# relevant, not that it's non-redundant: two kept features can both be
# individually relevant while carrying almost the same information. This
# step only merges a group of kept features when BOTH the rank-based
# (spearman_corr.py) and linear (pearson_corr.py) correlation reports agree
# every pair in the group is at or above REDUNDANCY_CORR_THRESHOLD --
# requiring agreement (rather than acting on either report alone) avoids
# collapsing a pair that is e.g. strongly monotonic but only weakly linear,
# where the two methods are picking up genuinely different structure. Within
# an agreed-redundant group, the representative kept is whichever member has
# the highest combined score of:
#   - relevance:  mean of adjusted_r and adjusted_auc (both rescaled to
#                 [0, 1], direction-agnostic) -- how strongly the feature
#                 relates to the fault label.
#   - robustness: 1 - (rank_diff / max rank_diff) -- how well the feature's
#                 r-based and AUC-based rankings from fault_feature.py agree
#                 with each other, i.e. how much its apparent relevance
#                 depends on which single statistic is used to judge it.
# Every merge -- the redundant set, each member's score, which one was
# selected, and which were dropped -- is printed when this step runs.
