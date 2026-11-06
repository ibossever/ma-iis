# %%
"""
SHAP explainability for detection.py's IsolationForest anomaly scores.

Continues branch 1 of the pipeline (detection.py -> this script): for every
EV flagged as faulty by detection.py, decompose *why* it got that anomaly
score into per-feature contributions, so a flagged EV isn't just a number
but a short list of "these features look off relative to normal operation".

--- Why TreeExplainer needs a sign flip -----------------------------------
SHAP's TreeExplainer supports IsolationForest, but it only ever explains the
model's *raw* per-tree output -- average isolation path length across the
forest -- not the anomaly_score detection.py actually thresholds on
(anomaly_score = -model.score_samples(X), higher = more anomalous). Path
length and anomaly_score move in opposite directions (a SHORT average path
means MORE anomalous), and the map from one to the other is a monotonic but
NONLINEAR normalization (sklearn's 2^(-depth/c(n)) transform), not a
constant offset. So:
  1. every raw TreeExplainer value here is negated before use, so a
     POSITIVE signed value means "pushes this EV's anomaly_score UP" --
     consistent with detection.py's convention -- instead of the raw
     "pushes path length up" (= more normal) convention shap returns by
     default.
  2. because the depth->score transform is nonlinear, the signed values are
     an additive decomposition of path length, not an exact additive
     decomposition of anomaly_score itself -- feature RANKING and DIRECTION
     are trustworthy, exact magnitudes/sums are not. This is checked below
     (validate_ranking_faithfulness) via the Spearman correlation between
     each EV's total signed contribution and its actual anomaly_score,
     which should be strongly negative... no -- strongly monotonic (see
     that function for the precise claim) rather than assumed.

--- Why the background is "healthy-per-detection.py", not "all EVs" -------
TreeExplainer's `feature_perturbation="interventional"` mode needs a
reference/background dataset: SHAP values become "how does this EV's
feature vector differ FROM THE BACKGROUND", so the choice of background is
a real modeling decision, not a default to accept blindly. Two candidates:
  - all 393 EVs (fleet-wide baseline): mixes healthy and already-anomalous
    EVs into "normal", so a flagged EV gets explained relative to a
    baseline that's itself partly anomalous.
  - only the EVs detection.py did NOT flag (is_fault == False): explains a
    flagged EV purely in terms of "how does this deviate from what
    detection.py itself currently calls normal operation" -- directly
    actionable, and self-consistent with detection.py's own is_fault split
    rather than importing the separate id>=335 ground-truth convention used
    elsewhere in 01_fault_detection/ for validation only.
This script uses the second (healthy-per-detection.py) as the default.
See ALTERNATIVE METHODS below for how to switch back to the first, and for
a slower but exact-target cross-check.

--- ALTERNATIVE METHODS (why these, and how to switch) --------------------
1. (default, used below) TreeExplainer, feature_perturbation="interventional",
   background = detection.py's healthy EVs, sign-flipped. Fast (milliseconds
   for all 393 EVs), exact Shapley values -- but for path length, not
   anomaly_score (see above).
2. feature_perturbation="tree_path_dependent" (set BACKGROUND_MODE below):
   drops the need to pick a background population at all -- SHAP instead
   walks each tree's own training-data split proportions. Marginally faster,
   and removes the "which background?" decision entirely, at the cost of
   implicitly using the *whole* fit population (all 393 EVs, healthy +
   faulty) as the implied reference, same tradeoff as the "all EVs"
   background option above.
3. (implemented below, RUN_PERMUTATION_CROSSCHECK) shap.explainers.Permutation
   directly on the callable anomaly_score function (-model.score_samples),
   rather than through the tree structure. This is the model-agnostic
   alternative: it explains the ACTUAL anomaly_score, not a path-length
   proxy, so it doesn't need the sign flip or the nonlinearity caveat above
   -- but Shapley values are estimated by sampling feature coalitions
   (approximate, not exact) and it's orders of magnitude slower, so it's
   only run here for a handful of the most-anomalous flagged EVs, as a
   cross-check on whether TreeExplainer's fast proxy agrees with the real
   target where it matters most.

Input:
  extracted_data/fault_detection/filtered_features.pkl (features.py) --
      one row per EV (index = ev_id), one column per scalar feature.
  extracted_data/fault_detection/fault_scores.pkl (detection.py) -- used to
      pick the flagged EVs to explain and the healthy background; a fresh
      IsolationForest is refit here (same config) rather than re-loading a
      saved model object, and cross-checked against this file to catch the
      two scripts' configs drifting out of sync.
Output:
  extracted_data/early_warning/shap_values.pkl -- signed per-EV per-feature
      SHAP contributions to anomaly_score (TreeExplainer, sign-flipped),
      indexed like filtered_features.pkl, plus a base_value column.
  experiments/02_early_warning/figures/shap_beeswarm_all_evs.png
  experiments/02_early_warning/figures/shap_bar_flagged_mean_abs.png
  experiments/02_early_warning/figures/shap_faithfulness_check.png
  experiments/02_early_warning/figures/shap_dependence/shap_dependence_{feature}.png --
      per-feature scatter of feature value vs. signed SHAP contribution, for
      the DEPENDENCE_TOP_K globally most important features -- surfaces
      thresholds ("once feature X drops below Y, its contribution to
      anomaly_score turns sharply positive"), not just a global ranking.
  experiments/02_early_warning/figures/shap_heatmap_flagged.png -- EVs (rows,
      most anomalous first) x features (columns) heatmap of signed SHAP
      values over the flagged EVs, capped at HEATMAP_MAX_EVS -- shows which
      features recur as drivers across many EVs vs. idiosyncratic single-EV
      faults.
  experiments/02_early_warning/figures/shap_waterfalls/shap_waterfall_ev{id}.png
  experiments/02_early_warning/figures/shap_decision_top_flagged.png -- one
      cumulative base_value -> anomaly_score path per top-DECISION_TOP_N
      flagged EV, overlaid, to compare whether the same features drive every
      flagged EV or different EVs have different failure signatures.
  experiments/02_early_warning/figures/shap_permutation_crosscheck.png (if
      RUN_PERMUTATION_CROSSCHECK)
  experiments/02_early_warning/figures/shap_windowed_trend_ev{id}.png (if the
      windowed track's artifacts exist) -- signed SHAP contribution of that
      EV's own top contributing features across its window history, with
      fault-flagged windows shaded, to check whether those contributions
      already rise BEFORE the alert first fires (the literal early-warning
      question) rather than only once it does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This file is itself named shap.py, which would otherwise shadow the
# installed `shap` package -- python auto-inserts this file's own directory
# at sys.path[0] when run directly, and that entry gets checked before
# site-packages. Drop it before importing the real library.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
import shap
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/02_.../shap.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
FEATURES_IN = EXTRACTED_DATA / "fault_detection" / "filtered_features.pkl"  # from features.py
FAULT_SCORES_IN = EXTRACTED_DATA / "fault_detection" / "fault_scores.pkl"  # from detection.py
OUT_DIR = EXTRACTED_DATA / "early_warning"
OUT_DIR.mkdir(exist_ok=True)
SHAP_VALUES_OUT = OUT_DIR / "shap_values.pkl"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
WATERFALL_DIR = FIGURES_DIR / "shap_waterfalls"
WATERFALL_DIR.mkdir(exist_ok=True)

# Isolation Forest params -- MUST match detection.py's own CONFIG exactly,
# so the model refit here reproduces detection.py's saved fault_scores.pkl
# bit-for-bit (checked below rather than assumed).
N_ESTIMATORS = 100
RANDOM_STATE = 0

# feature_perturbation mode for the primary TreeExplainer -- see
# "ALTERNATIVE METHODS" above. "interventional" needs BACKGROUND (below);
# "tree_path_dependent" ignores it.
BACKGROUND_MODE = "interventional"  # "interventional" | "tree_path_dependent"

TOP_N_WATERFALL = 10  # most-anomalous flagged EVs to render individual waterfall plots for

DEPENDENCE_TOP_K = 6  # most globally important features (by mean |signed SHAP|, all EVs) to render dependence scatter plots for
DEPENDENCE_DIR = FIGURES_DIR / "shap_dependence"
DEPENDENCE_DIR.mkdir(exist_ok=True)

HEATMAP_MAX_EVS = 60  # cap on flagged EVs shown in the heatmap so per-EV rows stay legible; keeps the most anomalous ones if there are more

DECISION_TOP_N = TOP_N_WATERFALL  # most-anomalous flagged EVs to overlay on the decision plot

RUN_PERMUTATION_CROSSCHECK = True   # alternative 3 above -- slower, exact-target sanity check
PERMUTATION_TOP_N = 5               # only cross-check the N most anomalous flagged EVs
PERMUTATION_BACKGROUND_SIZE = 50    # background sample size for the permutation explainer
PERMUTATION_RANDOM_STATE = 0

FIGURE_DPI = 300

plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def fit_isolation_forest(features: pd.DataFrame) -> IsolationForest:
    """Same fit as detection.py's detect_faults() -- refit here (rather than
    loading a saved model object, since detection.py doesn't pickle one) so
    SHAP has a live model to walk. Deterministic given the same RANDOM_STATE
    and input, so this reproduces detection.py's fault_scores.pkl exactly;
    verified below rather than assumed.
    """
    model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(features)
    return model


def explain_anomaly_contributions(
    model: IsolationForest, features: pd.DataFrame, background: pd.DataFrame | None,
) -> shap.Explanation:
    """Build the primary (TreeExplainer) explanation, sign-flipped into
    "contribution to anomaly_score" convention (positive = more anomalous)
    -- see module docstring for why the flip is needed and what it does and
    doesn't guarantee.
    """
    if BACKGROUND_MODE == "interventional":
        # Explicit Independent masker with max_samples = full background size --
        # otherwise shap silently subsamples to its default of 100, which would
        # make the background (and therefore every SHAP value) depend on an
        # unconfigured, unreported subsample instead of all healthy EVs.
        masker = shap.maskers.Independent(background, max_samples=len(background))
        explainer = shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")
    elif BACKGROUND_MODE == "tree_path_dependent":
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    else:
        raise ValueError(f"unknown BACKGROUND_MODE {BACKGROUND_MODE!r}")

    raw_values = explainer.shap_values(features)  # explains avg path length, NOT anomaly_score -- see docstring
    raw_base = np.asarray(explainer.expected_value).reshape(-1)[0]

    return shap.Explanation(
        values=-raw_values,
        base_values=np.full(len(features), -raw_base),
        data=features.to_numpy(),
        feature_names=list(features.columns),
    )


def validate_ranking_faithfulness(shap_values: shap.Explanation, anomaly_score: np.ndarray, out_path: Path) -> float:
    """Sanity check for the sign-flip claim in the module docstring: total
    signed contribution per EV (sum of signed SHAP values + base value)
    should be a MONOTONIC function of the actual anomaly_score, even though
    it isn't an exact additive reconstruction of it (nonlinear depth->score
    transform). Spearman rank correlation is the right statistic for
    "monotonic but not linear"; a value near 1 confirms the sign-flip is
    doing its job (higher total signed contribution <-> higher actual
    anomaly_score), a value far from 1 would mean TreeExplainer's ranking
    can't be trusted as a stand-in for anomaly_score and the permutation
    cross-check below should be weighted more heavily.
    """
    total_signed = shap_values.values.sum(axis=1) + shap_values.base_values
    rho, _ = spearmanr(total_signed, anomaly_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(total_signed, anomaly_score, s=18, alpha=0.7, color=RWTHColorScales.blue[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("total signed SHAP contribution (sum of signed values + base value)")
    ax.set_ylabel("actual anomaly_score (detection.py)")
    ax.set_title(f"SHAP ranking faithfulness check -- Spearman ρ = {rho:.3f}")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return rho


def run_permutation_crosscheck(
    model: IsolationForest, features: pd.DataFrame, background: pd.DataFrame, target_ev_ids: list,
) -> shap.Explanation:
    """Alternative 3 from the module docstring: model-agnostic SHAP directly
    on the actual anomaly_score callable, for a handful of EVs only (this is
    ~100-1000x slower per row than TreeExplainer, so it's not run for all
    393 EVs). Returns an Explanation restricted to `target_ev_ids`, in the
    same row order.
    """
    def anomaly_score_fn(X: np.ndarray) -> np.ndarray:
        return -model.score_samples(X)

    bg_sample = shap.sample(background, min(PERMUTATION_BACKGROUND_SIZE, len(background)), random_state=PERMUTATION_RANDOM_STATE)
    explainer = shap.explainers.Permutation(
        anomaly_score_fn, bg_sample, feature_names=list(features.columns), seed=PERMUTATION_RANDOM_STATE,
    )
    target_rows = features.loc[target_ev_ids]
    return explainer(target_rows, max_evals=2 * features.shape[1] + 1)


def plot_permutation_comparison(
    tree_values: shap.Explanation, permutation_values: shap.Explanation,
    target_ev_ids: list, out_path: Path,
) -> float:
    """Scatter of (TreeExplainer signed value, Permutation value) across
    every (EV, feature) pair in the cross-check set, with a y=x reference
    line -- points near the line mean the fast proxy and the slow
    exact-target method agree on both direction and rough magnitude for
    that feature/EV.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    x_vals, y_vals = [], []
    for row_i in range(len(target_ev_ids)):
        x_vals.extend(tree_values.values[row_i])
        y_vals.extend(permutation_values.values[row_i])
    x_vals, y_vals = np.array(x_vals), np.array(y_vals)

    rho, _ = spearmanr(x_vals, y_vals)
    lim = max(np.abs(x_vals).max(), np.abs(y_vals).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1, label="y = x")
    ax.scatter(x_vals, y_vals, s=20, alpha=0.7, color=RWTHColorScales.red[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("TreeExplainer signed value (fast, path-length proxy)")
    ax.set_ylabel("Permutation value (slow, exact anomaly_score target)")
    ax.set_title(f"TreeExplainer vs. Permutation SHAP agreement ({len(target_ev_ids)} EVs) -- Spearman ρ = {rho:.3f}")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return rho


def plot_shap_dependence(shap_values: shap.Explanation, is_fault: pd.Series, top_k: int, out_dir: Path) -> list:
    """Per-feature scatter of feature value vs. signed SHAP contribution, for
    the top_k globally most important features (by mean |signed SHAP| across
    all EVs). The global bar/beeswarm plots answer "which features matter",
    this answers "at what feature VALUE does the contribution turn positive"
    -- the actionable, threshold-shaped claim an early-warning system needs,
    since a ranking alone doesn't say when to raise an alarm. Points are
    colored by detection.py's is_fault split (not the id>=335 ground truth)
    for consistency with the rest of this script.
    """
    is_fault_arr = is_fault.to_numpy()
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top_feature_idx = np.argsort(mean_abs)[::-1][:top_k]

    saved_features = []
    for feat_idx in top_feature_idx:
        feat_name = shap_values.feature_names[feat_idx]
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, mask, color in (
            ("healthy (detection.py)", ~is_fault_arr, RWTHColorScales.blue[0]),
            ("flagged", is_fault_arr, RWTHColorScales.red[0]),
        ):
            ax.scatter(
                shap_values.data[mask, feat_idx], shap_values.values[mask, feat_idx],
                s=20, alpha=0.7, color=color, edgecolor="white", linewidth=0.3, label=label,
            )
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("SHAP contribution to anomaly_score")
        ax.set_title(f"SHAP dependence -- {feat_name}\n(positive = pushes anomaly_score up)")
        ax.legend()
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        fig.tight_layout()
        safe_name = feat_name.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"shap_dependence_{safe_name}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        saved_features.append(feat_name)
    return saved_features


def plot_shap_heatmap(
    shap_values: shap.Explanation, flagged_ev_ids: list, ev_id_to_row: dict, max_evs: int, out_path: Path,
) -> int:
    """Fleet-wide EV x feature heatmap of signed SHAP values, restricted to
    flagged EVs and ordered by anomaly_score (most anomalous first, capped at
    max_evs). Where the per-EV waterfalls tell one EV's story, this shows
    whether the same handful of features recur as drivers across MANY
    flagged EVs (a candidate fleet-wide early-warning indicator) or whether
    each EV's flag is driven by a different, idiosyncratic feature.
    """
    ev_ids_by_severity = flagged_ev_ids[:max_evs]  # already sorted by anomaly_score descending
    rows = [ev_id_to_row[eid] for eid in ev_ids_by_severity]
    subset = shap.Explanation(
        values=shap_values.values[rows],
        base_values=shap_values.base_values[rows],
        data=shap_values.data[rows],
        feature_names=shap_values.feature_names,
    )
    # instance_order=identity keeps the anomaly_score ordering already applied
    # above, instead of shap's default hierarchical-clustering reorder.
    shap.plots.heatmap(subset, show=False, instance_order=np.arange(len(rows)), max_display=15)
    fig = plt.gcf()
    # shap.plots.heatmap's own axes sit mid-canvas on a much taller fixed
    # figure -- a fig.suptitle() above y=1.0 makes bbox_inches="tight" stretch
    # to include that far-away point, reintroducing all the blank canvas
    # space it would otherwise crop. Titling the actual (widest) content axes
    # keeps the title inside the region tight-bbox already crops to.
    main_ax = max(fig.axes, key=lambda a: a.get_position().width)
    main_ax.set_title(
        f"SHAP heatmap, {len(rows)} flagged EVs ordered by anomaly_score\n(most anomalous first, left to right)",
        pad=14,
    )
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return len(rows)


def plot_shap_decision(
    shap_values: shap.Explanation, flagged_ev_ids: list, ev_id_to_row: dict, top_n: int, out_path: Path,
) -> list:
    """Decision plot overlaying the cumulative base_value -> anomaly_score
    path (one line per EV, features ordered by importance) for the top_n
    most anomalous flagged EVs. Waterfalls show one EV in isolation; this
    overlays several so it's visible at a glance whether flagged EVs share a
    common failure path or diverge onto different features partway through.
    """
    top_ev_ids = flagged_ev_ids[:top_n]
    rows = [ev_id_to_row[eid] for eid in top_ev_ids]
    base_value = float(shap_values.base_values[rows[0]])  # same scalar for every EV -- see explain_anomaly_contributions
    shap.decision_plot(
        base_value,
        shap_values.values[rows],
        feature_names=shap_values.feature_names,
        feature_order="importance",
        legend_labels=[f"EV {eid}" for eid in top_ev_ids],
        legend_location="lower right",
        show=False,
    )
    fig = plt.gcf()
    # Same fig.suptitle(y>1.0)-vs-bbox_inches="tight" issue as plot_shap_heatmap
    # above -- title the actual content axes instead.
    main_ax = max(fig.axes, key=lambda a: a.get_position().width)
    main_ax.set_title(
        f"SHAP decision plot -- top {len(rows)} most anomalous flagged EVs\n"
        "(cumulative path from base_value to anomaly_score, one line per EV)",
        pad=14,
    )
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return top_ev_ids


# %% 1. load features + fault_scores, refit the model, verify it reproduces
# detection.py's saved scores (config drift between the two scripts would
# silently invalidate the healthy/flagged split used below otherwise).
features = pd.read_pickle(FEATURES_IN)
if features.isna().any().any():
    n_before = len(features)
    features = features.dropna()
    print(f"dropped {n_before - len(features)} EVs with missing feature values")

fault_scores = pd.read_pickle(FAULT_SCORES_IN).loc[features.index]

model = fit_isolation_forest(features)
anomaly_score = -model.score_samples(features)

if not np.allclose(anomaly_score, fault_scores["anomaly_score"].to_numpy(), atol=1e-9):
    raise RuntimeError(
        "refit IsolationForest does not reproduce fault_scores.pkl -- "
        "detection.py's CONFIG (N_ESTIMATORS/RANDOM_STATE) has likely drifted "
        "out of sync with this script's copy, or fault_scores.pkl is stale. "
        "Re-run detection.py, then re-check this script's CONFIG block."
    )
print(f"refit IsolationForest matches fault_scores.pkl for all {len(features)} EVs")

is_fault = fault_scores["is_fault"]
flagged_ev_ids = fault_scores.loc[is_fault].sort_values("anomaly_score", ascending=False).index.tolist()
healthy_features = features.loc[~is_fault]
ev_id_to_row = {ev_id: i for i, ev_id in enumerate(features.index)}
print(f"{len(flagged_ev_ids)} EVs flagged by detection.py, {len(healthy_features)} healthy background EVs")

# %% 2. primary explanation -- TreeExplainer, sign-flipped, healthy background
shap_values = explain_anomaly_contributions(
    model, features, healthy_features if BACKGROUND_MODE == "interventional" else None,
)

# %% 3. faithfulness check -- does the sign-flipped ranking actually track
# the real anomaly_score across all 393 EVs?
rho = validate_ranking_faithfulness(shap_values, anomaly_score, FIGURES_DIR / "shap_faithfulness_check.png")
print(f"\nSHAP ranking faithfulness (Spearman ρ vs. actual anomaly_score): {rho:.3f}")
if rho < 0.9:
    print("  WARNING: ρ < 0.9 -- TreeExplainer's path-length proxy is not tracking anomaly_score "
          "as closely as expected here; weight the permutation cross-check below more heavily.")

# %% 4. save signed per-EV per-feature contributions for reuse downstream
shap_table = pd.DataFrame(shap_values.values, index=features.index, columns=features.columns)
shap_table.insert(0, "base_value", shap_values.base_values)
shap_table.insert(1, "anomaly_score", anomaly_score)
shap_table.insert(2, "is_fault", is_fault.to_numpy())
shap_table.to_pickle(SHAP_VALUES_OUT)
print(f"saved signed SHAP contributions for {len(shap_table)} EVs to {SHAP_VALUES_OUT}")

# %% 5. global plots -- beeswarm over all EVs (general feature-importance
# context), bar of mean |signed shap| restricted to flagged EVs only (what's
# actually driving the flags detection.py raised)
shap.plots.beeswarm(shap_values, show=False, max_display=15)
plt.gcf().suptitle("SHAP contributions to anomaly_score, all EVs\n(positive = pushes anomaly_score up)")
plt.gcf().tight_layout()
plt.gcf().savefig(FIGURES_DIR / "shap_beeswarm_all_evs.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(plt.gcf())

flagged_mask = np.isin(features.index, flagged_ev_ids)
shap_values_flagged = shap.Explanation(
    values=shap_values.values[flagged_mask],
    base_values=shap_values.base_values[flagged_mask],
    data=shap_values.data[flagged_mask],
    feature_names=shap_values.feature_names,
)
shap.plots.bar(shap_values_flagged, show=False, max_display=15)
plt.gcf().suptitle("Mean |SHAP contribution to anomaly_score|, flagged EVs only")
plt.gcf().tight_layout()
plt.gcf().savefig(FIGURES_DIR / "shap_bar_flagged_mean_abs.png", dpi=FIGURE_DPI, bbox_inches="tight")
plt.close(plt.gcf())

# %% 5b. dependence plots -- for each of the DEPENDENCE_TOP_K globally most
# important features, feature value vs. its signed SHAP contribution, to
# surface THRESHOLDS rather than just a global ranking
dependence_features = plot_shap_dependence(shap_values, is_fault, DEPENDENCE_TOP_K, DEPENDENCE_DIR)
print(f"saved {len(dependence_features)} SHAP dependence plots to {DEPENDENCE_DIR}: {dependence_features}")

# %% 5c. fleet-wide heatmap -- flagged EVs x features, ordered by anomaly_score,
# to see which features recur as drivers across many EVs vs. one-off faults
n_heatmap_evs = plot_shap_heatmap(shap_values, flagged_ev_ids, ev_id_to_row, HEATMAP_MAX_EVS, FIGURES_DIR / "shap_heatmap_flagged.png")
print(f"saved SHAP heatmap over {n_heatmap_evs} flagged EVs to {FIGURES_DIR / 'shap_heatmap_flagged.png'}")

# %% 6. per-EV waterfall plots for the TOP_N_WATERFALL most anomalous flagged EVs
for ev_id in flagged_ev_ids[:TOP_N_WATERFALL]:
    row = ev_id_to_row[ev_id]
    shap.plots.waterfall(shap_values[row], show=False, max_display=12)
    # shap.plots.waterfall lays out its own figure -- adding a suptitle and
    # then calling tight_layout() re-flows (and cramps) that layout, so just
    # reserve headroom for the title instead of re-triggering tight_layout.
    fig = plt.gcf()
    fig.suptitle(f"EV {ev_id} -- anomaly_score = {anomaly_score[row]:.3f}", y=1.04)
    fig.savefig(WATERFALL_DIR / f"shap_waterfall_ev{ev_id}.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
print(f"saved {min(TOP_N_WATERFALL, len(flagged_ev_ids))} waterfall plots to {WATERFALL_DIR}")

# %% 6b. decision plot -- overlay of the top DECISION_TOP_N flagged EVs'
# cumulative base_value -> anomaly_score paths, to compare failure signatures
plot_shap_decision(shap_values, flagged_ev_ids, ev_id_to_row, DECISION_TOP_N, FIGURES_DIR / "shap_decision_top_flagged.png")
print(f"saved SHAP decision plot for top {min(DECISION_TOP_N, len(flagged_ev_ids))} flagged EVs to {FIGURES_DIR / 'shap_decision_top_flagged.png'}")

# %% 7. text summary for the single most anomalous EV -- a short,
# human-readable explanation, not just a plot
top_ev_id = flagged_ev_ids[0]
top_row = ev_id_to_row[top_ev_id]
top_contribs = pd.Series(shap_values.values[top_row], index=features.columns).sort_values(key=np.abs, ascending=False)
print(f"\n--- Most anomalous flagged EV: {top_ev_id} (anomaly_score = {anomaly_score[top_row]:.3f}) ---")
print("Top feature contributions to its anomaly_score (positive = more anomalous):")
print(top_contribs.head(8).to_string())

# %% 8. optional permutation cross-check -- alternative 3, slower but exact-target
if RUN_PERMUTATION_CROSSCHECK:
    crosscheck_ev_ids = flagged_ev_ids[:PERMUTATION_TOP_N]
    print(f"\nrunning permutation cross-check for {len(crosscheck_ev_ids)} EVs "
          f"(background sample = {PERMUTATION_BACKGROUND_SIZE} healthy EVs)...")
    permutation_values = run_permutation_crosscheck(model, features, healthy_features, crosscheck_ev_ids)

    tree_rows = np.array([ev_id_to_row[ev_id] for ev_id in crosscheck_ev_ids])
    tree_values_subset = shap.Explanation(
        values=shap_values.values[tree_rows],
        base_values=shap_values.base_values[tree_rows],
        data=shap_values.data[tree_rows],
        feature_names=shap_values.feature_names,
    )
    agreement_rho = plot_permutation_comparison(
        tree_values_subset, permutation_values, crosscheck_ev_ids,
        FIGURES_DIR / "shap_permutation_crosscheck.png",
    )
    print(f"TreeExplainer vs. Permutation agreement (Spearman ρ across all EV x feature pairs): {agreement_rho:.3f}")


# ===========================================================================
# WINDOW-SCALE EXPLANATION -- separate, additive section ====================
# ===========================================================================
# A single reusable function for sliding_windows.py to call only when its
# alert rule fires on a live window -- NOT a batch script. Everything below
# (model, background, TreeExplainer) is built ONCE at import time; explain_
# window() then reuses all of it per call instead of rebuilding anything.
# Kept structurally separate from the whole-session batch report above,
# which it does not read from, modify, or depend on.

WINDOWED_MODEL_IN = EXTRACTED_DATA / "early_warning" / "detection_model_windowed.pkl"  # from detection.py's windowed track
WINDOWED_SCORES_IN = EXTRACTED_DATA / "early_warning" / "fault_scores_windowed.pkl"  # from detection.py's windowed track
WINDOW_FEATURES_IN = EXTRACTED_DATA / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py

CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"  # same convention as detection.py's windowed track

WINDOWED_BACKGROUND_SIZE = 300   # background sample size for the windowed TreeExplainer -- same style as PERMUTATION_BACKGROUND_SIZE above
WINDOWED_BACKGROUND_RANDOM_STATE = 0

TREND_EV_ID = None       # ev_id to plot the windowed SHAP trend for; None = auto-pick the windowed track's most anomalous flagged EV
TREND_TOP_K_FEATURES = 4  # how many of that EV's own top contributing features to plot as separate lines

# Loaded once, at import time, not per-call (sliding_windows.py calls
# explain_window() only when its alert rule fires on a live window). Guarded
# with try/except rather than a bare read: this file's whole-session logic
# above must keep working even before detection.py's windowed track has been
# run once (i.e. before detection_model_windowed.pkl exists) -- the artifact
# is only required once explain_window() is actually called.
try:
    _windowed_bundle = pd.read_pickle(WINDOWED_MODEL_IN)
except FileNotFoundError:
    _windowed_bundle = None
    print(f"\nnote: {WINDOWED_MODEL_IN} not found yet -- run detection.py's windowed track "
          "before calling explain_window()")

if _windowed_bundle is not None:
    _windowed_model = _windowed_bundle["model"]
    _windowed_scaler = _windowed_bundle["scaler"]
    _windowed_feature_columns = _windowed_bundle["feature_columns"]

    _windowed_scores = pd.read_pickle(WINDOWED_SCORES_IN)

    # Reproduce detection.py's windowed cleaning (sort by ev_id/window_start,
    # per-EV forward-fill of CAPACITY_COLUMN, then drop any row still NaN in
    # a feature column) rather than reading filtered_features_windows.pkl as
    # if it were already clean -- it still carries the pre-cleaning NaNs, and
    # fault_scores_windowed.pkl's row order/index does NOT line up with this
    # file's (rows were sorted and dropped in between), so pulling background
    # rows by raw index would risk silently grabbing a stale/NaN row instead
    # of the actual window a healthy score belongs to.
    _windowed_raw_features = pd.read_pickle(WINDOW_FEATURES_IN)
    _windowed_raw_features = _windowed_raw_features.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
    _windowed_raw_features[CAPACITY_COLUMN] = _windowed_raw_features.groupby("ev_id")[CAPACITY_COLUMN].ffill()
    _windowed_raw_features = _windowed_raw_features.dropna(subset=_windowed_feature_columns).reset_index(drop=True)

    _windowed_healthy_keys = _windowed_scores.loc[~_windowed_scores["is_fault"], ["ev_id", "window_start", "window_end"]]
    _windowed_background_keys = _windowed_healthy_keys.sample(
        n=min(WINDOWED_BACKGROUND_SIZE, len(_windowed_healthy_keys)),
        random_state=WINDOWED_BACKGROUND_RANDOM_STATE,
    )
    _windowed_background_raw = _windowed_raw_features.merge(
        _windowed_background_keys, on=["ev_id", "window_start", "window_end"], how="inner",
    )[_windowed_feature_columns]
    if len(_windowed_background_raw) != len(_windowed_background_keys):
        raise RuntimeError(
            "windowed SHAP background: merge against filtered_features_windows.pkl dropped row(s) -- "
            "fault_scores_windowed.pkl is likely stale relative to filtered_features_windows.pkl; "
            "re-run detection.py's windowed track."
        )

    _windowed_background_scaled = _windowed_scaler.transform(_windowed_background_raw)

    # Same interventional + healthy-background approach as the whole-session
    # explain_anomaly_contributions() above, with an explicit Independent
    # masker at max_samples = the full sampled background size (otherwise
    # shap silently subsamples to its own default of 100).
    _windowed_masker = shap.maskers.Independent(_windowed_background_scaled, max_samples=len(_windowed_background_scaled))
    _windowed_explainer = shap.TreeExplainer(_windowed_model, data=_windowed_masker, feature_perturbation="interventional")
    print(f"\nwindow-scale SHAP: built TreeExplainer with a {len(_windowed_background_scaled)}-window healthy "
          f"background (sampled from {len(_windowed_healthy_keys)} healthy windows total)")


def explain_window(x_raw: np.ndarray, top_n: int = 5) -> dict:
    """Explain a single live window's anomaly score. Meant to be called by
    sliding_windows.py only when its alert rule fires -- NOT a batch
    operation -- and reuses the module-level windowed model/scaler/explainer
    built once above; nothing here is refit per call.

    x_raw: one window's raw (unscaled) feature vector, in
        _windowed_feature_columns order (the same order as
        detection_model_windowed.pkl's saved feature_columns).

    Returns {"base_value": float, "top_contributions": [(feature_name,
    signed_value), ...]}, sorted by |signed_value| descending, length top_n.
    Same sign-flip convention as explain_anomaly_contributions() above --
    positive = pushes anomaly_score up (see module docstring for why the
    flip is needed and what it does/doesn't guarantee).
    """
    if _windowed_bundle is None:
        raise RuntimeError(
            f"{WINDOWED_MODEL_IN} not found -- run detection.py's windowed track to produce it "
            "before calling explain_window()."
        )

    x_scaled = _windowed_scaler.transform(np.asarray(x_raw, dtype=float).reshape(1, -1))
    raw_values = _windowed_explainer.shap_values(x_scaled)[0]  # explains avg path length, NOT anomaly_score -- see docstring
    raw_base = np.asarray(_windowed_explainer.expected_value).reshape(-1)[0]

    signed_values = -raw_values
    base_value = -raw_base

    top_contributions = (
        pd.Series(signed_values, index=_windowed_feature_columns)
        .sort_values(key=np.abs, ascending=False)
        .head(top_n)
    )
    return {
        "base_value": float(base_value),
        "top_contributions": list(top_contributions.items()),
    }


def plot_windowed_shap_trend(ev_id, top_k: int, out_path: Path) -> list:
    """Batch, analysis-only companion to explain_window() above -- NOT part
    of sliding_windows.py's live per-alert path, and doesn't change what
    explain_window() does or how it's called. Explains every window of a
    single EV's history in one batch call to the module-level windowed
    explainer, then plots the signed SHAP contribution of that EV's own
    top_k most important features across window_start, with windows the
    windowed track flagged as faulty shaded.

    This is the most literal "early warning" check available in this script:
    do the features that end up mattering already start climbing in the
    windows BEFORE the alert first fires, or only once it does? A feature
    whose contribution visibly rises several windows ahead of the first
    shaded region is a candidate leading indicator; one that jumps only
    inside the shaded region is confirming a fault already flagged, not
    anticipating it.
    """
    ev_windows = _windowed_raw_features.loc[_windowed_raw_features["ev_id"] == ev_id].sort_values("window_start").reset_index(drop=True)
    if ev_windows.empty:
        raise ValueError(f"no windowed feature rows found for ev_id={ev_id}")

    x_scaled = _windowed_scaler.transform(ev_windows[_windowed_feature_columns])
    raw_values = _windowed_explainer.shap_values(x_scaled)  # explains avg path length, NOT anomaly_score -- see module docstring
    signed_values = -raw_values  # same sign-flip convention as explain_window()/explain_anomaly_contributions() above

    # top_k by this EV's OWN mean |signed shap| across its windows, not the
    # whole-session global ranking -- this plot is about one EV's failure
    # story, which need not be driven by the fleet's globally top features.
    mean_abs = np.abs(signed_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    ev_scores = _windowed_scores.loc[_windowed_scores["ev_id"] == ev_id, ["window_start", "is_fault"]]
    is_fault_by_window = ev_windows[["window_start"]].merge(ev_scores, on="window_start", how="left")["is_fault"].fillna(False).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    for feat_idx in top_idx:
        ax.plot(
            ev_windows["window_start"], signed_values[:, feat_idx],
            marker="o", markersize=3, linewidth=1.3, label=_windowed_feature_columns[feat_idx],
        )
    for window_start in ev_windows.loc[is_fault_by_window, "window_start"]:
        ax.axvline(window_start, color=RWTHColorScales.red[0], alpha=0.15, linewidth=6, zorder=0)

    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("window_start")
    ax.set_ylabel("signed SHAP contribution to windowed anomaly_score")
    ax.set_title(
        f"Windowed SHAP trend -- EV {ev_id}\n"
        f"top {len(top_idx)} own-most-important features; shaded = windows flagged faulty"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [_windowed_feature_columns[i] for i in top_idx]


# %% 9. optional time-aligned SHAP trend for the windowed track -- runs only
# if the windowed track's artifacts already exist (same guard as everything
# else in this section)
if _windowed_bundle is not None:
    _trend_ev_id = TREND_EV_ID
    if _trend_ev_id is None:
        _flagged_windowed = _windowed_scores.loc[_windowed_scores["is_fault"]]
        _trend_ev_id = (
            _flagged_windowed.sort_values("anomaly_score", ascending=False)["ev_id"].iloc[0]
            if len(_flagged_windowed) else None
        )
    if _trend_ev_id is not None:
        _trend_out_path = FIGURES_DIR / f"shap_windowed_trend_ev{_trend_ev_id}.png"
        _trend_features = plot_windowed_shap_trend(_trend_ev_id, TREND_TOP_K_FEATURES, _trend_out_path)
        print(f"\nsaved windowed SHAP trend plot for EV {_trend_ev_id} ({_trend_features}) to {_trend_out_path}")
    else:
        print("\nno windowed-track EVs flagged as faulty -- skipping windowed SHAP trend plot")
