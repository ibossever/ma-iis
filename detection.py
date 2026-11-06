# %%
"""
Unsupervised fault detection across EVs using Isolation Forest.

Input: extracted_data/fault_detection/filtered_features.pkl (produced by
features.py) -- one row per EV (index = ev_id), one column per scalar
feature, already filtered down to features that pass the fault-relevance
screening in feature_fault_analysis.csv (significance, effect size, and
AUC-ROC discrimination -- see features.py for the exact criteria). The
column set is expected to keep changing as more features get added/removed
upstream; nothing here depends on which, or how many, columns are present.

IsolationForest isolates each EV by repeatedly splitting on random features/
thresholds -- an EV that sits off in feature space away from the bulk of the
fleet gets isolated in fewer splits, so it scores as more anomalous. Tree
splits only compare a feature to itself across samples, so no
standardization is needed across features of very different scales/units.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../detection.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG -- the knobs you are most likely to change between runs
# ===========================================================================
FEATURES_IN = PROJECT_ROOT / "extracted_data" / "fault_detection" / "filtered_features.pkl"  # from features.py
OUT_PATH = PROJECT_ROOT / "extracted_data" / "fault_detection" / "fault_scores.pkl"
DETECTION_MODEL_OUT = PROJECT_ROOT / "extracted_data" / "fault_detection" / "detection_model.pkl"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

N_ESTIMATORS = 100
RANDOM_STATE = 0
TOP_N = 10

# Fraction of EVs to flag as faulty -- the anomaly_score threshold is derived
# from this as the (1 - FAULTY_PERCENTAGE) quantile of the score distribution,
# so e.g. 0.15 always flags the most-anomalous 15% of EVs regardless of how
# the score distribution itself shifts as features/data change, rather than
# a fixed score cutoff that would flag a different fraction each time.
# Set to match the labeled fault prevalence (58/393 EVs = 14.8%, see
# fault_feature.py) rather than an arbitrary round number.
FAULTY_PERCENTAGE = 0.15

# --- windowed track config (see WINDOWED TRACK section at the bottom) ------
WINDOW_FEATURES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
WINDOWED_MODEL_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "detection_model_windowed.pkl"
WINDOWED_SCORES_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "fault_scores_windowed.pkl"
CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"
FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as the PLAYGROUND section below / kmeans.py


plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def detect_faults(features: pd.DataFrame, faulty_percentage: float) -> tuple[pd.DataFrame, float, IsolationForest]:
    """Fit an IsolationForest on `features` (EVs as rows, scalar numeric
    features as columns) and return:
      - a same-indexed DataFrame with:
          anomaly_score -- higher = more anomalous
          is_fault      -- anomaly_score > threshold, as a bool
      - the anomaly_score threshold, set so that exactly `faulty_percentage`
        of EVs (rounded to the nearest quantile) are flagged as faulty.
      - the fitted model itself, so callers (e.g. the windowed track at the
        bottom of this file) can persist/reuse it, not just its output table.
    """
    model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(features)
    anomaly_score = -model.score_samples(features)  # score_samples: higher = more normal, so negate
    threshold = np.quantile(anomaly_score, 1 - faulty_percentage)

    fault_scores = pd.DataFrame({
        "anomaly_score": anomaly_score,
        "is_fault": anomaly_score > threshold,
    }, index=features.index)
    return fault_scores, threshold, model


def plot_ev_level_primary_metric(tp: int, tn: int, fp: int, fn: int,
                                  accuracy: float, precision: float, recall: float, f1: float,
                                  n_evs: int, ev_id_cutoff: int, out_path: Path) -> None:
    """EV-level 'at least one alert' headline result: a confusion matrix
    (green = correct, red = incorrect) next to the derived accuracy/
    precision/recall/F1 bars, so the two block-3 print statements above have
    a figure to go with them."""
    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(10, 4.5))

    correct_color = RWTHColorScales.green[0]
    incorrect_color = RWTHColorScales.red[0]
    cm_counts = np.array([[tp, fn], [fp, tn]])
    cm_colors = np.array([[correct_color, incorrect_color], [incorrect_color, correct_color]])

    for row in range(2):
        for col in range(2):
            ax_cm.add_patch(plt.Rectangle((col, 1 - row), 1, 1, facecolor=cm_colors[row, col], edgecolor="white"))
            ax_cm.text(col + 0.5, 1 - row + 0.5, str(cm_counts[row, col]),
                       ha="center", va="center", fontsize=16, color="white", fontweight="bold")
    ax_cm.set_xlim(0, 2)
    ax_cm.set_ylim(0, 2)
    ax_cm.set_xticks([0.5, 1.5])
    ax_cm.set_xticklabels(["predicted faulty", "predicted normal"])
    ax_cm.set_yticks([0.5, 1.5])
    ax_cm.set_yticklabels(["actual normal", "actual faulty"])
    ax_cm.set_title(f"EV-level confusion matrix\n(n={n_evs} EVs, ground truth: ev_id >= {ev_id_cutoff})")
    ax_cm.set_aspect("equal")

    metric_names = ["Accuracy", "Precision", "Recall", "F1 score"]
    metric_values = [accuracy, precision, recall, f1]
    bars = ax_bar.barh(metric_names, [v * 100 for v in metric_values], color=RWTHColorScales.blue[0])
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("%")
    ax_bar.invert_yaxis()
    ax_bar.set_title("Derived metrics")
    for bar, value in zip(bars, metric_values):
        ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}", va="center", fontsize=10)

    fig.suptitle("Windowed track PRIMARY metric: EV-level 'at least one alert' vs. ground truth")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_distribution(fault_scores: pd.DataFrame, threshold: float,
                             faulty_percentage: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    normal = fault_scores.loc[~fault_scores["is_fault"], "anomaly_score"]
    faulty = fault_scores.loc[fault_scores["is_fault"], "anomaly_score"]

    # Thin bars (rwidth < 1) with a shared fine-grained bin edge set so
    # near-adjacent normal/faulty bars stay visually distinct instead of
    # blurring into each other.
    bin_edges = np.histogram_bin_edges(fault_scores["anomaly_score"], bins=80)
    ax.hist(normal, bins=bin_edges, alpha=0.8, rwidth=0.6, label="normal")
    ax.hist(faulty, bins=bin_edges, alpha=0.8, rwidth=0.6, label="flagged fault")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold ({threshold:.3f}, top {faulty_percentage:.1%})")
    ax.set_xlabel("IsolationForest anomaly score (higher = more anomalous)")
    ax.set_ylabel("# EVs")
    ax.set_title(f"Fault detection score distribution\n"
                 f"biased fault rate: {faulty_percentage:.1%}, threshold: {threshold:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# %%
features = pd.read_pickle(FEATURES_IN)

# IsolationForest can't handle NaNs; filtered_features has none today, but
# the upstream feature set is still being edited, so guard against it
# rather than letting fit_predict raise.
if features.isna().any().any():
    n_before = len(features)
    features = features.dropna()
    print(f"dropped {n_before - len(features)} EVs with missing feature values")

# %%
fault_scores, fault_threshold, fault_model = detect_faults(features, FAULTY_PERCENTAGE)
fault_scores.to_pickle(OUT_PATH)

# Persist the fitted model itself (not just its output table) so other
# tracks -- e.g. the windowed track at the bottom of this file -- can reuse
# its decision threshold instead of deriving their own from windowed fault
# prevalence.
pd.to_pickle({
    "model": fault_model,
    "feature_columns": features.columns.tolist(),
    "anomaly_score_threshold": fault_threshold,  # anomaly_score = -score_samples; see windowed track below
}, DETECTION_MODEL_OUT)
print(f"saved fitted whole-session IsolationForest to {DETECTION_MODEL_OUT}")

plot_score_distribution(fault_scores, fault_threshold, FAULTY_PERCENTAGE, FIGURES_DIR / "fault_scores.png")

# %%
faulty = fault_scores[fault_scores["is_fault"]].sort_values("anomaly_score", ascending=False)
print(f"flagged {len(faulty)}/{len(fault_scores)} EVs as faulty "
      f"(top {FAULTY_PERCENTAGE:.1%}, anomaly_score > {fault_threshold:.3f})")
print(f"\n--- EVs above threshold, by anomaly score ---")
print(faulty.to_string())

print(f"\n--- Top {TOP_N} EVs by anomaly score ---")
print(fault_scores.sort_values("anomaly_score", ascending=False).head(TOP_N).to_string())


# %%
# ===========================================================================
# PLAYGROUND -- not part of the detection script, just for manually checking
# results against a known rule. Ground truth: EV number >= 335 is faulty,
# EV number <= 334 is healthy.
# ===========================================================================
TP = 0
TN = 0
FP = 0
FN = 0

for ev_id, row in fault_scores.iterrows():
    predicted_fault = row["is_fault"]
    actual_fault = ev_id >= 335

    if predicted_fault and actual_fault:
        TP += 1
    elif not predicted_fault and not actual_fault:
        TN += 1
    elif predicted_fault and not actual_fault:
        FP += 1
    else:
        FN += 1

print("\n--- TP/TN/FP/FN vs. ground truth (EV >= 335 is faulty) ---")
print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")

# accuracy, precision, recall etc. -- guard each denominator individually
# since TP/TN/FP/FN can each independently be 0 depending on the run.
accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else float("nan")
precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")  # a.k.a. sensitivity / TPR
specificity = TN / (TN + FP) if (TN + FP) > 0 else float("nan")  # a.k.a. TNR
fpr = FP / (FP + TN) if (FP + TN) > 0 else float("nan")
fnr = FN / (FN + TP) if (FN + TP) > 0 else float("nan")
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
balanced_accuracy = (recall + specificity) / 2

print("\n--- Detection metrics ---")
print(f"Accuracy:            {accuracy:.1%}")
print(f"Precision (PPV):     {precision:.1%}")
print(f"Recall (Sensitivity):{recall:.1%}")
print(f"Specificity (TNR):   {specificity:.1%}")
print(f"False positive rate: {fpr:.1%}")
print(f"False negative rate: {fnr:.1%}")
print(f"F1 score:            {f1:.1%}")
print(f"Balanced accuracy:   {balanced_accuracy:.1%}")


# %%
# ===========================================================================
# WINDOWED TRACK -- same IsolationForest approach as above, but fit on
# per-(ev_id, window) rows from filtered_features_windows.pkl instead of one
# row per EV.
#
# Deliberately does NOT fit its own contamination-derived cutoff from
# windowed fault prevalence: passing contamination=<windowed fault fraction>
# to IsolationForest would mechanically force exactly that fraction of
# windows to be flagged as faulty every run, regardless of whether the score
# distribution actually supports it. Instead, this track transplants the
# whole-session detector's own decision threshold (persisted above as
# DETECTION_MODEL_OUT) onto the windowed model's score_samples output.
# ===========================================================================

# %% load + clean windowed features. Same "drop rather than impute"
# convention as the whole-session block above for every feature column
# EXCEPT charge_capacity_per_100soc__avg_ah_per_100soc, which goes NaN
# whenever a window has no qualifying charging session (capac2.py's
# MIN_SOC_INCREASE guard -- see sliding_window_fleet_features.py) and is
# forward-filled from that same EV's previous window instead of dropped
# outright -- per-EV via groupby so no value leaks across EVs at the
# boundary between them. An EV's leading NaN(s) (no prior window yet to
# forward-fill from) are left NaN and still caught by the general drop below.
window_features_all = pd.read_pickle(WINDOW_FEATURES_IN)
missing_window_columns = set(features.columns) - set(window_features_all.columns)
if missing_window_columns:
    raise RuntimeError(
        f"filtered_features_windows.pkl is missing feature column(s) {sorted(missing_window_columns)} "
        "relative to the current filtered_features.pkl -- re-run sliding_window_fleet_features.py first."
    )

window_features_all = window_features_all.sort_values(["ev_id", "window_start"]).reset_index(drop=True)

valid_before_capacity_ffill = window_features_all[features.columns].notna().all(axis=1)
window_features_all[CAPACITY_COLUMN] = (
    window_features_all.groupby("ev_id")[CAPACITY_COLUMN].ffill()
)
valid_after_capacity_ffill = window_features_all[features.columns].notna().all(axis=1)

n_recovered = int((~valid_before_capacity_ffill & valid_after_capacity_ffill).sum())
print(f"\nwindowed track: {CAPACITY_COLUMN} forward-fill recovered {n_recovered} window(s) "
      "that would otherwise have been dropped for missing capacity data")

n_windows_dropped = int((~valid_after_capacity_ffill).sum())
window_features_all = window_features_all.loc[valid_after_capacity_ffill].reset_index(drop=True)
print(f"windowed track: dropped {n_windows_dropped} window(s) with >=1 NaN feature value "
      "(unrecoverable -- either no earlier same-EV capacity value to forward-fill from, or a "
      f"different feature column), {len(window_features_all)} windows remain")

# %% transplant the whole-session detector's decision threshold. anomaly_score
# = -score_samples (see detect_faults above), so the equivalent cutoff on
# raw score_samples is the negation of the saved anomaly_score threshold.
whole_session_bundle = pd.read_pickle(DETECTION_MODEL_OUT)
whole_session_model = whole_session_bundle["model"]
whole_session_anomaly_threshold = whole_session_bundle["anomaly_score_threshold"]
whole_session_score_threshold = -whole_session_anomaly_threshold

# %% fit the windowed IsolationForest. contamination is NOT used for
# thresholding below (is_fault uses the transplanted threshold instead), so
# any valid value here only affects irrelevant internals like .predict()/
# .offset_, which this track never calls.
scaler_windowed = StandardScaler()
X_windowed = scaler_windowed.fit_transform(window_features_all[features.columns])

windowed_model = IsolationForest(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, contamination="auto")
windowed_model.fit(X_windowed)
windowed_score_samples = windowed_model.score_samples(X_windowed)
whole_session_score_samples = whole_session_model.score_samples(features[whole_session_bundle["feature_columns"]])

is_fault_windowed = windowed_score_samples < whole_session_score_threshold


def _score_summary(scores: np.ndarray) -> str:
    q = np.quantile(scores, [0, 0.25, 0.5, 0.75, 1.0])
    return f"min={q[0]:.4f} p25={q[1]:.4f} median={q[2]:.4f} p75={q[3]:.4f} max={q[4]:.4f}"


print("\n--- score_samples distributions (higher = more normal) -- sanity check before "
      "trusting the transplanted threshold ---")
print(f"whole-session: {_score_summary(whole_session_score_samples)}")
print(f"windowed:      {_score_summary(windowed_score_samples)}")
print(f"transplanted threshold (score_samples scale): {whole_session_score_threshold:.4f} "
      f"(anomaly_score scale: {whole_session_anomaly_threshold:.4f})")

pct_flagged_windowed = float(is_fault_windowed.mean()) * 100
print(f"windowed track: {pct_flagged_windowed:.1f}% of windows flagged as fault under the "
      "transplanted threshold")
if pct_flagged_windowed < 1.0 or pct_flagged_windowed > 99.0:
    print(f"WARNING: windowed track flags {pct_flagged_windowed:.1f}% of windows under the "
          "transplanted threshold -- the windowed score_samples distribution barely overlaps "
          "the whole-session threshold, which is a sign the two models' score scales may not be "
          "directly comparable. Treat the windowed is_fault labels with caution.")

# %% save windowed model + scaler + transplanted threshold, and per-window results
pd.to_pickle({
    "model": windowed_model,
    "scaler": scaler_windowed,
    "feature_columns": features.columns.tolist(),
    "score_threshold": whole_session_score_threshold,
    "source_whole_session_anomaly_threshold": whole_session_anomaly_threshold,
}, WINDOWED_MODEL_OUT)
print(f"\nsaved windowed IsolationForest + scaler + transplanted threshold to {WINDOWED_MODEL_OUT}")

windowed_fault_scores = pd.DataFrame({
    "ev_id": window_features_all["ev_id"].to_numpy(),
    "window_start": window_features_all["window_start"].to_numpy(),
    "window_end": window_features_all["window_end"].to_numpy(),
    "anomaly_score": -windowed_score_samples,
    "is_fault": is_fault_windowed,
})
windowed_fault_scores.to_pickle(WINDOWED_SCORES_OUT)
print(f"saved per-window fault scores to {WINDOWED_SCORES_OUT}")


# %%
# ===========================================================================
# WINDOWED TRACK evaluation, part 1/3 -- per-window vs. PROPAGATED EV label
# (ev_id >= FAULT_EV_ID_CUTOFF applied to every window of that EV). NOT
# RELIABLE as a headline number: it scores an eventually-faulty EV's early,
# pre-onset windows as should-have-been-flagged, so a detector that
# correctly stays quiet until symptoms actually appear gets penalized for
# doing the right thing. Kept printed, clearly labeled, for comparison
# against the two metrics below -- see the note after this block for why
# those replace this one as the headline result.
# ===========================================================================
actual_fault_windowed = windowed_fault_scores["ev_id"] >= FAULT_EV_ID_CUTOFF
predicted_fault_windowed = windowed_fault_scores["is_fault"]

TP_w = int((predicted_fault_windowed & actual_fault_windowed).sum())
TN_w = int((~predicted_fault_windowed & ~actual_fault_windowed).sum())
FP_w = int((predicted_fault_windowed & ~actual_fault_windowed).sum())
FN_w = int((~predicted_fault_windowed & actual_fault_windowed).sum())

accuracy_w = (TP_w + TN_w) / (TP_w + TN_w + FP_w + FN_w) if (TP_w + TN_w + FP_w + FN_w) > 0 else float("nan")
precision_w = TP_w / (TP_w + FP_w) if (TP_w + FP_w) > 0 else float("nan")
recall_w = TP_w / (TP_w + FN_w) if (TP_w + FN_w) > 0 else float("nan")
f1_w = 2 * precision_w * recall_w / (precision_w + recall_w) if (precision_w + recall_w) > 0 else float("nan")

print(f"\n--- Windowed track, per-window vs. PROPAGATED EV label -- NOT RELIABLE, "
      f"see EV-level metrics below (ground truth: ev_id >= {FAULT_EV_ID_CUTOFF}) ---")
print(f"TP: {TP_w}  TN: {TN_w}  FP: {FP_w}  FN: {FN_w}")
print(f"Accuracy:  {accuracy_w:.1%}")
print(f"Precision: {precision_w:.1%}")
print(f"Recall:    {recall_w:.1%}")
print(f"F1 score:  {f1_w:.1%}")

print("\nNOTE: the numbers above penalize the detector for correctly staying quiet on early, "
      "pre-onset windows from an eventually-faulty EV, since every one of that EV's windows is "
      "scored as if it should have fired. They are replaced as the headline result by two "
      "metrics below: (1) EV-level 'at least one alert' -- did detection.py catch this EV at "
      "all, the same logic sliding_windows.py already validates against -- and (2) a per-window "
      "metric validated against unsupervised cluster structure instead of the propagated label, "
      "so a window is only expected to fire if it actually looks like the fault-like cluster(s), "
      "not merely because it belongs to an EV that eventually failed.")

# %%
# ===========================================================================
# WINDOWED TRACK evaluation, part 2/3 -- PRIMARY metric: EV-level "at least
# one alert" (same logic as sliding_windows.py's own validation step),
# applied directly to detection.py's raw is_fault flags rather than to an
# alert rule layered on top.
# ===========================================================================
ev_level_summary = windowed_fault_scores.groupby("ev_id").agg(
    n_windows=("is_fault", "size"), any_fault=("is_fault", "any"),
)
ev_level_summary["is_confirmed_faulty"] = ev_level_summary.index >= FAULT_EV_ID_CUTOFF

TP_ev = int((ev_level_summary["any_fault"] & ev_level_summary["is_confirmed_faulty"]).sum())
TN_ev = int((~ev_level_summary["any_fault"] & ~ev_level_summary["is_confirmed_faulty"]).sum())
FP_ev = int((ev_level_summary["any_fault"] & ~ev_level_summary["is_confirmed_faulty"]).sum())
FN_ev = int((~ev_level_summary["any_fault"] & ev_level_summary["is_confirmed_faulty"]).sum())

accuracy_ev = (TP_ev + TN_ev) / (TP_ev + TN_ev + FP_ev + FN_ev) if (TP_ev + TN_ev + FP_ev + FN_ev) > 0 else float("nan")
precision_ev = TP_ev / (TP_ev + FP_ev) if (TP_ev + FP_ev) > 0 else float("nan")
recall_ev = TP_ev / (TP_ev + FN_ev) if (TP_ev + FN_ev) > 0 else float("nan")
f1_ev = 2 * precision_ev * recall_ev / (precision_ev + recall_ev) if (precision_ev + recall_ev) > 0 else float("nan")

print(f"\n--- Windowed track PRIMARY metric: EV-level 'at least one alert' vs. ground truth "
      f"(ev_id >= {FAULT_EV_ID_CUTOFF}) ---")
print(f"TP: {TP_ev}  TN: {TN_ev}  FP: {FP_ev}  FN: {FN_ev}  (n={len(ev_level_summary)} EVs)")
print(f"Accuracy:  {accuracy_ev:.1%}")
print(f"Precision: {precision_ev:.1%}")
print(f"Recall:    {recall_ev:.1%}")
print(f"F1 score:  {f1_ev:.1%}")

ev_primary_metric_fig_path = FIGURES_DIR / "windowed_ev_level_primary_metric.png"
plot_ev_level_primary_metric(
    TP_ev, TN_ev, FP_ev, FN_ev, accuracy_ev, precision_ev, recall_ev, f1_ev,
    n_evs=len(ev_level_summary), ev_id_cutoff=FAULT_EV_ID_CUTOFF, out_path=ev_primary_metric_fig_path,
)
print(f"saved EV-level primary metric figure to {ev_primary_metric_fig_path}")

pct_flagged_per_faulty_ev = (
    windowed_fault_scores.loc[actual_fault_windowed]
    .groupby("ev_id")["is_fault"]
    .mean() * 100
)
print(f"\n--- % of windows flagged as fault, per confirmed-faulty EV (ev_id >= {FAULT_EV_ID_CUTOFF}) ---")
print(pct_flagged_per_faulty_ev.to_string(float_format=lambda v: f"{v:.1f}"))

# %%
# ===========================================================================
# WINDOWED TRACK evaluation, part 3/3 -- SECONDARY metric: per-window,
# validated against UNSUPERVISED cluster structure (kmeans.py's windowed
# track), not the propagated EV label. Ground truth per window here is
# whether its own unsupervised cluster assignment is fault-like (pct_faulty
# >= FAULT_LIKE_CLUSTER_PCT_THRESHOLD), which does not assume every window
# of an eventually-faulty EV looks anomalous -- only windows actually
# resembling the fault-like cluster(s) count as positive.
# ===========================================================================
UNSUPERVISED_CLUSTERS_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "kmeans_windowed_unsupervised.csv"  # from kmeans.py's windowed track
# Cross-references kmeans.py's own FAULT_LIKE_CLUSTER_PCT_THRESHOLD (which
# actually determines is_fault_like in that CSV) -- duplicated here rather
# than imported since detection.py only reads kmeans.py's CSV output, not
# the module itself. Keep the two values in sync by hand.
FAULT_LIKE_CLUSTER_PCT_THRESHOLD = 50.0

try:
    unsupervised_clusters = pd.read_csv(UNSUPERVISED_CLUSTERS_IN)
except FileNotFoundError:
    unsupervised_clusters = None
    print(f"\nnote: {UNSUPERVISED_CLUSTERS_IN} not found -- run kmeans.py's windowed track "
          "(set FINAL_K_WINDOWED) to produce it before this secondary metric can be computed")

if unsupervised_clusters is not None:
    windowed_vs_unsupervised = windowed_fault_scores.merge(
        unsupervised_clusters[["ev_id", "window_start", "window_end", "is_fault_like"]],
        on=["ev_id", "window_start", "window_end"], how="inner",
    )
    if len(windowed_vs_unsupervised) != len(windowed_fault_scores):
        print(f"WARNING: only {len(windowed_vs_unsupervised)}/{len(windowed_fault_scores)} windows matched "
              f"{UNSUPERVISED_CLUSTERS_IN} by (ev_id, window_start, window_end) -- the two artifacts may be "
              "out of sync; re-run kmeans.py's windowed track against the current filtered_features_windows.pkl.")

    predicted_fault_unsup = windowed_vs_unsupervised["is_fault"]
    actual_fault_like_unsup = windowed_vs_unsupervised["is_fault_like"]

    TP_u = int((predicted_fault_unsup & actual_fault_like_unsup).sum())
    TN_u = int((~predicted_fault_unsup & ~actual_fault_like_unsup).sum())
    FP_u = int((predicted_fault_unsup & ~actual_fault_like_unsup).sum())
    FN_u = int((~predicted_fault_unsup & actual_fault_like_unsup).sum())

    precision_u = TP_u / (TP_u + FP_u) if (TP_u + FP_u) > 0 else float("nan")
    recall_u = TP_u / (TP_u + FN_u) if (TP_u + FN_u) > 0 else float("nan")
    f1_u = 2 * precision_u * recall_u / (precision_u + recall_u) if (precision_u + recall_u) > 0 else float("nan")

    print(f"\n--- Windowed track SECONDARY metric: per-window, validated against unsupervised "
          f"structure, not propagated labels (ground truth: cluster pct_faulty >= "
          f"{FAULT_LIKE_CLUSTER_PCT_THRESHOLD:.0f}%) ---")
    print(f"TP: {TP_u}  TN: {TN_u}  FP: {FP_u}  FN: {FN_u}  (n={len(windowed_vs_unsupervised)} windows)")
    print(f"Precision: {precision_u:.1%}")
    print(f"Recall:    {recall_u:.1%}")
    print(f"F1 score:  {f1_u:.1%}")
