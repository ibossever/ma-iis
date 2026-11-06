# %%
"""
Sliding-window early warning -- REPLAY over already-scored windowed data.

This does NOT re-score anything: detection.py's windowed track already
produced anomaly_score/is_fault per (ev_id, window) in
fault_scores_windowed.pkl. This script replays that sequence, in
chronological order per EV, through two additional layers built on top of
it:
  1. per-window fault-subtype diagnosis (centroid.py's windowed nearest-
     centroid assignment -- which known fault signature is this window
     closest to, and how close), and
  2. an alert rule that fires either on detection.py's own hard flag, or on
     a WORSENING TREND over the EV's most recent windows -- catching an EV
     that is drifting toward a fault signature before any single window's
     score crosses the hard threshold.
SHAP explanations (shap.py's explain_window) are computed only for windows
that actually alert, since that call is far more expensive than the
diagnosis step and is only useful once there's something to explain.

--- CLUSTER_SOURCE: which centroid.py function drives diagnosis ------------
"faulty_only"  -> centroid.assign_nearest_centroid_windowed, backed by
                  kmeans_centroids_windowed.pkl (K-means fit ONLY on
                  confirmed-faulty windows, kmeans.py's main windowed
                  track). This is the track whose precision/recall has
                  actually been checked (detection.py's own windowed
                  TP/TN/FP/FN printout) -- run this first.
"unsupervised" -> centroid.assign_nearest_centroid_windowed_unsupervised,
                  backed by kmeans_windowed_unrestricted_diagnostic.pkl
                  (K-means fit on ALL windows, healthy + faulty, unfiltered
                  by label -- kmeans.py's W4 block, flagged there as
                  "diagnostic only, NOT used anywhere in the early-warning
                  system"). Using it here as an alternative early-warning
                  signal is new territory -- its precision/recall has not
                  been validated the way "faulty_only" has, so treat a run
                  with this setting as exploratory until checked.

--- Alert rule ---------------------------------------------------------------
check_alert_rule(buffer, current_is_fault) fires if:
  - "hard": current_is_fault is True. This is detection.py's own windowed
    is_fault flag, reused as-is -- NOT re-derived from a threshold here.
  - "trend": once the EV's rolling buffer (deque, maxlen=ROLLING_WINDOW_N)
    has at least ROLLING_WINDOW_N (anomaly_score, distance_to_nearest_
    cluster) points, a linear regression of either series against window
    position crosses TREND_SLOPE_THRESHOLD in the WORSENING direction --
    anomaly_score rising, or distance_to_nearest_cluster falling (drifting
    closer to a known fault signature). Each series' raw slope is
    normalized by its own buffer mean (fraction of level per window step)
    before comparing to TREND_SLOPE_THRESHOLD, since anomaly_score
    (~0-1 range) and distance_to_nearest_cluster (standardized-Euclidean
    range) live on very different absolute scales -- this makes one shared
    threshold meaningful for both instead of only ever firing on whichever
    metric happens to have the larger raw slope.
  "hard" takes precedence in alert_reason when both conditions hold in the
  same window.

--- Reproducing detection.py's own windowed cleaning ------------------------
fault_scores_windowed.pkl's rows were produced from a CLEANED version of
filtered_features_windows.pkl (forward-fill charge_capacity_per_100soc__
avg_ah_per_100soc per EV, then drop any row still NaN in a feature column --
see detection.py's WINDOWED TRACK section). This script reproduces that same
cleaning on its own read of filtered_features_windows.pkl before joining, so
the raw feature vectors handed to centroid.py/shap.py line up 1:1 with the
exact windows detection.py actually scored (checked below via an exact
row-count assertion, not assumed).

Input:
  extracted_data/early_warning/fault_scores_windowed.pkl (detection.py's
      windowed track) -- ev_id, window_start, window_end, anomaly_score,
      is_fault.
  extracted_data/early_warning/filtered_features_windows.pkl
      (sliding_window_fleet_features.py) -- raw per-window feature vectors.
  extracted_data/early_warning/detection_model_windowed.pkl (detection.py's
      windowed track) -- feature_columns (canonical order) + score_threshold
      (anomaly_score scale, for the trend plot's reference line).
  extracted_data/early_warning/kmeans_centroids_windowed.pkl /
      kmeans_windowed_unrestricted_diagnostic.pkl (kmeans.py's windowed
      track) -- read indirectly via centroid.py's assign_nearest_centroid_
      windowed[_unsupervised].
Output:
  extracted_data/early_warning/sliding_window_results_{CLUSTER_SOURCE}.pkl
  experiments/02_early_warning/figures/sliding_window_trend_examples.png
"""
import importlib.util
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import centroid  # noqa: E402  -- runs centroid.py's own whole-session pipeline as an import side effect (see its module docstring); this script only actually uses its windowed assign_nearest_centroid_windowed[_unsupervised] functions.

# shap.py is itself named "shap", which would collide with the real shap
# package (sys.modules["shap"] would resolve to our own half-initialized
# module instead of site-packages -- see shap.py's own module docstring for
# why IT has to strip its directory from sys.path before importing the real
# library). Loading it under a private alias here sidesteps that collision
# entirely: the real "shap" name is never claimed by our own module, so
# shap.py's internal `import shap` still finds the genuine package.
_shap_spec = importlib.util.spec_from_file_location("_early_warning_shap_module", HERE / "shap.py")
shap_module = importlib.util.module_from_spec(_shap_spec)
sys.modules[_shap_spec.name] = shap_module
_shap_spec.loader.exec_module(shap_module)  # runs shap.py's own whole-session pipeline as an import side effect too

# ===========================================================================
# CONFIG
# ===========================================================================
EXTRACTED_DATA = PROJECT_ROOT / "extracted_data" / "early_warning"
WINDOW_SCORES_IN = EXTRACTED_DATA / "fault_scores_windowed.pkl"       # from detection.py's windowed track
WINDOW_FEATURES_IN = EXTRACTED_DATA / "filtered_features_windows.pkl"  # from sliding_window_fleet_features.py
WINDOWED_MODEL_IN = EXTRACTED_DATA / "detection_model_windowed.pkl"    # from detection.py's windowed track
CENTROID_ARTIFACT_IN = {
    "faulty_only": EXTRACTED_DATA / "kmeans_centroids_windowed.pkl",
    "unsupervised": EXTRACTED_DATA / "kmeans_windowed_unrestricted_diagnostic.pkl",
}

FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

FAULT_EV_ID_CUTOFF = 335  # EV number >= this is confirmed-faulty -- same ground truth as detection.py/kmeans.py

CLUSTER_SOURCE = "faulty_only"  # "faulty_only" | "unsupervised" -- see module docstring
_DIAGNOSE_FN = {
    "faulty_only": centroid.assign_nearest_centroid_windowed,
    "unsupervised": centroid.assign_nearest_centroid_windowed_unsupervised,
}[CLUSTER_SOURCE]

ROLLING_WINDOW_N = 5  # recent windows kept per EV for the trend rule -- tunable
TREND_SLOPE_THRESHOLD = 0.15  # relative slope (fraction of the buffer's own mean level, per window step) counted as "worsening" in check_alert_rule -- tunable; see the trend-vs-hard breakdown printed below to sanity-check against a first run. A sweep against the first run's results (0.05 -> 1.0) showed 0.05's default was far too permissive (precision 0.147, 64% of alerts trend-only, 313 FPs); 0.15 keeps most of the trend layer's added recall over the hard rule alone (0.818 vs. 0.691) while roughly doubling precision and cutting trend-only alerts to 31%. Precision/recall plateau at 38 TP/94 FP/17 FN once TREND_SLOPE_THRESHOLD exceeds ~0.3 (trend condition stops firing at all, pure hard-rule behavior).

CAPACITY_COLUMN = "charge_capacity_per_100soc__avg_ah_per_100soc"  # same forward-fill convention as detection.py's windowed track

RESULTS_OUT = EXTRACTED_DATA / f"sliding_window_results_{CLUSTER_SOURCE}.pkl"
TREND_FIGURE_OUT = FIGURES_DIR / "sliding_window_trend_examples.png"

N_ESCALATION_EXAMPLES = 3     # confirmed-faulty EVs with >=1 alert
N_FALSE_NEGATIVE_EXAMPLES = 1  # confirmed-faulty EV with NO alert, if any exist
N_HEALTHY_EXAMPLES = 2         # healthy control EVs

FIGURE_DPI = 300
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def check_alert_rule(buffer: deque, current_is_fault: bool) -> tuple[bool, str | None]:
    """See module docstring for the full rule. buffer holds the current
    EV's most recent (up to ROLLING_WINDOW_N) (anomaly_score,
    distance_to_nearest_cluster) pairs, CURRENT WINDOW ALREADY INCLUDED
    (append before calling). Returns (fired, reason) with reason in
    {"hard", "trend", None}.
    """
    if current_is_fault:
        return True, "hard"

    if len(buffer) < ROLLING_WINDOW_N:
        return False, None

    scores = np.array([point[0] for point in buffer])
    distances = np.array([point[1] for point in buffer])
    x = np.arange(len(buffer))

    score_slope = linregress(x, scores).slope if np.std(scores) > 0 else 0.0
    distance_slope = linregress(x, distances).slope if np.std(distances) > 0 else 0.0

    score_relative_slope = score_slope / (np.abs(scores.mean()) + 1e-9)
    distance_relative_slope = distance_slope / (np.abs(distances.mean()) + 1e-9)

    worsening = (score_relative_slope > TREND_SLOPE_THRESHOLD) or (distance_relative_slope < -TREND_SLOPE_THRESHOLD)
    return (True, "trend") if worsening else (False, None)


def plot_trend_examples(
    results: pd.DataFrame, examples: list[tuple[int, str]], score_threshold: float, out_path: Path,
) -> None:
    """Grid of (anomaly_score, distance_to_nearest_cluster) stacked subplot
    pairs, one pair per example EV. Alert-firing windows are marked in red
    on both rows; the anomaly_score row also gets detection.py's windowed
    score_threshold as a horizontal reference line.
    """
    n = len(examples)
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(9, 2.7 * n), squeeze=False,
        gridspec_kw={"height_ratios": [1.3, 1] * n},
    )
    axes = axes[:, 0]

    for i, (ev_id, label) in enumerate(examples):
        ax_score, ax_dist = axes[2 * i], axes[2 * i + 1]
        ev_rows = results.loc[results["ev_id"] == ev_id].sort_values("window_start")
        t = ev_rows["window_start_hours"]
        alerted = ev_rows["alert"].to_numpy()

        ax_score.plot(t, ev_rows["anomaly_score"], color=RWTHColorScales.blue[0], marker="o", markersize=3, linewidth=1, label="anomaly_score")
        ax_score.scatter(t[alerted], ev_rows.loc[alerted, "anomaly_score"], color=RWTHColorScales.red[0], s=32, zorder=3, label="alert")
        ax_score.axhline(score_threshold, color="black", linestyle="--", linewidth=1, label="detection.py score_threshold")
        ax_score.set_ylabel("anomaly_score")
        ax_score.set_title(f"EV {ev_id} -- {label}", fontsize=10, loc="left")
        ax_score.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        if i == 0:
            ax_score.legend(loc="best", fontsize=7)

        ax_dist.plot(t, ev_rows["distance_to_nearest_cluster"], color=RWTHColorScales.green[0], marker="o", markersize=3, linewidth=1)
        ax_dist.scatter(t[alerted], ev_rows.loc[alerted, "distance_to_nearest_cluster"], color=RWTHColorScales.red[0], s=32, zorder=3)
        ax_dist.set_ylabel("dist. to\nnearest cluster")
        ax_dist.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    axes[-1].set_xlabel("elapsed time (hours, window start)")
    fig.suptitle(f"Sliding-window early warning -- example EVs (CLUSTER_SOURCE={CLUSTER_SOURCE!r})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %% 1. load inputs, reproduce detection.py's windowed cleaning on our own
# read of filtered_features_windows.pkl (see module docstring), assert the
# result lines up 1:1 with fault_scores_windowed.pkl rather than assuming it
windowed_model_bundle = pd.read_pickle(WINDOWED_MODEL_IN)
FEATURE_COLUMNS = windowed_model_bundle["feature_columns"]
SCORE_THRESHOLD = windowed_model_bundle["source_whole_session_anomaly_threshold"]  # anomaly_score scale

centroid_artifact = pd.read_pickle(CENTROID_ARTIFACT_IN[CLUSTER_SOURCE])
if centroid_artifact["feature_columns"] != FEATURE_COLUMNS:
    raise RuntimeError(
        f"{CENTROID_ARTIFACT_IN[CLUSTER_SOURCE]}'s feature_columns don't match {WINDOWED_MODEL_IN}'s -- "
        "the windowed detection and windowed clustering tracks have drifted out of sync; "
        "re-run detection.py's and kmeans.py's windowed tracks against the current "
        "filtered_features_windows.pkl before replaying."
    )

fault_scores = pd.read_pickle(WINDOW_SCORES_IN)

window_features = pd.read_pickle(WINDOW_FEATURES_IN)
window_features = window_features.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
window_features[CAPACITY_COLUMN] = window_features.groupby("ev_id")[CAPACITY_COLUMN].ffill()
valid_mask = window_features[FEATURE_COLUMNS].notna().all(axis=1)
n_dropped = int((~valid_mask).sum())
window_features = window_features.loc[valid_mask].reset_index(drop=True)
print(f"loaded {len(fault_scores)} scored windows; dropped {n_dropped} raw window(s) with >=1 unrecoverable "
      f"NaN feature value while reproducing detection.py's windowed cleaning, {len(window_features)} remain")

results = fault_scores.merge(
    window_features[["ev_id", "window_start", "window_end", "window_start_hours"] + FEATURE_COLUMNS],
    on=["ev_id", "window_start", "window_end"], how="inner",
)
if len(results) != len(fault_scores):
    raise RuntimeError(
        f"joining fault_scores_windowed.pkl ({len(fault_scores)} rows) against the cleaned "
        f"filtered_features_windows.pkl ({len(window_features)} rows) on (ev_id, window_start, window_end) "
        f"produced {len(results)} rows -- one of the two is stale relative to the other. "
        "Re-run detection.py's windowed track against the current filtered_features_windows.pkl."
    )
results = results.sort_values(["ev_id", "window_start"]).reset_index(drop=True)
print(f"joined to {len(results)} (ev_id, window) rows across {results['ev_id'].nunique()} EVs, "
      f"CLUSTER_SOURCE={CLUSTER_SOURCE!r}, ROLLING_WINDOW_N={ROLLING_WINDOW_N}, "
      f"TREND_SLOPE_THRESHOLD={TREND_SLOPE_THRESHOLD}")

# %% 2. replay in chronological order per EV: diagnose -> update rolling
# buffer -> check alert rule -> explain only if it fires
buffers: dict[int, deque] = {}
nearest_cluster_col = np.empty(len(results), dtype=int)
distance_col = np.empty(len(results), dtype=float)
alert_col = np.zeros(len(results), dtype=bool)
alert_reason_col: list[str | None] = [None] * len(results)
shap_contrib_col: list[list | None] = [None] * len(results)

feature_matrix = results[FEATURE_COLUMNS].to_numpy(dtype=float)
ev_ids = results["ev_id"].to_numpy()
is_fault_arr = results["is_fault"].to_numpy()

n_explained = 0
for i in range(len(results)):
    x_raw = feature_matrix[i]
    diagnosis = _DIAGNOSE_FN(x_raw)
    nearest_cluster_col[i] = diagnosis["nearest_cluster"]
    distance_col[i] = diagnosis["distance_to_nearest_cluster"]

    buffer = buffers.setdefault(ev_ids[i], deque(maxlen=ROLLING_WINDOW_N))
    buffer.append((results.at[i, "anomaly_score"], distance_col[i]))

    fired, reason = check_alert_rule(buffer, bool(is_fault_arr[i]))
    alert_col[i] = fired
    alert_reason_col[i] = reason
    if fired:
        shap_contrib_col[i] = shap_module.explain_window(x_raw)["top_contributions"]
        n_explained += 1

results["nearest_cluster"] = nearest_cluster_col
results["distance_to_nearest_cluster"] = distance_col
results["alert"] = alert_col
results["alert_reason"] = alert_reason_col
results["shap_top_contributions"] = shap_contrib_col
results["cluster_source"] = CLUSTER_SOURCE

n_hard = int((results["alert_reason"] == "hard").sum())
n_trend = int((results["alert_reason"] == "trend").sum())
print(f"\n{results['alert'].sum()} alerting window(s) ({n_hard} hard, {n_trend} trend), "
      f"{n_explained} SHAP explanation(s) computed")

results = results.drop(columns=["is_fault"])
results = results[[
    "ev_id", "window_start", "window_end", "window_start_hours", "anomaly_score",
    "nearest_cluster", "distance_to_nearest_cluster", "alert", "alert_reason",
    "shap_top_contributions", "cluster_source",
]]
results.to_pickle(RESULTS_OUT)
print(f"saved replay results to {RESULTS_OUT}")

# %% 3. trend-vs-hard breakdown, per EV -- sanity check for TREND_SLOPE_THRESHOLD
alerted_evs = results.loc[results["alert"], ["ev_id", "alert_reason"]]
reason_sets = alerted_evs.groupby("ev_id")["alert_reason"].agg(set)
n_alerted_evs = len(reason_sets)
n_trend_only = int((reason_sets == {"trend"}).sum())
n_hard_only = int((reason_sets == {"hard"}).sum())
n_both = n_alerted_evs - n_trend_only - n_hard_only
print(f"\n--- Trend vs. hard breakdown, TREND_SLOPE_THRESHOLD={TREND_SLOPE_THRESHOLD} ---")
print(f"{n_alerted_evs} EV(s) alerted at least once:")
if n_alerted_evs:
    print(f"  trend-only (never hit the hard rule): {n_trend_only} ({n_trend_only / n_alerted_evs:.1%})")
    print(f"  hard-only (never hit the trend rule):  {n_hard_only} ({n_hard_only / n_alerted_evs:.1%})")
    print(f"  both:                                  {n_both} ({n_both / n_alerted_evs:.1%})")

# %% 4. validation summary -- EV-level precision/recall, first-alert timing
window_index = results.groupby("ev_id").cumcount()  # 0-based position within that EV's own chronological window sequence

ev_summary = results.groupby("ev_id").agg(n_windows=("alert", "size"), any_alert=("alert", "any"))
ev_summary["is_confirmed_faulty"] = ev_summary.index >= FAULT_EV_ID_CUTOFF

alerted_rows = results.loc[results["alert"]].copy()
alerted_rows["window_index"] = window_index.loc[alerted_rows.index]
first_alert_window_index = alerted_rows.groupby("ev_id")["window_index"].min()
ev_summary["first_alert_window_index"] = first_alert_window_index.reindex(ev_summary.index)

tp = ev_summary[ev_summary["any_alert"] & ev_summary["is_confirmed_faulty"]]
fp = ev_summary[ev_summary["any_alert"] & ~ev_summary["is_confirmed_faulty"]]
fn = ev_summary[~ev_summary["any_alert"] & ev_summary["is_confirmed_faulty"]]
tn = ev_summary[~ev_summary["any_alert"] & ~ev_summary["is_confirmed_faulty"]]

precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else float("nan")
recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else float("nan")

print(f"\n--- Validation: 'at least one alert' as the predicted-positive rule, "
      f"ground truth ev_id >= {FAULT_EV_ID_CUTOFF} ---")
print(f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  TN={len(tn)}  "
      f"(n={len(ev_summary)} EVs)")
print(f"precision={precision:.3f}  recall={recall:.3f}")
print(f"\ntrue positives -- first alert fired at window index (0-based, within that EV's own sequence):")
print(tp[["n_windows", "first_alert_window_index"]].sort_values("first_alert_window_index").to_string())

# %% 5. select example EVs, print why, and plot
ev_max_score = results.groupby("ev_id")["anomaly_score"].max()

escalation_ids = ev_max_score.loc[tp.index].sort_values(ascending=False).head(N_ESCALATION_EXAMPLES).index.tolist()
false_negative_ids = ev_max_score.loc[fn.index].sort_values(ascending=False).head(N_FALSE_NEGATIVE_EXAMPLES).index.tolist()
healthy_pool = tn.index if len(tn) else ev_summary.loc[~ev_summary["is_confirmed_faulty"]].index
healthy_ids = ev_summary.loc[healthy_pool, "n_windows"].sort_values(ascending=False).head(N_HEALTHY_EXAMPLES).index.tolist()

print(f"\n--- Example EVs selected for {TREND_FIGURE_OUT.name} ---")
print(f"escalation (confirmed-faulty, >=1 alert, highest max anomaly_score): {escalation_ids}")
if false_negative_ids:
    print(f"false negative (confirmed-faulty, NO alert, highest max anomaly_score): {false_negative_ids}")
else:
    print("false negative: none -- every confirmed-faulty EV received at least one alert")
print(f"healthy controls (not confirmed-faulty, zero alerts, most windows): {healthy_ids}")

examples = (
    [(ev_id, "escalation (confirmed-faulty, alerted)") for ev_id in escalation_ids]
    + [(ev_id, "false negative (confirmed-faulty, never alerted)") for ev_id in false_negative_ids]
    + [(ev_id, "healthy control") for ev_id in healthy_ids]
)
plot_trend_examples(results, examples, SCORE_THRESHOLD, TREND_FIGURE_OUT)
print(f"\nsaved trend plot to {TREND_FIGURE_OUT}")
