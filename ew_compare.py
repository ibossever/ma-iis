# %%
"""
Early-warning comparison: windowed Isolation Forest vs. centroid-distance
(prototype) score, under ONE shared alert rule, on BOTH window sources (120h
and 24h), cross-validated over vehicles. Implements thesis_revision_plan.pdf
A6 (windowed IF without the transplanted threshold), A7 (prototype warning,
"compared, not combined") and the A10 chance test.

--- Per CV fold (vehicles split, stratified on label x pack_type) ----------
  scaler     -- StandardScaler fit on ALL train windows (healthy + faulty).
  IF score   -- IsolationForest(n_estimators=100) fit on all train windows,
                unsupervised, score = -score_samples (higher = more anomalous)
                -- same model settings as detection.py's windowed track.
  proto score-- KMeans(kF) on train-faulty windows (fault prototypes),
                KMeans(kH) on train-healthy windows (healthy prototypes),
                kF/kH from select_k.py. Per window:
                  dF = distance to nearest fault prototype
                  dH = distance to nearest healthy prototype
                  s  = dH / (dF + dH)  in [0, 1], > 0.5 = closer to a fault
                Euclidean, standardized space. No IF flag involved.

--- Shared alert rule (identical for both methods) --------------------------
  tau  = quantile q of the method's score over TRAIN-HEALTHY windows
  flag = test-window score > tau
  alert on a vehicle = >= m consecutive flagged windows
q is swept (Q_SWEEP) to trace recall vs. healthy-alert-rate; the headline
operating point (Q_MAIN, M_MAIN) is fixed here in advance.
Caveat: tau comes from windows the models were fit on (in-sample), equally
for both methods -- it biases tau a little low for both, not one vs. the other.

--- Outputs -------------------------------------------------------------------
  extracted_data/revision/ew_window_scores.pkl -- every out-of-fold test
      window: source, seed, fold, ev_id, window_start(_hours), if_score,
      dF, dH, s, nearest_fault_prototype
  extracted_data/revision/ew_thresholds.csv -- tau per source/seed/fold/method/q
  extracted_data/revision/ew_sweep.csv -- EV-level metrics per source, method,
      q, m, seed (TP/FP/FN/TN, recall, precision, healthy alert rate, TP inside
      / outside small pack, lead time)
  extracted_data/revision/ew_summary.csv -- headline table (Q_MAIN, M_MAIN),
      mean +/- std over seeds, + Fisher test on large-pack vehicles (A10)
  extracted_data/revision/ew_overlap.csv -- which faulty EVs each method catches
  experiments/03_revision/figures/ew_operating_characteristic.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from ew_common import REVISION_OUT, ev_table, load_windows, vehicle_folds  # noqa: E402
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402

# ===========================================================================
# CONFIG
# ===========================================================================
WINDOW_SOURCES = ["120h", "24h"]
K_SELECTION = json.loads((REVISION_OUT / "select_k.json").read_text())  # from select_k.py
N_SPLITS = 5
SEEDS = range(10)
N_ESTIMATORS = 100  # == detection.py
N_INIT = 10

METHODS = ["isolation_forest", "prototype"]
SCORE_COLUMN = {"isolation_forest": "if_score", "prototype": "s"}
Q_SWEEP = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]
M_SWEEP = [1, 2, 3]
Q_MAIN = 0.95   # headline operating point, fixed in advance
M_MAIN = 3

FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_DPI = 300
METHOD_COLORS = {"isolation_forest": RWTHColorScales.blue[0], "prototype": RWTHColorScales.red[0]}

sns.set_theme(style="whitegrid")


def score_fold(train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], k_f: int, k_h: int,
               seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit both methods on train windows; return (test window scores,
    train-healthy window scores) -- the latter only for thresholding."""
    scaler = StandardScaler().fit(train[feature_columns])
    x_train = scaler.transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])
    train_faulty = train["is_faulty_ev"].to_numpy()

    forest = IsolationForest(n_estimators=N_ESTIMATORS, random_state=seed).fit(x_train)
    fault_protos = KMeans(n_clusters=k_f, n_init=N_INIT, random_state=seed).fit(x_train[train_faulty])
    healthy_protos = KMeans(n_clusters=k_h, n_init=N_INIT, random_state=seed).fit(x_train[~train_faulty])

    def scores(x: np.ndarray) -> pd.DataFrame:
        dist_f = fault_protos.transform(x)
        d_f = dist_f.min(axis=1)
        d_h = healthy_protos.transform(x).min(axis=1)
        return pd.DataFrame({"if_score": -forest.score_samples(x), "dF": d_f, "dH": d_h,
                             "s": d_h / (d_f + d_h), "nearest_fault_prototype": dist_f.argmin(axis=1)})

    test_scores = pd.concat([test[["ev_id", "window_start", "window_start_hours"]].reset_index(drop=True),
                             scores(x_test)], axis=1)
    return test_scores, scores(x_train[~train_faulty])


def ev_alerts(scores: pd.DataFrame, flagged: np.ndarray, m: int) -> pd.DataFrame:
    """Per EV: alerted (>= m consecutive flagged windows) and the
    window_start_hours at which that first happens. scores must be sorted by
    (ev_id, window_start)."""
    flagged = pd.Series(flagged, index=scores.index)
    new_ev = scores["ev_id"].ne(scores["ev_id"].shift())
    run_block = (~flagged | new_ev).cumsum()
    run_length = flagged.astype(int).groupby(run_block).cumsum()
    alert_window = run_length >= m
    first_alert = scores.loc[alert_window].groupby("ev_id")["window_start_hours"].min()
    out = pd.DataFrame(index=pd.Index(scores["ev_id"].unique(), name="ev_id"))
    out["alerted"] = out.index.isin(first_alert.index)
    out["first_alert_hours"] = first_alert
    return out


def ev_metrics(alerts: pd.DataFrame, evs: pd.DataFrame) -> dict:
    a = alerts.join(evs)
    faulty, alerted, small = a["is_faulty_ev"], a["alerted"], a["pack_type"] == "small"
    tp, fp = int((faulty & alerted).sum()), int((~faulty & alerted).sum())
    fn, tn = int((faulty & ~alerted).sum()), int((~faulty & ~alerted).sum())
    lead = (a["last_window_start_hours"] - a["first_alert_hours"])[faulty & alerted]
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
        "healthy_alert_rate": fp / max(fp + tn, 1),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "TP_small_pack": int((faulty & alerted & small).sum()),
        "TP_large_pack": int((faulty & alerted & ~small).sum()),
        "FN_large_pack": int((faulty & ~alerted & ~small).sum()),
        "recall_large_pack": (faulty & alerted & ~small).sum() / max((faulty & ~small).sum(), 1),
        "median_lead_hours": float(lead.median()) if len(lead) else np.nan,
    }


# %% 1. out-of-fold scores for every source x seed x fold
all_scores, all_thresholds, ev_tables = [], [], {}
for source in WINDOW_SOURCES:
    windows, feature_columns = load_windows(source)
    evs = ev_table(windows)
    ev_tables[source] = evs
    k_f, k_h = K_SELECTION[source]["kF"], K_SELECTION[source]["kH"]
    print(f"\n=== {source}: {len(windows)} windows, {len(evs)} EVs "
          f"({int(evs['is_faulty_ev'].sum())} faulty, {int((evs['pack_type'] == 'small').sum())} small-pack), "
          f"kF={k_f}, kH={k_h} ===")
    for seed in SEEDS:
        for fold, (train_evs, test_evs) in enumerate(vehicle_folds(evs, N_SPLITS, seed)):
            train = windows[windows["ev_id"].isin(train_evs)]
            test = windows[windows["ev_id"].isin(test_evs)]
            test_scores, train_healthy_scores = score_fold(train, test, feature_columns, k_f, k_h, seed)
            all_scores.append(test_scores.assign(source=source, seed=seed, fold=fold))
            for method in METHODS:
                taus = np.quantile(train_healthy_scores[SCORE_COLUMN[method]], Q_SWEEP)
                all_thresholds += [{"source": source, "seed": seed, "fold": fold, "method": method, "q": q,
                                    "tau": tau} for q, tau in zip(Q_SWEEP, taus)]
        print(f"  seed {seed} done")

window_scores = (pd.concat(all_scores, ignore_index=True)
                 .sort_values(["source", "seed", "ev_id", "window_start"]).reset_index(drop=True))
thresholds = pd.DataFrame(all_thresholds)
window_scores.to_pickle(REVISION_OUT / "ew_window_scores.pkl")
thresholds.to_csv(REVISION_OUT / "ew_thresholds.csv", index=False)

# %% 2. apply the shared alert rule over the (q, m) grid
sweep_rows, main_alerts = [], {}
for (source, seed), scores in window_scores.groupby(["source", "seed"], sort=False):
    scores = scores.reset_index(drop=True)
    tau_lookup = thresholds[(thresholds["source"] == source) & (thresholds["seed"] == seed)]
    for method in METHODS:
        for q in Q_SWEEP:
            tau_by_fold = tau_lookup[(tau_lookup["method"] == method) & (tau_lookup["q"] == q)].set_index("fold")["tau"]
            flagged = (scores[SCORE_COLUMN[method]] > scores["fold"].map(tau_by_fold)).to_numpy()
            for m in M_SWEEP:
                alerts = ev_alerts(scores, flagged, m)
                sweep_rows.append({"source": source, "method": method, "q": q, "m": m, "seed": seed,
                                   **ev_metrics(alerts, ev_tables[source])})
                if q == Q_MAIN and m == M_MAIN:
                    main_alerts[(source, method, seed)] = alerts["alerted"]
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv(REVISION_OUT / "ew_sweep.csv", index=False)

# %% 3. headline table at (Q_MAIN, M_MAIN) + A10 Fisher test on large-pack vehicles.
# Fisher uses each EV's alert status in the majority of seeds, so every EV
# counts once.
summary_rows, overlap_parts = [], []
main = sweep[(sweep["q"] == Q_MAIN) & (sweep["m"] == M_MAIN)]
for (source, method), group in main.groupby(["source", "method"]):
    metrics = group.drop(columns=["source", "method", "q", "m", "seed"])
    row = {"source": source, "method": method, "q": Q_MAIN, "m": M_MAIN}
    row.update({f"{c}_mean": metrics[c].mean() for c in metrics.columns})
    row.update({f"{c}_std": metrics[c].std() for c in ["recall", "precision", "healthy_alert_rate",
                                                        "recall_large_pack"]})

    evs = ev_tables[source]
    alerted_majority = pd.concat([main_alerts[(source, method, s)] for s in SEEDS], axis=1).mean(axis=1) >= 0.5
    large = evs[evs["pack_type"] == "large"].join(alerted_majority.rename("alerted"))
    table = pd.crosstab(large["is_faulty_ev"], large["alerted"]).reindex(
        index=[True, False], columns=[True, False], fill_value=0)
    odds_ratio, p_value = fisher_exact(table.to_numpy(), alternative="greater")
    row.update({"large_pack_faulty_alerted": int(table.loc[True, True]), "large_pack_faulty": int(table.loc[True].sum()),
                "large_pack_healthy_alerted": int(table.loc[False, True]), "large_pack_healthy": int(table.loc[False].sum()),
                "fisher_odds_ratio": odds_ratio, "fisher_p_one_sided": p_value})
    summary_rows.append(row)
    overlap_parts.append(alerted_majority.rename(f"{source}_{method}"))

summary = pd.DataFrame(summary_rows)
summary.to_csv(REVISION_OUT / "ew_summary.csv", index=False)
pd.set_option("display.width", 200)
print(f"\n--- headline: q={Q_MAIN}, m={M_MAIN}, mean over {len(SEEDS)} seeds ---")
print(summary[["source", "method", "TP_mean", "FP_mean", "FN_mean", "recall_mean", "precision_mean",
               "healthy_alert_rate_mean", "TP_small_pack_mean", "TP_large_pack_mean", "median_lead_hours_mean"]]
      .round(3).to_string(index=False))
print("\n--- A10: large-pack vehicles only (majority-of-seeds alert), one-sided Fisher exact ---")
print(summary[["source", "method", "large_pack_faulty_alerted", "large_pack_faulty", "large_pack_healthy_alerted",
               "large_pack_healthy", "fisher_odds_ratio", "fisher_p_one_sided"]].round(4).to_string(index=False))

# %% 4. which faulty EVs each method catches (per source), split by pack type
overlap = pd.concat(overlap_parts, axis=1)
faulty_evs = pd.concat(ev_tables.values()).groupby(level=0)[["is_faulty_ev", "pack_type"]].first()
overlap = overlap.join(faulty_evs)
overlap.to_csv(REVISION_OUT / "ew_overlap.csv")
for source in WINDOW_SOURCES:
    f = overlap[overlap["is_faulty_ev"]].dropna(subset=[f"{source}_isolation_forest"])
    by_if, by_proto = f[f"{source}_isolation_forest"].astype(bool), f[f"{source}_prototype"].astype(bool)
    print(f"\n{source} faulty EVs caught -- by pack type:")
    print(pd.crosstab(f["pack_type"], (by_if.map({True: "IF", False: ""}) + "|" +
                                      by_proto.map({True: "proto", False: ""})).rename("caught by")))

# %% 5. operating characteristic: recall vs. healthy alert rate, sweeping q at m = M_MAIN
curves = sweep[sweep["m"] == M_MAIN].groupby(["source", "method", "q"])[
    ["recall", "recall_large_pack", "healthy_alert_rate"]].mean().reset_index()
fig, axes = plt.subplots(len(WINDOW_SOURCES), 2, figsize=(10, 4.2 * len(WINDOW_SOURCES)), squeeze=False)
for row, source in enumerate(WINDOW_SOURCES):
    for col, (recall_col, label) in enumerate([("recall", "all vehicles"), ("recall_large_pack", "large-pack only")]):
        ax = axes[row, col]
        for method in METHODS:
            c = curves[(curves["source"] == source) & (curves["method"] == method)]
            ax.plot(c["healthy_alert_rate"], c[recall_col], marker="o", color=METHOD_COLORS[method], label=method)
            main_point = c[c["q"] == Q_MAIN]
            ax.scatter(main_point["healthy_alert_rate"], main_point[recall_col], s=120, facecolors="none",
                       edgecolors="black", zorder=5)
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="chance")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("healthy vehicles alerted (share)")
        ax.set_ylabel(f"recall ({label})")
        ax.set_title(f"{source}, m = {M_MAIN}, {label}", fontsize=10)
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "ew_operating_characteristic.png", dpi=FIGURE_DPI)
plt.close(fig)
print(f"\nsaved {FIGURES_DIR / 'ew_operating_characteristic.png'}")
