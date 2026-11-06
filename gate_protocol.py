"""Early-warning comparison, 120 h: Isolation Forest vs centroid score under one rule.
Threshold: quantile of healthy training windows, per fold. Alert: >= m consecutive flagged
windows. Headline q = 0.95, m = 3 (ew_compare.py); the sweep is kept for Figure 9."""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from scipy.stats.contingency import odds_ratio
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import alert
import common
import ew_common

Q_SWEEP = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99]  # budgets 20, 15, 10, 7.5, 5, 2.5, 1 %
M_SWEEP = [1, 2, 3]
Q_MAIN, M_MAIN = 0.95, 3
METHODS = {"isolation_forest": "if_score", "centroid": "s"}
OUT_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"
DATA_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "data"
REVISION = common.DATA / "revision"
OUT_DIR.mkdir(exist_ok=True)

windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
evs["recorded_hours"] = windows.groupby("ev_id")["recorded_hours"].max() + 120.0

stored = pd.read_pickle(REVISION / "ew_window_scores.pkl").query("source == '120h'").drop(columns=["s", "dF", "dH", "nearest_fault_prototype"])  # old ew_compare.py score must not shadow the documented s
centroid = pd.read_pickle(DATA_DIR / "ew_centroid_scores_120h.pkl")
scores = stored.merge(centroid, on=["seed", "ev_id", "window_start"], validate="one_to_one", suffixes=("", "_c"))
assert len(scores) == 62090 and (scores["fold"] == scores["fold_c"]).all(), "folds differ between methods"
assert np.allclose(scores["s"], scores["d_healthy"] / (scores["d_fault"] + scores["d_healthy"])), "s is not the documented score"
scores = scores.sort_values(["seed", "ev_id", "window_start"]).reset_index(drop=True)

thresholds = pd.read_csv(REVISION / "ew_thresholds.csv").query("source == '120h' and method == 'isolation_forest'")
tau = {"isolation_forest": thresholds[["seed", "fold", "q", "tau"]], "centroid": pd.read_csv(DATA_DIR / "ew_centroid_thresholds_120h.csv")}

# check 1: stored IF scores still come from the current features
train_evs, test_evs = next(ew_common.vehicle_folds(evs, 5, 0))
train, test = windows[windows["ev_id"].isin(train_evs)], windows[windows["ev_id"].isin(test_evs)]
scaler = StandardScaler().fit(train[columns])
refit = -IsolationForest(n_estimators=100, random_state=0).fit(scaler.transform(train[columns])).score_samples(scaler.transform(test[columns]))
check = scores[(scores["seed"] == 0) & (scores["fold"] == 0)]
print("IF scores reproduced on current features:", np.allclose(np.sort(refit), np.sort(check["if_score"])))

# check 2: folds are disjoint and every vehicle is tested once per seed
print("every vehicle tested once per seed:", (scores.groupby(["seed", "ev_id"])["fold"].nunique() == 1).all())
print("test vehicles per fold, seed 0:", scores[scores["seed"] == 0].groupby("fold")["ev_id"].nunique().to_dict())

rows, main_alerts, first_alerts = [], {}, []
window_rates = []
for seed, part in scores.groupby("seed", sort=False):
    part = part.reset_index(drop=True)
    healthy_window = ~part["ev_id"].map(evs["is_faulty_ev"]).to_numpy()
    for method, column in METHODS.items():
        lookup = tau[method][tau[method]["seed"] == seed]
        for q in Q_SWEEP:
            by_fold = lookup[lookup["q"] == q].set_index("fold")["tau"]
            flagged = (part[column] > part["fold"].map(by_fold)).to_numpy()
            window_rates.append({"seed": seed, "method": method, "q": q, "healthy_window_flag_rate": flagged[healthy_window].mean()})
            for m in M_SWEEP:
                alerts = alert.ev_alerts(part, flagged, m)
                rows.append({"method": method, "q": q, "m": m, "seed": seed, **alert.ev_metrics(alerts, evs)})
                if q == Q_MAIN and m == M_MAIN:
                    main_alerts[(method, seed)] = alerts["alerted"]
                    first_alerts.append(alerts[alerts["alerted"]].assign(method=method, seed=seed).reset_index())
sweep = pd.DataFrame(rows).merge(pd.DataFrame(window_rates), on=["seed", "method", "q"])
sweep.to_csv(OUT_DIR / "sweep_120h.csv", index=False)

# check 3: the ported rule reproduces the stored sweep for the Isolation Forest
old = pd.read_csv(REVISION / "ew_sweep.csv").query("source == '120h' and method == 'isolation_forest' and q in @Q_SWEEP")
joined = sweep[sweep["method"] == "isolation_forest"].merge(old, on=["q", "m", "seed"], suffixes=("", "_old"))
same = all((joined[c] == joined[f"{c}_old"]).all() for c in ["TP", "FP", "FN", "TN", "TP_small_pack", "TP_large_pack"])
print(f"ported rule reproduces ew_sweep.csv: {same} ({len(joined)} rows), lead equal:", np.allclose(joined["median_lead_hours"], joined["median_lead_hours_old"], equal_nan=True))

first = pd.concat(first_alerts, ignore_index=True).join(evs[["is_faulty_ev", "pack_type", "last_window_start_hours"]], on="ev_id")
first["hours_before_end"] = first["last_window_start_hours"] - first["first_alert_hours"]
first[["method", "seed", "ev_id", "is_faulty_ev", "pack_type", "first_alert_window_start", "first_alert_hours", "hours_before_end", "episodes"]].to_csv(OUT_DIR / "first_alerts_b5_m3.csv", index=False)

main = sweep[(sweep["q"] == Q_MAIN) & (sweep["m"] == M_MAIN)]
metrics = [c for c in main.columns if c not in ("method", "q", "m", "seed")]
table = main.groupby("method", sort=False)[metrics].agg(["mean", "std"]).round(3)
table.to_csv(OUT_DIR / "comparison_q95_m3.csv")
pd.set_option("display.width", 170)
print(table.T.to_string())

print("\nmatched comparison, m = 3, interpolated per seed from the sweep:")
curves = sweep[sweep["m"] == M_MAIN]
matched = []
for (method, seed), part in curves.groupby(["method", "seed"], sort=False):
    part = part.sort_values("healthy_alert_rate")
    for share in (0.05, 0.10):
        matched.append({"method": method, "seed": seed, "healthy_share": share, **{c: float(np.interp(share, part["healthy_alert_rate"], part[c]))
                                                                                  for c in ("recall", "recall_small_pack", "recall_large_pack", "precision")},
                        "budget_b_pct": float(np.interp(share, part["healthy_alert_rate"], 100 * (1 - part["q"])))})
matched = pd.DataFrame(matched)
matched_table = matched.groupby(["healthy_share", "method"], sort=False)[["recall", "recall_small_pack", "recall_large_pack", "precision", "budget_b_pct"]].agg(["mean", "std"]).round(3)
matched_table.to_csv(OUT_DIR / "matched_comparison.csv")
print(matched_table.to_string())

print("\nlarge-pack chance test (majority of seeds, one-sided Fisher as in ew_compare.py):")
majority = {}
for method in METHODS:
    votes = pd.concat([main_alerts[(method, s)] for s in range(10)], axis=1).mean(axis=1) >= 0.5
    majority[method] = votes
    large = evs[evs["pack_type"] == "large"].join(votes.rename("alerted"))
    cells = pd.crosstab(large["is_faulty_ev"], large["alerted"]).reindex(index=[True, False], columns=[True, False], fill_value=0).to_numpy()
    result = odds_ratio(cells)
    interval = result.confidence_interval(0.95)
    print(f"  {method}: faulty {cells[0, 0]}/{cells[0].sum()}, healthy {cells[1, 0]}/{cells[1].sum()}, "
          f"OR {result.statistic:.2f} [{interval.low:.2f}, {interval.high:.2f}], p = {fisher_exact(cells, alternative='greater')[1]:.3f}")
pd.DataFrame(majority).join(evs[["is_faulty_ev", "pack_type"]]).to_csv(OUT_DIR / "alerted_majority.csv")
evs["group"] = np.where(~evs["is_faulty_ev"], "healthy", "faulty_" + evs["pack_type"])
print("\nrecord length [h] by group:\n", evs.groupby("group")["recorded_hours"].describe()[["count", "min", "50%", "max"]].round(0).to_string())
print("vehicles that cannot alert at m = 3 (under 3 windows):", evs.index[evs["n_windows"] < 3].tolist())
