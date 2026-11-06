"""Appendix comparisons, 3-window rule, same folds and seeds. (1) Contamination threshold:
quantile over ALL training windows, 5 % and 10 %, both methods. (2) No size rule: all 13
clusters labelled, centroid score only, at b = 5 % and at the matched 5 % point."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import alert
import common
import ew_common
import prototypes as pr

M, TARGET = 3, 0.05
BUDGETS = np.round(np.arange(0.2, 8.01, 0.05), 2)
CONTAMINATION = (0.05, 0.10)
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"
COLUMNS = {"isolation_forest": "if_score", "centroid": "s", "centroid_no_size_rule": "s_all"}


def one_fold(windows, columns, train_evs, test_evs, seed, fold):
    train = windows[windows["ev_id"].isin(train_evs)].reset_index(drop=True)
    test = windows[windows["ev_id"].isin(test_evs)].reset_index(drop=True)
    scaler = StandardScaler().fit(train[columns])
    x, x_test = scaler.transform(train[columns]), scaler.transform(test[columns])
    healthy = ~train["is_faulty_ev"].to_numpy()
    with threadpool_limits(1):
        forest = IsolationForest(n_estimators=100, random_state=seed).fit(x)
        fit = pr.fit_clusters(x, train, seed)
        fit_all = pr.fit_clusters(x, train, seed, size_rule=False)
    counts = fit_all["table"]["label"].value_counts().reindex(pr.LABELS, fill_value=0).to_dict()
    train_if, train_s = -forest.score_samples(x), pr.score_windows(x, fit)["s"].to_numpy()
    train_all = pr.score_windows(x, fit_all)["s"].to_numpy()
    tests = test[["ev_id", "window_start"]].assign(seed=seed, fold=fold, s_all=pr.score_windows(x_test, fit_all)["s"].to_numpy())
    return {"seed": seed, "fold": fold, "counts": counts, "test": tests,
            "all": {"isolation_forest": train_if, "centroid": train_s},
            "healthy": {"isolation_forest": train_if[healthy], "centroid": train_s[healthy], "centroid_no_size_rule": train_all[healthy]}}


def evaluate(part, evs, column, taus, label, value, method, seed):
    flagged = (part[column] > part["fold"].map(taus)).to_numpy()
    alerts = alert.ev_alerts(part, flagged, M)
    healthy_window = ~part["ev_id"].map(evs["is_faulty_ev"]).to_numpy()
    return {"variant": label, "value": value, "method": method, "seed": seed, "healthy_window_flag_rate": flagged[healthy_window].mean(), **alert.ev_metrics(alerts, evs)}


def matched(part, evs, folds, seed):
    best = None
    for b in BUDGETS:
        taus = {f: np.quantile(folds[(seed, f)]["healthy"]["centroid_no_size_rule"], 1 - b / 100) for f in range(5)}
        row = evaluate(part, evs, "s_all", taus, "no_size_rule_matched", 5, "centroid_no_size_rule", seed)
        if best is None or abs(row["healthy_alert_rate"] - TARGET) < abs(best["healthy_alert_rate"] - TARGET) - 1e-12:
            best = {**row, "b_pct": b}
    return best


windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
evs["recorded_hours"] = windows.groupby("ev_id")["recorded_hours"].max() + 120.0
jobs = [(train, test, seed, fold) for seed in range(10) for fold, (train, test) in enumerate(ew_common.vehicle_folds(evs, 5, seed))]
done = Parallel(n_jobs=24)(delayed(one_fold)(windows, columns, *job) for job in jobs)
folds = {(d["seed"], d["fold"]): d for d in done}
extra = pd.concat([d["test"] for d in done], ignore_index=True)
scores, _ = alert.load_scores()
scores = scores.merge(extra, on=["seed", "fold", "ev_id", "window_start"], validate="one_to_one")
assert len(scores) == 62090

counts = pd.DataFrame([{"seed": d["seed"], "fold": d["fold"], **d["counts"]} for d in done])
print("no size rule, clusters per fold:", counts[list(pr.LABELS)].agg(["mean", "min", "max"]).round(2).to_dict(), "| fits lacking fault or healthy:", int(((counts["fault"] == 0) | (counts["healthy"] == 0)).sum()))

rows, match = [], []
for seed, part in scores.groupby("seed", sort=False):
    part = part.reset_index(drop=True)
    for c in CONTAMINATION:
        for method in ("isolation_forest", "centroid"):
            taus = {f: np.quantile(folds[(seed, f)]["all"][method], 1 - c) for f in range(5)}
            rows.append(evaluate(part, evs, COLUMNS[method], taus, "contamination", 100 * c, method, seed))
    for method in ("centroid", "centroid_no_size_rule"):
        taus = {f: np.quantile(folds[(seed, f)]["healthy"][method], 0.95) for f in range(5)}
        rows.append(evaluate(part, evs, COLUMNS[method], taus, "budget", 5, method, seed))
    match.append(matched(part, evs, folds, seed))
table = pd.concat([pd.DataFrame(rows), pd.DataFrame(match)], ignore_index=True)
table.to_csv(OUT / "appendix_comparisons_per_seed.csv", index=False)
metrics = ["TP", "FP", "recall", "precision", "f1", "healthy_alert_rate", "healthy_window_flag_rate", "recall_small_pack", "recall_large_pack", "median_lead_hours", "b_pct"]
summary = table.groupby(["variant", "value", "method"], sort=False)[metrics].agg(["mean", "std"]).round(3)
summary.to_csv(OUT / "appendix_comparisons.csv")
pd.set_option("display.width", 200)
print(summary.xs("mean", axis=1, level=1).to_string())
print(summary.xs("std", axis=1, level=1).to_string())

fmt = lambda frame, key, column, scale=100, digits=1: f"{frame.loc[key, (column, 'mean')] * scale:.{digits}f} $\\pm$ {frame.loc[key, (column, 'std')] * scale:.{digits}f}"
name = {"isolation_forest": "Isolation Forest", "centroid": "Centroid score", "centroid_no_size_rule": "Centroid, no size rule"}
lines = [r"\begin{tabular}{llccccc}", r"\toprule", r"Threshold & Method & Recall [\%] & Precision [\%] & Healthy warned [\%] & Recall small [\%] & Recall large [\%] \\", r"\midrule"]
rule = {("contamination", 5.0): "Contamination 5\\%", ("contamination", 10.0): "Contamination 10\\%", ("budget", 5.0): "Budget $b = 5\\,\\%$", ("no_size_rule_matched", 5.0): "Matched, 5\\% healthy"}
for key in summary.index:
    cells = [fmt(summary, key, c) for c in ("recall", "precision", "healthy_alert_rate", "recall_small_pack", "recall_large_pack")]
    lines.append(f"{rule[key[:2]]} & {name[key[2]]} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(common.PROJECT_ROOT / "figures" / "thesis" / "tableA_appendix_comparisons.tex").write_text("\n".join(lines) + "\n")
