"""Warning margin before end of record at the matched operating point: per seed and method,
the budget b (quantile of healthy training windows, as in the alert rule) whose 3-window
rule warns the share of healthy test vehicles closest to 5 %. Descriptive: b is read from
test results, as any matched comparison is. Alert time = end of the third flagged window."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import alert
import common
import ew_common
import prototypes as pr

SEEDS, N_SPLITS, M, TARGET = range(10), 5, 3, 0.05
BUDGETS = np.round(np.arange(0.2, 8.01, 0.05), 2)
WINDOW_HOURS, STRIDE_HOURS = 120.0, 12.0
EARLIEST_HOURS = WINDOW_HOURS + (M - 1) * STRIDE_HOURS
METHODS = {"isolation_forest": "if_score", "centroid": "s"}
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"


def healthy_training_scores(windows, columns, train_evs, seed, fold):
    train = windows[windows["ev_id"].isin(train_evs)].reset_index(drop=True)
    x = StandardScaler().fit_transform(train[columns])
    faulty = train["is_faulty_ev"].to_numpy()
    with threadpool_limits(1):
        forest = IsolationForest(n_estimators=100, random_state=seed).fit(x)
        fit = pr.fit_clusters(x, train, seed)
    return seed, fold, {"isolation_forest": -forest.score_samples(x[~faulty]), "centroid": pr.score_windows(x[~faulty], fit)["s"].to_numpy()}


def search(part, train_scores, evs, method, seed):
    """Budget whose warned-healthy share is closest to the target; smaller b on ties."""
    healthy_evs = evs.index[~evs["is_faulty_ev"]]
    column, best = METHODS[method], None
    for b in BUDGETS:
        taus = {fold: np.quantile(train_scores[(seed, fold)][method], 1 - b / 100) for fold in range(N_SPLITS)}
        alerts = alert.ev_alerts(part, (part[column] > part["fold"].map(taus)).to_numpy(), M)
        rate = alerts.loc[healthy_evs, "alerted"].mean()
        if best is None or abs(rate - TARGET) < abs(best["rate"] - TARGET) - 1e-12:
            best = {"b": b, "rate": rate, "alerts": alerts, "taus": taus}
    return method, seed, best


windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
n_samples = pd.read_pickle(common.DATA / "relative_capacity" / "charging_sessions.pkl")["n_samples"]
evs["record_hours"] = n_samples.reindex(evs.index) * common.SAMPLE_INTERVAL_S / 3600.0
assert evs["record_hours"].notna().all()

jobs = [(train, seed, fold) for seed in SEEDS for fold, (train, _) in enumerate(ew_common.vehicle_folds(evs, N_SPLITS, seed))]
done = Parallel(n_jobs=24)(delayed(healthy_training_scores)(windows, columns, train, seed, fold) for train, seed, fold in jobs)
train_scores = {(seed, fold): values for seed, fold, values in done}

scores, taus = alert.load_scores()
stored = taus["isolation_forest"][np.isclose(taus["isolation_forest"]["q"], 0.95)].set_index(["seed", "fold"])["tau"]
check = all(np.isclose(np.quantile(train_scores[key]["isolation_forest"], 0.95), stored[key]) for key in train_scores)
stored_c = taus["centroid"][np.isclose(taus["centroid"]["q"], 0.95)].set_index(["seed", "fold"])["tau"]
check_c = all(np.isclose(np.quantile(train_scores[key]["centroid"], 0.95), stored_c[key]) for key in train_scores)
print("training thresholds reproduce stored values: IF", check, "| centroid", check_c)

parts = {seed: part.reset_index(drop=True) for seed, part in scores.groupby("seed")}
found = Parallel(n_jobs=20)(delayed(search)(parts[seed], train_scores, evs, method, seed) for seed in SEEDS for method in METHODS)

rows, points, tau_rows = [], [], []
for method, seed, best in found:
    a = best["alerts"].join(evs[["is_faulty_ev", "pack_type", "record_hours"]])
    a = a[a["alerted"]]
    a["first_alert_recorded_h"] = a["first_alert_window_start"] * common.SAMPLE_INTERVAL_S / 3600.0 + WINDOW_HOURS
    a["margin_h"] = a["record_hours"] - a["first_alert_recorded_h"]
    rows.append(a.reset_index().assign(method=method, seed=seed))
    points.append({"method": method, "seed": seed, "b_pct": best["b"], "healthy_warned": best["rate"]})
    tau_rows += [{"method": method, "seed": seed, "fold": fold, "b_pct": best["b"], "tau": tau} for fold, tau in best["taus"].items()]
alerts_all = pd.concat(rows, ignore_index=True)
alerts_all.to_csv(OUT / "matched5_first_alerts.csv", index=False)
points = pd.DataFrame(points)
points.to_csv(OUT / "matched5_budgets.csv", index=False)
pd.DataFrame(tau_rows).to_csv(OUT / "matched5_thresholds.csv", index=False)
pd.set_option("display.width", 170)
print(points.groupby("method")[["b_pct", "healthy_warned"]].agg(["mean", "std", "min", "max"]).round(4).to_string())

detected = alerts_all[alerts_all["is_faulty_ev"]]
per_seed = []
for (method, seed, pack), part in pd.concat([detected.assign(pack_type="all"), detected]).groupby(["method", "seed", "pack_type"]):
    q = part[["margin_h", "first_alert_recorded_h"]].quantile([0.25, 0.5, 0.75])
    per_seed.append({"method": method, "pack": pack, "seed": seed, "detected": len(part),
                     "margin_median": q.loc[0.5, "margin_h"], "margin_q25": q.loc[0.25, "margin_h"], "margin_q75": q.loc[0.75, "margin_h"],
                     "first_median": q.loc[0.5, "first_alert_recorded_h"], "first_q25": q.loc[0.25, "first_alert_recorded_h"], "first_q75": q.loc[0.75, "first_alert_recorded_h"],
                     "earliest_pct": 100 * np.isclose(part["first_alert_recorded_h"], EARLIEST_HOURS).mean()})
per_seed = pd.DataFrame(per_seed)
per_seed.to_csv(OUT / "matched5_margin_per_seed.csv", index=False)
measures = ["detected", "margin_median", "margin_q25", "margin_q75", "first_median", "first_q25", "first_q75", "earliest_pct"]
table = per_seed.groupby(["pack", "method"])[measures].agg(["mean", "std"]).round(1)
table["seeds_with_detection"] = per_seed.groupby(["pack", "method"])["seed"].nunique()
table.to_csv(OUT / "matched5_margin_table.csv")
print(table.to_string())

paired = []
for (seed, pack), part in pd.concat([detected.assign(pack_type="all"), detected]).groupby(["seed", "pack_type"]):
    wide = part.pivot(index="ev_id", columns="method", values="first_alert_recorded_h").reindex(columns=list(METHODS)).dropna()
    difference = wide["centroid"] - wide["isolation_forest"] if len(wide) else pd.Series(dtype=float)
    nonzero = difference[difference != 0]
    paired.append({"seed": seed, "pack": pack, "pairs": len(wide), "median_diff_h": difference.median() if len(wide) else np.nan,
                   "centroid_earlier": int((difference < 0).sum()), "if_earlier": int((difference > 0).sum()), "tied": int((difference == 0).sum()),
                   "p": wilcoxon(nonzero).pvalue if len(nonzero) >= 6 else np.nan})
paired = pd.DataFrame(paired)
paired.to_csv(OUT / "matched5_paired_per_seed.csv", index=False)
print(paired.groupby("pack")[["pairs", "median_diff_h", "centroid_earlier", "if_earlier", "tied"]].agg(["mean", "std"]).round(1).to_string())
print("per-seed p (all): median", round(paired.loc[paired["pack"] == "all", "p"].median(), 4), "range", paired.loc[paired["pack"] == "all", "p"].min(), paired.loc[paired["pack"] == "all", "p"].max(),
      "| small: median", round(paired.loc[paired["pack"] == "small", "p"].median(), 4))

# each vehicle once: mean first-alert time over the seeds where both methods warn it
both = detected.pivot_table(index=["seed", "ev_id"], columns="method", values="first_alert_recorded_h").dropna()
vehicle = both.groupby("ev_id").mean().join(both.groupby("ev_id").size().rename("seeds")).join(evs["pack_type"])
vehicle["difference"] = vehicle["centroid"] - vehicle["isolation_forest"]
vehicle.to_csv(OUT / "matched5_paired_per_vehicle.csv")
for pack, part in (("all", vehicle), ("small", vehicle[vehicle["pack_type"] == "small"]), ("large", vehicle[vehicle["pack_type"] == "large"])):
    nonzero = part["difference"][part["difference"] != 0]
    result = wilcoxon(nonzero) if len(nonzero) >= 6 else None
    print(f"vehicle level, {pack}: n = {len(part)}, median difference {part['difference'].median() if len(part) else float('nan'):.1f} h, IQR {part['difference'].quantile(.25) if len(part) else float('nan'):.1f}..{part['difference'].quantile(.75) if len(part) else float('nan'):.1f}, centroid earlier {(part['difference'] < 0).sum()}, IF earlier {(part['difference'] > 0).sum()}, tied {(part['difference'] == 0).sum()}, Wilcoxon W = {result.statistic if result else float('nan')}, p = {result.pvalue if result else float('nan'):.4f}")
