"""Gate 2: per seed and fold, fit on training vehicles only, label clusters, store the
score of every test window. Reports training-side diagnostics; no test metric is computed."""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import ew_common
import prototypes as pr

SEEDS, N_SPLITS = range(10), 5
Q_SWEEP = [0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 0.99, 0.995]  # == ew_compare.py
OUT_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate2"
DATA_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "data"


def one_fold(windows, columns, train_evs, test_evs, seed, fold):
    train = windows[windows["ev_id"].isin(train_evs)].reset_index(drop=True)
    test = windows[windows["ev_id"].isin(test_evs)].reset_index(drop=True)
    assert not set(train["ev_id"]) & set(test["ev_id"])
    scaler = StandardScaler().fit(train[columns])
    x_train, x_test = scaler.transform(train[columns]), scaler.transform(test[columns])
    faulty = train["is_faulty_ev"].to_numpy()
    with threadpool_limits(1):
        fit = pr.fit_clusters(x_train, train, seed)
    table = fit["table"].assign(seed=seed, fold=fold)
    counts = table["label"].value_counts().reindex([*pr.LABELS, "excluded"], fill_value=0)
    row = {"seed": seed, "fold": fold, **counts.to_dict(), "no_healthy_assigned": int(np.isinf(table["ratio"]).sum())}
    if counts["fault"] and counts["healthy"]:
        train_scores = pr.score_windows(x_train, fit)
        limit = np.percentile(train_scores.loc[faulty, "d_nearest"], 95)
        row["healthy_nearest_fault"] = float((train_scores.loc[~faulty, "nearest_label"] == "fault").mean())
        row["healthy_nearest_transition"] = float((train_scores.loc[~faulty, "nearest_label"] == "transition").mean())
        row["healthy_nearest_healthy"] = float((train_scores.loc[~faulty, "nearest_label"] == "healthy").mean())
        row["healthy_far"] = float((train_scores.loc[~faulty, "d_nearest"] > limit).mean())
        row["distance_limit"] = float(limit)
        test_scores = pd.concat([test[["ev_id", "window_start"]], pr.score_windows(x_test, fit)], axis=1).assign(seed=seed, fold=fold)
        taus = np.quantile(train_scores.loc[~faulty, "s"], Q_SWEEP)  # threshold from healthy training windows only
        thresholds = [{"seed": seed, "fold": fold, "q": q, "tau": t} for q, t in zip(Q_SWEEP, taus)]
    else:
        test_scores, thresholds = None, []
    return row, table, test_scores, thresholds


windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
jobs = [(train, test, seed, fold) for seed in SEEDS for fold, (train, test) in enumerate(ew_common.vehicle_folds(evs, N_SPLITS, seed))]
results = Parallel(n_jobs=24)(delayed(one_fold)(windows, columns, *job) for job in jobs)
OUT_DIR.mkdir(exist_ok=True)
folds = pd.DataFrame([r[0] for r in results])
clusters = pd.concat([r[1] for r in results]).reset_index()
folds.to_csv(OUT_DIR / "fold_summary.csv", index=False)
clusters.to_csv(OUT_DIR / "clusters_per_fold.csv", index=False)

pd.set_option("display.width", 170)
stopped = folds[(folds["fault"] == 0) | (folds["healthy"] == 0)]
print(f"{len(folds)} fits")
print(folds[[*pr.LABELS, "excluded"]].agg(["mean", "min", "max"]).round(2).to_string())
print("fault counts", folds["fault"].value_counts().sort_index().to_dict())
print("healthy counts", folds["healthy"].value_counts().sort_index().to_dict())
print("transition counts", folds["transition"].value_counts().sort_index().to_dict())
print("clusters with no healthy window assigned per fit:", folds["no_healthy_assigned"].mean().round(2))
print("STOP RULE:", "HIT\n" + stopped[["seed", "fold", *pr.LABELS, "excluded"]].to_string(index=False) if len(stopped) else "not hit")
print(folds[["healthy_nearest_fault", "healthy_nearest_transition", "healthy_nearest_healthy", "healthy_far", "distance_limit"]].agg(["mean", "min", "max"]).round(3).T.to_string())
clusters["origin"] = clusters["large_pack_share"].map(pr.origin)
print(clusters.groupby("label")[["windows", "vehicles", "healthy_assigned"]].agg(["count", "min", "median", "max"]).round(0).to_string())
print(100 * pd.crosstab(clusters["label"], clusters["origin"], normalize="index").round(3))
print(pd.crosstab(clusters["label"], clusters["origin"]))
print("share of faulty training windows by label:", (clusters.groupby("label")["windows"].sum() / clusters["windows"].sum()).round(3).to_dict())
print("share of large-pack faulty windows by label:", ((clusters["windows"] * clusters["large_pack_share"]).groupby(clusters["label"]).sum() / (clusters["windows"] * clusters["large_pack_share"]).sum()).round(3).to_dict())
if stopped.empty:
    DATA_DIR.mkdir(exist_ok=True)
    scores = pd.concat([r[2] for r in results], ignore_index=True)
    pd.DataFrame([t for r in results for t in r[3]]).to_csv(DATA_DIR / "ew_centroid_thresholds_120h.csv", index=False)
    scores.to_pickle(DATA_DIR / "ew_centroid_scores_120h.pkl")
    print("stored", len(scores), "test-window scores; 10 seeds cover every window once:", (scores.groupby("seed").size() == len(windows)).all())

for fault_min, healthy_max in ((pr.FAULT_RATIO, pr.HEALTHY_RATIO), (1.5, 0.67), (3.0, 0.33)):
    kept = clusters[clusters["label"] != "excluded"]
    label = np.select([kept["ratio"] >= fault_min, kept["ratio"] <= healthy_max], ["fault", "healthy"], "transition")
    counts = pd.crosstab([kept["seed"], kept["fold"]], label).reindex(columns=list(pr.LABELS), fill_value=0)
    print(f"thresholds r >= {fault_min} / <= {healthy_max}:", {c: (round(counts[c].mean(), 2), int(counts[c].min()), int(counts[c].max())) for c in counts},
          "| fits without fault or healthy cluster:", int(((counts["fault"] == 0) | (counts["healthy"] == 0)).sum()))
    counts.reset_index().assign(fault_min=fault_min, healthy_max=healthy_max).to_csv(OUT_DIR / f"robustness_labels_{fault_min}_{healthy_max}.csv", index=False)
