"""Figure 12 data: per seed and fold, test windows flagged by both methods at the matched
operating point; SHAP of the fold's Isolation Forest (interventional, healthy training
background, sign flipped, as shap.py) and centroid contributions
(x_j - c_H,j)^2 - (x_j - c_F,j)^2 with the nearest healthy and nearest fault centroid."""
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import ew_common
import prototypes as pr

BACKGROUND_SIZE, TOP = 300, 5  # background as in shap.py
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"
METHOD_ORDER = ("isolation_forest", "centroid")


def top_set(values: np.ndarray, positive_only: bool = False) -> set:
    order = np.argsort(-(values if positive_only else np.abs(values)))
    return set(order[:TOP])


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def one_fold(windows, columns, train_evs, test_evs, seed, fold, tau_if, tau_c):
    train = windows[windows["ev_id"].isin(train_evs)].reset_index(drop=True)
    test = windows[windows["ev_id"].isin(test_evs)].reset_index(drop=True)
    scaler = StandardScaler().fit(train[columns])
    x_train, x_test = scaler.transform(train[columns]), scaler.transform(test[columns])
    healthy = ~train["is_faulty_ev"].to_numpy()
    with threadpool_limits(1):
        forest = IsolationForest(n_estimators=100, random_state=seed).fit(x_train)
        fit = pr.fit_clusters(x_train, train, seed)
    scores = pr.score_windows(x_test, fit)
    flagged = (-forest.score_samples(x_test) > tau_if) & (scores["s"].to_numpy() > tau_c)
    if not flagged.any():
        return None
    x = x_test[flagged]
    background = x_train[healthy][np.random.default_rng(seed).choice(healthy.sum(), BACKGROUND_SIZE, replace=False)]
    with threadpool_limits(1):
        explainer = shap.TreeExplainer(forest, data=shap.maskers.Independent(background, max_samples=BACKGROUND_SIZE), feature_perturbation="interventional")
        shap_values = -explainer.shap_values(x, check_additivity=False)
    table = fit["table"]
    fault = fit["centres"][np.flatnonzero(table["label"] == "fault")]
    healthy_c = fit["centres"][np.flatnonzero(table["label"] == "healthy")]
    c_f, c_h = fault[pr.nearest(x, fault)[0]], healthy_c[pr.nearest(x, healthy_c)[0]]
    contributions = (x - c_h) ** 2 - (x - c_f) ** 2
    meta = test.loc[flagged, ["ev_id", "window_start"]].reset_index(drop=True).assign(seed=seed, fold=fold)
    return meta, shap_values, contributions


windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
taus = pd.read_csv(OUT / "matched5_thresholds.csv").pivot(index=["seed", "fold"], columns="method", values="tau")
jobs = [(train, test, seed, fold, taus.loc[(seed, fold), "isolation_forest"], taus.loc[(seed, fold), "centroid"])
        for seed in range(10) for fold, (train, test) in enumerate(ew_common.vehicle_folds(evs, 5, seed))]
done = [r for r in Parallel(n_jobs=24)(delayed(one_fold)(windows, columns, *job) for job in jobs) if r is not None]
meta = pd.concat([r[0] for r in done], ignore_index=True)
shap_values, contributions = np.vstack([r[1] for r in done]), np.vstack([r[2] for r in done])
meta["jaccard"] = [jaccard(top_set(a), top_set(b)) for a, b in zip(shap_values, contributions)]
meta["jaccard_positive"] = [jaccard(top_set(a, True), top_set(b, True)) for a, b in zip(shap_values, contributions)]
rng = np.random.default_rng(0)
shuffled = rng.permutation(len(meta))
meta["jaccard_shuffled_windows"] = [jaccard(top_set(a), top_set(b)) for a, b in zip(shap_values, contributions[shuffled])]
meta.to_csv(OUT / "explanations_meta.csv", index=False)
np.savez_compressed(OUT / "explanations_values.npz", shap=shap_values, contributions=contributions, columns=np.array(columns))
print(len(meta), "jointly flagged windows", meta.groupby("seed").size().describe()[["min", "50%", "max"]].to_dict())
