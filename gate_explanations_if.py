"""Figure 7b data: as gate_explanations.py, but for every test window the Isolation Forest
flags at the matched operating point, not only windows also flagged by the centroid score.
That restriction would remove most healthy-cluster windows and so favour the clusters.
Stores SHAP plus the out-of-fold nearest cluster label. Asserts that the jointly flagged
subset reproduces the stored Figure 12 values."""
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

BACKGROUND_SIZE = 300  # as gate_explanations.py
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"


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
    flagged = -forest.score_samples(x_test) > tau_if
    if not flagged.any():
        return None
    background = x_train[healthy][np.random.default_rng(seed).choice(healthy.sum(), BACKGROUND_SIZE, replace=False)]
    with threadpool_limits(1):
        explainer = shap.TreeExplainer(forest, data=shap.maskers.Independent(background, max_samples=BACKGROUND_SIZE), feature_perturbation="interventional")
        shap_values = -explainer.shap_values(x_test[flagged], check_additivity=False)
    meta = test.loc[flagged, ["ev_id", "window_start"]].reset_index(drop=True).assign(
        seed=seed, fold=fold, centroid_flagged=scores["s"].to_numpy()[flagged] > tau_c,
        oof_label=scores["nearest_label"].to_numpy()[flagged])
    return meta, shap_values


windows, columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
taus = pd.read_csv(OUT / "matched5_thresholds.csv").pivot(index=["seed", "fold"], columns="method", values="tau")
jobs = [(train, test, seed, fold, taus.loc[(seed, fold), "isolation_forest"], taus.loc[(seed, fold), "centroid"])
        for seed in range(10) for fold, (train, test) in enumerate(ew_common.vehicle_folds(evs, 5, seed))]
done = [r for r in Parallel(n_jobs=24)(delayed(one_fold)(windows, columns, *job) for job in jobs) if r is not None]
meta = pd.concat([r[0] for r in done], ignore_index=True)
shap_values = np.vstack([r[1] for r in done])

stored = np.load(OUT / "explanations_values.npz")
stored_meta = pd.read_csv(OUT / "explanations_meta.csv")
assert stored["columns"].tolist() == list(columns)
keys = ["seed", "fold", "ev_id", "window_start"]
joint = meta.reset_index().loc[meta["centroid_flagged"].to_numpy()].merge(stored_meta[keys].reset_index(), on=keys, how="outer", suffixes=("", "_stored"), indicator=True)
assert (joint["_merge"] == "both").all(), joint["_merge"].value_counts()
assert np.allclose(shap_values[joint["index"].to_numpy()], stored["shap"][joint["index_stored"].to_numpy()])

meta.to_csv(OUT / "explanations_if_meta.csv", index=False)
np.savez_compressed(OUT / "explanations_if_values.npz", shap=shap_values, columns=np.array(columns))
print(len(meta), "window-seed rows flagged by the Isolation Forest;", int(meta["centroid_flagged"].sum()), "also by the centroid score; Figure 12 values reproduced")
