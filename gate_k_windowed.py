"""Gate 1 diagnostics: k for k-means on all windows. Label-free: labels are never
used for selection, only for describing clusters afterwards (gate_k_report.py).

SELECTION RULE, fixed before any result was seen
  purpose   resolve operating regimes across the fleet
  range     k = 2..15; k = 2 is diagnostic only, selection runs over k = 3..15
  size      smallest cluster holds at least 2 % of windows
  stability mean pairwise ARI >= 0.75 over 50 vehicle resamples
  choice    highest silhouette among k passing size and stability;
            if none reaches 0.75, the most stable k passing size

Usage: gate_k_windowed.py 120h|24h  -> figures/thesis/gate1/sweep_<source>.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common

K_RANGE = range(2, 16)
SELECTABLE_MIN_K = 3
MIN_CLUSTER_SHARE, MIN_STABILITY = 0.02, 0.75
N_BOOTSTRAP, N_INIT, SEED = 50, 10, 0
N_JOBS = 24
OUT_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate1"


def fit_labels(x: np.ndarray, fit_rows: np.ndarray, k: int, seed: int) -> np.ndarray:
    with threadpool_limits(1):
        return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(x[fit_rows]).predict(x).astype(np.int16)


def ari(a: np.ndarray, b: np.ndarray) -> float:
    return adjusted_rand_score(a, b)


def sweep(x: np.ndarray, ev_ids: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    unique = np.unique(ev_ids)
    rows_of = {ev: np.flatnonzero(ev_ids == ev) for ev in unique}
    resamples = [np.concatenate([rows_of[ev] for ev in rng.choice(unique, len(unique), replace=True)])
                 for _ in range(N_BOOTSTRAP)]
    results = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=SEED).fit(x)
        shares = np.bincount(model.labels_, minlength=k) / len(x)
        labelings = Parallel(n_jobs=N_JOBS)(delayed(fit_labels)(x, rows, k, b) for b, rows in enumerate(resamples))
        aris = Parallel(n_jobs=N_JOBS)(delayed(ari)(a, b) for a, b in combinations(labelings, 2))
        results.append({"k": k, "silhouette": silhouette_score(x, model.labels_), "stability": float(np.mean(aris)),
                        "stability_std": float(np.std(aris)), "smallest_share": float(shares.min()),
                        "inertia": float(model.inertia_)})
        print(results[-1], flush=True)
    return pd.DataFrame(results)


def apply_rule(table: pd.DataFrame) -> dict:
    allowed = table[(table["k"] >= SELECTABLE_MIN_K) & (table["smallest_share"] >= MIN_CLUSTER_SHARE)]
    if allowed.empty:
        return {"k": None, "fallback": True}  # the rule does not define a k here; reported, not patched
    stable = allowed[allowed["stability"] >= MIN_STABILITY]
    if len(stable):
        return {"k": int(stable.loc[stable["silhouette"].idxmax(), "k"]), "fallback": False}
    return {"k": int(allowed.loc[allowed["stability"].idxmax(), "k"]), "fallback": True}


if __name__ == "__main__":
    source = sys.argv[1]
    windows, feature_columns = common.load_windows(source)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    x_all = StandardScaler().fit_transform(windows[feature_columns])
    print(f"{source}: {len(windows)} windows, {windows['ev_id'].nunique()} vehicles, {len(feature_columns)} features")
    table = sweep(x_all, windows["ev_id"].to_numpy())
    table.to_csv(OUT_DIR / f"sweep_{source}.csv", index=False)
    print("rule implies", apply_rule(table))
