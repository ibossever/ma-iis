"""Centroid method (Gate 2, decisions.json): one k-means on windows of faulty vehicles,
size rule, enrichment labels, score. Used for the fold-wise fits and the descriptive fit."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

K = 13
MIN_WINDOWS, MIN_VEHICLES = 5, 3
FAULT_RATIO, HEALTHY_RATIO = 2.0, 0.5
N_INIT = 10
LABELS = ("fault", "transition", "healthy")


def nearest(x: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances = np.linalg.norm(x[:, None, :] - centres[None, :, :], axis=2)
    return distances.argmin(axis=1), distances.min(axis=1)


def fit_clusters(x: np.ndarray, windows: pd.DataFrame, seed: int, size_rule: bool = True) -> dict:
    """x: standardised training windows, row-aligned with `windows`. size_rule=False is
    for the appendix robustness comparison only."""
    faulty = windows["is_faulty_ev"].to_numpy()
    model = KMeans(n_clusters=K, n_init=N_INIT, random_state=seed).fit(x[faulty])
    centres = model.cluster_centers_
    members = windows[faulty].assign(cluster=model.labels_)
    table = members.groupby("cluster").agg(
        windows=("ev_id", "size"), vehicles=("ev_id", "nunique"),
        large_pack_share=("pack_type", lambda s: (s == "large").mean())).reindex(range(K))
    table["excluded"] = (table["windows"] < MIN_WINDOWS) | (table["vehicles"] < MIN_VEHICLES)

    if not size_rule:
        table["excluded"] = False
    remaining = np.flatnonzero(~table["excluded"].to_numpy())
    position, _ = nearest(x[~faulty], centres[remaining])
    healthy_counts = np.zeros(K, dtype=int)
    healthy_counts[remaining] = np.bincount(position, minlength=len(remaining))
    table["healthy_assigned"] = healthy_counts
    table["faulty_share"] = table["windows"] / faulty.sum()
    table["healthy_share"] = healthy_counts / (~faulty).sum()
    with np.errstate(divide="ignore"):
        table["ratio"] = table["faulty_share"] / table["healthy_share"]  # inf: no healthy window assigned
    table["label"] = np.select([table["excluded"], table["ratio"] >= FAULT_RATIO, table["ratio"] <= HEALTHY_RATIO],
                               ["excluded", "fault", "healthy"], "transition")
    return {"centres": centres, "table": table, "member_labels": model.labels_}


def score_windows(x: np.ndarray, fit: dict) -> pd.DataFrame:
    """Distances, score s, and the nearest non-excluded cluster with its label."""
    table, centres = fit["table"], fit["centres"]
    fault = np.flatnonzero(table["label"] == "fault")
    healthy = np.flatnonzero(table["label"] == "healthy")
    remaining = np.flatnonzero(table["label"] != "excluded")
    d_fault, d_healthy = nearest(x, centres[fault])[1], nearest(x, centres[healthy])[1]
    position, d_nearest = nearest(x, centres[remaining])
    cluster = remaining[position]
    return pd.DataFrame({"d_fault": d_fault, "d_healthy": d_healthy, "s": d_healthy / (d_fault + d_healthy),
                         "nearest_cluster": cluster, "nearest_label": table["label"].to_numpy()[cluster],
                         "d_nearest": d_nearest})


def origin(large_share: float) -> str:
    return "small only" if large_share == 0 else "large only" if large_share == 1 else "mostly small" if large_share < 0.5 else "mostly large"


def descriptive_fit(windows: pd.DataFrame, x: np.ndarray, seed: int) -> dict:
    """One fit on all vehicles under the same rule, scaler on all windows. Clusters are
    named F1.., T1.., H1.., X1.. (X = excluded) by faulty-window count within each label.
    `groups`: every window's nearest non-excluded cluster (Figure 7). `members`: the k-means
    cluster of each faulty-vehicle window, all 13 clusters (Figures 4b, 5b)."""
    fit = fit_clusters(x, windows, seed)
    table = fit["table"].copy()
    table["name"] = ""
    for label, prefix in (*zip(LABELS, "FTH"), ("excluded", "X")):
        part = table[table["label"] == label].sort_values("windows", ascending=False)
        table.loc[part.index, "name"] = [f"{prefix}{i + 1}" for i in range(len(part))]
    scores = score_windows(x, fit)
    groups = table["name"].to_numpy()[scores["nearest_cluster"].to_numpy()]
    members = table["name"].to_numpy()[fit["member_labels"]]
    in_order = lambda labels: pd.concat([table[table["label"] == v].sort_values("windows", ascending=False) for v in labels])
    return {"table": table, "ordered": in_order(LABELS), "all_ordered": in_order((*LABELS, "excluded")), "centres": fit["centres"],
            "groups": groups, "members": members, "scores": scores}
