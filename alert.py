"""Alert rule from 03_revision/ew_compare.py, ported because importing that script reruns
it. ev_alerts follows it line by line; ev_metrics adds F1, pack-type recall and alert rate.
gate_protocol.py checks these against the stored ew_sweep.csv before using them."""
import numpy as np
import pandas as pd


def ev_alerts(scores: pd.DataFrame, flagged: np.ndarray, m: int) -> pd.DataFrame:
    """Per EV: alerted (>= m consecutive flagged windows), first alert time, and the number
    of alert episodes (runs reaching m). scores: sorted by (ev_id, window_start)."""
    flagged = pd.Series(flagged, index=scores.index)
    new_ev = scores["ev_id"].ne(scores["ev_id"].shift())
    run_block = (~flagged | new_ev).cumsum()
    run_length = flagged.astype(int).groupby(run_block).cumsum()
    alert_window = run_length >= m
    first_alert = scores.loc[alert_window].groupby("ev_id")["window_start_hours"].min()
    episodes = (run_length == m).groupby(scores["ev_id"]).sum()
    out = pd.DataFrame(index=pd.Index(scores["ev_id"].unique(), name="ev_id"))
    out["alerted"] = out.index.isin(first_alert.index)
    out["first_alert_hours"] = first_alert
    out["first_alert_window_start"] = scores.loc[alert_window].groupby("ev_id")["window_start"].min()
    out["episodes"] = episodes
    return out


def ev_metrics(alerts: pd.DataFrame, evs: pd.DataFrame) -> dict:
    a = alerts.join(evs)
    faulty, alerted, small = a["is_faulty_ev"], a["alerted"], a["pack_type"] == "small"
    tp, fp = int((faulty & alerted).sum()), int((~faulty & alerted).sum())
    fn, tn = int((faulty & ~alerted).sum()), int((~faulty & ~alerted).sum())
    lead = (a["last_window_start_hours"] - a["first_alert_hours"])[faulty & alerted]
    recall, precision = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    healthy_hours = a.loc[~faulty, "recorded_hours"].sum()
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "recall": recall, "precision": precision,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "healthy_alert_rate": fp / max(fp + tn, 1),
        "balanced_accuracy": 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
        "TP_small_pack": int((faulty & alerted & small).sum()), "TP_large_pack": int((faulty & alerted & ~small).sum()),
        "FN_large_pack": int((faulty & ~alerted & ~small).sum()),
        "recall_small_pack": (faulty & alerted & small).sum() / max((faulty & small).sum(), 1),
        "recall_large_pack": (faulty & alerted & ~small).sum() / max((faulty & ~small).sum(), 1),
        "median_lead_hours": float(lead.median()) if len(lead) else np.nan,
        "healthy_alerts_per_100h": 100 * a.loc[~faulty, "episodes"].sum() / healthy_hours,
    }


def load_scores() -> tuple[pd.DataFrame, dict]:
    """Out-of-fold window scores of both methods (120 h, 10 seeds) and thresholds per method."""
    import common
    data = common.PROJECT_ROOT / "figures" / "thesis" / "data"
    revision = common.DATA / "revision"
    stored = pd.read_pickle(revision / "ew_window_scores.pkl").query("source == '120h'").drop(columns=["s", "dF", "dH", "nearest_fault_prototype"])  # old ew_compare.py score must not shadow the documented s
    centroid = pd.read_pickle(data / "ew_centroid_scores_120h.pkl")
    scores = stored.merge(centroid, on=["seed", "ev_id", "window_start"], validate="one_to_one", suffixes=("", "_c"))
    assert (scores["fold"] == scores["fold_c"]).all()
    assert np.allclose(scores["s"], scores["d_healthy"] / (scores["d_fault"] + scores["d_healthy"])), "s is not the documented score"
    scores = scores.sort_values(["seed", "ev_id", "window_start"]).reset_index(drop=True)
    thresholds = pd.read_csv(revision / "ew_thresholds.csv").query("source == '120h' and method == 'isolation_forest'")
    return scores, {"isolation_forest": thresholds[["seed", "fold", "q", "tau"]], "centroid": pd.read_csv(data / "ew_centroid_thresholds_120h.csv")}
