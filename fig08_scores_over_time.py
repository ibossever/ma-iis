"""Figure 8: Isolation Forest score and centroid score s over time for the three vehicles
of Figure 6, out-of-fold, one seed. Thresholds at b = 5 %, alert runs shaded."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alert
import common
import rwth_style as st

SEED, Q, M, WINDOW_HOURS = 0, 0.95, 3, 120.0
VEHICLES = [(389, "Detected faulty", "small pack"), (386, "Missed faulty", "large pack"), (334, "Healthy", "large pack")]
METHODS = {"isolation_forest": ("if_score", "Isolation Forest score [–]"), "centroid": ("s", "Centroid score s [–]")}
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"

scores, taus = alert.load_scores()
first = pd.read_csv(OUT / "first_alerts_b5_m3.csv")
scores = scores[scores["seed"] == SEED]

st.set_style()
fig, axes = plt.subplots(2, 3, figsize=(st.FULL_WIDTH, 9.0 * st.CM), sharex="col", sharey="row")
rows = []
for column, (ev, role, pack) in enumerate(VEHICLES):
    part = scores[scores["ev_id"] == ev].reset_index(drop=True)
    hours = part["window_start"].to_numpy() * common.SAMPLE_INTERVAL_S / 3600.0 + WINDOW_HOURS  # score is known at window end
    for row, (method, (score, ylabel)) in enumerate(METHODS.items()):
        ax, colour = axes[row, column], st.METHOD[method]
        table = taus[method]
        tau = table[(table["seed"] == SEED) & (table["fold"] == part["fold"].iloc[0]) & (np.isclose(table["q"], Q))]["tau"].iloc[0]
        flagged = part[score].to_numpy() > tau
        indices = np.flatnonzero(flagged)
        spans = [g for g in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(g) >= M]
        for group in spans:
            ax.axvspan(hours[group[0]] - 6, hours[group[-1]] + 6, color=colour, alpha=0.16, linewidth=0, zorder=0)
            ax.scatter(hours[group[M - 1]], 1.03, transform=ax.get_xaxis_transform(), marker="v", s=9, color="black", clip_on=False, zorder=5)
        ax.plot(hours, part[score], color=colour, lw=0.9, zorder=2)
        ax.scatter(hours[flagged], part[score].to_numpy()[flagged], s=7, color=colour, edgecolors="black", linewidths=0.25, zorder=3)
        ax.axhline(tau, color="black", lw=0.8, ls="--", zorder=1)
        ax.yaxis.grid(True)
        if row == 0:
            ax.set_title(f"{role}, EV {ev} ({pack})", fontsize=7.5, pad=9, loc="right")
        else:
            ax.set_xlabel("Recorded time [h]")
        if column == 0:
            ax.set_ylabel(ylabel)
        st.panel_label(ax, "abcdef"[row * 3 + column])
        seeds = int(((first["method"] == method) & (first["ev_id"] == ev)).sum())
        rows.append({"ev": ev, "method": method, "fold": int(part["fold"].iloc[0]), "tau": round(float(tau), 3), "windows": len(part), "flagged": int(flagged.sum()),
                     "alert_runs": len(spans), "first_alert_h": hours[spans[0][M - 1]] if spans else np.nan, "end_h": hours[-1], "seeds_warned_of_10": seeds})
handles = [plt.Line2D([], [], color="black", lw=0.8, ls="--", label="Threshold, b = 5 %"),
           plt.Line2D([], [], ls="", marker="o", ms=3, mfc=st.RWTH["grey"], mec="black", mew=0.25, label="Flagged window"),
           plt.Rectangle((0, 0), 1, 1, fc=st.RWTH["black25"], label="Alert: at least 3 consecutive flagged windows"),
           plt.Line2D([], [], ls="", marker="v", ms=3, color="black", label="Alert raised")]
fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=0.9, handlelength=1.2, handletextpad=0.4)
print(st.save_figure(fig, "fig08_scores_over_time"))
report = pd.DataFrame(rows)
print(report.to_string(index=False))
stored = first[first["seed"] == SEED].set_index(["method", "ev_id"])["first_alert_window_start"] * common.SAMPLE_INTERVAL_S / 3600.0 + WINDOW_HOURS
check = report.dropna(subset=["first_alert_h"]).set_index(["method", "ev"])["first_alert_h"]
print("first alerts match stored file:", all(abs(stored[key] - value) < 1e-6 for key, value in check.items()))
