"""Figure 11: median and IQR of both scores vs hours before end of record, by group.
Windows are aligned to each vehicle's last window. Scores are out-of-fold; each window's
score is the mean over the 10 seeds. Reference lines: mean matched threshold.
Second figure (fig11b): the same lines with one panel per group, both scores in each."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import alert
import common
import ew_common
import rwth_style as st

STRIDE_HOURS, MIN_VEHICLES, MAX_HOURS = 12.0, 5, 360.0
GROUPS = ["healthy", "faulty_large", "faulty_small"]
METHODS = {"isolation_forest": ("if_score", "Isolation Forest score [–]"), "centroid": ("s", "Centroid score s [–]")}
METHOD_NAMES = {"isolation_forest": "Isolation Forest score", "centroid": "Centroid score s"}
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"

scores, _ = alert.load_scores()
windows, _columns = common.load_windows("120h")
evs = ew_common.ev_table(windows)
evs["group"] = np.where(~evs["is_faulty_ev"], "healthy", "faulty_" + evs["pack_type"])
mean = scores.groupby(["ev_id", "window_start"])[["if_score", "s"]].mean().reset_index()
mean["hours_before_end"] = (mean.groupby("ev_id")["window_start"].transform("max") - mean["window_start"]) * common.SAMPLE_INTERVAL_S / 3600.0
mean["hours_before_end"] = (mean["hours_before_end"] / STRIDE_HOURS).round() * STRIDE_HOURS
mean["group"] = mean["ev_id"].map(evs["group"])
thresholds = pd.read_csv(OUT / "matched5_thresholds.csv").groupby("method")["tau"].agg(["mean", "std", "min", "max"])

st.set_style()
fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 7.6 * st.CM), sharex=True)
rows, lines = [], {}
for ax, (method, (column, ylabel)), letter in zip(axes, METHODS.items(), "ab"):
    for group in GROUPS:
        part = mean[(mean["group"] == group) & (mean["hours_before_end"] <= MAX_HOURS)]
        line = part.groupby("hours_before_end")[column].quantile([0.25, 0.5, 0.75]).unstack()
        count = part.groupby("hours_before_end")["ev_id"].nunique()
        line = line[count >= MIN_VEHICLES]
        lines[method, group] = line
        style = st.GROUP[group]
        ax.fill_between(line.index, line[0.25], line[0.75], color=style["color"], alpha=0.18, linewidth=0)
        ax.plot(line.index, line[0.5], color=style["color"], marker=style["marker"], ms=2.2, label=style["label"])
        rows += [{"method": method, "group": group, "hours": h, "median": line.loc[h, 0.5], "q25": line.loc[h, 0.25], "q75": line.loc[h, 0.75], "vehicles": int(count[h])} for h in line.index]
    ax.axhline(thresholds.loc[method, "mean"], color="black", lw=0.8, ls="--", label="Matched threshold (mean)")
    ax.set_xlim(MAX_HOURS, 0)
    ax.set_xlabel("Hours before end of record [h]")
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    st.panel_label(ax, letter)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, columnspacing=1.0, handlelength=1.3)
print(st.save_figure(fig, "fig11_scores_before_end"))

fig, axes = plt.subplots(1, 3, figsize=(st.FULL_WIDTH, 7.2 * st.CM), sharex=True, sharey=True)
group_sizes = evs["group"].value_counts()
for ax, group, letter in zip(axes, GROUPS, "abc"):
    for method, label in METHOD_NAMES.items():
        line, colour = lines[method, group], st.METHOD[method]
        ax.fill_between(line.index, line[0.25], line[0.75], color=colour, alpha=0.18, linewidth=0)
        ax.plot(line.index, line[0.5], color=colour, marker=st.METHOD_MARKER[method], ms=2.2, label=label)
        ax.axhline(thresholds.loc[method, "mean"], color=colour, lw=0.8, ls="--", label=f"Matched threshold (mean), {label}")
    ax.set_title(f"{st.GROUP[group]['label']} (n = {group_sizes[group]})", fontsize=st.LEGEND_PT, pad=6)
    ax.set_xlim(MAX_HOURS, 0)
    ax.set_xticks([300, 200, 100, 0])
    ax.yaxis.grid(True)
    st.panel_label(ax, letter)
axes[0].set_ylabel("Score [\u2013]")
fig.supxlabel("Hours before end of record [h]", fontsize=st.AXIS_LABEL_PT)
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, columnspacing=1.0, handlelength=1.3)
print(st.save_figure(fig, "fig11b_scores_before_end_by_group"))
report = pd.DataFrame(rows)
report.to_csv(OUT / "fig11_medians.csv", index=False)
print(thresholds.round(3).to_string())
for (method, group), part in report.groupby(["method", "group"], sort=False):
    pick = part.set_index("hours")
    values = {h: round(float(pick.loc[h, "median"]), 3) for h in (240.0, 120.0, 0.0) if h in pick.index}
    trend = np.corrcoef(-pick.index.to_numpy(), pick["median"])[0, 1]
    print(f"{method:17s} {group:13s} median at 240/120/0 h: {values} | trend corr with time {trend:+.2f} | vehicles at 0/240 h: {int(pick.loc[0.0, 'vehicles'])}/{int(pick.loc[240.0, 'vehicles']) if 240.0 in pick.index else 0}")

# same vehicles at both ends: change from 240 h before the end to the end of record
from scipy.stats import wilcoxon
for method, (column, _) in METHODS.items():
    wide = mean[mean["hours_before_end"].isin([0.0, 240.0])].pivot(index="ev_id", columns="hours_before_end", values=column).dropna()
    wide["group"] = evs["group"].reindex(wide.index)
    for group in GROUPS:
        part = wide[wide["group"] == group]
        if len(part) >= 6:
            change = part[0.0] - part[240.0]
            print(f"fixed cohort {method:17s} {group:13s} n = {len(part):3d}, median change {change.median():+.3f}, rising {int((change > 0).sum())}, p = {wilcoxon(change).pvalue:.4f}")
