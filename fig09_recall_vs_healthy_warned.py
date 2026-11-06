"""Figure 9: recall vs share of healthy vehicles warned over the budget sweep, 3-window
rule. (a) both methods, (b) by pack type with the chance line. Also writes the comparison
table as LaTeX (booktabs) and the appendix figure of metrics against budget b."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st

M, HEADLINE_Q = 3, 0.95
METHODS = ("isolation_forest", "centroid")
OUT = common.PROJECT_ROOT / "figures" / "thesis"
GATE = OUT / "gate3"
OFFSET = {"isolation_forest": (5, -9), "centroid": (-4, 6)}
X_MAX = 35  # both curves end below 30 % of healthy vehicles warned

sweep = pd.read_csv(GATE / "sweep_120h.csv")
sweep = sweep[sweep["m"] == M].assign(b=lambda d: (100 * (1 - d["q"])).round(1))
values = ["healthy_alert_rate", "recall", "recall_small_pack", "recall_large_pack", "precision", "healthy_alerts_per_100h"]
mean = sweep.groupby(["method", "b"])[values].mean() * [100, 100, 100, 100, 100, 1]
std = sweep.groupby(["method", "b"])[values].std() * [100, 100, 100, 100, 100, 1]


def budget_text(b: float) -> str:
    return f"{b:g}"


st.set_style()
fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 8.6 * st.CM), sharey=True)
for method in METHODS:
    colour, marker = st.METHOD[method], st.METHOD_MARKER[method]
    x, y, spread = mean.loc[method, "healthy_alert_rate"], mean.loc[method, "recall"], std.loc[method, "recall"]
    axes[0].fill_between(x, y - spread, y + spread, color=colour, alpha=0.18, linewidth=0)
    axes[0].plot(x, y, color=colour, marker=marker)
    for b in x.index:
        axes[0].annotate(budget_text(b), (x[b], y[b]), xytext=OFFSET[method], textcoords="offset points", fontsize=6.5, color=colour, fontweight="bold")
    axes[0].scatter(x[5.0], y[5.0], s=46, facecolors="none", edgecolors="black", linewidths=0.9, zorder=5)
    for column, style in (("recall_small_pack", "-"), ("recall_large_pack", "--")):
        y2, spread2 = mean.loc[method, column], std.loc[method, column]
        axes[1].fill_between(x, y2 - spread2, y2 + spread2, color=colour, alpha=0.13, linewidth=0)
        axes[1].plot(x, y2, color=colour, marker=marker, ms=2.5, ls=style)
        axes[1].scatter(x[5.0], y2[5.0], s=38, facecolors="none", edgecolors="black", linewidths=0.8, zorder=5, clip_on=False)
axes[1].plot([0, 100], [0, 100], color=st.RWTH["grey"], lw=0.9, zorder=1)
for x_text, y_text, text, method in ((19, 93, "small pack", "centroid"), (19, 71, "small pack", "isolation_forest"),
                                     (25, 5, "large pack", "centroid"), (12, 33, "large pack", "isolation_forest")):
    axes[1].text(x_text, y_text, text, color=st.METHOD[method], fontsize=st.LEGEND_PT, va="center")
for ax, letter in zip(axes, "ab"):
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Healthy vehicles warned [%]")
    ax.grid(True)
    st.panel_label(ax, letter)
axes[0].set_ylabel("Faulty vehicles warned (recall) [%]")
handles = [plt.Line2D([], [], color=st.METHOD[m], marker=st.METHOD_MARKER[m], label=st.METHOD_LABEL[m]) for m in METHODS]
handles += [plt.Line2D([], [], color="black", ls="-", label="Small pack"), plt.Line2D([], [], color="black", ls="--", label="Large pack"),
            plt.Line2D([], [], color=st.RWTH["grey"], label="Chance"),
            plt.Line2D([], [], ls="", marker="o", ms=5.5, mfc="none", mec="black", label="b = 5 %")]
fig.legend(handles=handles, loc="outside upper center", ncol=6, columnspacing=0.9, handlelength=1.5, handletextpad=0.4)
print(st.save_figure(fig, "fig09_recall_vs_healthy_warned"))

# appendix: metrics against budget
panels = [("recall", "Recall [%]"), ("precision", "Precision [%]"), ("healthy_alert_rate", "Healthy vehicles warned [%]"),
          ("healthy_alerts_per_100h", "Alerts per vehicle per 100 h [–]")]
fig, axes = plt.subplots(2, 2, figsize=(st.FULL_WIDTH, 11.0 * st.CM), sharex=True)
for ax, (column, label), letter in zip(axes.ravel(), panels, "abcd"):
    for method in METHODS:
        m, s = mean.loc[method, column], std.loc[method, column]
        ax.fill_between(m.index, m - s, m + s, color=st.METHOD[method], alpha=0.18, linewidth=0)
        ax.plot(m.index, m, color=st.METHOD[method], marker=st.METHOD_MARKER[method])
    ax.axvline(5, color="black", lw=0.7, ls=":")
    ax.set_ylabel(label)
    ax.yaxis.grid(True)
    st.panel_label(ax, letter)
for ax in axes[1]:
    ax.set_xlabel("Budget b [%]")
fig.legend(handles=handles[:2], loc="outside upper center", ncol=2)
st.save_figure(fig, "figA_metrics_vs_budget")

# LaTeX table: matched comparison and operating point
matched = pd.read_csv(GATE / "matched_comparison.csv", header=[0, 1], index_col=[0, 1])
main = sweep[np.isclose(sweep["q"], HEADLINE_Q)].groupby("method")
pm = lambda frame, method, column, scale=100, digits=1: f"{frame.loc[method, (column, 'mean')] * scale:.{digits}f} $\\pm$ {frame.loc[method, (column, 'std')] * scale:.{digits}f}"
summary = main[["recall", "precision", "f1", "balanced_accuracy", "healthy_alert_rate", "healthy_alerts_per_100h", "TP", "FP", "FN", "TN",
                "TP_small_pack", "TP_large_pack", "median_lead_hours"]].agg(["mean", "std"])
names = {"isolation_forest": "Isolation Forest", "centroid": "Centroid score"}
lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Metric & Isolation Forest & Centroid score \\", r"\midrule",
         r"\multicolumn{3}{l}{\textit{Matched: recall [\%] at equal share of healthy vehicles warned}} \\"]
actual = pd.read_csv(GATE / "matched5_first_alerts.csv").query("is_faulty_ev").groupby(["method", "seed"]).size().mul(100 / 55)
cells = [f"{actual[m].mean():.1f} $\\pm$ {actual[m].std():.1f}" for m in METHODS]
lines.append(f"Recall at 5\\% healthy warned & {cells[0]} & {cells[1]} \\\\")
cells = [f"{matched.loc[(0.1, m), ('recall', 'mean')] * 100:.1f} $\\pm$ {matched.loc[(0.1, m), ('recall', 'std')] * 100:.1f}" for m in METHODS]
lines.append(f"Recall at 10\\% healthy warned (interpolated) & {cells[0]} & {cells[1]} \\\\")
print("actual matched recall:", {m: (round(actual[m].mean(), 1), round(actual[m].std(), 1)) for m in METHODS})
lines += [r"\midrule", r"\multicolumn{3}{l}{\textit{Operating point: $b = 5\,\%$, three consecutive windows}} \\"]
rows = [("TP", "TP", 1, 1), ("FP", "FP", 1, 1), ("FN", "FN", 1, 1), ("TN", "TN", 1, 1), ("Recall [\\%]", "recall", 100, 1), ("Precision [\\%]", "precision", 100, 1),
        ("F1 [\\%]", "f1", 100, 1), ("Balanced accuracy [\\%]", "balanced_accuracy", 100, 1), ("Healthy vehicles warned [\\%]", "healthy_alert_rate", 100, 1),
        ("Alerts per vehicle per 100\\,h", "healthy_alerts_per_100h", 1, 3), ("TP small pack (of 38)", "TP_small_pack", 1, 1), ("TP large pack (of 17)", "TP_large_pack", 1, 1),
        ("First alert before end of record [h]", "median_lead_hours", 1, 0)]
for label, column, scale, digits in rows:
    lines.append(f"{label} & {pm(summary, 'isolation_forest', column, scale, digits)} & {pm(summary, 'centroid', column, scale, digits)} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(OUT / "fig09_comparison_table.tex").write_text("\n".join(lines) + "\n")
print(pd.concat({"mean": mean, "std": std}, axis=1).round(1).loc[:, [("mean", "healthy_alert_rate"), ("std", "healthy_alert_rate"), ("mean", "recall"), ("std", "recall"), ("mean", "recall_small_pack"), ("mean", "recall_large_pack")]].to_string())
