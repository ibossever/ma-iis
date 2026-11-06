"""Figure 1: whole-session Isolation Forest score of every vehicle, by group."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st

JITTER_SEED, JITTER = 0, 0.28  # vertical jitter is for visibility only; x values are the data

st.set_style()
d = common.load_whole_session()
v, threshold = d["vehicles"], d["threshold"]
rng = np.random.default_rng(JITTER_SEED)

fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 6.0 * st.CM))
rows = ["faulty_small", "faulty_large", "healthy"]
for y, g in enumerate(rows):
    s = v.loc[v["group"] == g, "anomaly_score"]
    ax.scatter(s, y + rng.uniform(-JITTER, JITTER, len(s)), s=9, marker=st.GROUP[g]["marker"],
               color=st.GROUP[g]["color"], alpha=0.75, linewidths=0, zorder=3)
ax.axvline(threshold, color="black", lw=0.9, ls="--", zorder=2)
ax.annotate("Alert threshold", xy=(threshold, len(rows) - 0.42), xytext=(4, 0), textcoords="offset points",
            fontsize=st.LEGEND_PT, va="center")
ax.set_yticks(range(len(rows)), [f"{st.GROUP[g]['label']} (n = {(v['group'] == g).sum()})" for g in rows])
ax.set_ylim(-0.55, len(rows) - 0.25)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True)
ax.set_xlabel("Isolation Forest anomaly score [–]")
print(st.save_figure(fig, "fig01_score_distribution"))
print("threshold", round(threshold, 4))
print(v.groupby("group")["anomaly_score"].describe()[["count", "min", "50%", "max"]].round(3).to_string())
print(v.assign(above=v["anomaly_score"] > threshold).groupby("group")["above"].sum().to_string())
