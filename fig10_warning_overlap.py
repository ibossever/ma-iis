"""Figure 10: which vehicles each method warns at the matched operating point (5 % of healthy
vehicles warned), 3-window rule. A vehicle counts as warned if at least half of the 10 seeds warn it."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st

OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"
table = pd.read_csv(OUT / "alerted_majority.csv", index_col=0)[["is_faulty_ev", "pack_type"]]  # vehicle list only
first = pd.read_csv(OUT / "matched5_first_alerts.csv")
seeds = first.groupby(["method", "ev_id"]).size().unstack(0).reindex(table.index).fillna(0).astype(int)
table["isolation_forest"] = seeds["isolation_forest"] >= 5
table["centroid"] = seeds["centroid"] >= 5
table["group"] = np.where(~table["is_faulty_ev"], "healthy", "faulty_" + table["pack_type"])
by_if, by_centroid = table["isolation_forest"].astype(bool), table["centroid"].astype(bool)
table["overlap"] = np.select([by_if & by_centroid, by_if & ~by_centroid, ~by_if & by_centroid], ["Both methods", "Isolation Forest only", "Centroid score only"], "Neither")
categories = ["Both methods", "Isolation Forest only", "Centroid score only", "Neither"]
colours = {"Both methods": st.RWTH["black"], "Isolation Forest only": st.METHOD["isolation_forest"], "Centroid score only": st.METHOD["centroid"], "Neither": st.RWTH["black25"]}
text_colour = {"Both methods": "white", "Isolation Forest only": "white", "Centroid score only": "black", "Neither": "black"}
groups = ["faulty_small", "faulty_large", "healthy"]
labels = {"faulty_small": "Faulty, small pack", "faulty_large": "Faulty, large pack", "healthy": "Healthy (large pack)"}
counts = pd.crosstab(table["group"], table["overlap"]).reindex(index=groups, columns=categories, fill_value=0)
shares = 100 * counts.div(counts.sum(axis=1), axis=0)
counts.to_csv(OUT / "fig10_overlap.csv")

st.set_style()
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 5.8 * st.CM))
left = np.zeros(len(groups))
positions = np.arange(len(groups))[::-1]
for category in categories:
    ax.barh(positions, shares[category], left=left, height=0.62, color=colours[category], edgecolor="white", label=category)
    for y, share, count, start in zip(positions, shares[category], counts[category], left):
        if count and share >= 2:
            ax.text(start + share / 2, y, str(count), ha="center", va="center", fontsize=st.LEGEND_PT, color=text_colour[category])
    left += shares[category].to_numpy()
ax.set_yticks(positions, [f"{labels[g]} (n = {counts.loc[g].sum()})" for g in groups])
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of vehicles [%] (numbers: vehicles)")
ax.xaxis.grid(True)
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=4, columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
print(st.save_figure(fig, "fig10_warning_overlap"))
print(counts.to_string())
print("\nseeds warned per vehicle, share of vehicles with a split vote (1-9 of 10):")
print(((seeds > 0) & (seeds < 10)).groupby(table["group"]).sum().to_string())
print("\nfaulty large warned:", table[(table["group"] == "faulty_large") & (by_if | by_centroid)].index.tolist())
print("faulty small missed by both:", table[(table["group"] == "faulty_small") & ~by_if & ~by_centroid].index.tolist())
print("seeds for those:", seeds.loc[table[(table["group"] == "faulty_small") & ~by_if & ~by_centroid].index].T.to_dict())
