"""Figure 5a: per cluster, share of windows from healthy and faulty vehicles, sorted;
clusters dominated by windows of faulty EVs (> 50 %) marked."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

import common
import rwth_style as st
from gate_k_windowed import OUT_DIR

K_FAULTY, DOMINANCE_PCT = 13, 50

table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
windows, columns = common.load_windows("120h")
assert (windows["ev_id"].to_numpy() == table["ev_id"].to_numpy()).all()
table["group"] = np.where(~table["is_faulty_ev"], "healthy", "faulty_" + table["pack_type"])
shares = 100 * pd.crosstab(table["cluster"], table["group"], normalize="index").reindex(columns=list(st.GROUP), fill_value=0)
shares["faulty"] = shares["faulty_large"] + shares["faulty_small"]
shares["windows"] = table.groupby("cluster").size()
shares["vehicles"] = table.groupby("cluster")["ev_id"].nunique()
shares = shares.sort_values("faulty", ascending=False)
shares.round(1).to_csv(OUT_DIR / "fig05a_cluster_shares.csv")

labels = pd.read_csv(common.PROJECT_ROOT / "figures" / "thesis" / "data" / "windows_120h_faulty_k13.csv")
ari = adjusted_rand_score(table.loc[windows["is_faulty_ev"].to_numpy(), "cluster"], labels["cluster_faulty_k13"])

st.set_style()
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 7.0 * st.CM))
positions, bottom = np.arange(len(shares)), np.zeros(len(shares))
for group in st.GROUP:
    ax.bar(positions, shares[group], bottom=bottom, width=0.68, color=st.GROUP[group]["color"], edgecolor="white",
           label=st.GROUP[group]["label"])
    bottom += shares[group].to_numpy()
dominated = shares["faulty"] > DOMINANCE_PCT
ax.scatter(positions[dominated], np.full(dominated.sum(), 105), marker="v", s=18, color="black", clip_on=False,
           label=f"Dominated by windows of faulty EVs (> {DOMINANCE_PCT} %)")
ax.set_xticks(positions, [f"{c}\n{int(n)}" for c, n in zip(shares.index, shares["windows"])])
ax.set_xlabel("Cluster number (top) and number of windows (bottom)")
ax.set_ylabel("Share of windows [%]")
ax.set_ylim(0, 100)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=4, columnspacing=0.8, handletextpad=0.4)
print(st.save_figure(fig, "fig05a_cluster_composition_all_k9"))
print(shares.round(1).to_string())
print(f"ARI faulty k={K_FAULTY} vs all k=9 on faulty windows: {ari:.3f}")
