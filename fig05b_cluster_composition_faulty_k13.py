"""Figure 5b: per faulty-only cluster (k = 13), share of windows by pack type. The
clustering holds faulty-vehicle windows only, so there is no healthy share to show."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st
from gate_k_windowed import OUT_DIR

table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
labels = pd.read_csv(common.PROJECT_ROOT / "figures" / "thesis" / "data" / "windows_120h_faulty_k13.csv")
table = table.merge(labels, on=["ev_id", "window_start"], validate="one_to_one")
assert len(table) == 1156 and table["is_faulty_ev"].all()
table["group"] = "faulty_" + table["pack_type"]
groups = ["faulty_large", "faulty_small"]
shares = 100 * pd.crosstab(table["cluster_faulty_k13"], table["group"], normalize="index").reindex(columns=groups, fill_value=0)
shares["windows"] = table.groupby("cluster_faulty_k13").size()
shares["vehicles"] = table.groupby("cluster_faulty_k13")["ev_id"].nunique()
shares = shares.sort_values(["faulty_small", "windows"], ascending=False)
shares.round(1).to_csv(OUT_DIR / "fig05b_cluster_shares.csv")

st.set_style()
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 7.0 * st.CM))
positions, bottom = np.arange(len(shares)), np.zeros(len(shares))
for group in groups:
    ax.bar(positions, shares[group], bottom=bottom, width=0.68, color=st.GROUP[group]["color"], edgecolor="white",
           label=st.GROUP[group]["label"])
    bottom += shares[group].to_numpy()
ax.set_xticks(positions, [f"{c}\n{int(n)}" for c, n in zip(shares.index, shares["windows"])])
ax.set_xlabel("Cluster number (top) and number of windows (bottom)")
ax.set_ylabel("Share of windows [%]")
ax.set_ylim(0, 100)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.0)
print(st.save_figure(fig, "fig05b_cluster_composition_faulty_k13"))
print(shares.round(1).to_string())
