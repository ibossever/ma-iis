"""Figure 4b: same PCA and UMAP coordinates as Figure 4a. Windows of faulty vehicles
coloured by the faulty-only clustering (k = 13), windows of healthy vehicles light grey."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import common
import rwth_style as st
from fleet_map import draw_labels
from gate_k_windowed import OUT_DIR

table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
labels = pd.read_csv(common.PROJECT_ROOT / "figures" / "thesis" / "data" / "windows_120h_faulty_k13.csv")
table = table.merge(labels, on=["ev_id", "window_start"], how="left", validate="one_to_one")
faulty = table["is_faulty_ev"].to_numpy()
assert table["cluster_faulty_k13"].notna().sum() == faulty.sum() == 1156
assert table.loc[faulty, "cluster_faulty_k13"].notna().all()

windows, columns = common.load_windows("120h")
variance = 100 * PCA(n_components=2, random_state=common.SEED).fit(StandardScaler().fit_transform(windows[columns])).explained_variance_ratio_

st.set_style()
fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 8.0 * st.CM))
panels = [("pc1", "pc2", f"PC 1 ({variance[0]:.0f} % of variance) [–]", f"PC 2 ({variance[1]:.0f} % of variance) [–]"),
          ("umap1", "umap2", "UMAP 1 [–]", "UMAP 2 [–]")]
coloured = table[faulty].rename(columns={"cluster": "cluster_all_k9", "cluster_faulty_k13": "cluster"})
order = pd.read_csv(OUT_DIR / "faulty_k13_clusters.csv")["name"].tolist()
colour_of = dict(zip(order, st.CLUSTER_COLORS_13))
for ax, (cx, cy, xl, yl), letter in zip(axes, panels, "ab"):
    ax.scatter(table.loc[~faulty, cx], table.loc[~faulty, cy], color=st.RWTH["black25"], s=3, linewidths=0, alpha=0.6)
    ax.scatter(coloured[cx], coloured[cy], c=coloured["cluster"].map(colour_of).tolist(), s=5, edgecolors="black", linewidths=0.25)
    draw_labels(ax, coloured, cx, cy)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    st.panel_label(ax, letter)
handles = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=st.RWTH["black25"], mec="none", label="Windows of healthy vehicles"),
           plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc=st.RWTH["black25"], mec="black", mew=0.4, label="Windows of faulty vehicles (outlined)")]
fig.legend(handles=handles, loc="outside upper center", ncol=2)
print(st.save_figure(fig, "fig04b_fleet_structure_faulty_k13"))
print(coloured["cluster"].value_counts().reindex(order).to_dict())
