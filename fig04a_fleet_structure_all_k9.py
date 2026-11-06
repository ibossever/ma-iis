"""Figure 4a: all 120 h windows in PCA and UMAP, k-means clusters (k = 9 on all
windows, decisions.json), windows of faulty vehicles outlined. Also stores the map
for Figures 5 and 6 in figures/thesis/gate1/."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import common
import rwth_style as st
from fleet_map import draw_labels
from gate_k_windowed import OUT_DIR

K_ALL = 9
UMAP_N_NEIGHBORS, UMAP_MIN_DIST = 15, 0.1  # same as wisek_window.py

windows, columns = common.load_windows("120h")
x = StandardScaler().fit_transform(windows[columns])
labels = common.cluster(x, K_ALL)
pca = PCA(n_components=2, random_state=common.SEED).fit(x)
pcs = pca.transform(x)
maps = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2,
                 random_state=common.SEED).fit_transform(x)
keep = ["ev_id", "window_start", "recorded_hours", "is_faulty_ev", "pack_type"]
table = windows[keep].assign(cluster=labels, pc1=pcs[:, 0], pc2=pcs[:, 1], umap1=maps[:, 0], umap2=maps[:, 1])
table.to_csv(OUT_DIR / "windows_120h_k9.csv", index=False)


st.set_style()
fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 8.0 * st.CM))
variance = 100 * pca.explained_variance_ratio_
panels = [("pc1", "pc2", f"PC 1 ({variance[0]:.0f} % of variance) [–]", f"PC 2 ({variance[1]:.0f} % of variance) [–]"),
          ("umap1", "umap2", "UMAP 1 [–]", "UMAP 2 [–]")]
faulty = table["is_faulty_ev"].to_numpy()
for ax, (cx, cy, xl, yl), letter in zip(axes, panels, "ab"):
    colours = np.array(st.CLUSTER_COLORS)[table["cluster"].to_numpy() - 1]
    ax.scatter(table.loc[~faulty, cx], table.loc[~faulty, cy], c=colours[~faulty], s=3, linewidths=0, alpha=0.7)
    ax.scatter(table.loc[faulty, cx], table.loc[faulty, cy], c=colours[faulty], s=5, edgecolors="black", linewidths=0.25)
    draw_labels(ax, table, cx, cy)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    st.panel_label(ax, letter)
handles = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=st.RWTH["blue50"], mec="none", label="Windows of healthy vehicles"),
           plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc=st.RWTH["blue50"], mec="black", mew=0.4, label="Windows of faulty vehicles (outlined)")]
fig.legend(handles=handles, loc="outside upper center", ncol=2)
print(st.save_figure(fig, "fig04a_fleet_structure_all_k9"))
print("variance", variance.round(1), "windows faulty", faulty.sum())
