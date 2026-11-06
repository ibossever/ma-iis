"""Figure 5c: how the faulty-only clusters (k = 13) map onto the all-vehicle clusters
(k = 9), on windows of faulty vehicles. Rows sum to 100 %."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

import common
import rwth_style as st
from gate_k_windowed import OUT_DIR

PRINT_MIN_PCT = 5

table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
labels = pd.read_csv(common.PROJECT_ROOT / "figures" / "thesis" / "data" / "windows_120h_faulty_k13.csv")
table = table.merge(labels, on=["ev_id", "window_start"], validate="one_to_one")
assert len(table) == 1156
shares = 100 * pd.crosstab(table["cluster_faulty_k13"], table["cluster"], normalize="index").reindex(columns=range(1, 10), fill_value=0)
ari = adjusted_rand_score(table["cluster"], table["cluster_faulty_k13"])

# diagonal order: columns by the faulty windows they hold, rows by their strongest column
column_order = table["cluster"].value_counts().index.tolist()
ordered = shares[column_order]
position = {c: i for i, c in enumerate(column_order)}
row_order = sorted(ordered.index, key=lambda r: (position[ordered.loc[r].idxmax()], -ordered.loc[r].max()))
ordered = ordered.loc[row_order]
ordered.round(1).to_csv(OUT_DIR / "fig05c_correspondence.csv")

st.set_style()
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 9.5 * st.CM))
image = ax.imshow(ordered.to_numpy(), cmap=st.CMAP_SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
for (r, c), value in np.ndenumerate(ordered.to_numpy()):
    if value >= PRINT_MIN_PCT:
        ax.text(c, r, f"{value:.0f}", ha="center", va="center", fontsize=st.LEGEND_PT,
                color="white" if value > 55 else "black")
ax.set_xticks(range(ordered.shape[1]), ordered.columns)
ax.set_yticks(range(ordered.shape[0]), ordered.index)
ax.set_xlabel(f"All-vehicle cluster (k = 9); ARI = {ari:.2f}")
ax.set_ylabel("Faulty-only cluster (k = 13)")
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
bar = fig.colorbar(image, ax=ax, aspect=24, pad=0.015)
bar.set_label("Share of the faulty-only cluster's windows [%]")
bar.outline.set_linewidth(0.4)
print(st.save_figure(fig, "fig05c_cluster_correspondence"))
print("ARI", round(ari, 3), "rows sum to 100:", np.allclose(ordered.sum(axis=1), 100))
print(ordered.round(0).astype(int).to_string())
