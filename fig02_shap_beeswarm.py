"""Figure 2: SHAP beeswarm of the whole-session Isolation Forest, all 393 vehicles."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import common
import rwth_style as st

N_BINS, ROW_HALF_HEIGHT = 120, 0.42


def swarm_offsets(x: np.ndarray) -> np.ndarray:
    """Spread points of similar x vertically, centred on the row; position only."""
    bins = np.floor((x - x.min()) / max(np.ptp(x), 1e-12) * N_BINS).astype(int)
    y = np.zeros(len(x))
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        y[idx] = (np.arange(len(idx)) + 1) // 2 * (-1.0) ** np.arange(len(idx))
    return y


st.set_style()
d = common.load_whole_session()
features, shap_table = d["features"], d["shap"]
columns = list(features.columns)
order = shap_table[columns].abs().mean().sort_values().index  # strongest ends on top

fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 11.5 * st.CM))
norm = Normalize(0, 1)
peak = max(np.abs(swarm_offsets(shap_table[c].to_numpy())).max() for c in columns)
for row, c in enumerate(order):
    x = shap_table[c].to_numpy()
    low, high = np.nanpercentile(features[c], [5, 95])  # colour per feature, robust to outliers
    colour = np.clip((features[c].to_numpy() - low) / max(high - low, 1e-12), 0, 1)
    ax.scatter(x, row + swarm_offsets(x) / peak * ROW_HALF_HEIGHT, c=colour, cmap=st.CMAP_DIVERGING, norm=norm,
               s=4, edgecolors=st.RWTH["grey"], linewidths=0.08, zorder=3)
ax.axvline(0, color=st.RWTH["grey"], lw=0.7, zorder=1)
ax.set_yticks(range(len(order)), [st.FEATURE_LABELS[c] for c in order])
ax.set_ylim(-0.6, len(order) - 0.4)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True)
ax.set_xlabel("SHAP value, positive = more anomalous [splits of mean path length]", loc="right")
bar = fig.colorbar(plt.cm.ScalarMappable(norm, st.CMAP_DIVERGING), ax=ax, shrink=0.45, aspect=18, pad=0.015)
bar.set_ticks([0, 1], labels=["Low", "High"])
bar.set_label("Feature value")
bar.outline.set_linewidth(0.4)
print(st.save_figure(fig, "fig02_shap_beeswarm"))
print(shap_table[columns].abs().mean().sort_values(ascending=False).round(4).to_string())
