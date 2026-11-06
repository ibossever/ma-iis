"""Shared cluster-number labels for the fleet maps (Figures 4a and 4b)."""
import numpy as np
import pandas as pd

import rwth_style as st

LABEL_MIN_GAP = 0.085
LABEL_NUDGE = np.array([0.045, 0.055])  # keeps a label off the points it names


def draw_labels(ax, data: pd.DataFrame, cx: str, cy: str) -> None:
    """Cluster number near each cluster centre, pushed apart in axes coordinates so
    labels do not overlap each other or hide small clusters; a thin line links them."""
    ax.margins(0.08)
    ax.figure.canvas.draw()
    to_axes = ax.transData + ax.transAxes.inverted()
    medians = data.groupby("cluster")[[cx, cy]].median()
    anchor = to_axes.transform(medians.to_numpy())
    place = anchor + LABEL_NUDGE
    for _ in range(300):
        for i in range(len(place)):
            for j in range(i + 1, len(place)):
                gap = place[j] - place[i]
                distance = np.hypot(*gap)
                if distance < LABEL_MIN_GAP:
                    push = (gap / max(distance, 1e-6)) * (LABEL_MIN_GAP - distance) / 2
                    place[i] -= push
                    place[j] += push
        place = np.clip(place, 0.05, 0.95)
    for c, a, b in zip(medians.index, anchor, place):
        ax.annotate(str(c), xy=a, xytext=b, xycoords="axes fraction", textcoords="axes fraction", ha="center",
                    va="center", fontsize=st.LEGEND_PT, fontweight="bold",
                    bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="black", lw=0.4),
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="black", shrinkA=0, shrinkB=0))
