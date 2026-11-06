"""Figure 6: paths of three vehicles through the UMAP map of Figure 4a (k = 9).
Top row: colour is recorded time. Bottom row: same paths, colour is the
k = 9 cluster (marker shape as well), so cluster changes are visible. Circle = first, square = last window.
Second figure (fig06b): the same vehicles' cluster per window, window number on x, cluster on y."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

import rwth_style as st
from gate_k_windowed import OUT_DIR

VEHICLES = [(389, "Detected faulty"), (386, "Missed faulty"), (334, "Healthy")]
CLUSTER_MARKER = {1: "o", 2: "^", 3: "s", 4: "v", 5: "D", 8: "P", 9: "X"}  # clusters these three vehicles visit
TIME_CMAP = LinearSegmentedColormap.from_list("time", [st.RWTH["blue25"], st.RWTH["blue"]])  # white start would vanish
FAULTY_DOMINATED_MIN_PCT = 50
STRIDE_HOURS = 12
EV_ROW_OFFSET = (0.27, 0, -0.27)  # same order as VEHICLES
EV_LINE_STYLE = ("-", "--", ":")

table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
faulty_share = 100 * table.groupby("cluster")["is_faulty_ev"].mean()
faulty_dominated = faulty_share[faulty_share > FAULTY_DOMINATED_MIN_PCT].index

st.set_style()
fig, axes = plt.subplots(2, 3, figsize=(st.FULL_WIDTH, 13.5 * st.CM), sharex=True, sharey=True)
rows = []


def draw_path(ax, path: pd.DataFrame, **point_style):
    ax.scatter(table["umap1"], table["umap2"], color=st.RWTH["black25"], s=1.5, linewidths=0, alpha=0.5, zorder=1)
    ax.plot(path["umap1"], path["umap2"], color=st.RWTH["grey"], lw=0.5, zorder=2)
    points = None
    if point_style:
        points = ax.scatter(path["umap1"], path["umap2"], s=11, edgecolors="black", linewidths=0.2, zorder=3, **point_style)
    else:
        for cluster, part in path.groupby("cluster"):
            ax.scatter(part["umap1"], part["umap2"], s=13, marker=CLUSTER_MARKER[cluster], color=palette[cluster - 1],
                       edgecolors="black", linewidths=0.2, zorder=3)
    for position, marker in ((0, "o"), (-1, "s")):
        ax.scatter(path["umap1"].iloc[position], path["umap2"].iloc[position], marker=marker, s=30, facecolors="none",
                   edgecolors="black", linewidths=0.9, zorder=4)
    return points


paths = {ev: table[table["ev_id"] == ev].sort_values("window_start") for ev, _ in VEHICLES}
palette = np.array(st.CLUSTER_COLORS)
for column, (ev, role) in enumerate(VEHICLES):
    path = paths[ev]
    hours = path["recorded_hours"].to_numpy()
    points = draw_path(axes[0, column], path, c=hours, cmap=TIME_CMAP, norm=Normalize(hours.min(), hours.max()))
    draw_path(axes[1, column], path)
    bar = fig.colorbar(points, ax=axes[0, column], orientation="horizontal", aspect=22, pad=0.02, shrink=0.9)
    bar.set_label(f"Recorded time [h]\n{role}, EV {ev}", fontsize=st.LEGEND_PT)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=st.TICK_PT)
    rows.append({"ev": ev, "role": role, "pack": path["pack_type"].iloc[0], "windows": len(path), "clusters_visited": path["cluster"].nunique(),
                 "changes": int(path["cluster"].ne(path["cluster"].shift()).sum() - 1),
                 "cluster_sequence": "-".join(map(str, path["cluster"].loc[path["cluster"].ne(path["cluster"].shift())])),
                 "faulty_dominated_pct": round(100 * path["cluster"].isin(faulty_dominated).mean(), 1),
                 "last_hours": hours.max()})
for ax in axes[1]:
    ax.set_xlabel("UMAP 1 [\u2013]")
for ax, letter in zip(axes.ravel(), "abcdef"):
    st.panel_label(ax, letter)
axes[0, 0].set_ylabel("UMAP 2 [\u2013]")
axes[1, 0].set_ylabel("UMAP 2 [\u2013]")
visited = sorted({c for path in paths.values() for c in path["cluster"]})
handles = [plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc="none", mec="black", label="First window"),
           plt.Line2D([], [], ls="", marker="s", ms=4.5, mfc="none", mec="black", label="Last window"),
           plt.Line2D([], [], ls="", marker="o", ms=2.5, mfc=st.RWTH["black25"], mec="none", label="All windows of the fleet")]
fig.legend(handles=handles, loc="outside upper center", ncol=3)
cluster_handles = [plt.Line2D([], [], ls="", marker=CLUSTER_MARKER[c], ms=3.8, mfc=palette[c - 1], mec="black", mew=0.2, label=f"Cluster {c}") for c in visited]
fig.legend(handles=cluster_handles, loc="outside lower center", ncol=len(visited), columnspacing=0.7, handletextpad=0.1)
print(st.save_figure(fig, "fig06_vehicle_trajectories"))

# Same vehicles, one axes: cluster per window. Each vehicle has its own row within every cluster lane.
lanes = {cluster: lane for lane, cluster in enumerate(visited)}
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 9.5 * st.CM))
for (ev, role), offset, line_style in zip(VEHICLES, EV_ROW_OFFSET, EV_LINE_STYLE):
    path = paths[ev]
    assert path["recorded_hours"].diff().dropna().eq(STRIDE_HOURS).all()  # no gaps, so row order is the window number
    window = np.arange(1, len(path) + 1)
    row = path["cluster"].map(lanes).to_numpy() + offset
    ax.step(window, row, where="mid", color=st.RWTH["grey"], lw=0.5, ls=line_style, zorder=2,
            label=f"{role}, EV {ev} ({len(path)} windows)")
    for cluster in path["cluster"].unique():
        here = path["cluster"].to_numpy() == cluster
        ax.scatter(window[here], row[here], s=9, marker=CLUSTER_MARKER[cluster], color=palette[cluster - 1],
                   edgecolors="black", linewidths=0.15, zorder=3)
for boundary in np.arange(len(visited) - 1) + 0.5:
    ax.axhline(boundary, color=st.RWTH["black25"], lw=0.5, zorder=1)
ax.set_yticks(list(lanes.values()), [f"Cluster {c}" for c in lanes])
for label, cluster in zip(ax.get_yticklabels(), lanes):
    label.set_color(palette[cluster - 1])
ax.set_ylim(-0.5, len(visited) - 0.5)
ax.set_xlim(0, max(len(path) for path in paths.values()) + 1)
ax.set_xlabel("Window number [–]")
ax.set_ylabel("Assigned cluster (k = 9)")
fig.legend(loc="outside upper center", ncol=3, columnspacing=1.0)
print(st.save_figure(fig, "fig06b_vehicle_cluster_sequence"))
print("faulty-dominated clusters", list(faulty_dominated))
print(pd.DataFrame(rows).to_string(index=False))
