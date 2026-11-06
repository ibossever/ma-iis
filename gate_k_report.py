"""Gate 1 report: appendix figures and tables from the k sweep. Describes clusters
with labels afterwards; ARI between populations is a consistency check only."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import common
import rwth_style as st
from common import cluster
from gate_k_windowed import K_RANGE, MIN_CLUSTER_SHARE, MIN_STABILITY, OUT_DIR, apply_rule

CANDIDATES = {"120h": [3, 7, 9], "24h": [3, 5, 10]}
SOURCE_LABEL = {"120h": "120 h windows", "24h": "24 h windows"}
Z_LIMIT = 3.0


def diagnostics_figure(tables: dict) -> None:
    panels = [("silhouette", "Silhouette [–]", None), ("stability", "Stability, mean ARI [–]", MIN_STABILITY),
              ("smallest_share", "Smallest cluster [%]", MIN_CLUSTER_SHARE * 100)]
    fig, axes = plt.subplots(2, 3, figsize=(st.FULL_WIDTH, 10.0 * st.CM), sharex=True)
    for row, source in enumerate(tables):
        table = tables[source]
        for col, (column, label, limit) in enumerate(panels):
            ax = axes[row, col]
            scale = 100 if column == "smallest_share" else 1
            ax.plot(table["k"], table[column] * scale, color=st.RWTH["blue"], marker="o")
            chosen = table[table["k"].isin(CANDIDATES[source])]
            ax.scatter(chosen["k"], chosen[column] * scale, s=34, facecolors="none", edgecolors=st.RWTH["red"],
                       linewidths=0.9, zorder=4)
            if limit is not None:
                ax.axhline(limit, color="black", lw=0.8, ls="--")
            if column == "smallest_share":
                ax.set_yscale("log")
            ax.set_ylabel(label)
            ax.yaxis.grid(True)
            ax.set_xticks([2, 5, 10, 15])
            if row == 1:
                ax.set_xlabel(f"k [–], {SOURCE_LABEL[source]}" if col == 1 else "k [–]")
            elif col == 1:
                ax.set_xlabel(f"k [–], {SOURCE_LABEL[source]}")
                ax.tick_params(labelbottom=True)
            st.panel_label(ax, "abcdef"[row * 3 + col])
    st.save_figure(fig, "figA1_k_diagnostics")


def profile_figure(source: str, x: np.ndarray, columns: list[str], labelings: dict) -> None:
    ks = CANDIDATES[source]
    height = (12.0 if len(columns) > 10 else 7.0) * st.CM
    fig, axes = plt.subplots(1, len(ks), figsize=(st.FULL_WIDTH, height), sharey=True, width_ratios=ks)
    for ax, k in zip(axes, ks):
        centroids = np.array([x[labelings[k] == c].mean(axis=0) for c in range(1, k + 1)]).T
        image = ax.imshow(centroids, cmap=st.CMAP_DIVERGING, vmin=-Z_LIMIT, vmax=Z_LIMIT, aspect="auto")
        ax.set_xticks(range(k), range(1, k + 1))
        ax.set_xlabel(f"k = {k}")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_yticks(range(len(columns)), [st.WINDOWED_FEATURE_LABELS[c] for c in columns])
    fig.supxlabel("Cluster number, sorted by size", fontsize=st.AXIS_LABEL_PT)
    bar = fig.colorbar(image, ax=axes, shrink=0.7, aspect=22, pad=0.015)
    bar.set_label("Standardised centroid value [–]")
    bar.outline.set_linewidth(0.4)
    st.save_figure(fig, f"figA2_cluster_profiles_{source}")


def composition(windows: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    group = np.where(~windows["is_faulty_ev"], "healthy", "faulty_" + windows["pack_type"])
    return pd.crosstab(labels, group, normalize="index").reindex(columns=list(st.GROUP), fill_value=0)


def k2_figure(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 5.5 * st.CM), sharey=True)
    for ax, source, letter in zip(axes, results, "ab"):
        windows, labels = results[source]
        table, counts, bottom = composition(windows, labels), np.bincount(labels)[1:], np.zeros(2)
        for group in table.columns:
            ax.bar(range(2), table[group] * 100, bottom=bottom, width=0.62, color=st.GROUP[group]["color"],
                   edgecolor="white", label=st.GROUP[group]["label"])
            bottom += table[group].to_numpy() * 100
        ax.set_xticks(range(2), [f"Cluster {c + 1}\n(n = {counts[c]})" for c in range(2)])
        ax.set_xlabel(SOURCE_LABEL[source])
        st.panel_label(ax, letter)
    axes[0].set_ylabel("Share of windows [%]")
    axes[0].set_ylim(0, 100)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    st.save_figure(fig, "figA3_k2_composition")


def ari_figure(rows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL_WIDTH, 6.0 * st.CM), sharey=True)
    for ax, (source, part), letter in zip(axes, rows.groupby("source", sort=False), "ab"):
        for i, (k_all, line) in enumerate(part.groupby("k_all")):
            ax.plot(line["k_faulty"], line["ari"], color=st.CLUSTER_COLORS[i], marker="ospD"[i], ms=3,
                    label=f"All vehicles, k = {k_all}")
        ax.set_xlabel(f"k of faulty-only clustering [–], {SOURCE_LABEL[source]}")
        ax.set_xticks([2, 5, 10, 15])
        ax.yaxis.grid(True)
        ax.legend(loc="upper right", handlelength=1.0, labelspacing=0.2)
        st.panel_label(ax, letter)
    axes[0].set_ylabel("ARI on faulty windows [–]")
    axes[0].set_ylim(bottom=0)
    st.save_figure(fig, "figA4_ari_consistency")


st.set_style()
tables, k2, ari_rows = {}, {}, []
for source, candidates in CANDIDATES.items():
    windows, columns = common.load_windows(source)
    x = StandardScaler().fit_transform(windows[columns])
    tables[source] = pd.read_csv(OUT_DIR / f"sweep_{source}.csv")
    print(f"\n=== {source}: rule -> {apply_rule(tables[source])}")
    labelings = {k: cluster(x, k) for k in candidates}
    k2[source] = (windows, cluster(x, 2))
    profile_figure(source, x, columns, labelings)

    summary = []
    for k, labels in {2: k2[source][1], **labelings}.items():
        for c in range(1, k + 1):
            part = windows[labels == c]
            summary.append({"k": k, "cluster": c, "windows": len(part), "share_pct": 100 * len(part) / len(windows),
                            "vehicles": part["ev_id"].nunique(), "top_vehicle_pct": 100 * part["ev_id"].value_counts().iloc[0] / len(part),
                            "faulty_pct": 100 * part["is_faulty_ev"].mean(), "small_pack_pct": 100 * (part["pack_type"] == "small").mean()})
    summary = pd.DataFrame(summary).round(1)
    summary.to_csv(OUT_DIR / f"clusters_{source}.csv", index=False)
    print(summary.to_string(index=False))

    faulty = windows["is_faulty_ev"].to_numpy()
    x_faulty = StandardScaler().fit_transform(windows.loc[faulty, columns])  # own scaler, as in aris.py
    faulty_labels = {k: cluster(x_faulty, k) for k in K_RANGE}
    ari_rows += [{"source": source, "k_all": ka, "k_faulty": kf, "ari": adjusted_rand_score(labelings[ka][faulty], lf)}
                 for ka in candidates for kf, lf in faulty_labels.items()]

ari_table = pd.DataFrame(ari_rows)
ari_table.to_csv(OUT_DIR / "ari_faulty_vs_all.csv", index=False)
print("\nARI at equal k:", ari_table[ari_table["k_all"] == ari_table["k_faulty"]].round(3).to_string(index=False))
print("\nARI range per candidate:\n", ari_table.groupby(["source", "k_all"], sort=False)["ari"].agg(["min", "max"]).round(3).to_string())
diagnostics_figure(tables)
k2_figure(k2)
ari_figure(ari_table)
