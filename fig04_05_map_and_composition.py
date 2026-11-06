"""Combined figures for the windowed structure of the Results chapter. Each holds PCA (a),
UMAP (b) and the cluster composition (c) in one plot:

  fig04_all_k9_map_composition: all 6,209 windows, k = 9. Coordinates and labels come
      from gate1/windows_120h_k9.csv, so panels (a, b) equal Figure 4a and the UMAP of
      Figure 6, and panel (c) equals Figure 5a.
  fig05_faulty_k13_map_composition: the 1,156 windows of faulty vehicles only, k = 13.
      PCA and UMAP are fitted on these windows only, in the space of the clustering
      (scaler fitted on all windows, as in gate_k_faulty.py). Panel (c) equals Figure 5b.

Writes new files only; existing figures and tables are not touched. OUT_DIR is set here
because importing gate_k_windowed would pull in an experiment script."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import common
import rwth_style as st
from fleet_map import draw_labels

UMAP_N_NEIGHBORS, UMAP_MIN_DIST = 15, 0.1  # same as Figure 4a
DOMINANCE_PCT = 50
HEIGHT_CM, HEIGHT_RATIOS = 14.0, [8.0, 6.0]
OUT_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate1"
DATA_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "data"


def new_figure():
    fig, axes = plt.subplot_mosaic([["a", "b"], ["c", "c"]], figsize=(st.FULL_WIDTH, HEIGHT_CM * st.CM),
                                   height_ratios=HEIGHT_RATIOS)
    return fig, axes


def panels_for(variance: np.ndarray) -> list[tuple]:
    return [("a", "pc1", "pc2", f"PC 1 ({variance[0]:.0f} % of variance) [–]", f"PC 2 ({variance[1]:.0f} % of variance) [–]"),
            ("b", "umap1", "umap2", "UMAP 1 [–]", "UMAP 2 [–]")]


def draw_bars(ax, shares: pd.DataFrame, groups: list[str]) -> np.ndarray:
    positions, bottom = np.arange(len(shares)), np.zeros(len(shares))
    for group in groups:
        ax.bar(positions, shares[group], bottom=bottom, width=0.68, color=st.GROUP[group]["color"], edgecolor="white",
               label=st.GROUP[group]["label"])
        bottom += shares[group].to_numpy()
    ax.set_xticks(positions, [f"{c}\n{int(n)}" for c, n in zip(shares.index, shares["windows"])])
    ax.set_xlabel("Cluster (top) and number of windows (bottom)")
    ax.set_ylabel("Share of windows [%]")
    ax.set_ylim(0, 100)
    st.panel_label(ax, "c")
    return positions


def composition(table: pd.DataFrame, cluster_column: str, groups: list[str]) -> pd.DataFrame:
    shares = 100 * pd.crosstab(table[cluster_column], table["group"], normalize="index").reindex(columns=groups, fill_value=0)
    shares["windows"] = table.groupby(cluster_column).size()
    return shares


def figure_all(windows: pd.DataFrame, columns: list[str]) -> None:
    table = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")
    assert (table["ev_id"].to_numpy() == windows["ev_id"].to_numpy()).all() and len(table) == 6209
    faulty = table["is_faulty_ev"].to_numpy()
    table["group"] = np.where(~faulty, "healthy", "faulty_" + table["pack_type"])
    variance = 100 * PCA(n_components=2, random_state=common.SEED).fit(StandardScaler().fit_transform(windows[columns])).explained_variance_ratio_

    shares = composition(table, "cluster", list(st.GROUP))
    shares["faulty"] = shares["faulty_large"] + shares["faulty_small"]
    shares = shares.sort_values("faulty", ascending=False)
    stored = pd.read_csv(OUT_DIR / "fig05a_cluster_shares.csv", index_col="cluster")
    assert np.allclose(shares[list(st.GROUP)].round(1), stored.loc[shares.index, list(st.GROUP)])

    st.set_style()
    fig, axes = new_figure()
    colours = np.array(st.CLUSTER_COLORS)[table["cluster"].to_numpy() - 1]
    for letter, cx, cy, xl, yl in panels_for(variance):
        ax = axes[letter]
        ax.scatter(table.loc[~faulty, cx], table.loc[~faulty, cy], c=colours[~faulty], s=3, linewidths=0, alpha=0.7)
        ax.scatter(table.loc[faulty, cx], table.loc[faulty, cy], c=colours[faulty], s=5, edgecolors="black", linewidths=0.25)
        draw_labels(ax, table, cx, cy)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        st.panel_label(ax, letter)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3, mfc=st.RWTH["blue50"], mec="none", label="Windows of healthy vehicles"),
               plt.Line2D([], [], ls="", marker="o", ms=3.5, mfc=st.RWTH["blue50"], mec="black", mew=0.4, label="Windows of faulty vehicles (outlined)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2)

    ax = axes["c"]
    positions = draw_bars(ax, shares, list(st.GROUP))
    dominated = shares["faulty"] > DOMINANCE_PCT
    ax.scatter(positions[dominated], np.full(dominated.sum(), 106), marker="v", s=18, color="black", clip_on=False,
               label=f"Dominated by windows of faulty EVs (> {DOMINANCE_PCT} %)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=4, columnspacing=0.8, handletextpad=0.4)
    print(st.save_figure(fig, "fig04_all_k9_map_composition"))
    print("all windows: variance", variance.round(1))


def figure_faulty(windows: pd.DataFrame, columns: list[str]) -> None:
    is_faulty = windows["is_faulty_ev"].to_numpy()
    x = StandardScaler().fit_transform(windows[columns])[is_faulty]
    table = windows.loc[is_faulty, ["ev_id", "window_start", "pack_type"]].reset_index(drop=True)
    labels = pd.read_csv(DATA_DIR / "windows_120h_faulty_k13.csv")
    table = table.merge(labels, on=["ev_id", "window_start"], how="left", validate="one_to_one")
    assert len(table) == 1156 and table["cluster_faulty_k13"].notna().all()
    assert adjusted_rand_score(common.cluster(x, 13), table["cluster_faulty_k13"]) > 0.999, "not the space of the clustering"

    pca = PCA(n_components=2, random_state=common.SEED).fit(x)
    pcs = pca.transform(x)
    maps = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, n_components=2, random_state=common.SEED).fit_transform(x)
    table = table.assign(pc1=pcs[:, 0], pc2=pcs[:, 1], umap1=maps[:, 0], umap2=maps[:, 1], group="faulty_" + table["pack_type"])
    table.to_csv(OUT_DIR / "windows_120h_faulty_k13_map.csv", index=False)

    groups = ["faulty_large", "faulty_small"]
    shares = composition(table, "cluster_faulty_k13", groups).sort_values(["faulty_small", "windows"], ascending=False)
    stored = pd.read_csv(OUT_DIR / "fig05b_cluster_shares.csv", index_col=0)
    assert np.allclose(shares[groups].round(1), stored.loc[shares.index, groups])
    assert list(shares.index) == list(stored.index)

    order = pd.read_csv(OUT_DIR / "faulty_k13_clusters.csv")["name"].tolist()
    colour_of = dict(zip(order, st.CLUSTER_COLORS_13))  # same colours as Figure 4b
    coloured = table.rename(columns={"cluster_faulty_k13": "cluster"})

    st.set_style()
    fig, axes = new_figure()
    for letter, cx, cy, xl, yl in panels_for(100 * pca.explained_variance_ratio_):
        ax = axes[letter]
        ax.scatter(coloured[cx], coloured[cy], c=coloured["cluster"].map(colour_of).tolist(), s=6, edgecolors="black", linewidths=0.25)
        draw_labels(ax, coloured, cx, cy)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        st.panel_label(ax, letter)

    ax = axes["c"]
    draw_bars(ax, shares, groups)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.0)
    print(st.save_figure(fig, "fig05_faulty_k13_map_composition"))
    print("faulty windows: variance", (100 * pca.explained_variance_ratio_).round(1))
    print(table.groupby("cluster_faulty_k13")[["pc1", "pc2", "umap1", "umap2"]].median().round(2).reindex(order).to_string())
    print(pd.Series(np.abs(pca.components_[0]), index=columns).sort_values(ascending=False).head(5).round(2).to_string())


if __name__ == "__main__":
    all_windows, feature_columns = common.load_windows("120h")
    figure_all(all_windows, feature_columns)
    figure_faulty(all_windows, feature_columns)
