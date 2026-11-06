"""Figure 12: (a) one window flagged by both methods at the matched operating point: SHAP
next to centroid contributions; (b) Jaccard@5 over all jointly flagged windows vs a random
baseline. The window is chosen by rule: EV 389, seed 0, Jaccard closest to the overall
median, earliest such window. Second figure (fig12b): panel (a) with both methods as share
of the window's total absolute attribution, so that they share one axis, next to the same
Jaccard@5 panel, which is (c) there."""
from math import comb

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import common
import rwth_style as st

EV, SEED, TOP, N_FEATURES = 389, 0, 5, 15
OUT = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"

meta = pd.read_csv(OUT / "explanations_meta.csv")
data = np.load(OUT / "explanations_values.npz")
values_shap, values_centroid, columns = data["shap"], data["contributions"], data["columns"].tolist()

# exact distribution of the Jaccard index between two random top-5 sets of 15 features
baseline = {k / (2 * TOP - k): comb(TOP, k) * comb(N_FEATURES - TOP, TOP - k) / comb(N_FEATURES, TOP) for k in range(TOP + 1)}
expected = sum(j * p for j, p in baseline.items())
median = meta["jaccard"].median()
candidates = meta[(meta["ev_id"] == EV) & (meta["seed"] == SEED)]
pick = candidates.assign(gap=(candidates["jaccard"] - median).abs()).sort_values(["gap", "window_start"]).index[0]
window = meta.loc[pick]
shap_row, centroid_row = pd.Series(values_shap[pick], columns), pd.Series(values_centroid[pick], columns)
shown = list(dict.fromkeys([*shap_row.abs().sort_values(ascending=False).index[:TOP], *centroid_row.abs().sort_values(ascending=False).index[:TOP]]))[::-1]
top_shap, top_centroid = set(shap_row.abs().sort_values(ascending=False).index[:TOP]), set(centroid_row.abs().sort_values(ascending=False).index[:TOP])

st.set_style()
fig = plt.figure(figsize=(st.FULL_WIDTH, 8.8 * st.CM))
grid = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.15])
ax_shap, ax_centroid = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
ax_b = fig.add_subplot(grid[2])
positions = np.arange(len(shown))


def draw_bars(ax, values: pd.Series, top: set, method: str, xlabel: str, title: str):
    bars = ax.barh(positions, values[shown], height=0.68, color=st.METHOD[method], edgecolor="black", linewidth=0.3)
    for bar, feature in zip(bars, shown):
        if feature not in top:
            bar.set_alpha(0.3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=6.8)
    ax.xaxis.grid(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return bars


draw_bars(ax_shap, shap_row, top_shap, "isolation_forest", "SHAP value [\u2013]", "Isolation Forest (SHAP)")
draw_bars(ax_centroid, centroid_row, top_centroid, "centroid", "Contribution [\u2013]", "Centroid score (exact contribution)")
ax_shap.set_yticks(positions, [st.WINDOWED_SHORT[c][0] for c in shown], fontsize=6.5)
ax_centroid.set_yticks(positions, [])
st.panel_label(ax_shap, "a")

levels = sorted(baseline)
observed = meta["jaccard"].round(6).value_counts(normalize=True)
heights = [observed.get(round(j, 6), 0) * 100 for j in levels]
x = np.array(levels)
means = [(meta["jaccard"].mean(), st.RWTH["grey"], "Mean, jointly flagged"), (meta["jaccard_shuffled_windows"].mean(), st.RWTH["petrol"], "Mean, mismatched windows"), (expected, "black", "Mean, random sets")]


def draw_jaccard(ax, letter: str):
    ax.bar(x - 0.02, heights, width=0.038, color=st.RWTH["grey"], label="Jointly flagged windows")
    ax.bar(x + 0.02, [baseline[j] * 100 for j in levels], width=0.038, color=st.RWTH["black25"], label="Random top-5 sets")
    for value, colour, label in means:
        ax.axvline(value, color=colour, lw=0.9, ls="--", label=f"{label} ({value:.2f})")
    ax.set_xticks(x, [f"{j:.2f}" for j in levels], fontsize=6.3, rotation=90)
    ax.set_xlim(-0.06, 1.06)
    ax.set_xlabel("Jaccard@5 [\u2013]")
    ax.set_ylabel("Share of windows [%]")
    ax.yaxis.grid(True)
    ax.set_ylim(0, 52)
    ax.figure.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=7, columnspacing=1.0, handlelength=1.4, labelspacing=0.25)
    st.panel_label(ax, letter)


draw_jaccard(ax_b, "b")
print(st.save_figure(fig, "fig12_explanations"))

# Figure 12b: the window of panel (a) on one scale. SHAP (path length) and the contributions (squared
# standardised distance) have different units, so each is given as share of the window's total |attribution|.
fig = plt.figure(figsize=(st.FULL_WIDTH, 8.8 * st.CM))
grid = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.15])
axes = [fig.add_subplot(grid[0])]
axes.append(fig.add_subplot(grid[1], sharex=axes[0], sharey=axes[0]))
XLABEL_SHARE = "Share of total\n|attribution| [%]"
share_shap, share_centroid = 100 * shap_row / shap_row.abs().sum(), 100 * centroid_row / centroid_row.abs().sum()
for ax, values, top, method, title, letter in ((axes[0], share_shap, top_shap, "isolation_forest", "Isolation Forest\n(SHAP)", "a"),
                                               (axes[1], share_centroid, top_centroid, "centroid", "Centroid score\n(exact contribution)", "b")):
    ax.bar_label(draw_bars(ax, values, top, method, XLABEL_SHARE, title), fmt="%.1f", padding=2, fontsize=6.3)
    st.panel_label(ax, letter)
axes[0].set_yticks(positions, [st.WINDOWED_SHORT[c][0] for c in shown], fontsize=6.5)
axes[1].tick_params(labelleft=False)
axes[0].set_xlim(-14, 64)
draw_jaccard(fig.add_subplot(grid[2]), "c")
print(st.save_figure(fig, "fig12b_explanations_same_scale"))

meta["group"] = np.where(meta["ev_id"] >= common.FAULT_EV_ID_CUTOFF, "faulty", "healthy")
print("windows", len(meta), "| per seed", meta.groupby("seed").size().median(), "| by group", meta["group"].value_counts().to_dict(), "| vehicles", meta["ev_id"].nunique())
print(f"Jaccard mean {meta['jaccard'].mean():.3f}, median {median:.3f}; positive-only mean {meta['jaccard_positive'].mean():.3f}; shuffled windows mean {meta['jaccard_shuffled_windows'].mean():.3f}; random expected {expected:.3f}")
print("share of windows above random expectation:", round(float((meta["jaccard"] > expected).mean()), 3), "| seed-level mean", meta.groupby("seed")["jaccard"].mean().agg(["mean", "std"]).round(3).to_dict())
print(meta.groupby("group")["jaccard"].agg(["count", "mean", "median"]).round(3).to_string())
print("distribution observed", {round(j, 2): round(h, 1) for j, h in zip(levels, heights)})
print("distribution random", {round(j, 2): round(p * 100, 1) for j, p in baseline.items()})
print("window:", window[["ev_id", "seed", "fold", "window_start", "jaccard"]].to_dict(), "hours end", window["window_start"] * 30 / 3600 + 120)
print("top SHAP:", list(shap_row.abs().sort_values(ascending=False).index[:TOP])); print("top centroid:", list(centroid_row.abs().sort_values(ascending=False).index[:TOP]))
print("shared:", sorted(top_shap & top_centroid))
top_counts = pd.DataFrame({"shap": pd.Series(np.argsort(-np.abs(values_shap), axis=1)[:, :TOP].ravel()).value_counts(normalize=True) * 100 * TOP / TOP, "centroid": pd.Series(np.argsort(-np.abs(values_centroid), axis=1)[:, :TOP].ravel()).value_counts(normalize=True) * 100}).fillna(0)
top_counts.index = [columns[i] for i in top_counts.index]
print((top_counts * TOP).round(0).sort_values("shap", ascending=False).head(8).to_string())
vehicle = meta.groupby("ev_id")["jaccard"].mean()
print("vehicle level: n", len(vehicle), "mean", round(vehicle.mean(), 3), "median", round(vehicle.median(), 3), "min", round(vehicle.min(), 3), "max", round(vehicle.max(), 3),
      "| windows per vehicle median", meta.groupby("ev_id").size().median(), "max", meta.groupby("ev_id").size().max(), "| vehicles above random", int((vehicle > expected).sum()))

for name, values in (("SHAP", values_shap), ("centroid", values_centroid)):
    shares = 100 * np.sort(np.abs(values), axis=1)[:, ::-1] / np.abs(values).sum(axis=1, keepdims=True)
    print(f"{name} share of total |attribution|, all jointly flagged windows: top feature median {np.median(shares[:, 0]):.1f} %, top five median {np.median(shares[:, :TOP].sum(axis=1)):.1f} %")
print("this window, top feature / top five share:", round(share_shap.abs().max(), 1), round(share_shap.abs().nlargest(TOP).sum(), 1), "(SHAP)", round(share_centroid.abs().max(), 1), round(share_centroid.abs().nlargest(TOP).sum(), 1), "(centroid)")
