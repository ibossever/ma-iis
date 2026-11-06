"""Figure 7: physical-unit violins of the six features with the largest one-way ANOVA F
across clusters, grouped by label. One descriptive fit on all vehicles (decisions.json).
Second figure (fig07b): SHAP reasons of Isolation Forest alarms against the clusters of the
same windows, data from gate_explanations_if.py."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import cluster_violins as cv
import common
import prototypes as pr
import rwth_style as st

OUT_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate2"

windows, columns = common.load_windows("120h")
x = StandardScaler().fit_transform(windows[columns])
fit = pr.descriptive_fit(windows, x, common.SEED)
fit["table"].to_csv(OUT_DIR / "descriptive_fit_clusters.csv")
windows = windows.assign(group=fit["groups"])
table = fit["ordered"]
names = list(table["name"])
scores = cv.anova(windows, names, columns)
scores.round(1).to_csv(OUT_DIR / "fig07_anova_f.csv", header=["F"])

st.set_style()
fig, axes = plt.subplots(2, 3, figsize=(st.FULL_WIDTH, 11.0 * st.CM))
scales = [cv.draw_panel(ax, windows, feature, table) for ax, feature in zip(axes.ravel(), scores.index[:6])]
cv.add_legend(fig)
print(st.save_figure(fig, "fig07_prototype_profiles"))
print(pd.DataFrame(scales).to_string(index=False))
print(pd.DataFrame({"F": scores.round(1)}).head(6).to_string())
counts = windows["group"].value_counts().reindex(names)
print("windows per cluster:", counts.to_dict())
near = windows.groupby("group")
report = table.set_index("name")[["label", "windows", "vehicles", "large_pack_share", "ratio"]].assign(
    nearest_windows=counts, nearest_faulty_windows=near["is_faulty_ev"].sum(), nearest_faulty_vehicles=near.apply(lambda g: g.loc[g["is_faulty_ev"], "ev_id"].nunique()))
report.to_csv(OUT_DIR / "fig07_clusters.csv")
print(report.round(2).to_string())

# Figure 7b: what SHAP reports about an alarm against the state the alarm is in. Every window the
# Isolation Forest flags in at least MIN_SEEDS of the seeds (the rule of Figure 10); not filtered
# by the centroid score, which would remove most healthy-cluster alarms. Shown reason and second
# feature follow fixed rules, so that nothing is picked by how it looks.
EXPLAIN_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "gate3"
MIN_SEEDS, MIN_REASON_WINDOWS = 5, 5
KEYS = ["ev_id", "window_start"]

meta = pd.read_csv(EXPLAIN_DIR / "explanations_if_meta.csv")
shap_table = pd.DataFrame(np.load(EXPLAIN_DIR / "explanations_if_values.npz")["shap"], columns=columns)
by_window = shap_table.groupby([meta[key] for key in KEYS])
alarm_shap = by_window.mean()[by_window.size() >= MIN_SEEDS]
oof_label = meta.groupby(KEYS)["oof_label"].agg(lambda s: s.mode().iloc[0])
alarms = windows.set_index(KEYS, drop=False).loc[alarm_shap.index].assign(reason=alarm_shap.abs().idxmax(axis=1), oof_label=oof_label)
counts = pd.crosstab(alarms["reason"], alarms["group"]).reindex(columns=names, fill_value=0)
counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
counts.rename(index=lambda c: st.WINDOWED_SHORT[c][0]).to_csv(EXPLAIN_DIR / "fig07b_reason_by_cluster.csv")

# reason shared by the most clusters (ties: most alarms); second feature: largest F within that reason
sharing = pd.DataFrame({"clusters": (counts >= MIN_REASON_WINDOWS).sum(axis=1), "alarms": counts.sum(axis=1)})
reason = sharing.sort_values(["clusters", "alarms"], ascending=False).index[0]
reason_groups = [name for name in names if counts.loc[reason, name] >= MIN_REASON_WINDOWS]
family = alarms[(alarms["reason"] == reason) & alarms["group"].isin(reason_groups)]
second = cv.anova(family, reason_groups, [c for c in columns if c != reason]).index[0]
reference = table[table["label"] == "healthy"]
shown = table[table["name"].isin(reason_groups) | table["name"].isin(reference["name"])]
shown_windows = pd.concat([family, windows[windows["group"].isin(reference["name"])]])

fig = plt.figure(figsize=(st.FULL_WIDTH, 17.0 * st.CM))
outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1])
ax_table = fig.add_subplot(outer[0])
bottom = outer[1].subgridspec(1, 3, width_ratios=[len(reason_groups) + 1, len(shown) + 2, len(shown) + 2])
ax_shap, ax_reason, ax_second = (fig.add_subplot(bottom[i]) for i in range(3))

positions, end = cv.layout(table)
info = table.set_index("name")
reason_rows = list(counts.index)
summary = {"Alarms": alarms.groupby("group").size(), "Vehicles": alarms.groupby("group")["ev_id"].nunique(),
           "Faulty [%]": 100 * alarms.groupby("group")["is_faulty_ev"].mean()}
for row, feature in enumerate(reason_rows):
    for name, x_position in positions.items():
        share = counts.loc[feature, name] / max(counts[name].sum(), 1)
        if counts.loc[feature, name]:
            ax_table.add_patch(plt.Rectangle((x_position - 0.5, row - 0.5), 1, 1, fc=st.CMAP_SEQUENTIAL(share), ec="white", lw=0.5))
            ax_table.text(x_position, row, counts.loc[feature, name], ha="center", va="center", fontsize=5.8, color="white" if share > 0.5 else "black")
    subset = alarms[alarms["reason"] == feature]
    ax_table.text(end + 0.4, row, len(subset), ha="center", va="center", fontsize=5.8)
    ax_table.text(end + 2.0, row, f"{100 * subset['is_faulty_ev'].mean():.0f}", ha="center", va="center", fontsize=5.8)
for row, (title, values) in enumerate(summary.items(), start=len(reason_rows)):
    for name, x_position in positions.items():
        if name in values.index:
            ax_table.text(x_position, row + 0.15, f"{values[name]:.0f}", ha="center", va="center", fontsize=5.8)
ax_table.axhline(len(reason_rows) - 0.4, color=st.RWTH["grey"], lw=0.5)
ax_table.text(end + 0.4, -1, "Alarms", ha="center", va="center", fontsize=5.8)
ax_table.text(end + 2.0, -1, "Faulty [%]", ha="center", va="center", fontsize=5.8)
ax_table.set_xlim(-0.6, end + 2.9)
ax_table.set_ylim(len(reason_rows) + len(summary) - 0.3, -0.6)
ax_table.set_xticks(list(positions.values()), list(positions), fontsize=6.5)
ax_table.xaxis.tick_top()
for tick, name in zip(ax_table.get_xticklabels(), positions):
    tick.set_color(cv.LABEL_COLOUR[info.loc[name, "label"]])
ax_table.set_yticks([*range(len(reason_rows)), *(row + 0.15 for row in range(len(reason_rows), len(reason_rows) + len(summary)))],
                    [st.WINDOWED_SHORT[c][0] for c in reason_rows] + list(summary), fontsize=6.3)
ax_table.tick_params(length=0)
ax_table.spines[:].set_visible(False)
st.panel_label(ax_table, "a")

reason_table = table[table["name"].isin(reason_groups)]
reason_positions, reason_end = cv.layout(reason_table)
for name, x_position in reason_positions.items():
    cv.violin(ax_shap, alarm_shap.loc[family.index[family["group"] == name], reason].to_numpy(), x_position, cv.LABEL_COLOUR[info.loc[name, "label"]])
ax_shap.set_title("SHAP value", fontsize=st.LEGEND_PT - 1, pad=8)
ax_shap.set_ylabel(f"SHAP of {st.WINDOWED_SHORT[reason][0].lower()} [–]")
ax_shap.set_ylim(0, None)
ax_shap.yaxis.grid(True)
ax_shap.set_xticks(list(reason_positions.values()), list(reason_positions), fontsize=5.8)
ax_shap.set_xlim(-0.7, reason_end - 0.3)
st.panel_label(ax_shap, "b")
scales = [cv.draw_panel(ax, shown_windows, feature, shown, title_size=st.LEGEND_PT - 1) for ax, feature in ((ax_reason, reason), (ax_second, second))]
st.panel_label(ax_reason, "c")
st.panel_label(ax_second, "d")
cv.add_legend(fig)
print(st.save_figure(fig, "fig07b_states_vs_shap"))


def purity(key: str) -> float:
    """Share of alarms in the majority class of their group; 1 = groups are all faulty or all healthy."""
    parts = alarms.groupby(key)["is_faulty_ev"].agg(["sum", "size"])
    return round(float(np.maximum(parts["sum"], parts["size"] - parts["sum"]).sum() / len(alarms)), 3)


print(pd.DataFrame(scales).to_string(index=False))
print("alarms", len(alarms), "| vehicles", alarms["ev_id"].nunique(), "| faulty share", round(alarms["is_faulty_ev"].mean(), 3), "| distinct reasons", alarms["reason"].nunique())
print("purity: reason", purity("reason"), "| cluster", purity("group"), "| out-of-fold label", purity("oof_label"))
print(alarms.groupby("oof_label")["is_faulty_ev"].agg(["size", "mean"]).round(3).to_string())
print("reason", reason, "| second feature", second)
print(family.assign(shap=alarm_shap.loc[family.index, reason]).groupby("group").agg(windows=("ev_id", "size"), vehicles=("ev_id", "nunique"), shap_median=("shap", "median"), shap_mean=("shap", "mean"), shap_q25=("shap", lambda s: s.quantile(0.25)), shap_q75=("shap", lambda s: s.quantile(0.75)), value_median=(reason, "median"), value_mean=(reason, "mean"), second_median=(second, "median")).round(4).to_string())
print(counts.rename(index=lambda c: st.WINDOWED_SHORT[c][0]).assign(total=counts.sum(axis=1).to_numpy()).to_string())
print("reasons per cluster:", (counts > 0).sum().to_dict())
