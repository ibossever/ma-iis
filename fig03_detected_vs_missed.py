"""Figure 3: detected and missed vehicles in original feature space.

Main figure: the feature pair named in the plan.
Alternative: the two features ranked highest by mean |SHAP|, for comparison.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st
from common import CAPACITY_COLUMN

TEMPERATURE = "pack_temp__mean_pack_temp"
DRAW_ORDER = ["TN", "FP", "FN", "TP"]


def scatter(x_col: str, y_col: str, name: str, width: float = st.HALF_WIDTH,
            y_scale: float = 1.0, y_label: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(width, 7.0 * st.CM))
    for outcome in DRAW_ORDER:
        rows = v["outcome"] == outcome
        style = {**st.OUTCOME[outcome], "label": f"{outcome} ({rows.sum()})"}
        ax.scatter(features.loc[rows, x_col], features.loc[rows, y_col] * y_scale, zorder=3, **style)
    ax.set_xlabel(st.FEATURE_LABELS[x_col])
    ax.set_ylabel(y_label or st.FEATURE_LABELS[y_col])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, columnspacing=0.6, handletextpad=0.1,
              borderaxespad=0.1)
    print(st.save_figure(fig, name))


st.set_style()
d = common.load_whole_session()
features, v = d["features"], d["vehicles"]
ranked = d["shap"][list(features.columns)].abs().mean().sort_values(ascending=False).index
scatter(CAPACITY_COLUMN, TEMPERATURE, "fig03_detected_vs_missed")
scatter(ranked[0], ranked[1], "fig03_alt_top_shap", width=st.FULL_WIDTH, y_scale=1e5,
        y_label="SOC spectrum dev., 7.5\u201330 h$^{-1}$,\nmean [10$^{-5}$]")
print(features.groupby(v["outcome"])[[CAPACITY_COLUMN, TEMPERATURE]].median().round(1).to_string())
print("max capacity TP", features.loc[v["outcome"] == "TP", CAPACITY_COLUMN].max().round(1),
      "| min capacity of FN/FP/TN", features.loc[v["outcome"] != "TP", CAPACITY_COLUMN].min().round(1))
