"""Violin panels by cluster label, shared by Figure 7 and the appendix figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

import rwth_style as st

GROUP_GAP, LOG_DECADES, SCALE_BELOW = 0.8, 2, 0.01
LABEL_COLOUR = {"fault": st.FAULTY, "transition": st.SMALL_PACK, "healthy": st.HEALTHY}


def layout(table: pd.DataFrame) -> tuple[dict, float]:
    positions, cursor, previous = {}, 0.0, None
    for _, row in table.iterrows():
        cursor += GROUP_GAP if previous is not None and row["label"] != previous else 0
        positions[row["name"]] = cursor
        cursor += 1
        previous = row["label"]
    return positions, cursor


def violin(ax, values: np.ndarray, position: float, colour: str) -> None:
    body = ax.violinplot(values, positions=[position], widths=0.82, showextrema=False)["bodies"][0]
    body.set(facecolor=colour, edgecolor="black", linewidth=0.3, alpha=1)
    low, median, high = np.percentile(values, [25, 50, 75])
    ax.vlines(position, low, high, color="black", lw=1.2, zorder=3)
    ax.scatter([position], [median], s=4, color="white", edgecolors="black", linewidths=0.3, zorder=4)


def anova(windows: pd.DataFrame, names: list[str], columns: list[str]) -> pd.Series:
    return pd.Series({c: f_oneway(*[windows.loc[windows["group"] == n, c].to_numpy() for n in names]).statistic
                      for c in columns}).sort_values(ascending=False)


def draw_panel(ax, windows: pd.DataFrame, feature: str, table: pd.DataFrame, title_size: float = st.LEGEND_PT) -> dict:
    """Returns how the axis was scaled, for the report."""
    positions, end = layout(table)
    info = table.set_index("name")
    title, ylabel = st.WINDOWED_SHORT[feature]
    values = windows[feature].to_numpy()
    use_log = values.min() > 0 and values.max() / values.min() > 10 ** LOG_DECADES
    exponent = 0
    if not use_log and np.abs(values).max() < SCALE_BELOW:
        exponent = int(np.floor(np.log10(np.abs(values).max())))
        ylabel = ylabel.replace("[–]", f"[10$^{{{exponent}}}$]") if "[–]" in ylabel else ylabel.replace(" [", f" [10$^{{{exponent}}}$ ")
    factor = 10.0 ** -exponent
    for name, position in positions.items():
        data = windows.loc[windows["group"] == name, feature].to_numpy()
        violin(ax, np.log10(data) if use_log else data * factor, position, LABEL_COLOUR[info.loc[name, "label"]])
    if use_log:
        ticks = np.arange(np.floor(np.log10(values.min())), np.ceil(np.log10(values.max())) + 1)
        ax.set_yticks(ticks, [f"$10^{{{int(t)}}}$" for t in ticks])
    for name, position in positions.items():
        if info.loc[name, "large_pack_share"] == 0:
            ax.scatter(position, 1.015, transform=ax.get_xaxis_transform(), marker="v", s=7, color="black", clip_on=False, zorder=5)
    ax.set_title(title, fontsize=title_size, pad=8)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    ax.set_xticks(list(positions.values()), list(positions), fontsize=5.8)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_xlim(-0.7, end - 0.3)
    return {"feature": title, "log": use_log, "scale_exponent": exponent}


def add_legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, fc=st.FAULTY, ec="black", lw=0.3, label="Fault cluster"),
               plt.Rectangle((0, 0), 1, 1, fc=st.SMALL_PACK, ec="black", lw=0.3, label="Transition cluster"),
               plt.Rectangle((0, 0), 1, 1, fc=st.HEALTHY, ec="black", lw=0.3, label="Healthy cluster"),
               plt.Line2D([], [], ls="", marker="v", ms=3, color="black", label="Faulty members from small-pack vehicles only")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
