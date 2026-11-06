# %%
"""
Volcano plot: Cohen's d (effect size) vs. Mann-Whitney U significance, for
every feature in extracted_data/feature_fault_analysis.csv (produced by
fault_feature.py).

x-axis: cohens_d -- signed effect size (positive = elevated in faulty EVs).
y-axis: -log10(mannwhitney_p) -- significance, higher = more significant.

Points are colored by whether they clear both the significance and
effect-size thresholds, split by direction (elevated in faulty vs. healthy),
same two-line threshold-guide convention as fault_feature.py's r-vs-AUC
scatter plots.

Input:  extracted_data/feature_fault_analysis.csv
Output: experiments/01_fault_detection/figures/volcano_plot.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make the repo-root "plot_fnc" package importable regardless of where this
# script lives (experiments/01_.../volcano_plot.py -> repo root is 2 levels up).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

EXTRACTED_DATA = PROJECT_ROOT / "extracted_data"
SUMMARY_CSV = EXTRACTED_DATA / "feature_fault_analysis.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ===========================================================================
# CONFIG -- same screening defaults as fault_feature.py, so the volcano plot's
# threshold guides line up with the summary table's "significant"/"large_effect"
# flags.
# ===========================================================================
ALPHA = 0.05               # significance threshold on mannwhitney_p
COHENS_D_THRESHOLD = 0.5   # |cohens_d| above this counts as a large effect

FIGURE_DPI = 300

COLOR_UP = RWTHColorScales.red[0]      # significant & elevated in faulty (d > threshold)
COLOR_DOWN = RWTHColorScales.blue[0]   # significant & elevated in healthy (d < -threshold)
COLOR_NS = RWTHColorScales.blue[3]     # not significant / not a large effect

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def classify_point(row: pd.Series) -> str:
    """Volcano-plot hit category from this row's own mannwhitney_p/cohens_d
    (not the CSV's precomputed "significant" column, which is based on
    point-biserial p -- this keeps the coloring consistent with the two
    statistics actually shown on the plot's axes).
    """
    if row["mannwhitney_p"] < ALPHA and row["cohens_d"] > COHENS_D_THRESHOLD:
        return "up"
    if row["mannwhitney_p"] < ALPHA and row["cohens_d"] < -COHENS_D_THRESHOLD:
        return "down"
    return "ns"


def plot_volcano(summary: pd.DataFrame, out_path: Path) -> None:
    neg_log10_p = -np.log10(summary["mannwhitney_p"])
    category = summary.apply(classify_point, axis=1)
    colors = category.map({"up": COLOR_UP, "down": COLOR_DOWN, "ns": COLOR_NS})

    fig, ax = plt.subplots(figsize=(9, 7))

    # threshold guides: horizontal at the significance cutoff, vertical at
    # +/- the effect-size cutoff -- same dashed-line convention as
    # fault_feature.py's r-vs-AUC scatter plots.
    ax.axhline(-np.log10(ALPHA), color="black", linewidth=0.7, linestyle="--", zorder=1)
    ax.axvline(COHENS_D_THRESHOLD, color="black", linewidth=0.7, linestyle="--", zorder=1)
    ax.axvline(-COHENS_D_THRESHOLD, color="black", linewidth=0.7, linestyle="--", zorder=1)

    ax.scatter(summary["cohens_d"], neg_log10_p, c=colors,
               s=42, edgecolor="white", linewidth=0.5, zorder=2)
    for x, y, feature in zip(summary["cohens_d"], neg_log10_p, summary["feature"]):
        ax.annotate(feature, (x, y), fontsize=6.5, xytext=(3, 3),
                    textcoords="offset points", zorder=3)

    ax.set_xlabel("Cohen's d (effect size, faulty vs. healthy)")
    ax.set_ylabel(r"$-\log_{10}$(Mann-Whitney U p-value)")
    ax.set_title("Volcano plot: effect size vs. significance (feature vs. fault label)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_UP,
                   label=f"significant, elevated in faulty (p < {ALPHA}, d > {COHENS_D_THRESHOLD})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_DOWN,
                   label=f"significant, elevated in healthy (p < {ALPHA}, d < -{COHENS_D_THRESHOLD})"),
        plt.Line2D([], [], marker="o", linestyle="", color=COLOR_NS, label="not significant / small effect"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# %%
summary = pd.read_csv(SUMMARY_CSV)
plot_volcano(summary, FIGURES_DIR / "volcano_plot.png")
print(f"saved {FIGURES_DIR / 'volcano_plot.png'}")
