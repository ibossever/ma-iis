"""Shared thesis figure style. Every script in experiments/04_figures/ imports this.

Single switch for the thesis font: FONT_FAMILY.
Colours are read from plot_fnc/rwth_colors.py; only the four tones missing there
are defined here. Figures are saved at their exact print size (no bbox "tight").
"""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_hex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales as S  # noqa: E402

OUT_DIR = PROJECT_ROOT / "figures" / "thesis"

# --- switch the thesis font here ----------------------------------------------
FONT_FAMILY = "Arial"
FONT_FALLBACKS = ["Helvetica", "Liberation Sans"]  # Liberation Sans = Arial metrics

# --- sizes --------------------------------------------------------------------
CM = 1 / 2.54
FULL_WIDTH = 15.5 * CM
HALF_WIDTH = 7.5 * CM
AXIS_LABEL_PT, TICK_PT, LEGEND_PT, PANEL_PT = 9, 8, 8, 10

# --- RWTH corporate palette ---------------------------------------------------
RWTH = {
    "blue": to_hex(S.blue[0]), "blue75": to_hex(S.blue[1]), "blue50": to_hex(S.blue[2]),
    "blue25": to_hex(S.blue[3]), "blue10": to_hex(S.blue[4]),
    "red": to_hex(S.red[0]), "bordeaux": to_hex(S.bordeaux[0]), "orange": to_hex(S.orange[0]),
    "green": to_hex(S.green[0]), "maygreen": to_hex(S.maygreen[0]), "petrol": to_hex(S.petrol[0]),
    "turquoise": to_hex(S.turquoise[0]), "violet": to_hex(S.violet[0]), "purple": to_hex(S.lilac[0]),
    # not in rwth_colors.py
    "black": "#000000", "grey": "#646567", "black25": "#cfd1d2", "yellow": "#ffed00", "magenta": "#e30066",
}

# --- fixed meaning of colours (identical in all figures) ----------------------
HEALTHY, FAULTY, SMALL_PACK = RWTH["blue"], RWTH["red"], RWTH["orange"]
METHOD = {"isolation_forest": RWTH["violet"], "centroid": RWTH["green"], "hybrid": RWTH["grey"]}
METHOD_MARKER = {"isolation_forest": "o", "centroid": "s", "hybrid": "^"}  # never rely on hue alone
METHOD_LINESTYLE = {"isolation_forest": "-", "centroid": "-", "hybrid": "--"}
METHOD_LABEL = {"isolation_forest": "Isolation Forest", "centroid": "Centroid distance",
                "hybrid": "Hybrid (flag or trend)"}

# Plan: "missed = filled, false alarm = open"; brief: FN = Bordeaux, FP = blue open.
# Red and Bordeaux are close, so TP and FN also differ in marker shape.
OUTCOME = {
    "TP": dict(marker="o", s=16, facecolors=RWTH["red"], edgecolors=RWTH["red"], linewidths=0.5, label="TP"),
    "FN": dict(marker="D", s=16, facecolors=RWTH["bordeaux"], edgecolors="black", linewidths=0.5, label="FN (missed)"),
    "FP": dict(marker="s", s=16, facecolors="none", edgecolors=RWTH["blue"], linewidths=0.8, label="FP (false alarm)"),
    "TN": dict(marker="o", s=5, facecolors=RWTH["blue50"], edgecolors="none", linewidths=0, label="TN"),
}

CLUSTER_COLORS = [RWTH[c] for c in ("blue", "orange", "green", "violet", "turquoise", "bordeaux", "maygreen",
                                    "purple", "petrol", "magenta", "grey")]
# Figure 4b needs 13: the 11 base colours plus two tints.
CLUSTER_COLORS_13 = [*CLUSTER_COLORS, RWTH["blue50"], to_hex(S.green[2])]
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("rwth_seq", ["#ffffff", RWTH["blue"]])
CMAP_DIVERGING = LinearSegmentedColormap.from_list("rwth_div", [RWTH["blue"], "#ffffff", RWTH["red"]])

# Vehicle groups: colour plus marker, so identity never relies on hue alone.
GROUP = {
    "healthy": dict(color=HEALTHY, marker="o", label="Healthy"),
    "faulty_large": dict(color=FAULTY, marker="D", label="Faulty, large pack"),
    "faulty_small": dict(color=SMALL_PACK, marker="s", label="Faulty, small pack"),
}

# code_name -> "Label [unit]". Temperature channel is normalised per vehicle in vin_1.
FEATURE_LABELS: dict[str, str] = {
    "charge_capacity_per_100soc__avg_ah_per_100soc": "Charge capacity per 100 % SOC [Ah]",
    "current_fft__band1_mean_diff": "Current spectrum deviation, 0.01\u20130.5 h$^{-1}$, mean [A]",
    "current_fft__band1_std_diff": "Current spectrum deviation, 0.01\u20130.5 h$^{-1}$, std [A]",
    "current_fft__band2_std_diff": "Current spectrum deviation, 0.5\u20132.5 h$^{-1}$, std [A]",
    "soc_fft__band1_std_diff": "SOC spectrum deviation, 0.01\u20130.25 h$^{-1}$, std [\u2013]",
    "soc_fft__band2_std_diff": "SOC spectrum deviation, 0.25\u20137.5 h$^{-1}$, std [\u2013]",
    "soc_fft__band3_mean_diff": "SOC spectrum deviation, 7.5\u201330 h$^{-1}$, mean [\u2013]",
    "local_temp__min_val": "Max. temperature signal, normalised, minimum [\u2013]",
    "local_temp__avg_amp": "Max. temperature signal, normalised, mean amplitude [\u2013]",
    "local_temp__mean_diff": "Max. temperature spectrum deviation, normalised, \u2265 5 h$^{-1}$, mean [\u2013]",
    "pack_temp__mean_pack_temp": "Mean pack temperature [\u00b0C]",
    "resistance__std_of_std_R": "Variability of cell resistance spread [\u03a9]",
    "voltage_features__ev_std_std": "Variability of cell voltage spread [V]",
    "voltage_features__ev_std_max": "Maximum cell voltage spread [V]",
    "voltage_features__max_dif_max": "Maximum cell deviation from pack mean [V]",
}

# Same code name, different meaning in the windowed table.
WINDOWED_FEATURE_LABELS = {**FEATURE_LABELS,
                           "charge_capacity_per_100soc__avg_ah_per_100soc": "Relative charge capacity [\u2013]"}

# code_name -> (short title, "quantity [unit]") for windowed figures.
WINDOWED_SHORT = {
    "charge_capacity_per_100soc__avg_ah_per_100soc": ("Relative charge capacity", "Relative capacity [\u2013]"),
    "current_fft__band1_mean_diff": ("Current deviation mean, 0.01\u20130.5 h$^{-1}$", "Deviation [A]"),
    "current_fft__band1_std_diff": ("Current deviation std, 0.01\u20130.5 h$^{-1}$", "Std [A]"),
    "current_fft__band2_std_diff": ("Current deviation std, 0.5\u20132.5 h$^{-1}$", "Std [A]"),
    "soc_fft__band1_std_diff": ("SOC deviation std, 0.01\u20130.25 h$^{-1}$", "Std [\u2013]"),
    "soc_fft__band2_std_diff": ("SOC deviation std, 0.25\u20137.5 h$^{-1}$", "Std [\u2013]"),
    "soc_fft__band3_mean_diff": ("SOC deviation mean, 7.5\u201330 h$^{-1}$", "Deviation [\u2013]"),
    "local_temp__min_val": ("Max. temperature minimum", "Normalised temperature [\u2013]"),
    "local_temp__avg_amp": ("Max. temperature amplitude", "Normalised amplitude [\u2013]"),
    "local_temp__mean_diff": ("Max. temp. spectrum, \u2265 5 h$^{-1}$", "Normalised deviation [\u2013]"),
    "pack_temp__mean_pack_temp": ("Mean pack temperature", "Temperature [\u00b0C]"),
    "resistance__std_of_std_R": ("Cell resistance spread variability", "Resistance [\u03a9]"),
    "voltage_features__ev_std_std": ("Cell voltage spread variability", "Voltage [V]"),
    "voltage_features__ev_std_max": ("Maximum cell voltage spread", "Voltage [V]"),
    "voltage_features__max_dif_max": ("Maximum cell deviation", "Voltage [V]"),
}


def set_style() -> None:
    installed = {f.name for f in fm.fontManager.ttflist}
    families = [f for f in [FONT_FAMILY, *FONT_FALLBACKS] if f in installed]
    if not families:
        raise RuntimeError(f"none of {[FONT_FAMILY, *FONT_FALLBACKS]} is installed")
    if families[0] != FONT_FAMILY:
        print(f"rwth_style: {FONT_FAMILY} not installed, using {families[0]}")
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": families, "font.size": TICK_PT,
        "mathtext.default": "regular", "mathtext.fontset": "custom", "mathtext.rm": families[0],
        "mathtext.it": f"{families[0]}:italic", "mathtext.bf": f"{families[0]}:bold",
        "axes.labelsize": AXIS_LABEL_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT, "legend.frameon": False, "legend.handlelength": 1.4,
        "axes.titlesize": AXIS_LABEL_PT, "axes.linewidth": 0.6, "axes.edgecolor": RWTH["grey"],
        "axes.labelcolor": "black", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "axes.axisbelow": True, "grid.color": RWTH["black25"], "grid.linewidth": 0.4,
        "axes.prop_cycle": mpl.cycler(color=CLUSTER_COLORS),
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.color": RWTH["grey"], "ytick.color": RWTH["grey"],
        "xtick.labelcolor": "black", "ytick.labelcolor": "black",
        "lines.linewidth": 1.2, "lines.markersize": 3.5, "patch.linewidth": 0.5,
        "figure.constrained_layout.use": True, "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, text stays editable
    })


def panel_label(ax, letter: str) -> None:
    """(a), (b) ... above the top-left axes corner, bold 10 pt."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(-2, 5),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=PANEL_PT, fontweight="bold", annotation_clip=False)


def save_figure(fig, name: str, out_dir: Path = OUT_DIR) -> list[Path]:
    """Write <name>.pdf (vector) and <name>.png (300 dpi) at exact figsize."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    fig.savefig(paths[0], metadata={"CreationDate": None})
    fig.savefig(paths[1], dpi=300)
    return paths
