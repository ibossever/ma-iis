"""
Windowed RELATIVE charge capacity per EV -- standalone test script, does NOT
touch the pipeline (sliding_window_fleet_features.py,
filtered_features_windows*.pkl, capac2.py outputs all stay as they are).

window value      = mean ah_per_100soc of all qualifying charging sessions
                    inside the window (several sessions -> averaged first)
reference(ev)     = median of the EV's first N_REF VALID window values
                    (N_REF = 3 for 120h windows, 5 for 24h windows) --
                    median, so one bad charging session (BMS SoC
                    correction) in the reference can't shift it
relative capacity = window value / reference(ev), computed only for windows
                    AFTER the N_REF-th valid one (the reference windows
                    themselves get no value). Fixed once set -- no updating.
-> 1.0 = same capacity as at start, < 1.0 = capacity fade.

Charging sessions are found exactly as in capac2.py
(find_stable_charging_windows + MIN_SOC_INCREASE), but ONCE on each EV's
full recording (after normal_hour_h), not re-detected per window slice. A
session belongs to a window only if it lies fully inside
[max(window_start, start_of_operation), window_end), so a window never
gets a value from a charging session that was cut off at its boundary.

Also per window: charging C-rate = mean current of the window's sessions /
the EV's reference capacity (constant per EV, so a bad SoC correction can't
leak into it), and relative C-rate = C-rate / median C-rate of the same
reference windows. Note c_rate uses the reference capacity, which is only
known after the reference windows -- use it downstream only from there on.

Storage:
  - up to and including the N_REF-th valid window: rel_charge_capacity = NaN
  - after that, windows without a qualifying session carry the last valid
    rel_charge_capacity forward
  rel_charge_capacity_valid is True only for windows with a real measurement
  after the reference. EVs with <= N_REF valid windows stay NaN throughout.

Window grid is identical to sliding_window_fleet_features.py (starts at raw
index 0, fixed stride), so window_start/window_end line up with the
pipeline's rows and can be merged on (ev_id, window_start). Two configs:
    120h window / 12h stride, 24h window / 12h stride.

Session filter (FILTER_SESSIONS): before windowing, a session is dropped if
its SoC gain is more than MAX_SOC_GAIN_RATIO x (or less than 1/that) the
gain its charge explains with the EV's nominal pack capacity -- these are
BMS SoC corrections, not capacity. Filtered outputs carry OUTPUT_SUFFIX.

Stage 1 (slow, parallel, cached): per-EV qualifying charging sessions ->
    extracted_data/relative_capacity/charging_sessions.pkl
Stage 2 (fast): windowed feature per config, data checks, plots ->
    extracted_data/relative_capacity/rel_charge_capacity_windows_{120h,24h}.pkl
"""
import io
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import torch
import torch.storage

# the raw vin_*.pkl tensors were pickled on CUDA -- load them straight to CPU
# so parallel workers don't fill up (a possibly shared) GPU
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), weights_only=False, map_location="cpu")

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THIS_DIR))
import capac2  # noqa: E402  (find_stable_charging_windows, MIN_SOC_INCREASE, column indices)
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")
LABELS_FILE = DATA_ROOT / "Labels.xls"  # Num, Label (1 = faulty)
NORMAL_HOURS_IN = PROJECT_ROOT / "extracted_data" / "pack_current_features.pkl"  # normal_hour_h = start of operation

OUT_DIR = PROJECT_ROOT / "extracted_data" / "relative_capacity"
SESSIONS_CACHE = OUT_DIR / "charging_sessions.pkl"
FIG_DIR = THIS_DIR / "figures"

SAMPLE_INTERVAL_S = capac2.SAMPLE_INTERVAL_S
WINDOW_CONFIGS = {"120h": (120, 12, 3), "24h": (24, 12, 5)}  # name -> (window_hours, stride_hours, n_reference_windows)

N_WORKERS = 16

# session filter: drop sessions whose SoC gain disagrees with the charge that flowed (BMS SoC
# corrections). Outputs get OUTPUT_SUFFIX so the unfiltered figures/tables are kept.
FILTER_SESSIONS = True
MAX_SOC_GAIN_RATIO = 1.5       # keep if measured / expected SoC gain is within [1/1.5, 1.5]
SMALL_PACK_MAX_AH = 60         # per-EV lifetime median below this -> small pack (~43), else large (~85)
OUTPUT_SUFFIX = "_filtered" if FILTER_SESSIONS else ""

# data checks (report only -- nothing is filtered on these)
MAX_SOC_RATE_PER_H = 2.0      # a session gaining SoC faster than 2C (200 %/h) is physically implausible -> SoC jump/reset
FLAG_REFERENCE_SPREAD = 0.20  # reference windows differing by > 20 % (max/min - 1) is not real ageing
FLAG_TABLE_ROWS = 15

# trend (for the faulty-subgroup view)
MIN_VALID_FOR_SLOPE = 5
MIN_SPAN_H_FOR_SLOPE = 72

PLOT_BIN_HOURS = 48       # time-bin width for the group median / IQR band
PLOT_MIN_EVS_PER_BIN = 5  # don't draw a median from fewer EVs than this
PLOT_YLIM = (0.6, 1.4)
PLOT_YLIM_ABS = (0, 140)  # Ah per 100 % SoC
PLOT_YLIM_C_RATE = (0.02, 2.0)      # log axis: slow AC charging ~0.05 C up to DC fast charging ~1-1.5 C
PLOT_YLIM_REL_C_RATE = (0.2, 5.0)   # log axis
N_TOP_VARYING = 16        # healthy EVs shown per figure (most varying by std)

GROUPS = [(0, "healthy", RWTHColorScales.blue), (1, "faulty", RWTHColorScales.red)]


def h2idx(hours: float) -> int:
    return int(round(hours * 3600.0 / SAMPLE_INTERVAL_S))


def idx2h(idx) -> float:
    return idx * SAMPLE_INTERVAL_S / 3600.0


# ===========================================================================
# STAGE 1 -- per-EV qualifying charging sessions
# ===========================================================================
def ev_sessions(args: tuple[int, float]) -> tuple[int, int, list[dict]]:
    """All stable charging sessions after start of operation with a SoC rise
    >= MIN_SOC_INCREASE, with absolute raw indices. Returns (ev_id, n_samples, sessions)."""
    ev_id, normal_hour_h = args
    data = capac2.load_vin(ev_id)
    anchor = h2idx(normal_hour_h)
    current = data[anchor:, capac2.CURRENT_COL]

    sessions = []
    for s, e in capac2.find_stable_charging_windows(current):
        value = capac2.window_ah_per_100soc(data[anchor:], s, e)
        if value is None:
            continue
        sessions.append({
            "ev_id": ev_id,
            "start": anchor + s,
            "end": anchor + e,
            "soc_start": float(data[anchor + s, capac2.SOC_COL]),
            "soc_end": float(data[anchor + e - 1, capac2.SOC_COL]),
            "ah_per_100soc": value,
        })
    return ev_id, data.shape[0], sessions


def load_or_build_sessions(normal_hours: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if SESSIONS_CACHE.exists():
        cache = pd.read_pickle(SESSIONS_CACHE)
        return cache["sessions"], cache["n_samples"]

    ev_ids = sorted(int(p.name) for p in DATA_ROOT.iterdir()
                    if p.is_dir() and (p / capac2.VIN_FILE).exists() and int(p.name) in normal_hours.index)
    rows, n_samples = [], {}
    with Pool(N_WORKERS) as pool:
        for i, (ev_id, n, sessions) in enumerate(
                pool.imap_unordered(ev_sessions, [(ev, normal_hours.loc[ev]) for ev in ev_ids]), 1):
            rows.extend(sessions)
            n_samples[ev_id] = n
            if i % 50 == 0:
                print(f"  ...{i}/{len(ev_ids)} EVs")

    sessions = pd.DataFrame(rows).sort_values(["ev_id", "start"]).reset_index(drop=True)
    n_samples = pd.Series(n_samples, name="n_samples").sort_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"sessions": sessions, "n_samples": n_samples}, SESSIONS_CACHE)
    print(f"cached {len(sessions)} sessions for {len(n_samples)} EVs -> {SESSIONS_CACHE}")
    return sessions, n_samples


def add_session_plausibility(sessions: pd.DataFrame) -> pd.DataFrame:
    sessions = sessions.copy()
    duration_h = idx2h(sessions["end"] - sessions["start"])
    sessions["soc_rate_per_h"] = (sessions["soc_end"] - sessions["soc_start"]) / duration_h
    sessions["implausible"] = sessions["soc_rate_per_h"] > MAX_SOC_RATE_PER_H
    # Ah charged = ah_per_100soc * SoC gained  ->  mean |current| over the session
    sessions["mean_current_a"] = sessions["ah_per_100soc"] * (sessions["soc_end"] - sessions["soc_start"]) / duration_h
    return sessions


def add_session_filter(sessions: pd.DataFrame) -> pd.DataFrame:
    """expected SoC gain = Ah charged / the EV's nominal pack capacity, where
    nominal = fleet median of its pack type (small ~43 / large ~85 Ah per
    100 % SoC, pack type from the EV's lifetime median session -- a fixed
    vehicle attribute, and robust to the EV's own bad early sessions).
    soc_gain_ratio = measured / expected SoC gain = nominal / ah_per_100soc."""
    sessions = sessions.copy()
    ev_median = sessions.groupby("ev_id")["ah_per_100soc"].median()
    small = ev_median < SMALL_PACK_MAX_AH
    nominal = pd.Series(np.where(small, ev_median[small].median(), ev_median[~small].median()), ev_median.index)
    sessions["pack_nominal_ah"] = sessions["ev_id"].map(nominal)
    sessions["soc_gain_ratio"] = sessions["pack_nominal_ah"] / sessions["ah_per_100soc"]
    sessions["accepted"] = sessions["soc_gain_ratio"].between(1 / MAX_SOC_GAIN_RATIO, MAX_SOC_GAIN_RATIO)
    return sessions


# ===========================================================================
# STAGE 2 -- windowed relative capacity
# ===========================================================================
def relative_to_reference(values: np.ndarray, n_ref: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one EV's window series (NaN = no measurement): reference = median of
    the first n_ref valid values; relative value only for valid windows after
    them, carried forward over NaN windows. Returns (reference, rel, is_reference_window)."""
    valid_idx = np.flatnonzero(~np.isnan(values))
    reference = np.full(len(values), np.nan)
    rel = np.full(len(values), np.nan)
    is_ref = np.zeros(len(values), dtype=bool)
    is_ref[valid_idx[:n_ref]] = True
    if len(valid_idx) <= n_ref:
        return reference, rel, is_ref

    ref_value = np.median(values[valid_idx[:n_ref]])
    after = valid_idx[n_ref:]
    reference[valid_idx[n_ref - 1]:] = ref_value
    rel[after] = values[after] / ref_value
    rel = pd.Series(rel).ffill().to_numpy()
    return reference, rel, is_ref


def windowed_relative_capacity(sessions: pd.DataFrame, n_samples: pd.Series, normal_hours: pd.Series,
                               labels: pd.Series, window_h: float, stride_h: float, n_ref: int) -> pd.DataFrame:
    win, stride = h2idx(window_h), h2idx(stride_h)
    by_ev = dict(tuple(sessions.groupby("ev_id")))

    frames = []
    for ev_id, n in n_samples.items():
        anchor = h2idx(normal_hours.loc[ev_id])
        ev_s = by_ev.get(ev_id, sessions.iloc[:0])
        starts, ends = ev_s["start"].to_numpy(), ev_s["end"].to_numpy()
        vals, implausible = ev_s["ah_per_100soc"].to_numpy(), ev_s["implausible"].to_numpy()
        currents = ev_s["mean_current_a"].to_numpy()

        rows = []
        for pos in range(0, n - win + 1, stride):
            w_end = pos + win
            inside = (starts >= max(pos, anchor)) & (ends <= w_end)
            rows.append({
                "ev_id": ev_id,
                "label": int(labels.get(ev_id, -1)),
                "window_start": pos,
                "window_end": w_end,
                "window_start_hours": idx2h(pos - anchor),  # relative to start of operation
                "window_end_hours": idx2h(w_end - anchor),
                "n_sessions": int(inside.sum()),
                "n_implausible_sessions": int(implausible[inside].sum()),
                "ah_per_100soc": float(vals[inside].mean()) if inside.any() else np.nan,
                "mean_charge_current_a": float(currents[inside].mean()) if inside.any() else np.nan,
            })
        if not rows:
            continue
        ev_df = pd.DataFrame(rows)
        ref, rel, is_ref = relative_to_reference(ev_df["ah_per_100soc"].to_numpy(), n_ref)
        ev_df["reference_ah_per_100soc"] = ref
        ev_df["is_reference_window"] = is_ref
        ev_df["rel_charge_capacity"] = rel
        ev_df["rel_charge_capacity_valid"] = ev_df["ah_per_100soc"].notna() & ~is_ref & ~np.isnan(ref)

        # C-rate = mean charging current / the EV's reference capacity (constant per EV, so bad
        # SoC corrections don't leak into it); relative C-rate uses the same reference windows
        cap_ref = ref[~np.isnan(ref)][0] if (~np.isnan(ref)).any() else np.nan
        ev_df["c_rate"] = ev_df["mean_charge_current_a"] / cap_ref
        ref_c, rel_c, _ = relative_to_reference(ev_df["c_rate"].to_numpy(), n_ref)
        ev_df["reference_c_rate"] = ref_c
        ev_df["rel_c_rate"] = rel_c
        ev_df["rel_c_rate_valid"] = ev_df["c_rate"].notna() & ~is_ref & ~np.isnan(ref_c)
        frames.append(ev_df)
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# STATISTICS
# ===========================================================================
def check_reference_windows(df: pd.DataFrame, name: str, n_ref: int):
    """Report EVs whose reference windows look like a data problem: large
    spread between them, or implausible sessions inside them."""
    ref = df[df.is_reference_window]
    g = ref.groupby("ev_id")
    measured = df[df.rel_charge_capacity_valid].groupby("ev_id").rel_charge_capacity
    summary = pd.DataFrame({
        "label": g.label.first(),
        "n_ref": g.size(),
        "values_ah": g.ah_per_100soc.apply(lambda v: " / ".join(f"{x:.1f}" for x in v)),
        "spread": g.ah_per_100soc.max() / g.ah_per_100soc.min() - 1,
        "implausible_sessions": g.n_implausible_sessions.sum(),
        "rel_min_later": measured.min(),
        "rel_max_later": measured.max(),
    })
    no_value = summary[summary.n_ref < n_ref].index.union(summary[summary.rel_min_later.isna()].index)
    first = df[df.rel_charge_capacity_valid].groupby("ev_id").window_end_hours.first()
    flagged = summary[(summary.spread > FLAG_REFERENCE_SPREAD) | (summary.implausible_sessions > 0)]

    print(f"\n=== {name}: reference = median of first {n_ref} valid windows ===")
    print(f"EVs with a relative value: {len(first)}/{df.ev_id.nunique()}  "
          f"(no value -- <= {n_ref} valid windows: {sorted(set(df.ev_id) - set(first.index))})")
    q = first.quantile([0, 0.25, 0.5, 0.75, 1])
    print(f"first relative value, window end h after start of operation: min {q[0]:.0f} | p25 {q[0.25]:.0f} | "
          f"median {q[0.5]:.0f} | p75 {q[0.75]:.0f} | max {q[1]:.0f}")
    print(f"spread (max/min - 1) of reference windows: median {summary.spread.median():.1%}, "
          f"p90 {summary.spread.quantile(0.9):.1%}, max {summary.spread.max():.0%}")
    print(f"flagged EVs (spread > {FLAG_REFERENCE_SPREAD:.0%} or an implausible session with SoC rate > "
          f"{MAX_SOC_RATE_PER_H:.0%}/h): {len(flagged)} "
          f"(healthy {(flagged.label == 0).sum()}, faulty {(flagged.label == 1).sum()}); worst {FLAG_TABLE_ROWS}:")
    if len(flagged):
        print(flagged.sort_values("spread", ascending=False).head(FLAG_TABLE_ROWS).to_string(
            formatters={"spread": "{:.0%}".format, "rel_min_later": "{:.2f}".format, "rel_max_later": "{:.2f}".format}))


def ev_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per EV, over its measured windows: linear trend (% per 100 h), spread
    (std) and typical window-to-window change (median |step|, robust to single
    spikes)."""
    rows = []
    for ev_id, ev in df[df.rel_charge_capacity_valid].groupby("ev_id"):
        span = ev.window_end_hours.max() - ev.window_end_hours.min()
        if len(ev) < MIN_VALID_FOR_SLOPE or span < MIN_SPAN_H_FOR_SLOPE:
            continue
        y = ev.rel_charge_capacity
        rows.append({"ev_id": ev_id, "label": ev.label.iloc[0],
                     "slope_pct_per_100h": np.polyfit(ev.window_end_hours, y, 1)[0] * 1e4,
                     "std": y.std(), "median_step": y.diff().abs().median(),
                     "rel_last": y.iloc[-3:].median()})
    return pd.DataFrame(rows).set_index("ev_id")


def print_ev_stats_summary(stats: pd.DataFrame, name: str) -> float:
    healthy, faulty = stats[stats.label == 0], stats[stats.label == 1]
    thr = healthy.slope_pct_per_100h.quantile(0.05)
    below = faulty[faulty.slope_pct_per_100h < thr].sort_values("slope_pct_per_100h")
    print(f"\n=== {name}: relative charge capacity per EV (EVs with >= {MIN_VALID_FOR_SLOPE} valid windows) ===")
    print("  trend, % per 100 h:")
    for lab_name, sel in [("healthy", healthy), ("faulty", faulty)]:
        q = sel.slope_pct_per_100h.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"    {lab_name:<8} n={len(sel):<4} p5 {q[0.05]:+.2f} | p25 {q[0.25]:+.2f} | median {q[0.5]:+.2f} | "
              f"p75 {q[0.75]:+.2f} | p95 {q[0.95]:+.2f}")
    print("  variability (std of measured values | median window-to-window change):")
    for lab_name, sel in [("healthy", healthy), ("faulty", faulty)]:
        print(f"    {lab_name:<8} std median {sel['std'].median():.3f}, p90 {sel['std'].quantile(0.9):.3f} | "
              f"step median {sel.median_step.median():.3f}, p90 {sel.median_step.quantile(0.9):.3f}")
    print(f"  faulty EVs declining faster than 95% of healthy (slope < {thr:+.2f}): "
          f"{len(below)}/{len(faulty)} = {len(below) / max(len(faulty), 1):.0%}  (5% expected by chance)")
    if len(below):
        print("    " + ", ".join(f"EV {ev} ({r.slope_pct_per_100h:+.1f}, last {r.rel_last:.2f})" for ev, r in below.iterrows()))
    return thr


# ===========================================================================
# PLOTS
# ===========================================================================
def group_median(df: pd.DataFrame, col: str = "rel_charge_capacity", valid: str = "rel_charge_capacity_valid") -> pd.DataFrame:
    """Per time bin: one value per EV (its median in the bin, measured windows
    only), then median and 25th/75th percentile across EVs."""
    df = df[df[valid]]
    b = (df.window_end_hours // PLOT_BIN_HOURS) * PLOT_BIN_HOURS + PLOT_BIN_HOURS / 2
    per_ev = df.groupby([b.rename("bin"), "ev_id"])[col].median().dropna()
    g = per_ev.groupby("bin")
    out = pd.DataFrame({"median": g.median(), "q25": g.quantile(0.25), "q75": g.quantile(0.75), "n": g.size()})
    return out[out.n >= PLOT_MIN_EVS_PER_BIN]


def set_log_y(ax):
    """Log y-axis with plain-number tick labels (0.1, 0.2, 1, 2 ...)."""
    ax.set_yscale("log")
    for axis in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
        axis(FuncFormatter(lambda v, _: f"{v:g}"))


def style_axis(ax, ylim: tuple[float, float] = PLOT_YLIM, log: bool = False):
    ax.axhline(1.0, color="0.6", lw=0.8)
    if log:
        set_log_y(ax)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_fleet(windows: dict[str, pd.DataFrame], out_path: Path, col: str = "rel_charge_capacity",
               valid: str = "rel_charge_capacity_valid", ylabel: str = "relative charge capacity [-]",
               ylim: tuple[float, float] = PLOT_YLIM, log: bool = False):
    """Per window config: every EV as its own line, healthy and faulty in the
    same axes, with each group's median + IQR band and a marker where each
    EV's recording ends. Below it: how many EVs are still recording."""
    fig, axes = plt.subplots(2, len(windows), figsize=(8 * len(windows), 8.5), sharex="col",
                             gridspec_kw={"height_ratios": [3.2, 1]}, squeeze=False, layout="constrained")

    for c, (name, df) in enumerate(windows.items()):
        ax, ax_n = axes[0, c], axes[1, c]
        n_ref = WINDOW_CONFIGS[name][2]
        stored = df.dropna(subset=[col])
        x_max = df.window_end_hours.max()

        for lab, _, color in GROUPS:  # healthy first, faulty drawn on top
            for _, ev in stored[stored.label == lab].groupby("ev_id"):
                ax.plot(ev.window_end_hours, ev[col], color=color[2 if lab == 0 else 1],
                        lw=0.5 if lab == 0 else 0.7, alpha=0.35 if lab == 0 else 0.6, zorder=1 + lab)
        for lab, _, color in GROUPS:
            m = group_median(df[df.label == lab], col, valid)
            ax.fill_between(m.index, m.q25, m.q75, color=color[1], alpha=0.25, lw=0, zorder=3)
            ax.plot(m.index, m["median"], color=color[0], lw=2.2, zorder=4)
        for lab, _, color in GROUPS:  # end of each EV's recording, clipped into the y-range so none get lost
            ends = stored[stored.label == lab].groupby("ev_id").last()
            ax.scatter(ends.window_end_hours, ends[col].clip(*ylim), s=16,
                       marker="o" if lab == 0 else "X", color=color[0], edgecolor="white", linewidth=0.5, zorder=5)
        style_axis(ax, ylim, log)
        ax.set_title(f"{name} window, reference = median of first {n_ref} valid windows", fontsize=11)

        # EVs still recording (per group), at each window end on a 12 h grid
        grid = np.arange(0, x_max + 12, 12)
        last_end = df.groupby("ev_id").agg(label=("label", "first"), end=("window_end_hours", "max"))
        for lab, lab_name, color in GROUPS:
            ends = last_end[last_end.label == lab].end.to_numpy()
            ax_n.step(grid, (ends[None, :] >= grid[:, None]).sum(1), where="post", color=color[0], lw=1.5)
        ax_n.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax_n.spines[s].set_visible(False)
        ax_n.set_xlabel("window end, hours after start of operation")
        ax_n.set_xlim(0, x_max)
        if c == 0:
            ax.set_ylabel(ylabel)
            ax_n.set_ylabel("EVs still recording")

    legend = [
        Line2D([], [], color=RWTHColorScales.blue[2], lw=1, label="healthy EV"),
        Line2D([], [], color=RWTHColorScales.red[1], lw=1, label="faulty EV"),
        Line2D([], [], color=RWTHColorScales.blue[0], lw=2.2, label="healthy median (band = IQR)"),
        Line2D([], [], color=RWTHColorScales.red[0], lw=2.2, label="faulty median (band = IQR)"),
        Line2D([], [], color=RWTHColorScales.blue[0], marker="o", ls="", label="end of healthy EV's recording"),
        Line2D([], [], color=RWTHColorScales.red[0], marker="X", ls="", label="end of faulty EV's recording"),
    ]
    fig.legend(handles=legend, loc="outside upper center", ncol=3, fontsize=9, frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def plot_absolute(windows: dict[str, pd.DataFrame], out_path: Path, col: str = "ah_per_100soc",
                  ylabel: str = "charge capacity [Ah per 100 % SoC]", ylim: tuple[float, float] = PLOT_YLIM_ABS,
                  log: bool = False):
    """Absolute window capacity (Ah per 100 % SoC) of every EV over its whole
    recording, measured windows only (reference windows included), with a
    marker at each EV's first and last measurement. Right strip: distribution
    of the starting values per group."""
    fig, axes = plt.subplots(1, 2 * len(windows), figsize=(9 * len(windows), 6.5), sharey=True,
                             gridspec_kw={"width_ratios": [5, 1] * len(windows)}, layout="constrained")
    bins = np.geomspace(*ylim, 57) if log else np.linspace(*ylim, 57)

    for c, (name, df) in enumerate(windows.items()):
        ax, ax_h = axes[2 * c], axes[2 * c + 1]
        measured = df.dropna(subset=[col])

        for lab, _, color in GROUPS:  # healthy first, faulty drawn on top
            for _, ev in measured[measured.label == lab].groupby("ev_id"):
                ax.plot(ev.window_end_hours, ev[col], color=color[2 if lab == 0 else 1],
                        lw=0.5 if lab == 0 else 0.7, alpha=0.35 if lab == 0 else 0.6, zorder=1 + lab)
        for lab, _, color in GROUPS:  # clipped into the y-range so none get lost
            g = measured[measured.label == lab].groupby("ev_id")
            starts, ends = g.first(), g.last()
            ax.scatter(starts.window_end_hours, starts[col].clip(*ylim), s=22, marker="^",
                       color=color[0], edgecolor="white", linewidth=0.5, zorder=5)
            ax.scatter(ends.window_end_hours, ends[col].clip(*ylim), s=16,
                       marker="o" if lab == 0 else "X", color=color[0], edgecolor="white", linewidth=0.5, zorder=5)
            ax_h.hist(starts[col].clip(*ylim), bins=bins, orientation="horizontal",
                      color=color[0], alpha=0.6, label=f"{len(starts)}")

        if log:
            set_log_y(ax)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, df.window_end_hours.max())
        ax.set_title(f"{name} window", fontsize=11)
        ax.set_xlabel("window end, hours after start of operation")
        ax_h.set_title("starting values", fontsize=9)
        ax_h.set_xlabel("EVs")
        for a in (ax, ax_h):
            a.grid(alpha=0.25, lw=0.5)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    axes[0].set_ylabel(ylabel)

    legend = [
        Line2D([], [], color=RWTHColorScales.blue[2], lw=1, label="healthy EV"),
        Line2D([], [], color=RWTHColorScales.red[1], lw=1, label="faulty EV"),
        Line2D([], [], color="0.3", marker="^", ls="", label="first measurement"),
        Line2D([], [], color=RWTHColorScales.blue[0], marker="o", ls="", label="last measurement, healthy"),
        Line2D([], [], color=RWTHColorScales.red[0], marker="X", ls="", label="last measurement, faulty"),
    ]
    fig.legend(handles=legend, loc="outside upper center", ncol=5, fontsize=9, frameon=False)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"saved plot: {out_path}")


def plot_per_ev(df: pd.DataFrame, stats: pd.DataFrame, slope_thr: float, name: str,
                label: int, ev_order: list[int], subtitle: str, n_cols: int, out_path: Path):
    """One panel per EV in ev_order, on top of the healthy fleet's median and IQR band."""
    healthy_band = group_median(df[df.label == 0])
    color = RWTHColorScales.red if label == 1 else RWTHColorScales.blue
    lab_name = "Faulty" if label == 1 else "Healthy"

    n_rows = int(np.ceil(len(ev_order) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.1 * n_rows + 0.8), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    x_max = df[df.rel_charge_capacity_valid].window_end_hours.quantile(0.98)

    for ax, ev_id in zip(axes.flat, ev_order):
        ev = df[df.ev_id == ev_id]
        measured = ev[ev.rel_charge_capacity_valid]
        ax.fill_between(healthy_band.index, healthy_band.q25, healthy_band.q75, color="0.85", lw=0)
        ax.plot(healthy_band.index, healthy_band["median"], color="0.55", lw=1)
        ax.plot(ev.window_end_hours, ev.rel_charge_capacity, color=color[1], lw=1)
        ax.plot(measured.window_end_hours, measured.rel_charge_capacity, "o", ms=2, color=color[0])
        style_axis(ax)
        ax.set_xlim(0, x_max)

        declining = False
        if ev_id in stats.index:
            st = stats.loc[ev_id]
            declining = st.slope_pct_per_100h < slope_thr
            title = f"EV {ev_id}  {st.slope_pct_per_100h:+.1f}%/100h  sd {st['std']:.2f}" + (" *" if declining else "")
        else:
            title = f"EV {ev_id}  (too few windows)"
        y = measured.rel_charge_capacity
        off_scale = y.lt(PLOT_YLIM[0]).any() or y.gt(PLOT_YLIM[1]).any()
        ax.set_title(title + ("  [off-scale]" if off_scale else ""), fontsize=7,
                     fontweight="bold" if declining else "normal")
    for ax in axes.flat[len(ev_order):]:
        ax.set_visible(False)

    fig.supxlabel("window end, hours after start of operation")
    fig.supylabel("relative charge capacity [-]")
    fig.suptitle(f"{lab_name} EVs, relative charge capacity, {name} window -- {subtitle}\n"
                 f"dots = measured, line = stored (carried forward); grey = healthy median + IQR; "
                 f"sd = std of measured values\n"
                 f"* = declines faster than 95% of healthy EVs ({slope_thr:+.1f}%/100h), "
                 f"[off-scale] = has values outside the y-range", fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved plot: {out_path}")


if __name__ == "__main__":
    normal_hours = pd.read_pickle(NORMAL_HOURS_IN)["normal_hour_h"]
    labels = pd.read_excel(LABELS_FILE).set_index("Num")["Label"]

    print("Stage 1: charging sessions")
    sessions, n_samples = load_or_build_sessions(normal_hours)
    sessions = add_session_plausibility(sessions)
    if FILTER_SESSIONS:
        sessions = add_session_filter(sessions)
        rejected = sessions[~sessions.accepted]
        too_fast, too_slow = (rejected.soc_gain_ratio > 1).sum(), (rejected.soc_gain_ratio < 1).sum()
        print(f"session filter: rejected {len(rejected)}/{len(sessions)} sessions in {rejected.ev_id.nunique()} EVs "
              f"(SoC gain > {MAX_SOC_GAIN_RATIO}x expected: {too_fast}, < 1/{MAX_SOC_GAIN_RATIO}x: {too_slow}); "
              f"including {rejected.implausible.sum()}/{sessions.implausible.sum()} of the > {MAX_SOC_RATE_PER_H:.0%}/h ones")
        sessions = sessions[sessions.accepted]
    print(f"sessions with SoC rate > {MAX_SOC_RATE_PER_H:.0%}/h (implausible, NOT filtered): "
          f"{sessions.implausible.sum()}/{len(sessions)} in {sessions[sessions.implausible].ev_id.nunique()} EVs")

    print("Stage 2: windowed relative capacity")
    windows = {name: windowed_relative_capacity(sessions, n_samples, normal_hours, labels, w, s, n_ref)
               for name, (w, s, n_ref) in WINDOW_CONFIGS.items()}

    for name, df in windows.items():
        check_reference_windows(df, name, WINDOW_CONFIGS[name][2])
        stats = ev_stats(df)
        thr = print_ev_stats_summary(stats, name)

        faulty_order = stats[stats.label == 1].sort_values("slope_pct_per_100h").index.tolist()
        faulty_order += sorted(set(df[df.label == 1].ev_id) - set(faulty_order))  # too few windows -> at the end
        plot_per_ev(df, stats, thr, name, 1, faulty_order, "sorted by trend (steepest decline first)", 8,
                    FIG_DIR / f"relative_charge_capacity_faulty_per_ev_{name}{OUTPUT_SUFFIX}.png")

        healthy_order = stats[stats.label == 0].sort_values("std", ascending=False).index[:N_TOP_VARYING].tolist()
        plot_per_ev(df, stats, thr, name, 0, healthy_order,
                    f"top {N_TOP_VARYING} most varying (by sd) of {(stats.label == 0).sum()}", 4,
                    FIG_DIR / f"relative_charge_capacity_healthy_top{N_TOP_VARYING}_varying_{name}{OUTPUT_SUFFIX}.png")

        out_path = OUT_DIR / f"rel_charge_capacity_windows_{name}{OUTPUT_SUFFIX}.pkl"
        df.drop(columns="label").to_pickle(out_path)
        print(f"saved {len(df)} windows -> {out_path}")

    plot_fleet(windows, FIG_DIR / f"relative_charge_capacity_fleet{OUTPUT_SUFFIX}.png")
    plot_absolute(windows, FIG_DIR / f"charge_capacity_absolute_fleet{OUTPUT_SUFFIX}.png")

    for name, df in windows.items():
        ev = df.groupby("ev_id").agg(label=("label", "first"), cap=("ah_per_100soc", "median"),
                                     current=("mean_charge_current_a", "median"), c_rate=("c_rate", "median"))
        ev["pack"] = np.where(ev.cap < 60, "small", "large")
        print(f"\n=== {name}: charging C-rate, per-EV median, by pack size and label ===")
        print(ev.groupby(["pack", "label"]).agg(n=("c_rate", "size"), current_a=("current", "median"),
                                                c_rate=("c_rate", "median")).round(3).to_string())
    plot_fleet(windows, FIG_DIR / f"relative_c_rate_fleet{OUTPUT_SUFFIX}.png", "rel_c_rate", "rel_c_rate_valid",
               "relative charging C-rate [-] (log)", PLOT_YLIM_REL_C_RATE, log=True)
    plot_absolute(windows, FIG_DIR / f"c_rate_absolute_fleet{OUTPUT_SUFFIX}.png", "c_rate",
                  "charging C-rate [1/h] (log)", PLOT_YLIM_C_RATE, log=True)
