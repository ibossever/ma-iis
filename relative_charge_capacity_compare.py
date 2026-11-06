"""
Unfiltered vs session-filtered relative charge capacity for selected EVs, one
panel per EV. Reads the tables saved by relative_charge_capacity.py (run it
with FILTER_SESSIONS = False and True first), nothing is recomputed except the
session filter flags, used only to mark where rejected sessions were.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import relative_charge_capacity as rcc  # noqa: E402

EV_IDS = [220, 47]
WINDOW = "24h"
OUT_PATH = rcc.FIG_DIR / f"relative_charge_capacity_compare_ev{'_'.join(map(str, EV_IDS))}_{WINDOW}.png"


if __name__ == "__main__":
    old = pd.read_pickle(rcc.OUT_DIR / f"rel_charge_capacity_windows_{WINDOW}.pkl")
    new = pd.read_pickle(rcc.OUT_DIR / f"rel_charge_capacity_windows_{WINDOW}_filtered.pkl")
    sessions = rcc.add_session_filter(rcc.add_session_plausibility(pd.read_pickle(rcc.SESSIONS_CACHE)["sessions"]))
    normal_hours = pd.read_pickle(rcc.NORMAL_HOURS_IN)["normal_hour_h"]
    blue = rcc.RWTHColorScales.blue

    fig, axes = plt.subplots(1, len(EV_IDS), figsize=(7 * len(EV_IDS), 4.6), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, ev_id in zip(axes.flat, EV_IDS):
        anchor_h = normal_hours.loc[ev_id]
        for df, label, color, ls, marker in [(old, "unfiltered", "0.45", "--", "o"),
                                             (new, "session-filtered", blue[0], "-", "o")]:
            ev = df[df.ev_id == ev_id]
            measured = ev[ev.rel_charge_capacity_valid]
            ax.plot(ev.window_end_hours, ev.rel_charge_capacity, color=color, ls=ls, lw=1.3, label=label)
            ax.plot(measured.window_end_hours, measured.rel_charge_capacity, marker, ms=4, color=color,
                    mfc="white" if ls == "--" else color)

        # rejected sessions, at their end time (when they enter a window)
        rejected = sessions[(sessions.ev_id == ev_id) & ~sessions.accepted]
        for i, (_, s) in enumerate(rejected.iterrows()):
            ax.axvline(rcc.idx2h(s.end) - anchor_h, color=rcc.RWTHColorScales.red[0], lw=1, alpha=0.7,
                       label="rejected session ends" if i == 0 else None)
            ax.annotate(f"{rcc.idx2h(s.end - s.start):.1f} h, SoC {s.soc_start:.0%}->{s.soc_end:.0%}\n"
                        f"{s.soc_gain_ratio:.1f}x expected gain",
                        (rcc.idx2h(s.end) - anchor_h, 0.02 + 0.13 * i), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points", fontsize=7, color="0.25")

        ax.axhline(1.0, color="0.6", lw=0.8)
        ax.set_title(f"EV {ev_id}, {WINDOW} window", fontsize=11)
        ax.set_xlabel("window end, hours after start of operation")
        ax.grid(alpha=0.25, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    axes[0, 0].set_ylabel("relative charge capacity [-]")
    fig.suptitle("Relative charge capacity, unfiltered vs session-filtered "
                 "(filled dots = measured window, line = stored incl. carried forward)", fontsize=11)
    fig.savefig(OUT_PATH, dpi=250)
    print(f"saved plot: {OUT_PATH}")
    for ev_id in EV_IDS:
        for name, df in [("unfiltered", old), ("filtered", new)]:
            v = df[(df.ev_id == ev_id) & df.rel_charge_capacity_valid].rel_charge_capacity
            print(f"EV {ev_id} {name:10}: {len(v)} measured windows, min {v.min():.3f}, max {v.max():.3f}")
