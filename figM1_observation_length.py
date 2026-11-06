"""Methodology figure: recorded length per vehicle by group (ECDF), window length marked."""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
import rwth_style as st

WINDOW_HOURS = 120
v = common.load_whole_session()["vehicles"]
hours = pd.read_pickle(common.DATA / "relative_capacity" / "charging_sessions.pkl")["n_samples"].reindex(v.index) * common.SAMPLE_INTERVAL_S / 3600.0
st.set_style()


def plot_ecdf(curves, name):
    """Draw (values, color, linestyle, label) curves as ECDFs and save the figure under name."""
    fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 6.2 * st.CM))
    for values, color, ls, label in curves:
        values = np.sort(values)
        ax.step(values, 100 * np.arange(1, len(values) + 1) / len(values), where="post", color=color, ls=ls, label=f"{label} (n = {len(values)})")
    ax.axvline(WINDOW_HOURS, color="black", lw=0.8, ls=":", label=f"Window length ({WINDOW_HOURS} h)")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 1500], ["50", "100", "200", "500", "1000", "1500"])
    ax.minorticks_off()
    ax.set_xlabel("Recorded length per vehicle [h]")
    ax.set_ylabel("Share of vehicles [%]")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend(loc="lower right")
    print(st.save_figure(fig, name))


plot_ecdf([(hours[v["group"] == group].to_numpy(), st.GROUP[group]["color"], ls, st.GROUP[group]["label"])
           for group, ls in (("healthy", "-"), ("faulty_large", "--"), ("faulty_small", "-."))], "figM1_observation_length")
plot_ecdf([(hours[~v["faulty"]].to_numpy(), st.HEALTHY, "-", "Healthy"),
           (hours[v["faulty"]].to_numpy(), st.FAULTY, "--", "Faulty")], "figM1b_observation_length_healthy_faulty")
print(hours.describe()[["min", "50%", "max"]].round(0).to_dict(), hours.groupby(v["group"]).median().round(0).to_dict(), "below 120 h:", int((hours < WINDOW_HOURS).sum()))
print(hours.groupby(v["faulty"]).describe()[["count", "min", "50%", "max"]].round(0).to_dict("index"), "below 120 h by faulty:", (hours < WINDOW_HOURS).groupby(v["faulty"]).sum().to_dict(), "missing:", int(hours.isna().sum()))
