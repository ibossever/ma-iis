# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402

DATA_ROOT = Path("/media/ssd_2tb/jak-iis")
OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"
OUT_DIR.mkdir(exist_ok=True)
CAR_IDS = [101, 134, 276]  # QAS car ids to plot, 0-392 (see QAS_data_format.md) - edit freely
CURRENT_COL = 224
SAMPLE_INTERVAL_S = 30.0
ZOOM_START_H = 150.0  # zoomed subplot window, in operational hours - edit freely
ZOOM_END_H = 310.0


def get_pack_currents(car_ids):
    """Algorithm: returns one pack-current array per car. Swap this out to change what gets plotted."""
    currents = []
    for car_id in car_ids:
        tensor = pd.read_pickle(DATA_ROOT / str(car_id) / "vin_2.pkl")
        matrix = tensor.cpu().numpy()
        currents.append(matrix[:, CURRENT_COL])
    return currents


def plot_pack_currents(car_ids, currents):
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))
    zoom_values = []
    for car_id, current, color in zip(car_ids, currents, RWTHColorScales.cycle):
        hours = np.arange(len(current)) * SAMPLE_INTERVAL_S / 3600.0
        ax_full.plot(hours, current, color=color, label=f"EV {car_id}")
        ax_zoom.plot(hours, current, color=color, label=f"EV {car_id}")

        in_zoom = (hours >= ZOOM_START_H) & (hours <= ZOOM_END_H)
        zoom_values.append(current[in_zoom])

    ax_full.set_xlabel("Operational hours")
    ax_full.set_ylabel("Pack current [A]")
    ax_full.legend()

    ax_zoom.set_xlim(ZOOM_START_H, ZOOM_END_H)
    zoom_values = np.concatenate(zoom_values)
    if zoom_values.size:
        y_lo, y_hi = zoom_values.min(), zoom_values.max()
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 1.0
        ax_zoom.set_ylim(y_lo - margin, y_hi + margin)
    ax_zoom.set_xlabel("Operational hours")
    ax_zoom.set_ylabel("Pack current [A]")
    ax_zoom.set_title(f"Zoomed: {ZOOM_START_H}-{ZOOM_END_H} h")
    ax_zoom.legend()

    fig.tight_layout()
    ev_tag = "_".join(f"ev{car_id}" for car_id in car_ids)
    fig.savefig(OUT_DIR / f"pack_currents_{ev_tag}.png", bbox_inches="tight")
    plt.show()


# %%
currents = get_pack_currents(CAR_IDS)
plot_pack_currents(CAR_IDS, currents)
