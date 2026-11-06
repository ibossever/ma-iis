# %%
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("/media/ssd_2tb/jak-iis")
CURRENT_COL = 224
SAMPLE_INTERVAL_S = 30.0
TOP_N = 10

# All cars start pinned at ~-1000 A before jumping to normal operating
# levels. Any current above this threshold counts as "normal" - it sits
# well above the -1000 A plateau and well below it never reaching normal
# levels above it.
STARTUP_THRESHOLD_A = -500.0

# A handful of EVs (e.g. 223, 318) drop back down to the same ~-1000 A clip
# value later in the trip, not just at startup - treat any sample below this
# as a sensor artifact and exclude it from the max-charge-current search.
CHARGE_CLIP_THRESHOLD_A = -300.0


def get_pack_currents():
    """Algorithm: returns (car_id, pack-current array) for every EV under DATA_ROOT. Swap this out to change what gets analyzed."""
    car_ids, currents = [], []
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        pkl_path = ev_dir / "vin_2.pkl"
        if not ev_dir.is_dir() or not pkl_path.exists():
            continue
        tensor = pd.read_pickle(pkl_path)
        matrix = tensor.cpu().numpy()
        car_ids.append(int(ev_dir.name))
        currents.append(matrix[:, CURRENT_COL])
    return car_ids, currents


def analyze_current(current):
    """Algorithm: startup hour + max discharge/charge current, ignoring the initial ~-1000 A startup plateau."""
    normal_idx = np.argmax(current > STARTUP_THRESHOLD_A)
    normal_hour = normal_idx * SAMPLE_INTERVAL_S / 3600.0

    normal_current = current[normal_idx:]
    max_discharge_A = normal_current.max()  # positive = discharge

    chargeable = normal_current[normal_current > CHARGE_CLIP_THRESHOLD_A]
    max_charge_A = chargeable.min()         # negative = charge

    return normal_hour, max_discharge_A, max_charge_A


# %%
car_ids, currents = get_pack_currents()
results = [
    (car_id, *analyze_current(current))
    for car_id, current in zip(car_ids, currents)
]  # each: (car_id, normal_hour, max_discharge_A, max_charge_A)

# %%
# Data matrix of the 3 features per EV, for reuse in other scripts (e.g.
# charge/discharge capacity, C-rate). normal_hour_h is the input every
# downstream calculation needs to align on operational time.
FEATURES_OUT = Path(__file__).resolve().parents[2] / "extracted_data" / "pack_current_features.pkl"

features_df = pd.DataFrame(
    results, columns=["car_id", "normal_hour_h", "max_discharge_A", "max_charge_A"]
).set_index("car_id")
features_df.to_pickle(FEATURES_OUT)

# %%
ROW_FMT = "{:>6} {:>14.2f} {:>19.1f} {:>17.1f}"
HEADER = f"{'EV':>6} {'normal @ [h]':>14} {'max discharge [A]':>19} {'max charge [A]':>17}"


def print_table(title, rows):
    print(f"\n--- {title} ---")
    print(HEADER)
    for car_id, normal_hour, max_discharge_A, max_charge_A in rows:
        print(ROW_FMT.format(car_id, normal_hour, max_discharge_A, max_charge_A))


def print_top(title, key, reverse):
    ranked = sorted(results, key=key, reverse=reverse)[:TOP_N]
    print_table(f"Top {TOP_N} by {title}", ranked)


print_table("All EVs, sorted by EV id", sorted(results, key=lambda r: r[0]))

print_top("lowest max discharge current", key=lambda r: r[2], reverse=False)
print_top("highest max discharge current", key=lambda r: r[2], reverse=True)
print_top("lowest max charge current (most negative)", key=lambda r: r[3], reverse=False)
print_top("highest max charge current (closest to 0)", key=lambda r: r[3], reverse=True)

longest_normal_hour = max(r[1] for r in results)
print(f"\nLongest normal-onset duration across all EVs: {longest_normal_hour:.2f} h")
