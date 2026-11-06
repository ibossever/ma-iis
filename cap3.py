"""
Pack SOC and pack current charge/discharge state per timestep for all EVs,
from QAS vin_2.pkl (pack SOC + pack current -- see QAS_data_format.md).

SOC-derived state: compare pack SOC at each timestep to the previous one:
  +1 -> charging    (SOC rose,   energy flows into the pack)
  -1 -> discharging (SOC fell)
   0 -> unchanged   (SOC did not move)
The first timestep of each EV has no previous sample, so it is set to 0.

Current-derived state: sign of the instantaneous pack current at each
timestep (current > 0 = discharge, current < 0 = charge -- verified
empirically, see capac2.py):
  +1 -> charging    (current < 0)
  -1 -> discharging (current > 0)
   0 -> current == 0

In theory the two must agree (charging current -> rising SOC and vice
versa). Compare them per EV, index by index, to see how often they do.
The SOC state at index 0 is artificial (no previous sample to diff
against), so the comparison starts at index 1.

Collects each EV's (T,) state arrays into fleet-wide matrices
(n_evs, max_T), padded with NaN where an EV's recording is shorter than
the longest one in the fleet.

Capacity/efficiency estimate: every timestep where SOC state and current
state agree (both +1 charge or both -1 discharge) is treated
independently and summed across the whole recording -- not just the
single longest uninterrupted run -- giving a gross charge/discharge
capacity and SOC change per EV, normalized to a 100% SOC swing.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN_FILE = "vin_2.pkl"

SOC_COL = 223      # pack SOC, stored as a 0-1 fraction
CURRENT_COL = 224  # pack current [A], positive = discharge, negative = charge

SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

# Every EV in this dataset has a large contiguous block of |current| ~ 1000 A
# (checked: max real current across the whole fleet, excluding that cluster,
# never exceeds 300 A). That block is a sensor fault or placeholder value,
# not real current -- treat anything past this as invalid.
CURRENT_GLITCH_THRESHOLD_A = 500.0


def load_vin(car_id: int, vin_file: str = VIN_FILE) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain (T, n_cols) array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def iter_evs(vin_file: str = VIN_FILE):
    """Yield (ev_id, data) for every EV under DATA_ROOT that has `vin_file`."""
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        if not ev_dir.is_dir():
            continue
        pkl_path = ev_dir / vin_file
        if not pkl_path.exists():
            continue
        ev_id = int(ev_dir.name)
        yield ev_id, load_vin(ev_id, vin_file)


def soc_state(data: np.ndarray) -> np.ndarray:
    """(T,) array of {-1, 0, 1} marking discharge/no-change/charge per timestep,
    from consecutive pack SOC samples. Index 0 has no previous sample -> 0."""
    soc = data[:, SOC_COL]
    state = np.zeros(len(soc), dtype=np.int8)
    state[1:] = np.sign(np.diff(soc)).astype(np.int8)
    return state


def current_state(data: np.ndarray) -> np.ndarray:
    """(T,) array of {-1, 0, 1} marking discharge/no-change/charge per timestep,
    from the instantaneous pack current (current > 0 -> discharge -> -1,
    current < 0 -> charge -> +1, current == 0 -> 0). Samples with
    |current| > CURRENT_GLITCH_THRESHOLD_A are a sensor fault, not real
    signal (see the constant's definition) -- forced to 0 so they can't be
    picked up as a "charge" or "discharge" run downstream."""
    current = data[:, CURRENT_COL]
    state = (-np.sign(current)).astype(np.int8)
    state[np.abs(current) > CURRENT_GLITCH_THRESHOLD_A] = 0
    return state


def gross_matched_capacity(soc: np.ndarray, current: np.ndarray, soc_st: np.ndarray,
                            cur_st: np.ndarray, sign: int) -> dict | None:
    """Sum SOC change and Coulomb count over every timestep where SOC state
    and current state agree and both equal `sign` (+1 charge, -1 discharge)
    -- each matched timestep counted on its own, not just the longest
    uninterrupted run of matches. None if no timestep matches.

    For a matched timestep t: its SOC contribution is soc[t] - soc[t-1] (the
    same delta soc_st[t] is the sign of), its capacity contribution is
    |current[t]| * dt. Summed independently across every such t, whether or
    not they're contiguous -- gross totals over the whole recording."""
    mask = (soc_st == sign) & (cur_st == sign)
    if not mask.any():
        return None
    soc_diff = np.empty_like(soc)
    soc_diff[0] = 0.0
    soc_diff[1:] = np.diff(soc)
    dt_hours = SAMPLE_INTERVAL_S / 3600.0
    n_steps = int(mask.sum())
    return {
        "n_steps": n_steps,
        "duration_s": n_steps * SAMPLE_INTERVAL_S,
        "soc_change": float(np.sum(soc_diff[mask])),
        "capacity_change_ah": float(np.sum(np.abs(current[mask])) * dt_hours),
    }


def build_matrix(ev_id_to_state: dict[int, np.ndarray], ev_ids: list[int], max_t: int) -> np.ndarray:
    """Stack per-EV (T,) state arrays into an (n_evs, max_T) matrix, padded
    with NaN where an EV's recording is shorter than max_t."""
    matrix = np.full((len(ev_ids), max_t), np.nan)
    for row, ev_id in enumerate(ev_ids):
        state = ev_id_to_state[ev_id]
        matrix[row, :len(state)] = state
    return matrix


def _period_str(period: dict | None) -> str:
    if period is None:
        return "n_steps=n/a soc_change=n/a capacity_change=n/a"
    return (f"n_steps={period['n_steps']} ({period['duration_s'] / 60:.1f} min total) "
            f"soc_change={period['soc_change'] * 100:+.2f}% "
            f"capacity_change={period['capacity_change_ah']:.3f} Ah")


if __name__ == "__main__":
    soc_states = {}
    current_states = {}

    total_match = 0
    total_diff = 0
    period_rows = []

    for ev_id, data in iter_evs():
        soc = data[:, SOC_COL]
        current = data[:, CURRENT_COL]
        soc_st = soc_state(data)
        cur_st = current_state(data)
        soc_states[ev_id] = soc_st
        current_states[ev_id] = cur_st

        # index 0's soc_state is artificial (no previous sample) -> exclude it
        matches = soc_st[1:] == cur_st[1:]
        n_match = int(np.sum(matches))
        n_diff = int(matches.size - n_match)
        total_match += n_match
        total_diff += n_diff

        print(f"EV {ev_id}: T={len(soc_st)} match={n_match} diff={n_diff} "
              f"match_pct={100 * n_match / matches.size:.1f}%")

        # -----------------------------------------------------------------
        # Gross totals over every timestep where SOC state and current state
        # agree, separately for charge and discharge -- each matched
        # timestep summed independently, not just the longest uninterrupted
        # run (see gross_matched_capacity docstring).
        # -----------------------------------------------------------------
        charge_period = gross_matched_capacity(soc, current, soc_st, cur_st, sign=1)
        discharge_period = gross_matched_capacity(soc, current, soc_st, cur_st, sign=-1)

        print(f"  charge:    {_period_str(charge_period)}")
        print(f"  discharge: {_period_str(discharge_period)}")

        period_rows.append({
            "ev_id": ev_id,
            "charge_duration_min": charge_period["duration_s"] / 60 if charge_period else np.nan,
            "charge_soc_change_pct": charge_period["soc_change"] * 100 if charge_period else np.nan,
            "charge_capacity_ah": charge_period["capacity_change_ah"] if charge_period else np.nan,
            "discharge_duration_min": discharge_period["duration_s"] / 60 if discharge_period else np.nan,
            "discharge_soc_change_pct": discharge_period["soc_change"] * 100 if discharge_period else np.nan,
            "discharge_capacity_ah": discharge_period["capacity_change_ah"] if discharge_period else np.nan,
        })

    # ===================================================================
    # Fleet-wide matrices: (n_evs, max_T), padded with NaN for shorter EVs
    # ===================================================================
    ev_ids = sorted(soc_states)
    max_t = max(len(soc_states[ev_id]) for ev_id in ev_ids)

    soc_state_matrix = build_matrix(soc_states, ev_ids, max_t)
    current_state_matrix = build_matrix(current_states, ev_ids, max_t)

    print(f"\nSOC state matrix:     shape {soc_state_matrix.shape} (rows=EVs, cols=timesteps)")
    print(f"Current state matrix: shape {current_state_matrix.shape} (rows=EVs, cols=timesteps)")
    print(f"Row order (row index -> ev_id): {ev_ids}")

    print(f"\nSOC totals:     charge={int(np.sum(soc_state_matrix == 1))} "
          f"discharge={int(np.sum(soc_state_matrix == -1))} idle={int(np.sum(soc_state_matrix == 0))}")
    print(f"Current totals: charge={int(np.sum(current_state_matrix == 1))} "
          f"discharge={int(np.sum(current_state_matrix == -1))} idle={int(np.sum(current_state_matrix == 0))}")

    print(f"\nFleet-wide SOC-vs-current agreement (index 0 excluded per EV): "
          f"match={total_match} diff={total_diff} match_pct={100 * total_match / (total_match + total_diff):.2f}%")

    # ===================================================================
    # Per-EV gross matched charge/discharge summary (all matched timesteps
    # summed, not just the longest uninterrupted run)
    # ===================================================================
    periods_df = pd.DataFrame(period_rows)
    print("\nGross matched charge/discharge totals per EV:")
    print(periods_df.to_string(index=False))

    # ===================================================================
    # Normalized capacity per 100% SOC swing (charge_capacity_ah /
    # charge_soc_change_pct * 100, same idea discharge-side) + their ratio.
    # charge_soc_change_pct is already positive (SOC rose); discharge_soc_change_pct
    # is negative (SOC fell) -- abs() it so the discharge estimate comes out
    # positive too, same sign as the charge one. A period with 0% net SOC
    # change (can happen for a short, noisy discharge run) makes the
    # normalized estimate undefined -- NaN rather than dividing by zero.
    # ===================================================================
    norm_rows = []
    for row in period_rows:
        charge_norm_ah = (row["charge_capacity_ah"] / row["charge_soc_change_pct"] * 100
                           if row["charge_soc_change_pct"] else np.nan)
        discharge_norm_ah = (row["discharge_capacity_ah"] / abs(row["discharge_soc_change_pct"]) * 100
                              if row["discharge_soc_change_pct"] else np.nan)
        norm_rows.append({
            "ev_id": row["ev_id"],
            "charge_capacity_norm_ah": charge_norm_ah,
            "discharge_capacity_norm_ah": discharge_norm_ah,
            "eff": discharge_norm_ah / charge_norm_ah,
        })

    norm_df = pd.DataFrame(norm_rows)
    print("\nNormalized capacity per 100% SOC swing (charge/discharge) + eff = discharge/charge ratio:")
    print(norm_df.to_string(index=False))
