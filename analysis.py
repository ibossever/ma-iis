"""
Pack temperature (vin_2.pkl) analysis playground for the QAS dataset.

Loop DATA_ROOT for all EV ids, load each one's vin_2.pkl pack temperature
channel, and look at its frequency content via FFT: overlay a few selected
EVs' spectra against the pooled average spectrum across all EVs, then
scatter each EV's deviation from that pooled average (mean and std of the
diff, over the whole spectrum) with faulty EVs (ground-truth label,
Labels.xls) shown as red dots.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

# ===========================================================================
# CONFIG
# ===========================================================================
DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root (393 cars, id 0-392)
VIN2_FILE = "vin_2.pkl"
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PACK_TEMP_COL = 222  # vin_2.pkl: pack temperature [degC]
SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

EV_IDS = [380, 370]  # EVs to overlay in the FFT plot below

# FFT frequency axis (0..FFT_MAX_FREQ_PER_H) that every EV's spectrum gets
# resampled onto, so spectra of different trip lengths can be pooled/averaged.
FFT_MAX_FREQ_PER_H = 30.0  # Nyquist is 60/h; cycle frequencies of interest are well below that
FFT_COMMON_FREQS = np.linspace(0, FFT_MAX_FREQ_PER_H, 500)

# 4 cut points -> 5 frequency bands (below the first, between each
# consecutive pair, above the last) for the ranked lists/scatter grid below.
# The 3 middle bands (index 1, 2, 3 -- between breakpoints[0]-[1], [1]-[2],
# [2]-[3]) double as the low/mid/high frequency intervals used for the
# per-EV feature extraction at the bottom; the 2 outer bands are not used
# there.
BAND_BREAKPOINTS_PER_H = [0.1, 0.4, 1.2, 3.6]

def load_vin(car_id: int, vin_file: str) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain numpy array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def iter_evs():
    """Yield (ev_id, vin2_data) for every EV under DATA_ROOT that has vin_2.pkl."""
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        if not ev_dir.is_dir():
            continue
        vin2_path = ev_dir / VIN2_FILE
        if not vin2_path.exists():
            continue
        ev_id = int(ev_dir.name)
        yield ev_id, load_vin(ev_id, VIN2_FILE)


def dtdt_stats(vin2_data: np.ndarray) -> tuple[float, float, np.ndarray]:
    """dT/dt for the pack temperature channel, in degC/s, plus summary stats."""
    temp = vin2_data[:, PACK_TEMP_COL]
    time_seconds = np.arange(vin2_data.shape[0]) * SAMPLE_INTERVAL_S
    dtdt = np.gradient(temp, time_seconds)

    max_abs_dtdt = float(np.max(np.abs(dtdt)))
    dtdt_std = float(np.std(dtdt))

    return max_abs_dtdt, dtdt_std, dtdt


def mean_pack_temp(pack_temp: np.ndarray) -> float:
    """Plain time-domain mean of the pack temp channel -- same formula the
    __main__ block below inlines when building mean_pack_temps, factored
    out so it can be called on an arbitrary (e.g. windowed) segment."""
    return float(np.mean(pack_temp))


def fft_spectrum(pack_temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|FFT| of the linear-detrended, full-length pack temp array, plus its
    frequency axis in cycles/h. Detrending removes the slow overall
    temperature drift so it doesn't dominate the spectrum. Magnitude is
    normalized by array length so EVs of different trip length are
    comparable."""
    detrended = detrend(pack_temp)
    spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
    freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
    return freqs_per_h, spectrum


def bands_from_breakpoints(breakpoints: list[float], freqs: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Turn a sorted list of frequency breakpoints into len(breakpoints) + 1
    (title, mask) bands: below the first breakpoint, between each
    consecutive pair, and above the last."""
    edges = [None, *breakpoints, None]
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo is None:
            mask = freqs < hi
            title = f"freq < {hi:g}/h"
        elif hi is None:
            mask = freqs > lo
            title = f"freq > {lo:g}/h"
        else:
            mask = (freqs >= lo) & (freqs <= hi)
            title = f"{lo:g}/h <= freq <= {hi:g}/h"
        bands.append((title, mask))
    return bands


def band_diff_df(ev_ids: list[int], spectra: list[np.ndarray], fft_pool_avg: np.ndarray,
                  mask: np.ndarray) -> pd.DataFrame:
    """mean and std of (EV FFT - pool avg FFT), restricted to `mask`, per EV."""
    return pd.DataFrame({
        "ev_id": ev_ids,
        "mean_diff": [float(np.mean((s - fft_pool_avg)[mask])) for s in spectra],
        "std_diff": [float(np.std((s - fft_pool_avg)[mask])) for s in spectra],
    })


def print_top5(band_df: pd.DataFrame, title: str) -> None:
    """Print the top 5 EVs by mean_diff and, separately, by std_diff."""
    print(f"\n--- Top 5 EVs by mean_diff of FFT diff ({title}) ---")
    print(band_df.sort_values("mean_diff", ascending=False)
            .head(5).to_string(index=False))
    print(f"\n--- Top 5 EVs by std_diff of FFT diff ({title}) ---")
    print(band_df.sort_values("std_diff", ascending=False)
            .head(5).to_string(index=False))


if __name__ == "__main__":
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)

    # =======================================================================
    # FFT of pack temp for every EV (whole signal, no start offset),
    # resampled onto FFT_COMMON_FREQS so spectra can be pooled/averaged.
    # =======================================================================
    ev_ids, spectra, mean_pack_temps = [], [], []
    for ev_id, vin2_data in iter_evs():
        pack_temp = vin2_data[:, PACK_TEMP_COL]
        freqs_per_h, spectrum = fft_spectrum(pack_temp)
        ev_ids.append(ev_id)
        spectra.append(np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum))
        mean_pack_temps.append(float(np.mean(pack_temp)))

    fft_pool_avg = np.mean(spectra, axis=0)

    # =======================================================================
    # FFT plot: selected EVs overlaid against the pooled average spectrum
    # =======================================================================
    fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
    for ev_id in EV_IDS:
        if ev_id not in ev_ids:
            print(f"EV {ev_id}: no vin_2.pkl, skipping in FFT plot")
            continue
        pack_temp = load_vin(ev_id, VIN2_FILE)[:, PACK_TEMP_COL]
        freqs_per_h, spectrum = fft_spectrum(pack_temp)
        ax_fft.plot(freqs_per_h, spectrum, label=f"EV {ev_id}", linewidth=0.4)

    ax_fft.plot(FFT_COMMON_FREQS, fft_pool_avg, label="Pool average (all EVs)",
                color="black", linewidth=1.2, zorder=5)

    ax_fft.set_xlim(0, FFT_MAX_FREQ_PER_H)
    ax_fft.set_yscale("log")
    ax_fft.set_xlabel("Frequency [1/h]")
    ax_fft.set_ylabel("Magnitude (normalized, log scale)")
    ax_fft.set_title("Pack temp FFT, detrended (full signal)")
    ax_fft.legend(fontsize=8)
    fft_plot_path = FIGURES_DIR / f"pack_temp_fft_ev{'_'.join(str(e) for e in EV_IDS)}.png"
    fig_fft.savefig(fft_plot_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {fft_plot_path}")

    # =======================================================================
    # Per-EV FFT amplitude vs. pool average, over the whole spectrum (no
    # frequency threshold): diff(freq) = EV spectrum(freq) - fft_pool_avg(freq)
    # over all of FFT_COMMON_FREQS, then mean and std of that diff curve.
    # =======================================================================
    fft_diff_df = pd.DataFrame({
        "ev_id": ev_ids,
        "mean_diff": [float(np.mean(s - fft_pool_avg)) for s in spectra],
        "std_diff": [float(np.std(s - fft_pool_avg)) for s in spectra],
    })

    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    fft_diff_df = fft_diff_df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")
    faulty = fft_diff_df["Label"] == 1
    healthy = ~faulty

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
    ax_scatter.scatter(fft_diff_df.loc[healthy, "mean_diff"], fft_diff_df.loc[healthy, "std_diff"],
                        s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
    ax_scatter.scatter(fft_diff_df.loc[faulty, "mean_diff"], fft_diff_df.loc[faulty, "std_diff"],
                        s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
    ax_scatter.set_xlabel("mean(EV FFT amplitude - pool avg FFT amplitude)")
    ax_scatter.set_ylabel("std(EV FFT amplitude - pool avg FFT amplitude)")
    ax_scatter.set_title("Pack temp FFT: mean diff vs. std diff per EV (whole spectrum)")
    ax_scatter.grid(True, linewidth=0.4, alpha=0.5)
    ax_scatter.legend(fontsize=8)
    scatter_path = FIGURES_DIR / "pack_temp_fft_diff_scatter.png"
    fig_scatter.savefig(scatter_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {scatter_path}")

    # =======================================================================
    # Same mean/std diff, restricted to the 5 bands cut out of
    # BAND_BREAKPOINTS_PER_H: one scatter subplot per band (all in one
    # figure) plus, per band, the top 5 EVs by mean_diff and by std_diff.
    # =======================================================================
    bands = bands_from_breakpoints(BAND_BREAKPOINTS_PER_H, FFT_COMMON_FREQS)

    BAND_GRID_NCOLS = 3
    band_grid_nrows = -(-len(bands) // BAND_GRID_NCOLS)  # ceil division
    fig_bands, axes_bands = plt.subplots(band_grid_nrows, BAND_GRID_NCOLS,
                                          figsize=(6 * BAND_GRID_NCOLS, 5.5 * band_grid_nrows), dpi=150)
    for ax, (title, mask) in zip(axes_bands.flat, bands):
        band_df = band_diff_df(ev_ids, spectra, fft_pool_avg, mask)
        print_top5(band_df, title)

        band_df = band_df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")
        band_faulty = band_df["Label"] == 1
        band_healthy = ~band_faulty

        ax.scatter(band_df.loc[band_healthy, "mean_diff"], band_df.loc[band_healthy, "std_diff"],
                   s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
        ax.scatter(band_df.loc[band_faulty, "mean_diff"], band_df.loc[band_faulty, "std_diff"],
                   s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
        ax.set_xlabel("mean(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_ylabel("std(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    for ax in axes_bands.flat[len(bands):]:  # unused grid slots, if any
        ax.axis("off")

    fig_bands.suptitle("Pack temp FFT: mean diff vs. std diff per EV, by frequency band")
    fig_bands.tight_layout()
    bands_scatter_path = FIGURES_DIR / "pack_temp_fft_diff_scatter_bands.png"
    fig_bands.savefig(bands_scatter_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {bands_scatter_path}")

    # =======================================================================
    # Extract 7 scalar features per EV: mean pack temp, plus mean_diff and
    # std_diff (EV FFT - pool avg FFT) for the 3 middle bands cut out of
    # BAND_BREAKPOINTS_PER_H above (band1/band2/band3, low to high frequency
    # -- same naming convention as soc_fft.py / current_fft.py). Re-run with
    # different BAND_BREAKPOINTS_PER_H to extract different bands.
    # =======================================================================
    if len(bands) != 5:
        raise ValueError(
            f"Feature extraction needs exactly 4 breakpoints (5 bands) in "
            f"BAND_BREAKPOINTS_PER_H, got {len(BAND_BREAKPOINTS_PER_H)} "
            f"breakpoints ({len(bands)} bands)."
        )
    feature_bands = bands[1:4]  # band1, band2, band3 -- the 3 middle bands

    pack_temp_features_df = pd.DataFrame({
        "ev_id": ev_ids,
        "mean_pack_temp": mean_pack_temps,
    }).set_index("ev_id")
    for band_idx, (_, mask) in enumerate(feature_bands, start=1):
        pack_temp_features_df[f"band{band_idx}_mean_diff"] = [
            float(np.mean((s - fft_pool_avg)[mask])) for s in spectra
        ]
        pack_temp_features_df[f"band{band_idx}_std_diff"] = [
            float(np.std((s - fft_pool_avg)[mask])) for s in spectra
        ]

    PACK_TEMP_FEATURES_OUT = PROJECT_ROOT / "extracted_data" / "whole_features" / "pack_temp.pkl"
    pack_temp_features_df.to_pickle(PACK_TEMP_FEATURES_OUT)
    print(f"\nSaved pack temp features to: {PACK_TEMP_FEATURES_OUT}")

    plt.show()
