"""
Mean cell voltage (vin_2.pkl) FFT analysis playground for the QAS dataset.

Loop DATA_ROOT for all EV ids, load each one's vin_2.pkl, average the 110
per-cell voltage columns into one mean-cell-voltage signal per EV, and look
at its frequency content via FFT: overlay a few selected EVs' spectra
against the pooled average spectrum across all EVs, then scatter each EV's
deviation from that pooled average (mean and std of the diff, over the
whole spectrum) with faulty EVs (ground-truth label, Labels.xls) shown as
red dots.
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

CELL_VOLTAGE_COLS = slice(2, 112)  # vin_2.pkl: per-cell actual voltage, 110 cells [V]
SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

EV_IDS = [373, 392]  # EVs to overlay in the FFT plot below

# FFT frequency axis (0..FFT_MAX_FREQ_PER_H) that every EV's spectrum gets
# resampled onto, so spectra of different trip lengths can be pooled/averaged.
FFT_MAX_FREQ_PER_H = 30.0  # Nyquist is 60/h; cycle frequencies of interest are well below that
FFT_COMMON_FREQS = np.linspace(0, FFT_MAX_FREQ_PER_H, 500)


def load_vin(car_id: int, vin_file: str) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain numpy array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def mean_cell_voltage(vin2_data: np.ndarray) -> np.ndarray:
    """Mean across the 110 per-cell voltage columns, per timestep."""
    return vin2_data[:, CELL_VOLTAGE_COLS].mean(axis=1)


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


def fft_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """|FFT| of the linear-detrended, full-length signal, plus its frequency
    axis in cycles/h. Detrending removes the slow overall voltage drift so
    it doesn't dominate the spectrum. Magnitude is normalized by array
    length so EVs of different trip length are comparable."""
    detrended = detrend(signal)
    spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
    freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
    return freqs_per_h, spectrum


def rank_by_band(ev_ids, spectra, fft_pool_avg, mask, title):
    band_df = pd.DataFrame({
        "ev_id": ev_ids,
        "mean_diff": [float(np.mean((s - fft_pool_avg)[mask])) for s in spectra],
        "std_diff": [float(np.std((s - fft_pool_avg)[mask])) for s in spectra],
    })
    band_df["mean_std_prod"] = band_df["mean_diff"] * band_df["std_diff"]

    print(f"\n--- Top 10 EVs by mean_diff * std_diff of FFT diff ({title}) ---")
    print(band_df.sort_values("mean_std_prod", ascending=False)
            .head(10).to_string(index=False))
    return band_df


if __name__ == "__main__":
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)

    # =======================================================================
    # FFT of mean cell voltage for every EV (whole signal, no start offset),
    # resampled onto FFT_COMMON_FREQS so spectra can be pooled/averaged.
    # =======================================================================
    ev_ids, spectra = [], []
    for ev_id, vin2_data in iter_evs():
        voltage = mean_cell_voltage(vin2_data)
        freqs_per_h, spectrum = fft_spectrum(voltage)
        ev_ids.append(ev_id)
        spectra.append(np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum))

    fft_pool_avg = np.mean(spectra, axis=0)

    # =======================================================================
    # FFT plot: selected EVs overlaid against the pooled average spectrum
    # =======================================================================
    FFT_PLOT_MIN_FREQ_PER_H = 0.1  # exclude the near-DC spike so the rest of the spectrum is visible

    fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
    for ev_id in EV_IDS:
        if ev_id not in ev_ids:
            print(f"EV {ev_id}: no vin_2.pkl, skipping in FFT plot")
            continue
        voltage = mean_cell_voltage(load_vin(ev_id, VIN2_FILE))
        freqs_per_h, spectrum = fft_spectrum(voltage)
        plot_mask = freqs_per_h >= FFT_PLOT_MIN_FREQ_PER_H
        ax_fft.plot(freqs_per_h[plot_mask], spectrum[plot_mask], label=f"EV {ev_id}", linewidth=0.2)

    pool_mask = FFT_COMMON_FREQS >= FFT_PLOT_MIN_FREQ_PER_H
    ax_fft.plot(FFT_COMMON_FREQS[pool_mask], fft_pool_avg[pool_mask], label="Pool average (all EVs)",
                color="black", linewidth=1.0, zorder=5)

    ax_fft.set_xlim(FFT_PLOT_MIN_FREQ_PER_H, FFT_MAX_FREQ_PER_H)
    ax_fft.set_yscale("log")
    ax_fft.set_xlabel("Frequency [1/h]")
    ax_fft.set_ylabel("Magnitude (normalized, log scale)")
    ax_fft.set_title(f"Mean cell voltage FFT, detrended (full signal, freq >= {FFT_PLOT_MIN_FREQ_PER_H}/h)")
    ax_fft.legend(fontsize=8)
    fft_plot_path = FIGURES_DIR / f"mean_cell_voltage_fft_ev{'_'.join(str(e) for e in EV_IDS)}.png"
    fig_fft.savefig(fft_plot_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {fft_plot_path}")

    # =======================================================================
    # Per-EV FFT amplitude vs. pool average, per frequency band: diff(freq) =
    # EV spectrum(freq) - fft_pool_avg(freq) restricted to that band, then
    # mean and std of that diff curve -- ranked list + one scatter subplot
    # per band.
    # =======================================================================
    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})

    BANDS = [
        ("0.1/h <= freq <= 2.5/h", (FFT_COMMON_FREQS >= 0.1) & (FFT_COMMON_FREQS <= 2.5)),
        ("2.5/h < freq <= 15/h", (FFT_COMMON_FREQS > 2.5) & (FFT_COMMON_FREQS <= 15.0)),
        ("freq > 15/h", FFT_COMMON_FREQS > 15.0),
    ]

    fig_scatter, axes_scatter = plt.subplots(1, len(BANDS), figsize=(6 * len(BANDS), 5.5), dpi=150)
    for ax, (title, mask) in zip(axes_scatter, BANDS):
        band_df = rank_by_band(ev_ids, spectra, fft_pool_avg, mask, title)
        band_df = band_df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")
        faulty = band_df["Label"] == 1
        healthy = ~faulty

        ax.scatter(band_df.loc[healthy, "mean_diff"], band_df.loc[healthy, "std_diff"],
                   s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
        ax.scatter(band_df.loc[faulty, "mean_diff"], band_df.loc[faulty, "std_diff"],
                   s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
        ax.set_xlabel("mean(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_ylabel("std(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    fig_scatter.suptitle("Mean cell voltage FFT: mean diff vs. std diff per EV, by frequency band")
    fig_scatter.tight_layout()
    scatter_path = FIGURES_DIR / "mean_cell_voltage_fft_diff_scatter.png"
    fig_scatter.savefig(scatter_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {scatter_path}")

    plt.show()
