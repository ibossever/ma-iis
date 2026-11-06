"""
Pack current (vin_2.pkl) FFT analysis playground for the QAS dataset.

Loop DATA_ROOT for all EV ids, load each one's vin_2.pkl pack current
channel, and look at its frequency content via FFT: overlay a few selected
EVs' spectra against the pooled average spectrum across all EVs, then
scatter each EV's deviation from that pooled average (mean and std of the
diff, restricted to each of three frequency bands below) with faulty EVs
(ground-truth label, Labels.xls) shown as red dots.
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

CURRENT_COL = 224  # vin_2.pkl: pack current [A]
SAMPLE_INTERVAL_S = 30.0  # seconds between consecutive rows

EV_IDS = [63, 73]  # EVs to overlay in the FFT plot below

# FFT frequency axis (0..FFT_MAX_FREQ_PER_H) that every EV's spectrum gets
# resampled onto, so spectra of different trip lengths can be pooled/averaged.
FFT_MAX_FREQ_PER_H = 30.0  # Nyquist is 60/h; cycle frequencies of interest are well below that
FFT_COMMON_FREQS = np.linspace(0, FFT_MAX_FREQ_PER_H, 500)

# Moved to module level (was previously defined only inside the __main__
# block below) so it's importable by other scripts without triggering the
# full plotting/analysis run -- value unchanged.
BAND_INTERVALS_PER_H = [
    [0.01, 0.5],
    [0.5, 2.5],
    [2.5, 7.5],
]


def load_vin(car_id: int, vin_file: str) -> np.ndarray:
    """Load a vin_*.pkl file for one car and return it as a plain numpy array."""
    tensor = pd.read_pickle(DATA_ROOT / str(car_id) / vin_file)
    return tensor.cpu().numpy()


def pack_current(vin2_data: np.ndarray) -> np.ndarray:
    """Pack current channel, per timestep."""
    return vin2_data[:, CURRENT_COL]


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
    axis in cycles/h. Detrending removes the slow overall current drift so
    it doesn't dominate the spectrum. Magnitude is normalized by array
    length so EVs of different trip length are comparable."""
    detrended = detrend(signal)
    spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
    freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
    return freqs_per_h, spectrum


def band_diff_df(ev_ids, spectra, fft_pool_avg, mask):
    """mean and std of (EV FFT - pool avg FFT), restricted to `mask`, per EV."""
    return pd.DataFrame({
        "ev_id": ev_ids,
        "mean_diff": [float(np.mean((s - fft_pool_avg)[mask])) for s in spectra],
        "std_diff": [float(np.std((s - fft_pool_avg)[mask])) for s in spectra],
    })


def compute_fleet_pool_avg_spectrum() -> np.ndarray:
    """The fleet-wide reference spectrum every EV/window gets diffed
    against: whole-signal FFT (from index 0, no start-of-operation offset --
    same as the __main__ block below) for every EV under DATA_ROOT,
    resampled onto FFT_COMMON_FREQS and averaged. Not persisted anywhere by
    the __main__ block (only the derived per-EV band-diff features are), so
    callers that need it (e.g. a windowed feature extractor) must rebuild it
    via this function -- one full-fleet pass over raw pack current.
    """
    spectra = []
    for _ev_id, vin2_data in iter_evs():
        current = pack_current(vin2_data)
        freqs_per_h, spectrum = fft_spectrum(current)
        spectra.append(np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum))
    return np.mean(spectra, axis=0)


def band_diff_features(spectrum: np.ndarray, fft_pool_avg: np.ndarray, band_intervals_per_h: list) -> dict:
    """mean_diff/std_diff of (spectrum - fft_pool_avg) for one EV/window,
    restricted to each band in `band_intervals_per_h`, keyed band1_.._bandN_
    (same numbering/convention as the __main__ block's per-EV feature
    table). `spectrum` must already be resampled onto FFT_COMMON_FREQS
    (i.e. the output of fft_spectrum() + np.interp, same as fft_pool_avg).
    """
    out = {}
    for band_idx, (lo, hi) in enumerate(band_intervals_per_h, start=1):
        mask = (FFT_COMMON_FREQS >= lo) & (FFT_COMMON_FREQS <= hi)
        diff = (spectrum - fft_pool_avg)[mask]
        out[f"band{band_idx}_mean_diff"] = float(np.mean(diff))
        out[f"band{band_idx}_std_diff"] = float(np.std(diff))
    return out


if __name__ == "__main__":
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)

    # =======================================================================
    # FFT of pack current for every EV (whole signal, no start offset),
    # resampled onto FFT_COMMON_FREQS so spectra can be pooled/averaged.
    # =======================================================================
    ev_ids, spectra = [], []
    for ev_id, vin2_data in iter_evs():
        current = pack_current(vin2_data)
        freqs_per_h, spectrum = fft_spectrum(current)
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
        current = pack_current(load_vin(ev_id, VIN2_FILE))
        freqs_per_h, spectrum = fft_spectrum(current)
        plot_mask = freqs_per_h >= FFT_PLOT_MIN_FREQ_PER_H
        ax_fft.plot(freqs_per_h[plot_mask], spectrum[plot_mask], label=f"EV {ev_id}", linewidth=0.2)

    pool_mask = FFT_COMMON_FREQS >= FFT_PLOT_MIN_FREQ_PER_H
    ax_fft.plot(FFT_COMMON_FREQS[pool_mask], fft_pool_avg[pool_mask], label="Pool average (all EVs)",
                color="black", linewidth=1.0, zorder=5)

    ax_fft.set_xlim(FFT_PLOT_MIN_FREQ_PER_H, FFT_MAX_FREQ_PER_H)
    ax_fft.set_yscale("log")
    ax_fft.set_xlabel("Frequency [1/h]")
    ax_fft.set_ylabel("Magnitude (normalized, log scale)")
    ax_fft.set_title(f"Pack current FFT, detrended (full signal, freq >= {FFT_PLOT_MIN_FREQ_PER_H}/h)")
    ax_fft.legend(fontsize=8)
    fft_plot_path = FIGURES_DIR / f"pack_current_fft_ev{'_'.join(str(e) for e in EV_IDS)}.png"
    fig_fft.savefig(fft_plot_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {fft_plot_path}")

    # =======================================================================
    # Per-EV FFT amplitude vs. pool average, per frequency band: diff(freq) =
    # EV spectrum(freq) - fft_pool_avg(freq) restricted to that band, then
    # mean and std of that diff curve -- ranked list + one scatter subplot
    # per interval. Edit BAND_INTERVALS_PER_H (module level, top of file) to
    # try different [lo, hi] cuts (in cycles/h); titles and masks are built
    # from it automatically.
    # =======================================================================
    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})

    BANDS = [
        (f"{lo:g}/h <= freq <= {hi:g}/h", (FFT_COMMON_FREQS >= lo) & (FFT_COMMON_FREQS <= hi))
        for lo, hi in BAND_INTERVALS_PER_H
    ]

    band_dfs = [band_diff_df(ev_ids, spectra, fft_pool_avg, mask) for _, mask in BANDS]

    fig_scatter, axes_scatter = plt.subplots(1, len(BANDS), figsize=(6 * len(BANDS), 5.5), dpi=150)
    for ax, (title, mask), band_df in zip(axes_scatter, BANDS, band_dfs):
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

    fig_scatter.suptitle("Pack current FFT: mean diff vs. std diff per EV, by frequency band")
    fig_scatter.tight_layout()
    scatter_path = FIGURES_DIR / "pack_current_fft_diff_scatter_bands.png"
    fig_scatter.savefig(scatter_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {scatter_path}")

    # =======================================================================
    # mean_diff vs. std_diff scatter, one panel per interval plus the whole
    # frequency range, laid out as: whole range + band 1 on top, band 2 +
    # band 3 on bottom. mean_diff is shown on a symlog x-axis (log-scaled on
    # both sides of zero) so EVs below the pool average (negative mean_diff)
    # still show up, which a plain log axis would silently drop.
    # =======================================================================
    whole_range_mask = np.ones_like(FFT_COMMON_FREQS, dtype=bool)
    whole_range_df = band_diff_df(ev_ids, spectra, fft_pool_avg, whole_range_mask)

    def scatter_diff_panel(ax, df, title):
        df = df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")
        faulty = df["Label"] == 1
        healthy = ~faulty

        nonzero_abs_mean_diff = df.loc[df["mean_diff"] != 0, "mean_diff"].abs()
        linthresh = float(nonzero_abs_mean_diff.min()) if not nonzero_abs_mean_diff.empty else 1e-6

        ax.scatter(df.loc[healthy, "mean_diff"], df.loc[healthy, "std_diff"],
                   s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
        ax.scatter(df.loc[faulty, "mean_diff"], df.loc[faulty, "std_diff"],
                   s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
        ax.set_xscale("symlog", linthresh=linthresh)
        ax.axvline(0, color="grey", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("mean(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_ylabel("std(EV FFT amplitude - pool avg FFT amplitude)")
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)

    fig_combined, axes_combined = plt.subplots(2, 2, figsize=(12, 11), dpi=150)
    scatter_diff_panel(axes_combined[0, 0], whole_range_df, "Whole frequency range")
    scatter_diff_panel(axes_combined[0, 1], band_dfs[0], BANDS[0][0])
    scatter_diff_panel(axes_combined[1, 0], band_dfs[1], BANDS[1][0])
    scatter_diff_panel(axes_combined[1, 1], band_dfs[2], BANDS[2][0])

    fig_combined.suptitle("Pack current FFT: mean diff vs. std diff per EV, whole range and by frequency band")
    fig_combined.tight_layout()
    combined_path = FIGURES_DIR / "current_fft.png"
    fig_combined.savefig(combined_path, bbox_inches="tight", dpi=300)
    print(f"Saved figure to: {combined_path}")

    # =======================================================================
    # Save mean_diff/std_diff for each of the 3 bands as per-EV features,
    # one column pair per band (band1_mean_diff, band1_std_diff, ...).
    # =======================================================================
    current_fft_features_df = pd.DataFrame({"ev_id": ev_ids}).set_index("ev_id")
    for band_idx, band_df in enumerate(band_dfs, start=1):
        band_df = band_df.set_index("ev_id")
        current_fft_features_df[f"band{band_idx}_mean_diff"] = band_df["mean_diff"]
        current_fft_features_df[f"band{band_idx}_std_diff"] = band_df["std_diff"]

    CURRENT_FFT_FEATURES_OUT = PROJECT_ROOT / "extracted_data" / "whole_features" / "current_fft.pkl"
    current_fft_features_df.to_pickle(CURRENT_FFT_FEATURES_OUT)
    print(f"\nSaved current FFT features to: {CURRENT_FFT_FEATURES_OUT}")

    plt.show()
