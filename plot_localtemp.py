"""Plot local max temperature (vin_1.pkl) for a chosen set of EVs, plus a
max/min-after-startup ranking and scatter of that same column across all EVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402  (RWTH corporate colors)

DATA_ROOT = Path("/media/ssd_2tb/jak-iis")  # QAS dataset root
LABELS_FILE = DATA_ROOT / "Labels.xls"  # QAS ground truth: Num, Label (1 = faulty)
EV_IDS = [369, 373, 382, 376]  # EVs to overlay in the single plot below
MAX_TEMP_COL = 6  # vin_1.pkl: max (local/per-cell) temperature, normalized [0, 1]
SAMPLE_INTERVAL_S = 30.0
TOP_N = 10
PLOT_START_IDX = 1000  # multi-EV plot always starts here, regardless of each EV's actual start time
ANALYSIS_START_OFFSET_H = 1.0  # max/min + frequency/amplitude analysis starts this many hours after each EV's own operational start time

# normal_hour_h per car_id: hour at which pack current leaves the startup
# plateau (see current2.py) -- used here as each EV's operation start time.
CURRENT_FEATURES = Path(__file__).resolve().parents[2] / "extracted_data" / "pack_current_features.pkl"

OUT_DIR = Path(__file__).resolve().parent / "uselessfigures"

# FFT frequency axis (0..FFT_MAX_FREQ_PER_H) that every EV's spectrum gets
# resampled onto, so spectra of different trip lengths can be pooled/averaged.
FFT_MAX_FREQ_PER_H = 30.0  # Nyquist is 60/h; cycle frequencies of interest are well below that
FFT_COMMON_FREQS = np.linspace(0, FFT_MAX_FREQ_PER_H, 500)

FFT_DIFF_MIN_FREQ_PER_H = 5  # mean/std diff below excludes bins slower than this
FFT_DIFF_MASK = FFT_COMMON_FREQS >= FFT_DIFF_MIN_FREQ_PER_H


def signal_freq_amp(segment: np.ndarray) -> tuple[float, float]:
    """Avg. frequency [cycles/h] and avg. amplitude (peak-to-trough) of a
    segment, from its local maxima/minima (find_peaks)."""
    peaks, _ = find_peaks(segment)
    troughs, _ = find_peaks(-segment)
    if peaks.size == 0:
        return 0.0, 0.0

    duration_h = segment.size * SAMPLE_INTERVAL_S / 3600.0
    avg_freq = peaks.size / duration_h

    if troughs.size == 0:
        avg_amp = float(np.max(segment) - np.min(segment))
    else:
        avg_amp = float(np.mean(segment[peaks]) - np.mean(segment[troughs]))

    return avg_freq, avg_amp


def fft_diff_features(segment: np.ndarray, fft_pool_avg: np.ndarray, mask: np.ndarray = FFT_DIFF_MASK) -> tuple[float, float]:
    """mean_diff/std_diff of this segment's spectrum against a (fixed)
    pool-average reference spectrum, restricted to `mask` -- same
    computation as the fft_diff_results loop below, factored out so it can
    be called on an arbitrary (e.g. windowed) segment. `segment` is raw
    (not yet detrended/FFT'd); `fft_pool_avg` must already be resampled
    onto FFT_COMMON_FREQS (i.e. built the same way as
    compute_fleet_pool_avg_spectrum's output)."""
    detrended = detrend(segment)
    spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
    freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
    resampled = np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum)
    diff = (resampled - fft_pool_avg)[mask]
    return float(np.mean(diff)), float(np.std(diff))


def compute_fleet_pool_avg_spectrum() -> np.ndarray:
    """The fleet-wide reference spectrum every EV/window's local-max-temp
    diff features get compared against: whole-signal FFT (from each EV's
    own start_hours + ANALYSIS_START_OFFSET_H onward, detrended, same as
    the main analysis loop below) for every EV under DATA_ROOT that has a
    normal_hour_h entry, resampled onto FFT_COMMON_FREQS and averaged.
    Not persisted anywhere by the analysis below (only the derived per-EV
    diff features are), so callers that need it (e.g. a windowed feature
    extractor) must rebuild it via this function -- one full-fleet pass
    over raw vin_1.pkl local max temp.
    """
    start_hours = pd.read_pickle(CURRENT_FEATURES)["normal_hour_h"]
    spectra = []
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        vin1_path = ev_dir / "vin_1.pkl"
        if not ev_dir.is_dir() or not vin1_path.exists():
            continue
        car_id = int(ev_dir.name)
        if car_id not in start_hours.index:
            continue

        tensor = pd.read_pickle(vin1_path)
        temp = tensor.cpu().numpy()[:, 0, MAX_TEMP_COL]
        start_idx = int(round((start_hours[car_id] + ANALYSIS_START_OFFSET_H) * 3600.0 / SAMPLE_INTERVAL_S))
        segment = temp[start_idx:]
        if segment.size == 0:
            continue

        detrended = detrend(segment)
        spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
        freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
        spectra.append(np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum))

    return np.mean(spectra, axis=0)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)

    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)

    # normal_hour_h per car_id -- also used below to mark each selected EV's
    # start time on the multi-EV plot.
    start_hours = pd.read_pickle(CURRENT_FEATURES)["normal_hour_h"]

    # =======================================================================
    # Multi-EV plot -- start time marked with a dot + dashed vertical drop-line
    # =======================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    for ev_id in EV_IDS:
        tensor = pd.read_pickle(DATA_ROOT / str(ev_id) / "vin_1.pkl")
        local_max_temp = tensor.cpu().numpy()[:, 0, MAX_TEMP_COL]
        x = np.arange(PLOT_START_IDX, len(local_max_temp))
        line, = ax.plot(x, local_max_temp[PLOT_START_IDX:], label=f"EV {ev_id}", linewidth=0.8)

        if ev_id in start_hours.index:
            start_idx = int(round(start_hours[ev_id] * 3600.0 / SAMPLE_INTERVAL_S))
            start_val = local_max_temp[start_idx]
            ax.plot(start_idx, start_val, marker="o", color=line.get_color(), zorder=5)
            ax.vlines(start_idx, ymin=0, ymax=start_val,
                      color=line.get_color(), linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Sample")
    ax.set_ylabel("Local max temp (normalized)")
    ax.set_title("vin_1 local max temp (dot + dashed line = start of operation)")
    ax.legend(fontsize=8)
    fig.savefig(OUT_DIR / f"vin1_local_max_temp_ev{'_'.join(str(e) for e in EV_IDS)}.png")

    # =======================================================================
    # Max/min + avg. frequency/amplitude of local temp, all EVs, starting
    # ANALYSIS_START_OFFSET_H (1 h) after each EV's own operational start time
    # till end of operation
    # =======================================================================
    results = []  # (car_id, max_val, min_val, avg_freq, avg_amp)
    fft_spectra = []  # each EV's |FFT| resampled onto FFT_COMMON_FREQS, for the pooled average
    excluded_evs = []  # (car_id, reason) -- EVs left out of the analysis
    for ev_dir in sorted(DATA_ROOT.iterdir(), key=lambda p: p.name):
        vin1_path = ev_dir / "vin_1.pkl"
        if not ev_dir.is_dir() or not vin1_path.exists():
            continue
        car_id = int(ev_dir.name)

        if car_id not in start_hours.index:
            excluded_evs.append((car_id, "no operational start time in pack_current_features.pkl"))
            continue

        tensor = pd.read_pickle(vin1_path)
        temp = tensor.cpu().numpy()[:, 0, MAX_TEMP_COL]

        start_idx = int(round((start_hours[car_id] + ANALYSIS_START_OFFSET_H) * 3600.0 / SAMPLE_INTERVAL_S))
        segment = temp[start_idx:]
        if segment.size == 0:
            length_h = temp.size * SAMPLE_INTERVAL_S / 3600.0
            excluded_evs.append((car_id, f"no data left after start+{ANALYSIS_START_OFFSET_H:.0f}h (total length {length_h:.1f} h)"))
            continue

        avg_freq, avg_amp = signal_freq_amp(segment)
        results.append((car_id, float(np.max(segment)), float(np.min(segment)), avg_freq, avg_amp))

        detrended = detrend(segment)
        spectrum = np.abs(np.fft.rfft(detrended)) / detrended.size
        freqs_per_h = np.fft.rfftfreq(detrended.size, d=SAMPLE_INTERVAL_S) * 3600.0
        fft_spectra.append(np.interp(FFT_COMMON_FREQS, freqs_per_h, spectrum))

    fft_pool_avg = np.mean(fft_spectra, axis=0)

    print(f"\n--- EVs excluded from the analysis (start+{ANALYSIS_START_OFFSET_H:.0f}h) ---")
    for car_id, reason in excluded_evs:
        print(f"EV {car_id}: {reason}")

    ROW_FMT = "{:>6} {:>10.4f} {:>10.4f} {:>14.4f} {:>10.4f}"
    HEADER = f"{'EV':>6} {'max':>10} {'min':>10} {'freq [1/h]':>14} {'amp':>10}"

    def print_top(title, key, reverse, n=TOP_N):
        ranked = sorted(results, key=key, reverse=reverse)[:n]
        print(f"\n--- Top {n} by {title} ---")
        print(HEADER)
        for car_id, max_val, min_val, avg_freq, avg_amp in ranked:
            print(ROW_FMT.format(car_id, max_val, min_val, avg_freq, avg_amp))

    print_top("highest avg frequency", key=lambda r: r[3], reverse=True)
    print_top("lowest avg frequency", key=lambda r: r[3], reverse=False)
    print_top("highest avg amplitude", key=lambda r: r[4], reverse=True)
    print_top("lowest avg amplitude", key=lambda r: r[4], reverse=False)
    print_top("highest maxmin dif", key=lambda r: r[1] - r[2], reverse=True)
    print_top("highest avg frequency * avg amplitude", key=lambda r: r[3] * r[4], reverse=True, n=25)

    # =======================================================================
    # Per-EV FFT amplitude vs. pool average: for each EV, diff(freq) = EV
    # spectrum(freq) - fft_pool_avg(freq) over FFT_COMMON_FREQS restricted to
    # freq >= FFT_DIFF_MIN_FREQ_PER_H (2.5 /h), then mean and std of that diff
    # curve. Ranks EVs by how much their spectral shape varies around its own
    # mean deviation from the pool average.
    # =======================================================================
    fft_diff_results = []  # (car_id, mean_diff, std_diff)
    for (car_id, *_rest), spectrum in zip(results, fft_spectra):
        diff = (spectrum - fft_pool_avg)[FFT_DIFF_MASK]
        fft_diff_results.append((car_id, float(np.mean(diff)), float(np.std(diff))))

    FFT_DIFF_ROW_FMT = "{:>6} {:>14.6f} {:>14.6f}"
    FFT_DIFF_HEADER = f"{'EV':>6} {'mean diff':>14} {'std diff':>14}"

    ranked_by_std = sorted(fft_diff_results, key=lambda r: r[2], reverse=True)[:25]
    print("\n--- Top 25 EVs by std of (EV FFT amplitude - pool avg FFT amplitude) ---")
    print(FFT_DIFF_HEADER)
    for car_id, mean_diff, std_diff in ranked_by_std:
        print(FFT_DIFF_ROW_FMT.format(car_id, mean_diff, std_diff))

    ranked_by_mean_std = sorted(fft_diff_results, key=lambda r: r[1] * r[2], reverse=True)[:25]
    print("\n--- Top 25 EVs by mean diff * std diff ---")
    print(FFT_DIFF_HEADER)
    for car_id, mean_diff, std_diff in ranked_by_mean_std:
        print(FFT_DIFF_ROW_FMT.format(car_id, mean_diff, std_diff))

    # =======================================================================
    # FFT of local max temp for the selected EVs, starting ANALYSIS_START_OFFSET_H
    # (1 h) after each EV's own operational start time, plus the pooled average
    # spectrum across all EVs. Linear-detrended first -- otherwise the slow
    # overall temperature drift dominates the spectrum and hides the
    # charge/discharge-cycle peaks. Magnitude normalized by segment length so
    # EVs of different trip length are comparable.
    # =======================================================================
    fig_fft, ax_fft = plt.subplots(figsize=(12, 6))
    for ev_id in EV_IDS:
        if ev_id not in start_hours.index:
            print(f"EV {ev_id}: no operational start time, skipping in FFT plot")
            continue

        tensor = pd.read_pickle(DATA_ROOT / str(ev_id) / "vin_1.pkl")
        local_max_temp = tensor.cpu().numpy()[:, 0, MAX_TEMP_COL]
        start_idx = int(round((start_hours[ev_id] + ANALYSIS_START_OFFSET_H) * 3600.0 / SAMPLE_INTERVAL_S))
        raw_segment = local_max_temp[start_idx:]
        if raw_segment.size == 0:
            print(f"EV {ev_id}: no data left after start+{ANALYSIS_START_OFFSET_H:.0f}h, skipping in FFT plot")
            continue
        segment = detrend(raw_segment)

        spectrum = np.abs(np.fft.rfft(segment)) / segment.size
        freqs_per_h = np.fft.rfftfreq(segment.size, d=SAMPLE_INTERVAL_S) * 3600.0

        ax_fft.plot(freqs_per_h, spectrum, label=f"EV {ev_id}", linewidth=0.8)

    ax_fft.plot(FFT_COMMON_FREQS, fft_pool_avg, label="Pool average (all EVs)",
                color="black", linewidth=2.0, zorder=5)

    ax_fft.set_xlim(0, FFT_MAX_FREQ_PER_H)
    ax_fft.set_yscale("log")
    ax_fft.set_xlabel("Frequency [1/h]")
    ax_fft.set_ylabel("Magnitude (normalized, log scale)")
    ax_fft.set_title(f"vin_1 local max temp FFT, detrended (from start+{ANALYSIS_START_OFFSET_H:.0f}h)")
    ax_fft.legend(fontsize=8)
    fig_fft.savefig(OUT_DIR / f"vin1_local_max_temp_fft_ev{'_'.join(str(e) for e in EV_IDS)}.png")

    # =======================================================================
    # Scatters, faulty EVs labeled: (1) max vs. min, (2) avg freq vs. avg amp
    # =======================================================================
    results_df = pd.DataFrame(results, columns=["ev_id", "max_val", "min_val", "avg_freq", "avg_amp"])
    fft_diff_df = pd.DataFrame(fft_diff_results, columns=["ev_id", "mean_diff", "std_diff"])
    results_df = results_df.merge(fft_diff_df, on="ev_id", how="left")
    labels_df = pd.read_excel(LABELS_FILE).rename(columns={"Num": "ev_id"})
    results_df = results_df.merge(labels_df[["ev_id", "Label"]], on="ev_id", how="left")

    faulty = results_df["Label"] == 1
    healthy = ~faulty

    # =======================================================================
    # Save max/min + FFT mean/std diff + avg freq/amplitude per EV to
    # extracted_data/whole_features/local_temp.pkl
    # =======================================================================
    LOCAL_TEMP_OUT = PROJECT_ROOT / "extracted_data" / "whole_features" / "local_temp.pkl"

    local_temp_df = results_df[["ev_id", "max_val", "min_val", "mean_diff", "std_diff",
                                 "avg_freq", "avg_amp"]].set_index("ev_id")
    local_temp_df.to_pickle(LOCAL_TEMP_OUT)
    print(f"\nSaved local temp features to: {LOCAL_TEMP_OUT}")

    # =======================================================================
    # Pooled avg/std per feature, healthy vs. faulty
    # =======================================================================
    results_df["minmax_dif"] = results_df["max_val"] - results_df["min_val"]
    results_df["freq_amp"] = results_df["avg_freq"] * results_df["avg_amp"]

    STATS_FEATURES = ["max_val", "min_val", "avg_freq", "avg_amp", "minmax_dif", "freq_amp"]
    STATS_ROW_FMT = "{:>12} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}"
    STATS_HEADER = f"{'feature':>12} {'healthy avg':>12} {'healthy std':>12} {'faulty avg':>12} {'faulty std':>12}"

    print(f"\n--- Pooled avg / std per feature, healthy vs. faulty ---")
    print(STATS_HEADER)
    for feature in STATS_FEATURES:
        h = results_df.loc[healthy, feature]
        f = results_df.loc[faulty, feature]
        print(STATS_ROW_FMT.format(feature, h.mean(), h.std(), f.mean(), f.std()))

    def scatter_faulty(x_col, y_col, xlabel, ylabel, title, out_name):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(results_df.loc[healthy, x_col], results_df.loc[healthy, y_col],
                   s=20, color=RWTHColorScales.blue[0], alpha=0.7, label="healthy")
        ax.scatter(results_df.loc[faulty, x_col], results_df.loc[faulty, y_col],
                   s=20, color=RWTHColorScales.red[0], alpha=0.9, label="faulty")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)
        fig.savefig(OUT_DIR / out_name, bbox_inches="tight")

    scatter_faulty("max_val", "min_val",
                    f"max local temp (from start+{ANALYSIS_START_OFFSET_H:.0f}h)",
                    f"min local temp (from start+{ANALYSIS_START_OFFSET_H:.0f}h)",
                    "vin_1 local max temp: max vs. min per EV",
                    "vin1_local_max_temp_scatter.png")

    scatter_faulty("avg_freq", "avg_amp",
                    f"avg. frequency [1/h] (from start+{ANALYSIS_START_OFFSET_H:.0f}h)",
                    f"avg. amplitude (from start+{ANALYSIS_START_OFFSET_H:.0f}h)",
                    "vin_1 local max temp: avg. frequency vs. avg. amplitude per EV",
                    "vin1_local_max_temp_freq_amp_scatter.png")

    scatter_faulty("mean_diff", "std_diff",
                    "mean(EV FFT amplitude - pool avg FFT amplitude)",
                    "std(EV FFT amplitude - pool avg FFT amplitude)",
                    "vin_1 local max temp FFT: mean diff vs. std diff per EV",
                    "vin1_local_max_temp_fft_diff_scatter.png")
