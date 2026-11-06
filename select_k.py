# %%
"""
Pick k for the centroid-distance (prototype) early warning, for BOTH window
sources, with a criterion stated BEFORE looking at detection results
(thesis_revision_plan.pdf A7 + A12 -- replaces the post-hoc k=4 of kmeans.py's
W-track, k=2 of kmeans24.py and k=9 of wisek_window.py).

Two prototype sets per source, clustered separately:
  kF -- K-means on windows of confirmed-faulty EVs  (fault prototypes)
  kH -- K-means on windows of healthy EVs           (healthy prototypes)
Both in ONE standardized space: StandardScaler fit on ALL windows (healthy +
faulty), not on faulty windows only as kmeans.py's W2 does -- the distances to
fault and healthy prototypes must be measured with the same ruler.

For every k in K_RANGE:
  silhouette -- on the full population (sampled to SILHOUETTE_SAMPLE windows).
  stability  -- N_BOOTSTRAP resamples of VEHICLES (not windows: windows of
                one EV are strongly correlated), K-means refit on each
                resample's windows, every refit labels ALL windows, mean
                pairwise ARI between the refits' labelings.

Selection rule (fixed in advance):
  among k with stability >= MIN_STABILITY, take the highest silhouette;
  if no k reaches MIN_STABILITY, take the most stable k.
  kH may additionally be 1 (single healthy mean, == kmeans.py's
  healthy_reference) -- ew_compare.py can run kH=1 as a baseline either way.

k is chosen once on the full fleet (it only uses the faulty/healthy split, no
detection outcome); the prototypes themselves are refit per CV fold on train
vehicles only in ew_compare.py.

Output:
  extracted_data/revision/select_k_<source>.csv -- population, k, silhouette,
      stability (mean/std pairwise ARI), inertia
  extracted_data/revision/select_k.json -- {source: {"kF": .., "kH": ..}}
  experiments/03_revision/figures/select_k_<source>.png
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from ew_common import REVISION_OUT, load_windows  # noqa: E402
from plot_fnc.rwth_colors import RWTHColorScales  # noqa: E402

# ===========================================================================
# CONFIG
# ===========================================================================
WINDOW_SOURCES = ["120h", "24h"]
K_RANGE = range(2, 9)       # same sweep range as kmeans.py / kmeans24.py
N_INIT = 10
RANDOM_STATE = 0
N_BOOTSTRAP = 50
MIN_STABILITY = 0.75        # mean pairwise ARI -- "stable" threshold, fixed in advance
SILHOUETTE_SAMPLE = 4000

FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
FIGURE_DPI = 300

sns.set_theme(style="whitegrid")
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=RWTHColorScales.cycle)


def sweep_k(x: np.ndarray, ev_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Silhouette + bootstrap-over-vehicles stability for every k in K_RANGE."""
    unique_evs = np.unique(ev_ids)
    rows_by_ev = {ev: np.flatnonzero(ev_ids == ev) for ev in unique_evs}
    bootstrap_rows = [
        np.concatenate([rows_by_ev[ev] for ev in rng.choice(unique_evs, size=len(unique_evs), replace=True)])
        for _ in range(N_BOOTSTRAP)
    ]
    results = []
    for k in K_RANGE:
        full_model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(x)
        silhouette = silhouette_score(x, full_model.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(x)),
                                      random_state=RANDOM_STATE)
        labelings = [
            KMeans(n_clusters=k, n_init=N_INIT, random_state=b).fit(x[rows]).predict(x)
            for b, rows in enumerate(bootstrap_rows)
        ]
        aris = [adjusted_rand_score(a, b) for a, b in combinations(labelings, 2)]
        results.append({"k": k, "silhouette": silhouette, "stability": float(np.mean(aris)),
                        "stability_std": float(np.std(aris)), "inertia": full_model.inertia_})
        print(f"  k={k}: silhouette={silhouette:.3f}  stability={np.mean(aris):.3f} +/- {np.std(aris):.3f}")
    return pd.DataFrame(results)


def choose_k(sweep: pd.DataFrame) -> int:
    stable = sweep[sweep["stability"] >= MIN_STABILITY]
    if stable.empty:
        return int(sweep.loc[sweep["stability"].idxmax(), "k"])
    return int(stable.loc[stable["silhouette"].idxmax(), "k"])


def plot_sweep(sweeps: dict[str, pd.DataFrame], chosen: dict[str, int], source: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (population, sweep) in zip(axes, sweeps.items()):
        ax.plot(sweep["k"], sweep["silhouette"], marker="o", label="silhouette")
        ax.errorbar(sweep["k"], sweep["stability"], yerr=sweep["stability_std"], marker="s", capsize=3,
                    label="bootstrap stability (mean pairwise ARI)")
        ax.axhline(MIN_STABILITY, color="black", linestyle=":", linewidth=1, label=f"stability >= {MIN_STABILITY}")
        ax.axvline(chosen[population], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{source}: {population} windows (chosen k = {chosen[population]})")
        ax.set_xlabel("k")
        ax.set_xticks(list(K_RANGE))
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


# %% sweep both populations, both sources
selection = {}
for source in WINDOW_SOURCES:
    windows, feature_columns = load_windows(source)
    x_all = StandardScaler().fit_transform(windows[feature_columns])
    is_faulty = windows["is_faulty_ev"].to_numpy()
    ev_ids = windows["ev_id"].to_numpy()
    print(f"\n=== {source}: {len(windows)} windows, {windows['ev_id'].nunique()} EVs, "
          f"{len(feature_columns)} features ===")

    sweeps, chosen = {}, {}
    for population, mask in [("faulty", is_faulty), ("healthy", ~is_faulty)]:
        print(f"{population}: {mask.sum()} windows from {len(np.unique(ev_ids[mask]))} EVs")
        sweeps[population] = sweep_k(x_all[mask], ev_ids[mask], np.random.default_rng(RANDOM_STATE))
        chosen[population] = choose_k(sweeps[population])

    pd.concat(sweeps, names=["population"]).reset_index(level=0).to_csv(
        REVISION_OUT / f"select_k_{source}.csv", index=False)
    plot_sweep(sweeps, chosen, source, FIGURES_DIR / f"select_k_{source}.png")
    selection[source] = {"kF": chosen["faulty"], "kH": chosen["healthy"]}
    print(f"--> {source}: kF = {chosen['faulty']}, kH = {chosen['healthy']}")

# %% save the choice for ew_compare.py
(REVISION_OUT / "select_k.json").write_text(json.dumps(selection, indent=2))
print(f"\nsaved {REVISION_OUT / 'select_k.json'}: {selection}")
