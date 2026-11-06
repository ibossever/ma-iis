"""Chosen descriptive clustering: k = 13 on windows of faulty vehicles, scaler on all
windows (same space as the warning), names F/T/H/X as in Figure 7. Saves window labels,
the profile figure and stability. Fold-wise refits live in gate_prototypes.py."""
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import common
import prototypes as pr
import rwth_style as st
from gate_k_windowed import N_BOOTSTRAP, N_JOBS, OUT_DIR

K_ALL, Z_LIMIT = 9, 3.0
DATA_DIR = common.PROJECT_ROOT / "figures" / "thesis" / "data"

windows, columns = common.load_windows("120h")
x_all = StandardScaler().fit_transform(windows[columns])
fit = pr.descriptive_fit(windows, x_all, common.SEED)
is_faulty = windows["is_faulty_ev"].to_numpy()
faulty = windows[is_faulty].reset_index(drop=True).assign(cluster_faulty_k13=fit["members"])
faulty[["ev_id", "window_start", "cluster_faulty_k13"]].to_csv(DATA_DIR / "windows_120h_faulty_k13.csv", index=False)
x = x_all[is_faulty]
table = fit["all_ordered"]
names = list(table["name"])
summary = table.assign(top_vehicle_pct=[100 * faulty.loc[faulty["cluster_faulty_k13"] == n, "ev_id"].value_counts().iloc[0] / w for n, w in zip(names, table["windows"])]).set_index("name")[
    ["label", "windows", "vehicles", "top_vehicle_pct", "large_pack_share", "ratio"]].round(2)
summary.to_csv(OUT_DIR / "faulty_k13_clusters.csv")

previous = pd.read_csv(DATA_DIR / "windows_120h_faulty_k13_faulty_scaler.csv")["cluster_faulty_k13"].to_numpy()
all_labels = pd.read_csv(OUT_DIR / "windows_120h_k9.csv")["cluster"].to_numpy()[is_faulty]
ids = faulty["ev_id"].to_numpy()
unique = np.unique(ids)
rng = np.random.default_rng(common.SEED)
rows = {ev: np.flatnonzero(ids == ev) for ev in unique}
resamples = [np.concatenate([rows[ev] for ev in rng.choice(unique, len(unique))]) for _ in range(N_BOOTSTRAP)]


def refit(fit_rows, seed):
    with threadpool_limits(1):
        return KMeans(pr.K, n_init=pr.N_INIT, random_state=seed).fit(x[fit_rows]).predict(x)


fits = Parallel(n_jobs=N_JOBS)(delayed(refit)(r, b) for b, r in enumerate(resamples))
pairs = Parallel(n_jobs=N_JOBS)(delayed(adjusted_rand_score)(a, b) for a, b in combinations(fits, 2))

st.set_style()
fig, ax = plt.subplots(figsize=(st.FULL_WIDTH, 11.0 * st.CM))
centroids = fit["centres"][table.index.to_numpy()].T
image = ax.imshow(centroids, cmap=st.CMAP_DIVERGING, vmin=-Z_LIMIT, vmax=Z_LIMIT, aspect="auto")
ax.set_xticks(range(len(names)), names)
ax.set_yticks(range(len(columns)), [st.WINDOWED_FEATURE_LABELS[c] for c in columns])
ax.set_xlabel("Cluster")
ax.tick_params(length=0)
for side in ax.spines.values():
    side.set_visible(False)
bar = fig.colorbar(image, ax=ax, shrink=0.7, aspect=22, pad=0.015)
bar.set_label("Standardised centroid value [–]")
bar.outline.set_linewidth(0.4)
st.save_figure(fig, "figA5_faulty_k13_profiles")

print(summary.to_string())
print(f"ARI vs all-vehicle k = {K_ALL}: {adjusted_rand_score(all_labels, fit['members']):.3f}")
print(f"ARI vs previous clustering (faulty-only scaler): {adjusted_rand_score(previous, fit['members']):.3f}")
print(f"stability {np.mean(pairs):.3f} +- {np.std(pairs):.3f}, silhouette {silhouette_score(x, fit['members']):.3f}")
