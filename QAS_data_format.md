# QAS Dataset — Raw File Format

Each vehicle folder contains three pickle files: `vin_1.pkl`, `vin_2.pkl`, `vin_3.pkl`.

```
/media/ssd_4tb/pbu-spr/battery-data/QAS/{vehicle_id}/
    vin_1.pkl
    vin_2.pkl
    vin_3.pkl
```

395 vehicles total (335 healthy, 58 fault + 2 unlabelled). 30-second sampling interval. 110 cells per pack.

---

## vin_1.pkl

**Shape**: `(N, 1, 7)` — torch.Tensor, float64, normalized to [0, 1] per vehicle.

Pack-level BMS summary. Used as input to a bidirectional LSTM.

| col | Signal |
|-----|--------|
| 0 | Pack terminal voltage |
| 1 | Pack current |
| 2 | Pack SOC |
| 3 | Max single-cell voltage |
| 6 | Max temperature |

*(cols 4 & 5 not identified with confidence)*

---

## vin_2.pkl

**Shape**: `(N, 225)` — torch.Tensor, physical units (not normalized).

Layout: `[x(2) | y(110) | z(110) | q(3)]`

Used as input to the MC-AE voltage reconstruction model.

| cols | segment | Signal |
|------|---------|--------|
| 1 | x | EKF-predicted terminal voltage `U_pre` (V) |
| 2 – 111 | y | Per-cell actual voltage, 110 cells (V) ← reconstruction target |
| 112 – 221 | z | EKF per-cell voltage deviation `DU_i`, 110 cells (~0–0.1 V) ← physical prior |
| 222 | q | Temperature (°C) |
| 223 | q | Pack SOC (0–1) |
| 224 | q | Pack current (A, positive = discharge) |

*(col 0 not identified with confidence)*

---

## vin_3.pkl

**Shape**: `(N, 226)` — torch.Tensor, physical units. One column more than vin_2.

Layout: `[x(2) | y(110) | z(110) | q(4)]`

Used as input to the MC-AE SOC reconstruction model.

| cols | segment | Signal |
|------|---------|--------|
| 2 – 111 | y | Per-cell absolute SOC, 110 cells (%) ← reconstruction target |
| 112 – 221 | z | EKF per-cell SOC deviation `x_i_pre`, 110 cells (~0) ← physical prior |
| 222 | q | Temperature (°C) |
| 223 | q | Pack SOC |
| 224 | q | **always 65535 in QAS data (corrupt)*** |
| 225 | q | Pack current (A) |

*(cols 0 & 1 not identified with confidence)*

---

## Loading

The files are torch.Tensor objects. Load with:

```python
import pickle
import torch

with open("vin_2.pkl", "rb") as f:
    data = pickle.load(f)

arr = data.cpu().numpy()  # shape (N, 225)
cell_voltages = arr[:, 2:112]   # (N, 110)
temperature   = arr[:, 222]
soc           = arr[:, 223]
current       = arr[:, 224]
```

---

## What the current pipeline actually uses

`preprocess_qas.py` reads **vin_2 only**, extracts driving segments (charging periods removed), and saves:

```
qas_driving_segments/{healthy,fault}/{vehicle_id}.pkl
```

Each file: `{'vehicle_id', 'label', 'segments', 'segment_lengths', 'n_segments', 'n_frames_total'}`  
Each segment: `numpy.ndarray` of shape `(T, 225)`, same column layout as vin_2.
