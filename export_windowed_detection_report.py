# %%
"""
Human-readable export of the windowed IsolationForest results, for visual
sanity-checking (detection.py's WINDOWED TRACK output) before wiring up
sliding_windows.py.

Input: extracted_data/early_warning/fault_scores_windowed.pkl (from
detection.py's windowed track) -- one row per (ev_id, window), with
anomaly_score and is_fault.

Output:
  extracted_data/early_warning/windowed_detection_report.csv  -- plain data
  extracted_data/early_warning/windowed_detection_report.xlsx -- same data,
      each row colored by (is_fault, is_confirmed_faulty) so the four
      possible cases (true/false positive/negative) are visible at a glance
      regardless of what program opens the file.
"""
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ===========================================================================
# CONFIG
# ===========================================================================
SCORES_IN = PROJECT_ROOT / "extracted_data" / "early_warning" / "fault_scores_windowed.pkl"  # from detection.py
CSV_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "windowed_detection_report.csv"
XLSX_OUT = PROJECT_ROOT / "extracted_data" / "early_warning" / "windowed_detection_report.xlsx"

FAULT_EV_ID_CUTOFF = 335  # EV number >= this is faulty -- same ground truth as detection.py

# Row fill colors by (is_fault, is_confirmed_faulty), matching detection.py's
# is_fault convention (flagged by the transplanted whole-session threshold)
# against the ev_id >= FAULT_EV_ID_CUTOFF ground truth.
FILL_TRUE_POSITIVE = PatternFill(fill_type="solid", fgColor="FFC7CE")   # red   -- flagged, confirmed-faulty EV
FILL_FALSE_POSITIVE = PatternFill(fill_type="solid", fgColor="FFD9A0")  # amber -- flagged, healthy EV
FILL_FALSE_NEGATIVE = PatternFill(fill_type="solid", fgColor="FFF6B3")  # pale yellow -- not flagged, confirmed-faulty EV
FILL_TRUE_NEGATIVE = None                                               # no fill -- not flagged, healthy EV


def _row_fill(is_fault: bool, is_confirmed_faulty: bool) -> PatternFill | None:
    if is_fault and is_confirmed_faulty:
        return FILL_TRUE_POSITIVE
    if is_fault and not is_confirmed_faulty:
        return FILL_FALSE_POSITIVE
    if not is_fault and is_confirmed_faulty:
        return FILL_FALSE_NEGATIVE
    return FILL_TRUE_NEGATIVE


# %% load + build report table
report = pd.read_pickle(SCORES_IN)
report["is_confirmed_faulty"] = report["ev_id"] >= FAULT_EV_ID_CUTOFF
report = report.sort_values(["ev_id", "window_start"]).reset_index(drop=True)

report.to_csv(CSV_OUT, index=False)
print(f"saved plain CSV report to {CSV_OUT}")

# %% build the formatted workbook
wb = Workbook()
ws = wb.active
ws.title = "windowed_detection_report"

ws.append(report.columns.tolist())
for row in report.itertuples(index=False):
    ws.append(row)

# color each data row (openpyxl rows are 1-indexed, row 1 is the header)
for i, row in enumerate(report.itertuples(index=False), start=2):
    fill = _row_fill(row.is_fault, row.is_confirmed_faulty)
    if fill is None:
        continue
    for col in range(1, len(report.columns) + 1):
        ws.cell(row=i, column=col).fill = fill

# freeze header row
ws.freeze_panes = "A2"

# auto-size columns to the widest of header/content
for col_idx, col_name in enumerate(report.columns, start=1):
    max_len = max(len(str(col_name)), report[col_name].astype(str).map(len).max())
    ws.column_dimensions[get_column_letter(col_idx)].width = max_len + 2

wb.save(XLSX_OUT)
print(f"saved formatted XLSX report to {XLSX_OUT}")

# %% summary counts for the four categories
counts = report.groupby(["is_fault", "is_confirmed_faulty"]).size()
n_tp = int(counts.get((True, True), 0))
n_fp = int(counts.get((True, False), 0))
n_fn = int(counts.get((False, True), 0))
n_tn = int(counts.get((False, False), 0))

print(f"\n--- Windowed detection report: {len(report)} windows ---")
print(f"is_fault=True,  is_confirmed_faulty=True  (red, likely TP):    {n_tp}")
print(f"is_fault=True,  is_confirmed_faulty=False (amber, FP):         {n_fp}")
print(f"is_fault=False, is_confirmed_faulty=True  (pale yellow, FN):   {n_fn}")
print(f"is_fault=False, is_confirmed_faulty=False (no fill, TN):       {n_tn}")
