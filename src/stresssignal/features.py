"""Turn raw wristband signals into 60-second feature windows with stress labels.

Each feature is then expressed relative to the person's own rest period before the
first task (their personal baseline), the way a real wearable would calibrate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .data import TASKS, Recording, Signal

WIN = 60.0    # window length (s)
STEP = 15.0   # hop between windows (s)

# Feature name -> sensor group (used for the "which sensor matters" comparison).
GROUPS = {
    "eda_mean": "Skin conductance", "eda_std": "Skin conductance", "eda_slope": "Skin conductance",
    "eda_range": "Skin conductance", "eda_peaks": "Skin conductance", "eda_dmax": "Skin conductance",
    "hr_mean": "Heart", "hr_std": "Heart", "hr_slope": "Heart",
    "ibi_mean": "Heart", "ibi_sdnn": "Heart", "ibi_rmssd": "Heart",
    "temp_mean": "Skin temperature", "temp_slope": "Skin temperature", "temp_std": "Skin temperature",
    "acc_mean": "Movement", "acc_std": "Movement",
}
FEATURES = list(GROUPS)


def _seg(sig: Signal, a: float, b: float) -> np.ndarray:
    i, j = int((a - sig.t0) * sig.fs), int((b - sig.t0) * sig.fs)
    return sig.x[max(i, 0): max(j, 0)]


def _slope(x: np.ndarray, fs: float) -> float:
    if len(x) < 3:
        return np.nan
    t = np.arange(len(x)) / fs
    return float(np.polyfit(t, x, 1)[0]) * 60.0  # per minute


def window_features(rec: Recording, a: float, b: float) -> dict[str, float]:
    f: dict[str, float] = {}
    eda = _seg(rec.eda, a, b)
    if len(eda) >= rec.eda.fs * WIN * 0.8:
        f["eda_mean"], f["eda_std"] = eda.mean(), eda.std()
        f["eda_slope"], f["eda_range"] = _slope(eda, rec.eda.fs), eda.max() - eda.min()
        # Skin conductance responses: small, fast rises on top of the slow level.
        peaks, _ = find_peaks(eda, prominence=0.01)
        f["eda_peaks"] = len(peaks)
        f["eda_dmax"] = float(np.max(np.diff(eda)) * rec.eda.fs) if len(eda) > 1 else np.nan
    hr = _seg(rec.hr, a, b)
    if len(hr) >= WIN * 0.5:
        f["hr_mean"], f["hr_std"], f["hr_slope"] = hr.mean(), hr.std(), _slope(hr, rec.hr.fs)
    beats = rec.ibi[(rec.ibi.t >= a) & (rec.ibi.t < b)]["ibi"].to_numpy()
    if len(beats) >= 10:  # need enough clean beats for heart-rate variability
        f["ibi_mean"], f["ibi_sdnn"] = beats.mean() * 1000, beats.std() * 1000
        f["ibi_rmssd"] = float(np.sqrt(np.mean(np.diff(beats) ** 2)) * 1000)
    temp = _seg(rec.temp, a, b)
    if len(temp) >= rec.temp.fs * WIN * 0.8:
        f["temp_mean"], f["temp_std"], f["temp_slope"] = temp.mean(), temp.std(), _slope(temp, rec.temp.fs)
    acc = _seg(rec.acc, a, b)
    if len(acc):
        mag = np.linalg.norm(acc, axis=1) / 64.0  # units of g
        f["acc_mean"], f["acc_std"] = mag.mean(), mag.std()
    return f


def label_window(rec: Recording, a: float, b: float) -> tuple[float, str | None]:
    """Fraction of the window inside a stress task, and which task."""
    for name, (s, e) in zip(TASKS, rec.tasks):
        overlap = max(0.0, min(b, e) - max(a, s))
        if overlap > 0:
            return overlap / (b - a), name
    return 0.0, None


def build(rec: Recording) -> pd.DataFrame:
    start = rec.eda.t0
    end = max(e for _, e in rec.tasks) + 300  # include ~5 min of recovery after the last task
    rows = []
    a = start
    while a + WIN <= end:
        frac, task = label_window(rec, a, a + WIN)
        row = {"sid": rec.sid, "t": a, "t_min": (a - start) / 60.0, "stress_frac": frac, "task": task}
        row.update(window_features(rec, a, a + WIN))
        rows.append(row)
        a += STEP
    df = pd.DataFrame(rows).reindex(columns=["sid", "t", "t_min", "stress_frac", "task", *FEATURES])

    # Label: stress if mostly inside a task, rest if not at all; ambiguous edges are dropped from training.
    df["label"] = np.where(df.stress_frac >= 0.75, 1, np.where(df.stress_frac == 0, 0, -1))

    # Personal baseline: rest period before the first task.
    base = df[(df.t + WIN) <= rec.tasks[0][0]]
    for c in FEATURES:
        mu, sd = base[c].mean(), base[c].std()
        df[c + "_rel"] = (df[c] - mu) / sd if sd and np.isfinite(sd) and sd > 0 else df[c] - mu
    df["baseline_min"] = len(base) * STEP / 60.0
    return df
