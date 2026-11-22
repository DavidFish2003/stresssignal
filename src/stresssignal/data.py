"""Load Empatica E4 recordings and task timings from the Stress-Predict dataset.

Dataset: Iqbal et al., "Stress Monitoring Using Wearable Sensors: A Pilot Study and
Stress-Predict Dataset" (MIT licence), https://github.com/italha-d/Stress-Predict-Dataset
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "stress-predict" / "Raw_data"

# The first six tags are start/end of the three stress tasks, in this order.
# Verified against the authors' own per-second stress labels (Processed_data).
TASKS = ["Stroop colour-word test", "Interview", "Hyperventilation"]

# S01 has an irregular tag file and is excluded from the authors' processed data too.
EXCLUDED = {"S01"}


@dataclass
class Signal:
    t0: float          # unix start time (s)
    fs: float          # sample rate (Hz)
    x: np.ndarray      # samples (n,) or (n, k)

    @property
    def t(self) -> np.ndarray:
        return self.t0 + np.arange(len(self.x)) / self.fs


@dataclass
class Recording:
    sid: str
    eda: Signal
    temp: Signal
    hr: Signal
    acc: Signal
    ibi: pd.DataFrame          # columns: t (unix s), ibi (s)
    tasks: list[tuple[float, float]]   # (start, end) unix s for each task in TASKS


def _read_e4(path: Path) -> Signal:
    head = pd.read_csv(path, header=None, nrows=2)
    t0, fs = float(head.iloc[0, 0]), float(head.iloc[1, 0])
    x = pd.read_csv(path, header=None, skiprows=2).to_numpy(dtype=float)
    return Signal(t0, fs, x[:, 0] if x.shape[1] == 1 else x)


def _read_ibi(path: Path) -> pd.DataFrame:
    t0 = float(pd.read_csv(path, header=None, nrows=1).iloc[0, 0])
    try:
        d = pd.read_csv(path, header=None, skiprows=1, names=["dt", "ibi"])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["t", "ibi"])
    return pd.DataFrame({"t": t0 + d["dt"].astype(float), "ibi": d["ibi"].astype(float)})


def subjects() -> list[str]:
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and p.name not in EXCLUDED)


def load(sid: str) -> Recording:
    d = RAW / sid
    tags = [float(x) for x in (d / f"tags_{sid}.csv").read_text().split()]
    tasks = [(tags[0], tags[1]), (tags[2], tags[3]), (tags[4], tags[5])]
    return Recording(
        sid=sid,
        eda=_read_e4(d / "EDA.csv"),
        temp=_read_e4(d / "TEMP.csv"),
        hr=_read_e4(d / "HR.csv"),
        acc=_read_e4(d / "ACC.csv"),
        ibi=_read_ibi(d / "IBI.csv"),
        tasks=tasks,
    )
