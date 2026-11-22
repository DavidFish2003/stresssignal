"""Train and evaluate stress detection with leave-one-person-out cross-validation.

Every person is scored by a model trained only on the other people, so the numbers
reflect how the model would do on someone it has never seen.

Run: uv run python -m stresssignal.train   ->  web/data/results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from . import features as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "data" / "results.json"
CACHE = ROOT / "data" / "windows.parquet"

REL = [c + "_rel" for c in F.FEATURES]
LABELS = {
    "eda_mean": "Skin conductance level", "eda_std": "Skin conductance variability", "eda_slope": "Skin conductance trend",
    "eda_range": "Skin conductance range", "eda_peaks": "Sweat responses (count)", "eda_dmax": "Fastest sweat response",
    "hr_mean": "Heart rate", "hr_std": "Heart rate variability (short-term)", "hr_slope": "Heart rate trend",
    "ibi_mean": "Beat-to-beat interval", "ibi_sdnn": "HRV (SDNN)", "ibi_rmssd": "HRV (RMSSD)",
    "temp_mean": "Skin temperature", "temp_slope": "Skin temperature trend", "temp_std": "Skin temperature variability",
    "acc_mean": "Movement level", "acc_std": "Movement variability",
}


def hgb():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                          l2_regularization=1.0, random_state=0)


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def load_windows() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for sid in data.subjects():
        frames.append(F.build(data.load(sid)))
        print(f"  {sid}: {len(frames[-1])} windows")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(CACHE)
    return df


def loso(df: pd.DataFrame, cols: list[str], make, importance: bool = False):
    """Leave-one-person-out. Returns per-person metrics, out-of-fold probabilities, importances."""
    probs = pd.Series(np.nan, index=df.index)
    per, imps = [], []
    for sid in df.sid.unique():
        train = df[(df.sid != sid) & (df.label >= 0)]
        test_all = df[df.sid == sid]
        test = test_all[test_all.label >= 0]
        m = make().fit(train[cols], train.label)
        probs[test_all.index] = m.predict_proba(test_all[cols])[:, 1]
        y, p = test.label.to_numpy(), probs[test.index].to_numpy()
        per.append({"sid": sid, "auroc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, p >= 0.5),
                    "f1": f1_score(y, p >= 0.5)})
        if importance:
            r = permutation_importance(m, test[cols], test.label, scoring="roc_auc", n_repeats=3, random_state=0)
            imps.append(r.importances_mean)
    per = pd.DataFrame(per)
    imp = pd.Series(np.mean(imps, axis=0), index=cols) if importance else None
    return per, probs, imp


def summary(per: pd.DataFrame) -> dict:
    return {k: {"mean": round(float(per[k].mean()), 3), "sd": round(float(per[k].std()), 3)}
            for k in ["auroc", "bal_acc", "f1"]}


def main():
    t0 = time.time()
    df = load_windows()
    train_rows = df[df.label >= 0]
    print(f"{df.sid.nunique()} people, {len(df)} windows ({(train_rows.label == 1).sum()} stress, "
          f"{(train_rows.label == 0).sum()} rest)")

    results = {}
    # Main model: gradient boosting on personal-baseline features.
    per_main, probs, imp = loso(df, REL, hgb, importance=True)
    results["Gradient boosting (personal baseline)"] = summary(per_main)
    print("main", results["Gradient boosting (personal baseline)"])

    per, _, _ = loso(df, REL, logreg)
    results["Logistic regression (personal baseline)"] = summary(per)
    per, _, _ = loso(df, F.FEATURES, hgb)
    results["Gradient boosting (no baseline)"] = summary(per)
    for k, v in results.items():
        print(f"  {k}: AUROC {v['auroc']['mean']:.3f}  bal.acc {v['bal_acc']['mean']:.3f}")

    # Which sensors carry the signal?
    sensors = {}
    for g in ["Skin conductance", "Heart", "Skin temperature", "Movement"]:
        cols = [c + "_rel" for c, grp in F.GROUPS.items() if grp == g]
        per, _, _ = loso(df, cols, hgb)
        sensors[g] = summary(per)["auroc"]
        print(f"  only {g}: AUROC {sensors[g]['mean']:.3f}")
    no_move = [c + "_rel" for c, grp in F.GROUPS.items() if grp != "Movement"]
    per, _, _ = loso(df, no_move, hgb)
    sensors["All except movement"] = summary(per)["auroc"]
    sensors["All sensors"] = summary(per_main)["auroc"]

    # Per-task detection rate (how often stress windows in each task are caught).
    df["prob"] = probs
    stress = df[df.label == 1]
    by_task = {t: round(float((g.prob >= 0.5).mean()), 3) for t, g in stress.groupby("task")}
    false_alarm = round(float((df[df.label == 0].prob >= 0.5).mean()), 3)

    people = []
    for sid, g in df.groupby("sid"):
        rec = data.load(sid)
        start = rec.eda.t0
        m = per_main[per_main.sid == sid].iloc[0]
        people.append({
            "id": sid,
            "auroc": round(float(m.auroc), 3), "bal_acc": round(float(m.bal_acc), 3),
            "tasks": [{"name": n, "start": round((s - start) / 60, 2), "end": round((e - start) / 60, 2)}
                      for n, (s, e) in zip(data.TASKS, rec.tasks)],
            "t": [round(x, 2) for x in g.t_min + F.WIN / 120],  # window centre, minutes
            "p": [None if pd.isna(x) else round(float(x), 3) for x in g.prob],
            "hr": [None if pd.isna(x) else round(float(x), 1) for x in g.hr_mean],
            "eda": [None if pd.isna(x) else round(float(x), 3) for x in g.eda_mean],
        })

    imp_out = sorted(({"feature": LABELS[c.removesuffix("_rel")], "sensor": F.GROUPS[c.removesuffix("_rel")],
                       "importance": round(float(v), 4)} for c, v in imp.items()), key=lambda d: -d["importance"])

    out = {
        "dataset": {"name": "Stress-Predict", "people": int(df.sid.nunique()), "windows": int(len(train_rows)),
                    "stress_windows": int((train_rows.label == 1).sum()), "rest_windows": int((train_rows.label == 0).sum()),
                    "window_s": F.WIN, "step_s": F.STEP,
                    "baseline_min_median": round(float(df.groupby("sid").baseline_min.first().median()), 1)},
        "models": results, "sensors": sensors, "importance": imp_out,
        "by_task": by_task, "false_alarm_rate": false_alarm, "people": people,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
