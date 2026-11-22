# StressSignal

Detecting stress from a wristband's skin conductance, heart and temperature signals, evaluated on people the model has never seen.

**Live dashboard:** https://stresssignal.netlify.app

## What it does
- **Data:** the [Stress-Predict dataset](https://github.com/italha-d/Stress-Predict-Dataset) (Iqbal et al., 2022; MIT licence): 35 volunteers wearing an Empatica E4 wristband through a Stroop colour-word test, an interview and a hyperventilation task, with rest between tasks. S01 has irregular task markers and is excluded, as in the original study, leaving 34 people.
- **Windows:** the raw signals are cut into 60-second windows every 15 seconds. A window counts as stress if at least 75% of it falls inside a task, and as rest if none of it does. Windows that straddle a boundary are left out of training.
- **Features (17):** skin conductance level, trend, variability, and sweat responses; heart rate and heart-rate variability (SDNN, RMSSD); skin temperature; and movement.
- **Personal baseline:** every feature is expressed relative to the person's own rest period before the first task (median 9.8 minutes), the way a real wearable would calibrate.
- **Evaluation:** leave-one-person-out. Each person is scored by a model trained only on the other 33, so results reflect performance on new people.
- **Models:** gradient boosting (main model) and logistic regression. The evaluation also compares calibrated with uncalibrated features, and each sensor on its own.

## Results (leave-one-person-out, 34 people)

| Model | AUROC | Balanced accuracy |
|---|---|---|
| Gradient boosting, calibrated to each person | **0.79 ± 0.14** | 0.70 |
| Logistic regression, calibrated | 0.70 | 0.66 |
| Gradient boosting, not calibrated | 0.67 | 0.59 |

- **Calibration:** comparing to each person's own rest period lifts AUROC from 0.67 to 0.79.
- **By sensor (gradient boosting, calibrated):** skin conductance 0.72, heart 0.67, skin temperature 0.61, movement 0.58. Movement alone is weak, so the model isn't just detecting that people moved.
- **Most informative measurement:** skin conductance level, followed by skin temperature and beat-to-beat interval.
- **Per person:** median AUROC 0.81; 26 of 34 people reach 0.70 or higher.
- **By task:** 71% of stress minutes caught in the Stroop test, 70% in the interview, and 46% during hyperventilation. 25% of rest minutes are flagged (false alarms).

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone --depth 1 https://github.com/italha-d/Stress-Predict-Dataset.git data/stress-predict
uv run python -m stresssignal.train      # writes web/data/results.json (~8 min; on Windows set OMP_NUM_THREADS=1 first)
npx http-server web                      # view the dashboard
netlify deploy --prod --dir web          # publish (after `netlify login`)
```

## Code
- `src/stresssignal/data.py`: loads the E4 CSV files and the task timings. The first six tags per person are the start and end of the three tasks; this was checked against the authors' own stress labels.
- `src/stresssignal/features.py`: windowing, features, labels and the personal baseline.
- `src/stresssignal/train.py`: leave-one-person-out evaluation, sensor comparison, permutation importance, and export for the dashboard.
- `web/`: the static dashboard (Chart.js).

## Limits
- **Protocol:** these are lab-induced, short stressors in healthy volunteers. Real-life stress, and stress during concussion recovery, will look different.
- **Hyperventilation:** this task is partly a breathing manipulation, not purely psychological stress.
- **Heart-rate variability:** it comes from wrist PPG, which drops beats during movement, so some windows have no HRV features.
- **Not a medical device.**

## Citation
Iqbal, T., Simpkin, A.J., Roshan, D., et al. (2022). Stress Monitoring Using Wearable Sensors: A Pilot Study and Stress-Predict Dataset. *Sensors*, 22(21), 8135. https://doi.org/10.3390/s22218135
