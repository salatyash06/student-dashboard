# Scholar / Student Performance Studio

A light-theme Streamlit dashboard for the original Python for Data Science project. The supplied trained model, feature list, dataset, and notebook are preserved.

## Start on Windows

Install Python 3.12 or 3.13. Open a terminal **inside this extracted project folder**, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in the terminal (normally http://localhost:8501).
Run from this folder so Streamlit reads `.streamlit/config.toml` and uses the light theme.

macOS / Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

## Features

- **Prediction studio:** five original inputs, representative starting profile, tooltips, reset, score card, comparison with the dataset average, and score-distribution chart.
- **What-if lab:** edit two profiles, compare predictions, inspect the changed inputs, and download the comparison.
- **Data explorer:** combine school, motivation, and exam-score filters; explore interactive charts; inspect and download filtered records. Empty selections show a helpful empty state.
- **Prediction history:** submitted profiles and full-precision predictions, sortable table, CSV export, and clear-history control. History is per browser session, capped at 100 entries, and lost when the session ends. What-if calculations do not add entries.
- Responsive light styling, keyboard-accessible native controls, chart zoom/hover/image export, and reduced-motion support.

## Model behavior and interpretation

The original app supplies five numeric values and reindexes them to the saved model's 27 features, filling every other feature with zero. This dashboard preserves that transformation and the supplied model; it does not retrain it or claim improved prediction accuracy. The interface explains this assumption.

The initial score is an explicitly labeled example based on the default profile. Only clicking **Predict score** saves a prediction to history. The result card always describes the last submitted profile; changing form controls does not silently change a saved prediction.

Predictions are estimates, not guarantees, validated causal effects, or confidence intervals. A difference between predictions shows only the model's response with other encoded features fixed at zero. No accuracy percentage is fabricated.

The dataset includes an `Exam_Score` of 101. Charts and exports preserve source values. Model outputs are displayed without clipping; the decorative progress bar alone is limited to 0–100. Inputs outside observed data ranges are flagged as extrapolation.

The scikit-learn version is pinned to **1.8.0**, the version recorded in the original model. Load only trusted pickle model files.

## Project files

```text
app.py                    Dashboard and page interactions
dashboard_core.py         Original input transformation and validation
assets/style.css          Light design and responsive adjustments
.streamlit/config.toml    Native light theme
models/                   Original model and feature list
data/                     Original student dataset
notebooks/                Original analysis notebook
tests/                    Interaction and model regression checks
requirements.txt          Dashboard dependencies
requirement.txt           Original install filename compatibility
requirements-notebook.txt Optional notebook dependencies
```

## Run checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

To use the original notebook, also install `requirements-notebook.txt`.
The supplied `train_model.py` was empty and is retained unchanged.
The original machine-specific `venv` is omitted; create a fresh environment using the commands above.
