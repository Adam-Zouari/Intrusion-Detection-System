# Machine-learning workflow

This guide reproduces the CIC-IDS-2017 research workflow in chronological order. Static dataset semantics live in [DATASET.md](DATASET.md), metric definitions in [METRICS.md](METRICS.md), and the completed decision record in [RESULTS.md](RESULTS.md).

The raw and processed datasets, MLflow and Optuna databases, generated reports, native booster backup, and non-final model artifacts remain local and uncommitted. The frozen multiclass Joblib model used by the application is committed. The notebooks retain their executed outputs, and selected final diagnostics are published under `reports/published/`.

## 1. Environment setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The editable installation exposes the `ids_ml` package and its commands while keeping source edits immediately available.

### NVIDIA GPU setup

The standard dependency declaration installs the normal PyTorch package. On a CUDA-capable Windows machine, install the project-tested CUDA build afterward:

```powershell
python -m pip install --force-reinstall -r ml/requirements-gpu.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Neural screening and tuned XGBoost training use CUDA when required by their workflow. All recorded inference latency and throughput measurements use CPU inference for a consistent deployment comparison.

## 2. Merged source dataset

Download the official CIC-IDS-2017 flow CSV exports and concatenate them without cleaning into:

```text
ml/data/raw/cicids2017_merged.csv
```

This source file remains local and uncommitted. [DATASET.md](DATASET.md) records its provenance and feature meanings.

## 3. Data exploration and cleaning

Notebook 1 is executed with:

```powershell
jupyter lab ml/notebooks/01_data_exploration.ipynb
```

Notebook 1 investigates the target, class imbalance, missing and infinite values, constant features, invalid measurements, and duplicate records. It excludes seven records with invalid TCP-header measurements and writes the 2,824,745-row dataset to:

```text
ml/data/processed/cicids2017_cleaned.parquet
```

**Published benchmark:** Notebook 2 and the training commands below still expect the earlier 2,824,752-row export. Running the current Notebook 1 replaces that file; the old shape check will reject the revised data. To reproduce the published input, run Notebook 1 from revision `5425709` with the same merged CSV.

## 4. Feature preparation and protected split

Notebook 2 is executed with:

```powershell
jupyter lab ml/notebooks/02_feature_engineering.ipynb
```

Notebook 2 removes the four identifier or leakage-prone columns, defines the 71 eligible source features, creates the reproducible stratified partitions, calculates mutual-information and correlation diagnostics, and documents the optional seven-feature reduction. It does not fit a final model or use the protected test set for model selection.

## 5. Model-screening rounds

The complete baseline and tree matrices execute with:

```powershell
ids-run-baselines
ids-run-trees
```

The scikit-learn MLP stays on CPU. It uses the same explicit stratified 10% stopping split as the PyTorch challengers, with the scaler and any balanced weights fitted only on the inner-training rows. Both weighting modes select the epoch using unweighted macro F1 across the fixed 15 labels. The earliest highest-scoring epoch is retained; patience resets on an improvement of at least `1e-4` over the last qualifying score. A fresh model and preprocessor then fit all fitting rows for that epoch count, with balanced weights recalculated when applicable. The outer validation set is scored only after refitting.

Selection logs loss, inner-validation macro F1, and elapsed time per epoch in the terminal and MLflow. Refit logs loss and duration under separate names. The MLP logs both inner split fingerprints and per-class support in the run context. See [METRICS.md](METRICS.md) for the timing boundaries.

The neural families that produced the eight completed configurations in the published investigation execute with:

```powershell
ids-run-neural --models mlp resnet
```

The larger neural matrix was stopped midway. FT-Transformer and TabNet required substantial computation while their monitored validation behavior showed no meaningful improvement. This was a resource-allocation decision, not proof that those architectures can never perform well.

Every screening command trains models, records metrics and artifacts in MLflow, skips matching successful configurations by default, and prints a leaderboard. Use `--rerun` only to intentionally duplicate completed work.

To replace existing MLP baseline results with the explicit-split version without rerunning other models:

```powershell
ids-run-baselines --models mlp --rerun
```

Use the printed run-group summary to confirm **4/4** new fits finished. The leaderboard alone can fall back to older successful results for a failed configuration.

New training runs record the workflow command as **Source**, use the effective Git user name for **Created by** if configured, and include Git revision and an `environment.json` artifact. Without a Git name, MLflow supplies the creator. Search trials, verification/stability fits, and final refits use the same tracking. Memory is recorded per fit when observable, with measurement method and scope recorded alongside it. Earlier runs without telemetry are not backfilled.

## 6. Screening-results investigation

The committed read-only investigation opens with:

```powershell
jupyter lab ml/notebooks/03_model_screening_analysis.ipynb
```

It compares the completed screening evidence in decision order and contains no training or protected-test evaluation. Its conclusion hands off XGBoost and LightGBM to tuning.

For terminal-only inspection of local runs:

```powershell
ids-show-results
ids-show-results --round baseline tree
ids-show-results --round neural
```

These commands query existing MLflow records; they do not train or score models.
Generated leaderboard CSVs include available memory measurements. Missing, unobserved, and inapplicable values stay blank.

## 7. XGBoost and LightGBM tuning

First smoke-test the training paths without starting full studies:

```powershell
ids-tune-trees search --smoke-only
```

The resumable studies target 20 successful trials per model:

```powershell
ids-tune-trees search --target-trials 20
```

`--target-trials` is the total target, so interrupted studies resume without repeating successful trials. Optuna search uses an inner validation holdout and never evaluates the outer validation or protected test partitions.

XGBoost tuning uses CUDA. LightGBM tuning uses the deterministic CPU build installed for this project.
Each new Optuna trial measures memory during its own booster fit. The matrices are prepared once per search invocation and already resident at the start of each trial. The tuning report exports trial and verification CSVs with the available memory fields.

## 8. Strongest-trial verification

```powershell
ids-tune-trees verify
ids-tune-trees report
```

`verify` refits the three strongest trials from each study and evaluates them on the fixed outer validation partition. It then compares the selected XGBoost and LightGBM settings across development-split seeds `42`, `123`, and `2025`. The protected test partition is not scored.

The diagnostic original-versus-tuned XGBoost comparison is:

```powershell
ids-tune-trees compare-original-xgboost
```

This command trains only missing comparison configurations and does not participate directly in automatic final selection.

## 9. Selected-model freeze

```powershell
ids-tune-trees freeze-final
```

This read-only selection command chooses the verified tuned XGBoost run with the highest full-precision outer-validation macro F1 and writes `ml/final_model_spec.json`. It validates the dataset, split, feature, parameter, and iteration contracts without predicting on the test partition.

The committed specification records the completed project's frozen recipe. A fresh reproduction must recreate compatible source MLflow runs before it can regenerate or validate that recipe.

## 10. One-time final evaluation

```powershell
ids-tune-trees evaluate-final
ids-tune-trees final-report
```

`evaluate-final` is the only command allowed to predict on the protected 20% test partition. It refits the frozen recipe on the complete 80% development data, serializes and reload-verifies the pipeline, evaluates the test once, and blocks another successful evaluation for the same protected-test fingerprint.

The published project has already completed this step. Its result is final and must not be used to restart tuning. `final-report` is read-only.

## 11. Published evidence

[RESULTS.md](RESULTS.md) explains the actual decisions using the recorded baseline, tree, neural, tuning, stability, and protected-test results. The selected final diagnostic CSV and PNG artifacts are available in [reports/published](reports/published/README.md) without the local MLflow database.

## Application handoff

Model development ends after the frozen pipeline and protected-test result are produced. Runtime serving is owned by the separate [`ids_backend`](../backend/README.md) package, while the temporary replay producer belongs to [`ids_tools`](../tools/README.md). See the [application guide](../docs/APPLICATION.md) for the flow contract, setup, endpoints, and execution commands.

For ad-hoc inspection when local tracking state is available:

```powershell
mlflow ui --backend-store-uri sqlite:///ml/mlflow.db
```

## Implementation and tests

The [source-package guide](src/ids_ml/README.md) documents module ownership. The deeper [tree-tuning guide](src/ids_ml/tree_tuning/README.md) documents tuning dependencies and final-test isolation. The [test guide](../tests/README.md) explains the data-free suite used locally and in GitHub Actions.

```powershell
python -m pytest -q
```

The tests do not require CIC-IDS-2017, a GPU, or existing MLflow or Optuna state. Training-path tests use short synthetic fits and temporary tracking stores instead of running the benchmark suites.
