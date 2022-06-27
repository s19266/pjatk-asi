# Commands
```bash
conda env create -n asi --file environment.yaml
```

```bash
conda env export > environment.yaml
```

```bash
mlflow server --backend-store-uri=sqlite:///db/mlrunsdb15.db --default-artifact-root=file:mlruns --host 0.0.0.0 --port 5001
```

# Data
```
data/
├─ batches/ ← Batches of data to be trained/checked (0.csv should be initial data)
├─ source/ ← Source of data
├─ raw_batches/ ← Batches to be processed by generate_drift.py (should be ignored in DVC)
```
