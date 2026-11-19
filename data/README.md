# data/

This directory is git-ignored (except this file). Layout produced/expected by `python -m preprocessing.main`:

```
data/
  raw/                 <- Kaggle ASL Alphabet dataset (never modified by the pipeline)
  processed/
    manifests/         train.csv, validation.csv, test.csv (+ train_augmented.csv if offline augmentation)
    metadata/          labels.json, dataset_summary.json, preprocessing_config.json, run_state.json
    reports/           charts, corrupt/duplicate/split/validation reports, logs/
    augmented/         only if --offline-augment is used (train images only)
    _archive/          previous outputs moved here by --force
```

Manifest `image_path` values are relative to the dataset *class root* (the folder that contains `A`, `B`, ...);
the absolute class root used for a run is recorded as `class_root` in `metadata/dataset_summary.json`.
