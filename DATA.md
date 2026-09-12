# Data Policy

This public code package does not include raw EEG/PSG recordings, preprocessed epoch arrays, hospital data, model checkpoints, subject manifests, or generated logs.

To run the full experiments, prepare local NPZ files with the following schema:

```text
x           float32, [N, 1, 3750]
y           int64, labels 0=W, 1=N1, 2=N2, 3=N3, 4=REM
subject_ids per-epoch subject identifiers
```

Place authorized local files at:

```text
data/shhs_c4a1_125hz_epochs.npz
data/hospital_c4a1_125hz_epochs.npz
```

Do not commit these files to git. The repository `.gitignore` excludes common raw, preprocessed, checkpoint, log, and result formats.
