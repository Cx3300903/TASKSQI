# TaskSQI

Task-aware EEG usability estimation for reliable single-channel sleep staging.

This repository contains the code for training an AttnSleep teacher, training the lightweight TaskSQI estimator, and evaluating failure detection and selective sleep staging under signal degradations. Raw datasets, preprocessed NPZ files, trained checkpoints, logs, manifests, and generated result tables are intentionally not included.

## Repository Contents

- `src/`: models, datasets, corruptions, quality targets, metrics, and utilities.
- `scripts/`: preprocessing, training, evaluation, ablation, and figure-generation scripts.
- `configs/`: YAML configurations for teacher training, TaskSQI training, and evaluation.
- `tests/`: lightweight unit tests.
- `third_party/attnsleep_official/`: vendored AttnSleep source and license.

## Installation

```bash
python -m pip install -r requirements.txt
```

Optional dependencies may be needed if you reconvert raw EDF/RML data locally.

## Data

The default configs expect 30-s single-channel EEG epochs sampled at 125 Hz:

```text
data/shhs_c4a1_125hz_epochs.npz
data/hospital_c4a1_125hz_epochs.npz
```

Each NPZ should contain:

- `x`: float array shaped `[N, 1, 3750]`
- `y`: integer labels in `{0,1,2,3,4}` for `W,N1,N2,N3,REM`
- `subject_ids`: subject identifier per epoch

Dataset files are not distributed with this repository. Use the preprocessing scripts with locally authorized data access, or run the synthetic-data smoke test below.

## Smoke Test

```bash
python scripts/make_synthetic_data.py --out data/shhs_c4a1_125hz_epochs.npz --subjects 12 --samples 3750
python scripts/make_synthetic_data.py --out data/hospital_c4a1_125hz_epochs.npz --subjects 6 --samples 3750
python scripts/make_splits.py --data data/shhs_c4a1_125hz_epochs.npz --out-dir splits_official_like --train 8 --val 2 --test 2 --seed 42
python -m unittest discover -s tests -v
```

## Main Pipeline

```bash
python scripts/make_splits.py --data data/shhs_c4a1_125hz_epochs.npz --out-dir splits_official_like --train 350 --val 50 --test 100 --seed 42
python scripts/train_teacher.py --config configs/teacher.yaml
python scripts/generate_quality_targets.py --config configs/tasksqi.yaml
python scripts/train_tasksqi.py --config configs/tasksqi.yaml
python scripts/eval_seen_corruption.py --config configs/experiment.yaml
python scripts/eval_unseen_corruption.py --config configs/experiment.yaml
python scripts/eval_selective_shhs.py --config configs/experiment.yaml
python scripts/eval_selective_hospital.py --config configs/experiment.yaml
python scripts/run_ablation.py --config configs/experiment.yaml
```

Generated artifacts are written under `outputs/` and are ignored by git by default.

## Notes

- SHHS splits are subject-level.
- TaskSQI is trained on SHHS only; the hospital cohort is used for external evaluation.
- Clipping/saturation is held out from TaskSQI training and used only for unseen-corruption evaluation.
- Confidence intervals in the evaluation scripts use subject-level bootstrapping.
