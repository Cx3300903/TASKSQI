# TaskSQI 2.0 125Hz Official-Like Version

This folder is the 125Hz SHHS official-like TaskSQI project.

Main dataset:

- `data/shhs_c4a1_125hz_epochs.npz`
- Shape expected by the default teacher: `(N, 1, 3750)`
- Split directory: `splits_official_like`
- Teacher architecture: `official_shhs`

Default commands:

```bash
python scripts/make_splits.py --data data/shhs_c4a1_125hz_epochs.npz --out-dir splits_official_like --train 350 --val 50 --test 100 --seed 42
python scripts/train_teacher.py --config configs/teacher.yaml
python scripts/audit_teacher.py --config configs/teacher.yaml --npz data/shhs_c4a1_125hz_epochs.npz --split-dir splits_official_like --manifest outputs/logs/shhs_500_125hz_manifest.csv --checkpoint outputs/checkpoints/teacher.pt --out-dir outputs/teacher_audit
python scripts/generate_quality_targets.py --config configs/tasksqi.yaml
python scripts/train_tasksqi.py --config configs/tasksqi.yaml
```

