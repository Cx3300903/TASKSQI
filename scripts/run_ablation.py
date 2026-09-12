from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yaml

from eval_common import (
    apply_seen_corruption,
    apply_unseen_clipping,
    clean_correct_subset,
    load_teacher_tasksqi,
    reliability_dict,
    selective_rows,
    teacher_probs,
    test_dataset,
)
from src.metrics.failure import failure_metrics
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.reproducibility import seed_everything


TARGETS = ["severity", "binary_flip", "js", "retention"]


def make_tasksqi_config(exp_cfg, target):
    cfg = {
        "seed": exp_cfg["seed"],
        "device": exp_cfg["device"],
        "data": exp_cfg["data"],
        "teacher": exp_cfg["teacher"],
        "tasksqi": {
            "batch_size": exp_cfg["eval"]["batch_size"],
            "epochs": 30,
            "patience": 5,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "mixed_precision": True,
            "resume": None,
            "target_type": target,
            "clean_probability": 0.2,
            "stage_balanced": True,
        },
        "outputs": {
            "checkpoint": exp_cfg["tasksqi"]["ablation_checkpoints"][target],
            "target_cache": f"outputs/checkpoints/quality_targets_{target}.npz",
            "log_dir": exp_cfg["outputs"]["log_dir"],
        },
    }
    path = Path("outputs/logs") / f"tasksqi_{target}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def evaluate_target(exp_cfg, target, device):
    teacher, tasksqi = load_teacher_tasksqi(exp_cfg, device, exp_cfg["tasksqi"]["ablation_checkpoints"][target])
    batch_size = exp_cfg["eval"]["batch_size"]
    clean = clean_correct_subset(teacher, test_dataset(exp_cfg), device, batch_size)
    seen_scores = []
    for corruption in ["noise", "drift", "dropout"]:
        x_cor = apply_seen_corruption(clean["x"], corruption, device)
        prob = teacher_probs(teacher, x_cor, device, batch_size)
        rel = reliability_dict(tasksqi, x_cor, prob, device, batch_size)["TaskSQI"]
        failure = prob.argmax(axis=1) != clean["y"]
        seen_scores.append(failure_metrics(failure, 1.0 - rel)["auroc"])
    unseen_scores = []
    for k in exp_cfg["eval"]["clipping_k"]:
        x_cor = apply_unseen_clipping(clean["x"], k, device)
        prob = teacher_probs(teacher, x_cor, device, batch_size)
        rel = reliability_dict(tasksqi, x_cor, prob, device, batch_size)["TaskSQI"]
        failure = prob.argmax(axis=1) != clean["y"]
        unseen_scores.append(failure_metrics(failure, 1.0 - rel)["auroc"])
    ds = test_dataset(exp_cfg)
    x, y = ds.x.numpy(), ds.y.numpy()
    prob = teacher_probs(teacher, x, device, batch_size)
    rel = reliability_dict(tasksqi, x, prob, device, batch_size)["TaskSQI"]
    selective, _ = selective_rows(
        "ablation", y, prob.argmax(axis=1), {"TaskSQI": rel}, exp_cfg["eval"]["coverages"], ds.subject_ids, exp_cfg["eval"]["n_boot"], exp_cfg["seed"]
    )
    return {
        "target": target,
        "seen_auroc": float(np.nanmean(seen_scores)),
        "unseen_auroc": float(np.nanmean(unseen_scores)),
        "shhs_aurc": selective[0]["aurc"],
        "checkpoint": exp_cfg["tasksqi"]["ablation_checkpoints"][target],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--skip-train", action="store_true", help="Only evaluate existing ablation checkpoints.")
    args = ap.parse_args()
    exp_cfg = load_config(args.config)
    seed_everything(exp_cfg["seed"])
    for target in TARGETS:
        cfg_path = make_tasksqi_config(exp_cfg, target)
        if not args.skip_train:
            subprocess.check_call([sys.executable, "scripts/generate_quality_targets.py", "--config", str(cfg_path)])
            subprocess.check_call([sys.executable, "scripts/train_tasksqi.py", "--config", str(cfg_path)])
    device = resolve_device(exp_cfg["device"])
    rows = [evaluate_target(exp_cfg, target, device) for target in TARGETS]
    out = Path(exp_cfg["outputs"]["ablation_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_run_log(exp_cfg, Path(exp_cfg["outputs"]["log_dir"]) / "run_ablation.json", targets=TARGETS)
    print(rows)


if __name__ == "__main__":
    main()
