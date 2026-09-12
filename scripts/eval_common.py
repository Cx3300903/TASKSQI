from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.corruptions.clipping import apply_clipping
from src.corruptions.drift import add_baseline_drift
from src.corruptions.dropout import apply_signal_dropout
from src.corruptions.noise import add_gaussian_noise
from src.datasets import HospitalDataset, SHHSDataset
from src.metrics.failure import failure_metrics
from src.metrics.bootstrap import bootstrap_subject_level
from src.metrics.selective import aurc, risk_at_coverage, risk_coverage_curve
from src.models import AttnSleep, TaskSQI
from src.quality.baselines import msp_reliability, predictive_entropy_reliability, spectral_entropy_reliability
from src.utils.eval import predict_tasksqi, predict_teacher
from src.utils.training import load_model_weights


METHODS = ["MSP", "PredictiveEntropy", "SpectralEntropy", "TaskSQI"]


def load_teacher_tasksqi(cfg, device, tasksqi_checkpoint=None):
    teacher = AttnSleep(architecture=cfg["teacher"].get("architecture", "default")).to(device)
    load_model_weights(teacher, cfg["teacher"]["checkpoint"], map_location=device)
    tasksqi = TaskSQI().to(device)
    load_model_weights(tasksqi, tasksqi_checkpoint or cfg["tasksqi"]["checkpoint"], map_location=device)
    return teacher, tasksqi


def test_dataset(cfg):
    split_dir = Path(cfg["data"]["split_dir"])
    return SHHSDataset(
        cfg["data"]["shhs_npz"],
        split_dir / "shhs_test.txt",
        cfg["data"]["normalize"],
        expected_samples=cfg["data"].get("input_samples", 3000),
    )


def hospital_dataset(cfg):
    return HospitalDataset(cfg["data"]["hospital_npz"], None, cfg["data"]["normalize"], expected_samples=cfg["data"].get("hospital_input_samples", 3000))


def append_rows(path, rows, keys=("experiment", "method")):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if Path(path).exists():
        old = pd.read_csv(path)
        for _, row in new.iterrows():
            mask = np.ones(len(old), dtype=bool)
            for key in keys:
                if key in old.columns and key in row:
                    mask &= old[key] == row[key]
            old = old.loc[~mask]
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


@torch.no_grad()
def teacher_probs(teacher, x_np, device, batch_size):
    probs = []
    teacher.eval()
    starts = range(0, len(x_np), batch_size)
    total = (len(x_np) + batch_size - 1) // batch_size
    for start in tqdm(starts, desc="teacher probs", unit="batch", total=total, ascii=True):
        x = torch.as_tensor(x_np[start : start + batch_size], dtype=torch.float32, device=device)
        probs.append(torch.softmax(teacher(x), dim=1).cpu().numpy())
    return np.concatenate(probs)


def reliability_dict(tasksqi, x_np, prob_np, device, batch_size):
    return {
        "MSP": msp_reliability(prob_np),
        "PredictiveEntropy": predictive_entropy_reliability(prob_np),
        "SpectralEntropy": spectral_entropy_reliability(x_np),
        "TaskSQI": predict_tasksqi(tasksqi, x_np, device, batch_size),
    }


def apply_seen_corruption(x, name, device):
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    if name == "noise":
        out = add_gaussian_noise(xt, torch.empty(len(x), device=device).uniform_(-5, 20))
    elif name == "drift":
        out = add_baseline_drift(xt)
    elif name == "dropout":
        out = apply_signal_dropout(xt)
    else:
        raise ValueError(name)
    return out.cpu().numpy()


def apply_unseen_clipping(x, k, device):
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    return apply_clipping(xt, k).cpu().numpy()


def failure_rows(experiment, corruption, y, clean_pred, corrupt_prob, rels, subjects, n_boot=1000, seed=42):
    failure = corrupt_prob.argmax(axis=1) != y
    rows = []
    for method, rel in rels.items():
        vals = failure_metrics(failure, 1.0 - rel)
        auroc_ci = bootstrap_subject_level(
            lambda idx: failure_metrics(failure[idx], (1.0 - rel)[idx])["auroc"], subjects, n_boot=n_boot, seed=seed
        )
        auprc_ci = bootstrap_subject_level(
            lambda idx: failure_metrics(failure[idx], (1.0 - rel)[idx])["auprc"], subjects, n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "experiment": experiment,
                "corruption": corruption,
                "method": method,
                "auroc": vals["auroc"],
                "auroc_ci_low": auroc_ci["lo"],
                "auroc_ci_high": auroc_ci["hi"],
                "auprc": vals["auprc"],
                "auprc_ci_low": auprc_ci["lo"],
                "auprc_ci_high": auprc_ci["hi"],
                "n_epochs": len(y),
                "n_subjects": len(np.unique(subjects)),
            }
        )
    return rows


def stagewise_rows(experiment, corruption, y, corrupt_prob, rels):
    rows = []
    failure = corrupt_prob.argmax(axis=1) != y
    for stage in range(5):
        mask = y == stage
        if mask.sum() < 2:
            continue
        for method, rel in rels.items():
            vals = failure_metrics(failure[mask], 1.0 - rel[mask])
            rows.append({"experiment": experiment, "corruption": corruption, "stage": stage, "method": method, **vals, "n_epochs": int(mask.sum())})
    return rows


def selective_rows(experiment, y, pred, rels, coverages, subjects=None, n_boot=1000, seed=42):
    rows, curves = [], {}
    for method, rel in rels.items():
        curve = risk_coverage_curve(y, pred, rel, coverages)
        curves[method] = curve
        row = {"experiment": experiment, "method": method, "aurc": aurc(curve), "risk_at_90": risk_at_coverage(curve, 0.9)}
        if subjects is not None:
            aurc_ci = bootstrap_subject_level(
                lambda idx: aurc(risk_coverage_curve(y[idx], pred[idx], rel[idx], coverages)), subjects, n_boot=n_boot, seed=seed
            )
            risk90_ci = bootstrap_subject_level(
                lambda idx: risk_at_coverage(risk_coverage_curve(y[idx], pred[idx], rel[idx], coverages), 0.9),
                subjects,
                n_boot=n_boot,
                seed=seed,
            )
            row.update(
                {
                    "aurc_ci_low": aurc_ci["lo"],
                    "aurc_ci_high": aurc_ci["hi"],
                    "risk_at_90_ci_low": risk90_ci["lo"],
                    "risk_at_90_ci_high": risk90_ci["hi"],
                }
            )
        rows.append(row)
    return rows, curves


def clean_correct_subset(teacher, ds, device, batch_size):
    pred = predict_teacher(teacher, ds, device, batch_size)
    mask = pred["pred"] == pred["y"]
    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v for k, v in pred.items()}
