from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.corruptions.drift import add_baseline_drift
from src.corruptions.dropout import apply_signal_dropout
from src.corruptions.noise import add_gaussian_noise
from eval_common import (
    append_rows,
    apply_seen_corruption,
    clean_correct_subset,
    failure_rows,
    load_teacher_tasksqi,
    reliability_dict,
    stagewise_rows,
    teacher_probs,
    test_dataset,
)
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.reproducibility import seed_everything
from src.utils.results import plot_corruption_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    teacher, tasksqi = load_teacher_tasksqi(cfg, device)
    batch_size = cfg["eval"]["batch_size"]
    clean = clean_correct_subset(teacher, test_dataset(cfg), device, batch_size)
    rows, stage_rows = [], []
    for corruption in ["noise", "drift", "dropout"]:
        x_cor = apply_seen_corruption(clean["x"], corruption, device)
        prob = teacher_probs(teacher, x_cor, device, batch_size)
        rels = reliability_dict(tasksqi, x_cor, prob, device, batch_size)
        rows.extend(
            failure_rows(
                "E1_seen_corruption",
                corruption,
                clean["y"],
                clean["pred"],
                prob,
                rels,
                clean["subject_ids"],
                cfg["eval"]["n_boot"],
                cfg["seed"],
            )
        )
        stage_rows.extend(stagewise_rows("E1_seen_corruption", corruption, clean["y"], prob, rels))
    curve_rows = []
    xt = torch.as_tensor(clean["x"], dtype=torch.float32, device=device)
    for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        snr = 20.0 - 25.0 * sev
        drop = 0.05 + 0.45 * sev
        amp = 0.1 + 1.4 * sev
        variants = {
            "noise": add_gaussian_noise(xt, torch.full((len(xt),), snr, device=device)),
            "dropout": apply_signal_dropout(xt, torch.full((len(xt),), drop, device=device)),
            "drift": add_baseline_drift(
                xt,
                f=torch.full((len(xt),), 0.05 + 0.45 * sev, device=device),
                amplitude=torch.full((len(xt),), amp, device=device) * xt.std(dim=-1).squeeze(1).clamp_min(1e-6),
            ),
        }
        for name, x_var in variants.items():
            q = reliability_dict(tasksqi, x_var.cpu().numpy(), prob, device, batch_size)["TaskSQI"]
            curve_rows.append({"corruption": name, "severity": sev, "quality": float(q.mean())})
    plot_corruption_curve(curve_rows, cfg["outputs"]["corruption_curve_figure"])
    append_rows(cfg["outputs"]["main_table"], rows, keys=("experiment", "corruption", "method"))
    append_rows(cfg["outputs"]["stagewise_table"], stage_rows, keys=("experiment", "corruption", "stage", "method"))
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / "eval_seen_corruption.json", n_clean_correct=len(clean["y"]))
    print(rows)


if __name__ == "__main__":
    main()
