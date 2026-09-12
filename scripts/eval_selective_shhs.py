from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import append_rows, load_teacher_tasksqi, reliability_dict, selective_rows, teacher_probs, test_dataset
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.reproducibility import seed_everything
from src.utils.results import plot_risk_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    teacher, tasksqi = load_teacher_tasksqi(cfg, device)
    ds = test_dataset(cfg)
    x, y, subjects = ds.x.numpy(), ds.y.numpy(), ds.subject_ids
    prob = teacher_probs(teacher, x, device, cfg["eval"]["batch_size"])
    pred = prob.argmax(axis=1)
    rels = reliability_dict(tasksqi, x, prob, device, cfg["eval"]["batch_size"])
    rows, curves = selective_rows(
        "E3_selective_shhs", y, pred, rels, cfg["eval"]["coverages"], subjects, cfg["eval"]["n_boot"], cfg["seed"]
    )
    append_rows(cfg["outputs"]["main_table"], rows)
    plot_risk_coverage(curves, cfg["outputs"]["shhs_rc_figure"], "SHHS Raw EEG Selective Staging")
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / "eval_selective_shhs.json", n_epochs=len(y), n_subjects=len(set(subjects)))
    print(rows)


if __name__ == "__main__":
    main()
