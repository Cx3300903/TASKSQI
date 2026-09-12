from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_common import (
    append_rows,
    apply_unseen_clipping,
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
    for k in cfg["eval"]["clipping_k"]:
        x_cor = apply_unseen_clipping(clean["x"], float(k), device)
        prob = teacher_probs(teacher, x_cor, device, batch_size)
        rels = reliability_dict(tasksqi, x_cor, prob, device, batch_size)
        label = f"clipping_k={k}"
        rows.extend(
            failure_rows(
                "E2_unseen_corruption",
                label,
                clean["y"],
                clean["pred"],
                prob,
                rels,
                clean["subject_ids"],
                cfg["eval"]["n_boot"],
                cfg["seed"],
            )
        )
        stage_rows.extend(stagewise_rows("E2_unseen_corruption", label, clean["y"], prob, rels))
    append_rows(cfg["outputs"]["main_table"], rows, keys=("experiment", "corruption", "method"))
    append_rows(cfg["outputs"]["stagewise_table"], stage_rows, keys=("experiment", "corruption", "stage", "method"))
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / "eval_unseen_corruption.json", n_clean_correct=len(clean["y"]))
    print(rows)


if __name__ == "__main__":
    main()
