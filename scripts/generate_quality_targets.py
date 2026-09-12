from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.corruptions.sampler import sample_training_corruption
from src.datasets import SHHSDataset
from src.models import AttnSleep
from src.quality.targets import compute_quality_target
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.reproducibility import seed_everything
from src.utils.training import load_model_weights


@torch.no_grad()
def build_split(model, ds, cfg, device, split_name: str):
    loader = DataLoader(ds, batch_size=cfg["tasksqi"]["batch_size"], shuffle=False)
    xs, qs, ys, sids = [], [], [], []
    n_seen = 0
    n_correct = 0
    corruption_counts = {}
    model.eval()
    pbar = tqdm(loader, desc=f"quality targets {split_name}", unit="batch", ascii=True)
    for x, y, sid in pbar:
        x, y = x.to(device), y.to(device)
        clean_prob = torch.softmax(model(x), dim=1)
        correct = clean_prob.argmax(dim=1) == y
        n_seen += len(y)
        n_correct += int(correct.sum().item())
        if correct.sum() == 0:
            pbar.set_postfix(seen=n_seen, kept=n_correct)
            continue
        x, y, clean_prob = x[correct], y[correct], clean_prob[correct]
        sid = np.asarray(sid)[correct.cpu().numpy()]
        x_cor, cname, params = sample_training_corruption(x, cfg["tasksqi"]["clean_probability"])
        corruption_counts[cname] = corruption_counts.get(cname, 0) + len(y)
        corrupt_prob = torch.softmax(model(x_cor), dim=1)
        if cname == "clean":
            q = torch.ones(len(y), device=device)
        else:
            q = compute_quality_target(
                cfg["tasksqi"]["target_type"], clean_prob, corrupt_prob, y, corruption_name=cname, corruption_params=params
            )
        xs.append(x_cor.cpu().numpy())
        qs.append(q.cpu().numpy())
        ys.append(y.cpu().numpy())
        sids.append(sid.astype(str))
        pbar.set_postfix(seen=n_seen, kept=n_correct, corruption=cname)
    if not xs:
        raise RuntimeError("No teacher-clean-correct epochs available for TaskSQI target generation.")
    print(
        {
            "split": split_name,
            "epochs_seen": n_seen,
            "teacher_correct_kept": n_correct,
            "keep_rate": round(n_correct / max(1, n_seen), 4),
            "corruptions": corruption_counts,
        },
        flush=True,
    )
    return np.concatenate(xs), np.concatenate(qs), np.concatenate(ys), np.concatenate(sids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tasksqi.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    split_dir = Path(cfg["data"]["split_dir"])
    expected_samples = cfg["data"].get("input_samples", 3000)
    train_ds = SHHSDataset(cfg["data"]["shhs_npz"], split_dir / "shhs_train.txt", cfg["data"]["normalize"], expected_samples=expected_samples)
    val_ds = SHHSDataset(cfg["data"]["shhs_npz"], split_dir / "shhs_val.txt", cfg["data"]["normalize"], expected_samples=expected_samples)
    print(
        {
            "config": args.config,
            "device": str(device),
            "train_epochs": len(train_ds),
            "val_epochs": len(val_ds),
            "batch_size": cfg["tasksqi"]["batch_size"],
            "target_type": cfg["tasksqi"]["target_type"],
            "input_samples": expected_samples,
            "architecture": cfg["teacher"].get("architecture", "default"),
        },
        flush=True,
    )
    model = AttnSleep(architecture=cfg["teacher"].get("architecture", "default")).to(device)
    load_model_weights(model, cfg["teacher"]["checkpoint"], map_location=device)
    x_train, q_train, y_train, sid_train = build_split(model, train_ds, cfg, device, "train")
    x_val, q_val, y_val, sid_val = build_split(model, val_ds, cfg, device, "val")
    Path(cfg["outputs"]["target_cache"]).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg["outputs"]["target_cache"],
        x_train=x_train,
        q_train=q_train,
        y_train=y_train,
        subject_train=sid_train,
        x_val=x_val,
        q_val=q_val,
        y_val=y_val,
        subject_val=sid_val,
    )
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / f"generate_quality_targets_{cfg['tasksqi']['target_type']}.json")
    print({"target_cache": cfg["outputs"]["target_cache"], "x_train": x_train.shape, "x_val": x_val.shape}, flush=True)


if __name__ == "__main__":
    main()
