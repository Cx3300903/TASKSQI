from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.tasksqi import TaskSQI, count_parameters
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.reproducibility import seed_everything
from src.utils.training import EarlyStopping, load_model_weights, save_checkpoint


def make_loader(x, q, y, batch_size, shuffle, stage_balanced=False):
    ds = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(q).float(), torch.from_numpy(y).long())
    if stage_balanced:
        counts = np.bincount(y, minlength=5).astype(np.float32)
        weights = 1.0 / np.maximum(counts[y], 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, optimizer, scaler, criterion, device, train=True, epoch=0, total_epochs=0):
    model.train(train)
    total = 0.0
    seen = 0
    mode = "train" if train else "val"
    pbar = tqdm(loader, desc=f"tasksqi {mode} {epoch}/{total_epochs}", unit="batch", ascii=True)
    for x, q, _ in pbar:
        x, q = x.to(device), q.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            loss = criterion(model(x), q)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total += loss.item() * len(q)
        seen += len(q)
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total / max(1, seen):.4f}")
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tasksqi.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    cache = np.load(cfg["outputs"]["target_cache"], allow_pickle=True)
    train_loader = make_loader(
        cache["x_train"], cache["q_train"], cache["y_train"], cfg["tasksqi"]["batch_size"], True, cfg["tasksqi"]["stage_balanced"]
    )
    val_loader = make_loader(cache["x_val"], cache["q_val"], cache["y_val"], cfg["tasksqi"]["batch_size"], False)
    model = TaskSQI().to(device)
    n_params = count_parameters(model)
    print(
        {
            "config": args.config,
            "device": str(device),
            "target_cache": cfg["outputs"]["target_cache"],
            "train_targets": int(len(cache["q_train"])),
            "val_targets": int(len(cache["q_val"])),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "batch_size": cfg["tasksqi"]["batch_size"],
            "target_type": cfg["tasksqi"]["target_type"],
            "stage_balanced": cfg["tasksqi"]["stage_balanced"],
            "parameters": n_params,
        },
        flush=True,
    )
    if n_params >= 200_000:
        raise RuntimeError(f"TaskSQI parameter budget exceeded: {n_params}")
    optimizer = AdamW(model.parameters(), lr=cfg["tasksqi"]["lr"], weight_decay=cfg["tasksqi"]["weight_decay"])
    if cfg["tasksqi"].get("resume"):
        ckpt = load_model_weights(model, cfg["tasksqi"]["resume"], map_location=device)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["tasksqi"]["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["tasksqi"]["mixed_precision"] and device.type == "cuda")
    criterion = nn.SmoothL1Loss()
    stopper = EarlyStopping(cfg["tasksqi"]["patience"], "min")
    best = float("inf")
    for epoch in range(1, cfg["tasksqi"]["epochs"] + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, criterion, device, train=True, epoch=epoch, total_epochs=cfg["tasksqi"]["epochs"])
        with torch.no_grad():
            va = run_epoch(model, val_loader, optimizer, scaler, criterion, device, train=False, epoch=epoch, total_epochs=cfg["tasksqi"]["epochs"])
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"epoch={epoch} train_loss={tr:.4f} val_mae_proxy={va:.4f} best_val={min(best, va):.4f} lr={lr:.6g}", flush=True)
        if va < best:
            best = va
            save_checkpoint(cfg["outputs"]["checkpoint"], model, optimizer, epoch, va, cfg)
            print(f"saved best TaskSQI checkpoint: {cfg['outputs']['checkpoint']} epoch={epoch} val_mae_proxy={va:.4f}", flush=True)
        if stopper.step(va):
            print(f"early stopping at epoch={epoch}", flush=True)
            break
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / f"train_tasksqi_{cfg['tasksqi']['target_type']}.json", best_val=best, parameters=n_params)
    print({"checkpoint": cfg["outputs"]["checkpoint"], "parameters": n_params, "best_val": best}, flush=True)


if __name__ == "__main__":
    main()
