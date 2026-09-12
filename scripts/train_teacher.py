from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.datasets import SHHSDataset
from src.metrics import classification_metrics
from src.models import AttnSleep
from src.utils.config import load_config, resolve_device, write_run_log
from src.utils.eval import class_weights_from_dataset, predict_teacher
from src.utils.reproducibility import seed_everything
from src.utils.training import EarlyStopping, load_model_weights, save_checkpoint


def ensure_finite(name: str, value, epoch: int, batch_idx: int | None = None):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if not torch.isfinite(tensor).all():
        where = f"epoch={epoch}"
        if batch_idx is not None:
            where += f" batch={batch_idx}"
        raise FloatingPointError(f"Non-finite {name} detected at {where}.")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, epoch, total_epochs, grad_clip_norm=None):
    model.train()
    total = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"teacher train {epoch}/{total_epochs}", unit="batch", ascii=True)
    for batch_idx, (x, y, _) in enumerate(pbar, start=1):
        ensure_finite("input", x, epoch, batch_idx)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            ensure_finite("logits", logits, epoch, batch_idx)
            loss = criterion(logits, y)
        ensure_finite("loss", loss, epoch, batch_idx)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        seen += len(y)
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total / max(1, seen):.4f}")
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    total = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"teacher val {epoch}/{total_epochs}", unit="batch", ascii=True)
    for batch_idx, (x, y, _) in enumerate(pbar, start=1):
        ensure_finite("input", x, epoch, batch_idx)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ensure_finite("logits", logits, epoch, batch_idx)
        loss = criterion(logits, y)
        ensure_finite("loss", loss, epoch, batch_idx)
        total += loss.item() * len(y)
        seen += len(y)
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total / max(1, seen):.4f}")
    return total / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/teacher.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["device"])
    split_dir = Path(cfg["data"]["split_dir"])
    expected_samples = cfg["data"].get("input_samples", 3000)
    train_ds = SHHSDataset(cfg["data"]["shhs_npz"], split_dir / "shhs_train.txt", cfg["data"]["normalize"], expected_samples=expected_samples)
    val_ds = SHHSDataset(cfg["data"]["shhs_npz"], split_dir / "shhs_val.txt", cfg["data"]["normalize"], expected_samples=expected_samples)
    test_ds = SHHSDataset(cfg["data"]["shhs_npz"], split_dir / "shhs_test.txt", cfg["data"]["normalize"], expected_samples=expected_samples)
    train_loader = DataLoader(train_ds, batch_size=cfg["teacher"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["teacher"]["batch_size"], shuffle=False)
    print(
        {
            "config": args.config,
            "device": str(device),
            "train_epochs": len(train_ds),
            "val_epochs": len(val_ds),
            "test_epochs": len(test_ds),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "batch_size": cfg["teacher"]["batch_size"],
            "input_samples": expected_samples,
            "architecture": cfg["teacher"].get("architecture", "default"),
            "mixed_precision": cfg["teacher"]["mixed_precision"],
            "lr": cfg["teacher"]["lr"],
            "grad_clip_norm": cfg["teacher"].get("grad_clip_norm"),
        },
        flush=True,
    )
    model = AttnSleep(cfg["teacher"]["num_classes"], architecture=cfg["teacher"].get("architecture", "default")).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["teacher"]["lr"], weight_decay=cfg["teacher"]["weight_decay"])
    if cfg["teacher"].get("resume"):
        ckpt = load_model_weights(model, cfg["teacher"]["resume"], map_location=device)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
    weights = class_weights_from_dataset(train_ds, cfg["teacher"]["num_classes"]).to(device)
    train_counts = np.bincount(train_ds.y.numpy(), minlength=cfg["teacher"]["num_classes"])
    print(
        {
            "class_counts": {int(i): int(c) for i, c in enumerate(train_counts)},
            "class_weights": [round(float(w), 4) for w in weights.detach().cpu()],
        },
        flush=True,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["teacher"]["epochs"]))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["teacher"]["mixed_precision"] and device.type == "cuda")
    print({"amp_enabled": scaler.is_enabled()}, flush=True)
    stopper = EarlyStopping(cfg["teacher"]["patience"], mode="min")
    best = float("inf")
    for epoch in range(1, cfg["teacher"]["epochs"] + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            scaler.is_enabled(),
            epoch,
            cfg["teacher"]["epochs"],
            cfg["teacher"].get("grad_clip_norm"),
        )
        va = evaluate_loss(model, val_loader, criterion, device, epoch, cfg["teacher"]["epochs"])
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"epoch={epoch} train_loss={tr:.4f} val_loss={va:.4f} best_val={min(best, va):.4f} lr={lr:.6g}", flush=True)
        if va < best:
            best = va
            save_checkpoint(cfg["outputs"]["checkpoint"], model, optimizer, epoch, va, cfg)
            print(f"saved best teacher checkpoint: {cfg['outputs']['checkpoint']} epoch={epoch} val_loss={va:.4f}", flush=True)
        if stopper.step(va):
            print(f"early stopping at epoch={epoch}", flush=True)
            break
    load_model_weights(model, cfg["outputs"]["checkpoint"], map_location=device)
    print("evaluating teacher on test split...", flush=True)
    pred = predict_teacher(model, test_ds, device, cfg["teacher"]["batch_size"])
    metrics = classification_metrics(pred["y"], pred["pred"])
    Path(cfg["outputs"]["table"]).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**metrics, "checkpoint": cfg["outputs"]["checkpoint"]}]).to_csv(cfg["outputs"]["table"], index=False)
    write_run_log(cfg, Path(cfg["outputs"]["log_dir"]) / "train_teacher.json", best_val_loss=best, test=metrics)
    print(metrics)


if __name__ == "__main__":
    main()
