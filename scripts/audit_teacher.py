from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from scripts.convert_shhs_profusion import EdfHeader, parse_sleep_stages
from src.datasets import SHHSDataset
from src.metrics import classification_metrics
from src.models import AttnSleep
from src.utils.config import load_config, resolve_device
from src.utils.eval import predict_teacher
from src.utils.training import load_model_weights

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def read_split_ids(split_file: Path) -> set[str]:
    return {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def channel_audit(manifest: Path, out_dir: Path, sample_n: int = 20) -> dict:
    df = pd.read_csv(manifest)
    rows = []
    for row in df.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        edf_path = Path(getattr(row, "edf"))
        try:
            header = EdfHeader(edf_path)
            selected = str(getattr(row, "channel"))
            status = "PASS" if selected == "EEG" else "NEEDS_REVIEW"
            rows.append(
                {
                    "subject_id": subject_id,
                    "raw_channels": "|".join(header.labels),
                    "selected_channel": selected,
                    "expected_mapping": "NSRR SHHS1: EEG=C4-A1; EEG(sec)/EEG2=C3-A2",
                    "fs_raw": float(getattr(row, "fs_original")),
                    "fs_final": 100.0,
                    "status": status,
                    "edf": str(edf_path),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "subject_id": subject_id,
                    "raw_channels": "",
                    "selected_channel": str(getattr(row, "channel", "")),
                    "expected_mapping": "NSRR SHHS1: EEG=C4-A1; EEG(sec)/EEG2=C3-A2",
                    "fs_raw": "",
                    "fs_final": 100.0,
                    "status": f"FAIL: {exc}",
                    "edf": str(edf_path),
                }
            )
    write_csv(out_dir / "channel_audit.csv", rows)
    write_csv(out_dir / "channel_audit_sample20.csv", rows[:sample_n])
    status_counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    return {"rows": len(rows), "status_counts": status_counts, "sample_csv": str(out_dir / "channel_audit_sample20.csv")}


def alignment_audit(npz_path: Path, manifest: Path, out_dir: Path, sample_n: int = 5) -> dict:
    df = pd.read_csv(manifest).head(sample_n)
    data = np.load(npz_path)
    subject_ids = data["subject_ids"].astype(str)
    y_npz = data["y"].astype(np.int64)
    rows = []
    for row in df.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        xml_path = Path(getattr(row, "xml"))
        xml_y_raw = parse_sleep_stages(xml_path)
        xml_y = xml_y_raw[xml_y_raw >= 0]
        npz_y = y_npz[subject_ids == subject_id]
        n = min(len(xml_y), len(npz_y))
        agree = bool(n > 0 and np.array_equal(xml_y[:n], npz_y[:n]) and len(xml_y) == len(npz_y))
        rows.append(
            {
                "subject_id": subject_id,
                "xml_epochs_valid": int(len(xml_y)),
                "npz_epochs": int(len(npz_y)),
                "compared_epochs": int(n),
                "label_agreement": float((xml_y[:n] == npz_y[:n]).mean()) if n else np.nan,
                "exact_sequence_match": agree,
                "first50_xml": " ".join(map(str, xml_y[:50].tolist())),
                "first50_npz": " ".join(map(str, npz_y[:50].tolist())),
                "last20_xml": " ".join(map(str, xml_y[-20:].tolist())),
                "last20_npz": " ".join(map(str, npz_y[-20:].tolist())),
                "status": "PASS" if agree else "FAIL",
                "xml": str(xml_path),
            }
        )
    write_csv(out_dir / "alignment_audit.csv", rows)
    return {"rows": len(rows), "all_pass": all(r["status"] == "PASS" for r in rows)}


def stage_distribution(npz_path: Path, split_dir: Path, out_dir: Path) -> dict:
    data = np.load(npz_path)
    y = data["y"].astype(np.int64)
    subject_ids = data["subject_ids"].astype(str)
    rows = []
    for split in ["train", "val", "test"]:
        ids = read_split_ids(split_dir / f"shhs_{split}.txt")
        mask = np.fromiter((sid in ids for sid in subject_ids), dtype=bool, count=len(subject_ids))
        counts = np.bincount(y[mask], minlength=5)
        total = int(counts.sum())
        for idx, name in enumerate(STAGE_NAMES):
            rows.append(
                {
                    "split": split,
                    "stage": name,
                    "label_id": idx,
                    "epochs": int(counts[idx]),
                    "proportion": float(counts[idx] / max(1, total)),
                    "subjects": len(ids),
                    "total_epochs": total,
                }
            )
    write_csv(out_dir / "stage_distribution.csv", rows)
    return {"csv": str(out_dir / "stage_distribution.csv")}


def config_diff(config_path: Path, out_dir: Path) -> dict:
    cfg = load_config(config_path)
    points = [
        ("input_channel", "Official SHHS command selects EEG C4-A1", "Current converter selected EDF label EEG, which NSRR maps to C4-A1."),
        ("input_sampling", "Official SHHS preprocessing keeps SHHS EEG at 125 Hz, commonly yielding 3750 samples per 30 s epoch.", "Current NPZ is resampled to 100 Hz and uses 3000 samples per epoch."),
        ("architecture", "Official source comments: d_model=100 and MRCNN_SHHS for SHHS.", "Current wrapper uses the default official AttnSleep class: d_model=80 and MRCNN."),
        ("training_protocol", "Official README describes K-fold cross validation.", f"Current config uses one subject split: {cfg['data']['split_dir']}."),
        ("loss", "Official README allows configurable loss; the README does not document class weighting.", "Current script uses weighted cross entropy to compensate class imbalance."),
        ("lr_amp", "Official exact optimizer settings are not vendored in this local copy.", f"Current teacher lr={cfg['teacher']['lr']}, mixed_precision={cfg['teacher']['mixed_precision']}, grad_clip={cfg['teacher'].get('grad_clip_norm')}."),
    ]
    text = ["# Teacher Config Diff", ""]
    rows = []
    for item, official, current in points:
        rows.append({"item": item, "official_or_expected": official, "current": current})
        text.extend([f"## {item}", f"- Official/expected: {official}", f"- Current: {current}", ""])
    (out_dir / "config_diff.md").write_text("\n".join(text), encoding="utf-8")
    write_csv(out_dir / "config_diff.csv", rows)
    return {"md": str(out_dir / "config_diff.md")}


def plot_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), STAGE_NAMES)
    ax.set_yticks(range(5), STAGE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if cm.dtype.kind == "f" else str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def confusion_audit(config_path: Path, checkpoint: Path, out_dir: Path) -> dict:
    if not checkpoint.exists():
        return {"status": "NEEDS_REVIEW", "reason": f"checkpoint not found: {checkpoint}"}
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])
    split_dir = Path(cfg["data"]["split_dir"])
    model = AttnSleep(cfg["teacher"]["num_classes"], architecture=cfg["teacher"].get("architecture", "default")).to(device)
    load_model_weights(model, checkpoint, map_location=device)
    out = {"status": "PASS", "checkpoint": str(checkpoint)}
    reports = []
    for split in ["val", "test"]:
        ds = SHHSDataset(
            cfg["data"]["shhs_npz"],
            split_dir / f"shhs_{split}.txt",
            cfg["data"]["normalize"],
            expected_samples=cfg["data"].get("input_samples", 3000),
        )
        pred = predict_teacher(model, ds, device, cfg["teacher"]["batch_size"])
        cm = confusion_matrix(pred["y"], pred["pred"], labels=[0, 1, 2, 3, 4])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        pd.DataFrame(cm, index=STAGE_NAMES, columns=STAGE_NAMES).to_csv(out_dir / f"confusion_matrix_{split}.csv")
        pd.DataFrame(cm_norm, index=STAGE_NAMES, columns=STAGE_NAMES).to_csv(out_dir / f"confusion_matrix_{split}_normalized.csv")
        plot_confusion(cm_norm, out_dir / f"confusion_matrix_{split}.png", f"Teacher {split} confusion matrix")
        report = classification_report(pred["y"], pred["pred"], labels=[0, 1, 2, 3, 4], target_names=STAGE_NAMES, output_dict=True, zero_division=0)
        for stage in STAGE_NAMES:
            reports.append({"split": split, "stage": stage, **report[stage]})
        out[f"{split}_metrics"] = classification_metrics(pred["y"], pred["pred"])
    pd.DataFrame(reports).to_csv(out_dir / "per_stage_precision_recall_f1.csv", index=False)
    return out


def write_summary(out_dir: Path, results: dict) -> None:
    lines = [
        "# Teacher Audit Summary",
        "",
        "## P0 Channel Audit",
        f"Status: {'PASS' if results['channel']['status_counts'].get('PASS', 0) == results['channel']['rows'] else 'NEEDS REVIEW'}",
        f"Rows: {results['channel']['rows']}",
        f"Status counts: {results['channel']['status_counts']}",
        "",
        "## P0 Label Alignment Audit",
        f"Status: {'PASS' if results['alignment']['all_pass'] else 'FAIL'}",
        f"Checked nights: {results['alignment']['rows']}",
        "",
        "## P1 Config/Official Difference",
        "Status: FAIL for official-like reproduction, because current teacher uses 100 Hz / 3000 samples / default MRCNN, while the official SHHS note uses 125 Hz / SHHS-specific MRCNN settings.",
        "",
        "## P2 Confusion And Stage Distribution",
        f"Confusion status: {results['confusion'].get('status')}",
        f"Confusion details: {results['confusion']}",
        "",
        "## P3 Sanity Reproduction",
        "Status: NEEDS REVIEW. Run an official-like 125 Hz SHHS setting before formal TaskSQI generation.",
        "",
        "## P4 Go / No-Go",
        "Status: NO-GO for formal TaskSQI with the current 71.8% teacher. Main suspected cause is official SHHS architecture/preprocessing mismatch; N1 remains weak.",
        "",
    ]
    (out_dir / "teacher_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Run the PDF-ordered teacher audit checklist.")
    ap.add_argument("--config", default="configs/teacher.yaml")
    ap.add_argument("--npz", default="data/shhs_epochs.npz")
    ap.add_argument("--split-dir", default="splits")
    ap.add_argument("--manifest", default="outputs/logs/shhs_500_manifest.csv")
    ap.add_argument("--checkpoint", default="outputs/checkpoints/teacher.pt")
    ap.add_argument("--out-dir", default="outputs/teacher_audit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "channel": channel_audit(Path(args.manifest), out_dir),
        "alignment": alignment_audit(Path(args.npz), Path(args.manifest), out_dir),
        "stage_distribution": stage_distribution(Path(args.npz), Path(args.split_dir), out_dir),
        "config_diff": config_diff(Path(args.config), out_dir),
        "confusion": confusion_audit(Path(args.config), Path(args.checkpoint), out_dir),
    }
    (out_dir / "teacher_audit_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
