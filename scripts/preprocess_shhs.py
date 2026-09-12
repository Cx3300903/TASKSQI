from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import shutil

import numpy as np
import pandas as pd

from src.datasets.preprocessing import STAGE_TO_ID, bandpass_filter, load_epochs_npz, resample_to_100hz


def load_epoch_file(path: Path):
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    data = np.load(path, allow_pickle=True)
    if "x" in data:
        return data["x"].astype(np.float32)
    raise ValueError(f"{path} must be .npy or .npz with key x")


def preprocess_manifest(input_csv: Path, output: Path, default_fs: float):
    df = pd.read_csv(input_csv)
    required = {"subject_id", "label", "epoch_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    xs, ys, sids = [], [], []
    for row in df.itertuples(index=False):
        epoch_path = Path(getattr(row, "epoch_path"))
        if not epoch_path.is_absolute():
            epoch_path = input_csv.parent / epoch_path
        x = load_epoch_file(epoch_path).squeeze()
        fs = float(getattr(row, "fs", default_fs))
        x = resample_to_100hz(x, fs)
        if x.shape[-1] != 3000:
            raise ValueError(f"{epoch_path} has {x.shape[-1]} samples after resampling; expected 3000")
        x = bandpass_filter(x[None, :], fs=100.0)[0]
        label = str(getattr(row, "label"))
        if label not in STAGE_TO_ID:
            continue
        xs.append(x)
        ys.append(STAGE_TO_ID[label])
        sids.append(str(getattr(row, "subject_id")))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=np.asarray(xs)[:, None, :], y=np.asarray(ys), subject_ids=np.asarray(sids))


def main():
    ap = argparse.ArgumentParser(description="Convert SHHS EEG epochs to the project NPZ contract.")
    ap.add_argument("--input", required=True, help="Existing NPZ or CSV manifest with subject_id,label,epoch_path[,fs].")
    ap.add_argument("--output", default="data/shhs_epochs.npz")
    ap.add_argument("--fs", type=float, default=100.0)
    args = ap.parse_args()
    src = Path(args.input)
    out = Path(args.output)
    if src.suffix == ".npz":
        load_epochs_npz(src, normalize="none")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
    elif src.suffix == ".csv":
        preprocess_manifest(src, out, args.fs)
    else:
        raise ValueError("--input must be .npz or .csv")
    print(out)


if __name__ == "__main__":
    main()
