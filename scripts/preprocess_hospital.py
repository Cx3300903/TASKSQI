from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preprocess_shhs import preprocess_manifest

import shutil

from src.datasets.preprocessing import load_epochs_npz


def main():
    ap = argparse.ArgumentParser(description="Convert hospital PSG EEG epochs to the project NPZ contract.")
    ap.add_argument("--input", required=True, help="Existing NPZ or CSV manifest with subject_id,label,epoch_path[,fs].")
    ap.add_argument("--output", default="data/hospital_epochs.npz")
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
