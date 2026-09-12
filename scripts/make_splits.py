from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shhs_epochs.npz")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--train", type=int, default=350)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    data = np.load(args.data, allow_pickle=True)
    subjects = np.unique(data["subject_ids"].astype(str))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(subjects)
    need = args.train + args.val + args.test
    if len(subjects) < need:
        raise ValueError(f"Need {need} subjects, found {len(subjects)}")
    splits = {
        "shhs_train.txt": subjects[: args.train],
        "shhs_val.txt": subjects[args.train : args.train + args.val],
        "shhs_test.txt": subjects[args.train + args.val : need],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        (out / name).write_text("\n".join(ids) + "\n", encoding="utf-8")
    print({k: len(v) for k, v in splits.items()})


if __name__ == "__main__":
    main()
