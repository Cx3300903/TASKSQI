from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Create a subject-level subset from a TaskSQI NPZ file.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--max-epochs-per-subject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=False)
    x = data["x"]
    y = data["y"]
    subject_ids = data["subject_ids"].astype(str)
    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(args.seed)
    chosen_subjects = unique_subjects[: args.subjects]
    indices = []
    for sid in chosen_subjects:
        idx = np.flatnonzero(subject_ids == sid)
        if args.max_epochs_per_subject is not None and len(idx) > args.max_epochs_per_subject:
            idx = rng.choice(idx, size=args.max_epochs_per_subject, replace=False)
            idx.sort()
        indices.append(idx)
    indices = np.concatenate(indices)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=x[indices].astype(np.float32), y=y[indices].astype(np.int64), subject_ids=subject_ids[indices])
    print({"output": str(out), "subjects": len(chosen_subjects), "epochs": len(indices)})


if __name__ == "__main__":
    main()
