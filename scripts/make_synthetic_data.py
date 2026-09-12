from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=int, default=12)
    ap.add_argument("--epochs-per-subject", type=int, default=20)
    ap.add_argument("--samples", type=int, default=3750)
    ap.add_argument("--fs", type=float, default=125.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    xs, ys, sids = [], [], []
    t = np.arange(args.samples) / args.fs
    freqs = [9.0, 5.0, 12.0, 2.0, 7.0]
    for s in range(args.subjects):
        sid = f"S{s:04d}"
        bias = rng.normal(0, 0.2)
        for e in range(args.epochs_per_subject):
            y = int(rng.integers(0, 5))
            sig = np.sin(2 * np.pi * freqs[y] * t + rng.uniform(0, 2 * np.pi))
            sig += 0.4 * np.sin(2 * np.pi * (freqs[y] / 2) * t)
            sig += rng.normal(0, 0.4, size=t.shape) + bias
            xs.append(sig.astype(np.float32))
            ys.append(y)
            sids.append(sid)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, x=np.asarray(xs)[:, None, :], y=np.asarray(ys), subject_ids=np.asarray(sids))
    print(out)


if __name__ == "__main__":
    main()
