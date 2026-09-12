from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lxml import etree
from scipy import signal

from src.datasets.preprocessing import bandpass_filter

STAGE_MAP = {
    "Wake": 0,
    "NonREM1": 1,
    "NonREM2": 2,
    "NonREM3": 3,
    "NonREM4": 3,
    "REM": 4,
}

CHANNEL_PRIORITY = ("C4M1", "C4-A1", "C4A1", "C4-M1", "C3M2", "C3-A2", "C3A2")


def _clean_label(label: str) -> str:
    return label.replace(" ", "").replace("-", "").upper()


def parse_rml_stages(path: Path):
    root = etree.parse(str(path)).getroot()
    groups = []
    for parent in root.iter():
        stages = [child for child in parent if child.tag.split("}")[-1] == "Stage"]
        if stages:
            name = parent.tag.split("}")[-1]
            ancestors = [a.tag.split("}")[-1] for a in parent.iterancestors()]
            groups.append((name, stages, ancestors))
    if not groups:
        stages = [e for e in root.iter() if e.tag.split("}")[-1] == "Stage"]
        groups = [("all", stages, [])]
    user_groups = [g for g in groups if "UserStaging" in g[2]]
    candidate_groups = user_groups or groups
    # Prefer edited/user scoring. If absent, fall back to the longest available group.
    group_name, stages, ancestors = max(candidate_groups, key=lambda item: len(item[1]))
    if ancestors:
        group_name = "/".join(reversed(ancestors[:3])) + "/" + group_name
    parsed = []
    for stage in stages:
        typ = stage.attrib.get("Type")
        start = stage.attrib.get("Start")
        if typ is None or start is None:
            continue
        parsed.append((float(start), typ))
    parsed.sort(key=lambda x: x[0])
    return group_name, parsed


def labels_from_stages(stages, recording_seconds: float):
    n_epochs = int(recording_seconds // 30)
    y = np.full(n_epochs, -1, dtype=np.int64)
    if not stages:
        return y
    starts = [s for s, _ in stages] + [recording_seconds]
    types = [t for _, t in stages]
    for i, typ in enumerate(types):
        label = STAGE_MAP.get(typ, -1)
        start_epoch = max(0, int(np.floor(starts[i] / 30.0)))
        end_epoch = min(n_epochs, int(np.floor(starts[i + 1] / 30.0)))
        if end_epoch > start_epoch:
            y[start_epoch:end_epoch] = label
    return y


class EdfHeader:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            fixed = f.read(256)
            self.header_bytes = int(fixed[184:192].decode("latin1").strip())
            self.n_records = int(float(fixed[236:244].decode("latin1").strip()))
            self.record_duration = float(fixed[244:252].decode("latin1").strip())
            self.n_signals = int(fixed[252:256].decode("latin1").strip())
            rest = f.read(self.header_bytes - 256)
        header = fixed + rest
        off = 256
        fields = {}
        for name, width in [
            ("labels", 16),
            ("transducer", 80),
            ("physical_dimension", 8),
            ("phys_min", 8),
            ("phys_max", 8),
            ("dig_min", 8),
            ("dig_max", 8),
            ("prefilter", 80),
            ("samples_per_record", 8),
            ("reserved", 32),
        ]:
            vals = []
            for i in range(self.n_signals):
                vals.append(header[off + i * width : off + (i + 1) * width].decode("latin1").strip())
            fields[name] = vals
            off += width * self.n_signals
        self.labels = fields["labels"]
        self.phys_min = np.asarray(fields["phys_min"], dtype=float)
        self.phys_max = np.asarray(fields["phys_max"], dtype=float)
        self.dig_min = np.asarray(fields["dig_min"], dtype=float)
        self.dig_max = np.asarray(fields["dig_max"], dtype=float)
        self.samples_per_record = np.asarray(fields["samples_per_record"], dtype=int)

    @property
    def seconds(self) -> float:
        return self.n_records * self.record_duration

    def channel_index(self, preferred=CHANNEL_PRIORITY):
        cleaned = [_clean_label(x) for x in self.labels]
        for target in preferred:
            target_clean = _clean_label(target)
            for idx, label in enumerate(cleaned):
                if label == target_clean:
                    return idx
        for target in preferred:
            target_clean = _clean_label(target)
            for idx, label in enumerate(cleaned):
                if target_clean in label:
                    return idx
        raise ValueError(f"No preferred EEG channel found. Available labels: {self.labels}")


def read_edf_channel(path: Path, channel_idx: int):
    header = EdfHeader(path)
    spr = header.samples_per_record
    total_per_record = int(spr.sum())
    channel_samples = int(spr[channel_idx])
    scale = (header.phys_max[channel_idx] - header.phys_min[channel_idx]) / (
        header.dig_max[channel_idx] - header.dig_min[channel_idx]
    )
    offset = header.phys_min[channel_idx] - scale * header.dig_min[channel_idx]
    out = np.empty(header.n_records * channel_samples, dtype=np.float32)
    with path.open("rb") as f:
        f.seek(header.header_bytes)
        write = 0
        for _ in range(header.n_records):
            rec = np.frombuffer(f.read(total_per_record * 2), dtype="<i2")
            pos = int(spr[:channel_idx].sum())
            vals = rec[pos : pos + channel_samples].astype(np.float32)
            out[write : write + channel_samples] = vals * scale + offset
            write += channel_samples
    fs = channel_samples / header.record_duration
    return out, fs, header


def subject_files(subject_dir: Path):
    rmls = sorted(subject_dir.glob("*.rml"))
    edfs = sorted(p for p in subject_dir.glob("*.edf") if "-T.edf" not in p.name)
    if not rmls or not edfs:
        return None, []
    return rmls[0], edfs


def convert_subject(subject_dir: Path, preferred=CHANNEL_PRIORITY, target_fs: float = 100.0):
    rml, edfs = subject_files(subject_dir)
    if rml is None:
        raise ValueError("Missing RML or EDF")
    group_name, stages = parse_rml_stages(rml)
    chunks = []
    fs_values = []
    labels = None
    channel = None
    for edf in edfs:
        hdr = EdfHeader(edf)
        ch = hdr.channel_index(preferred)
        sig, fs, hdr = read_edf_channel(edf, ch)
        chunks.append(sig)
        fs_values.append(fs)
        labels = hdr.labels
        channel = hdr.labels[ch]
    if len(set(round(v, 6) for v in fs_values)) != 1:
        # Resample each segment to the first segment rate before concatenation.
        base_fs = fs_values[0]
        chunks = [signal.resample(x, int(round(len(x) * base_fs / fs))).astype(np.float32) for x, fs in zip(chunks, fs_values)]
        fs = base_fs
    else:
        fs = fs_values[0]
    raw = np.concatenate(chunks).astype(np.float32)
    if abs(fs - target_fs) > 1e-6:
        raw = signal.resample(raw, int(round(len(raw) * target_fs / fs))).astype(np.float32)
    fs = float(target_fs)
    raw = bandpass_filter(raw[None, :], fs=fs)[0]
    epoch_samples = int(round(fs * 30.0))
    n_epochs = len(raw) // epoch_samples
    x = raw[: n_epochs * epoch_samples].reshape(n_epochs, epoch_samples)
    y = labels_from_stages(stages, n_epochs * 30.0)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    keep = y >= 0
    return {
        "subject_id": subject_dir.name,
        "x": x[keep, None, :].astype(np.float32),
        "y": y[keep].astype(np.int64),
        "stage_group": group_name,
        "channel": channel,
        "edf_count": len(edfs),
        "fs_original": fs_values[0],
        "n_epochs_total": int(n),
        "n_epochs_kept": int(keep.sum()),
        "labels": labels,
    }


def plan_subject(subject_dir: Path, preferred=CHANNEL_PRIORITY):
    rml, edfs = subject_files(subject_dir)
    if rml is None:
        raise ValueError("Missing RML or EDF")
    group_name, stages = parse_rml_stages(rml)
    total_seconds = 0.0
    fs_values = []
    channel = None
    for edf in edfs:
        hdr = EdfHeader(edf)
        ch = hdr.channel_index(preferred)
        channel = hdr.labels[ch]
        fs_values.append(hdr.samples_per_record[ch] / hdr.record_duration)
        total_seconds += hdr.seconds
    n_epochs_total = int(total_seconds // 30)
    y = labels_from_stages(stages, n_epochs_total * 30.0)
    n_epochs_kept = int((y >= 0).sum())
    return {
        "subject_id": subject_dir.name,
        "channel": channel,
        "edf_count": len(edfs),
        "fs_original": fs_values[0],
        "stage_group": group_name,
        "n_epochs_total": n_epochs_total,
        "n_epochs_kept": n_epochs_kept,
    }


def save_npz_streaming(output: Path, x, y, subject_ids, compressed: bool):
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(output, mode="w", compression=compression, allowZip64=True) as zf:
        for name, arr in {"x": x, "y": y, "subject_ids": subject_ids}.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


def main():
    ap = argparse.ArgumentParser(description="Convert hospital EDF/RML folders to TaskSQI NPZ format.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", default="data/hospital_epochs.npz")
    ap.add_argument("--manifest", default="outputs/logs/hospital_npz_manifest.csv")
    ap.add_argument("--summary", default="outputs/logs/hospital_npz_summary.json")
    ap.add_argument("--limit-subjects", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--compressed", action="store_true", help="Use compressed NPZ. Slower, but smaller on disk.")
    ap.add_argument("--target-fs", type=float, default=100.0, help="Target sampling rate after resampling.")
    args = ap.parse_args()

    root = Path(args.root)
    subject_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.limit_subjects:
        subject_dirs = subject_dirs[: args.limit_subjects]
    rows = []
    xs, ys, sids = [], [], []
    failures = []
    for i, subject_dir in enumerate(subject_dirs, 1):
        try:
            result = convert_subject(subject_dir, target_fs=args.target_fs) if args.dry_run else plan_subject(subject_dir)
            rows.append(result if args.dry_run else result.copy())
            if args.dry_run:
                xs.append(result["x"])
                ys.append(result["y"])
                sids.append(np.full(len(result["y"]), result["subject_id"]))
            print(f"[{i}/{len(subject_dirs)}] {subject_dir.name}: {result['n_epochs_kept']} epochs, channel={result['channel']}")
        except Exception as exc:
            failures.append({"subject_id": subject_dir.name, "error": str(exc)})
            print(f"[{i}/{len(subject_dirs)}] {subject_dir.name}: ERROR {exc}", file=sys.stderr)
            if not args.allow_missing:
                raise
    if args.dry_run:
        print(json.dumps({"subjects_ok": len(rows), "subjects_failed": len(failures), "failures": failures[:5]}, indent=2))
        return
    if not rows:
        raise RuntimeError("No subjects planned.")
    total_epochs = int(sum(row["n_epochs_kept"] for row in rows))
    tmp_dir = Path("outputs/tmp_hospital_npz")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    epoch_samples = int(round(args.target_fs * 30.0))
    x = np.lib.format.open_memmap(
        tmp_dir / "hospital_x.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_epochs, 1, epoch_samples),
    )
    y = np.lib.format.open_memmap(tmp_dir / "hospital_y.npy", mode="w+", dtype=np.int64, shape=(total_epochs,))
    subject_ids = np.lib.format.open_memmap(tmp_dir / "hospital_subject_ids.npy", mode="w+", dtype="U64", shape=(total_epochs,))
    offset = 0
    ok_rows = []
    second_pass_failures = []
    planned = {row["subject_id"]: row for row in rows}
    for i, subject_dir in enumerate(subject_dirs, 1):
        if subject_dir.name not in planned:
            continue
        try:
            result = convert_subject(subject_dir, target_fs=args.target_fs)
            n = len(result["y"])
            x[offset : offset + n] = result["x"]
            y[offset : offset + n] = result["y"]
            subject_ids[offset : offset + n] = result["subject_id"]
            offset += n
            ok_rows.append({k: result[k] for k in ["subject_id", "channel", "edf_count", "fs_original", "stage_group", "n_epochs_total", "n_epochs_kept"]})
            print(f"[write {i}/{len(subject_dirs)}] {subject_dir.name}: {n} epochs")
        except Exception as exc:
            second_pass_failures.append({"subject_id": subject_dir.name, "error": str(exc)})
            print(f"[write {i}/{len(subject_dirs)}] {subject_dir.name}: ERROR {exc}", file=sys.stderr)
            if not args.allow_missing:
                raise
    if offset == 0:
        raise RuntimeError("No subjects converted.")
    x = x[:offset]
    y = y[:offset]
    subject_ids = subject_ids[:offset]
    out = Path(args.output)
    save_npz_streaming(out, x, y, subject_ids, args.compressed)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "channel", "edf_count", "fs_original", "stage_group", "n_epochs_total", "n_epochs_kept"])
        writer.writeheader()
        writer.writerows(ok_rows)
    failures.extend(second_pass_failures)
    summary = {
        "root": str(root),
        "output": str(out),
        "n_subjects_ok": len(ok_rows),
        "n_subjects_failed": len(failures),
        "n_epochs": int(offset),
        "target_fs": float(args.target_fs),
        "epoch_samples": int(epoch_samples),
        "shape": [int(offset), 1, int(epoch_samples)],
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "failures": failures,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
