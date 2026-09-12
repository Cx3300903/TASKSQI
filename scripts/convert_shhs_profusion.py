from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import signal

from src.datasets.preprocessing import bandpass_filter

STAGE_MAP = {
    0: 0,  # Wake
    1: 1,  # N1
    2: 2,  # N2
    3: 3,  # N3
    4: 3,  # N4 -> N3
    5: 4,  # REM
}

CHANNEL_PRIORITY = ("EEG", "C4-A1", "C4M1", "C4-M1", "EEG2", "C3-A2", "C3M2", "C3-M2")


def _clean_label(label: str) -> str:
    return label.replace(" ", "").replace("-", "").replace("_", "").upper()


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

    def channel_index(self, preferred=CHANNEL_PRIORITY) -> int:
        cleaned = [_clean_label(label) for label in self.labels]
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


def parse_sleep_stages(xml_path: Path) -> np.ndarray:
    root = ET.parse(xml_path).getroot()
    values = []
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "SleepStage" and elem.text is not None:
            try:
                raw = int(elem.text.strip())
            except ValueError:
                values.append(-1)
                continue
            values.append(STAGE_MAP.get(raw, -1))
    return np.asarray(values, dtype=np.int64)


def pair_files(edf_dir: Path, xml_dir: Path):
    edfs = {p.stem: p for p in edf_dir.rglob("*.edf")}
    xmls = {}
    for p in xml_dir.rglob("*-profusion.xml"):
        stem = p.name[: -len("-profusion.xml")]
        xmls[stem] = p
    ids = sorted(set(edfs) & set(xmls))
    return [(sid, edfs[sid], xmls[sid]) for sid in ids]


def plan_subject(subject_id: str, edf_path: Path, xml_path: Path, preferred=CHANNEL_PRIORITY, target_fs: float = 100.0):
    header = EdfHeader(edf_path)
    channel_idx = header.channel_index(preferred)
    stages = parse_sleep_stages(xml_path)
    n_signal_epochs = int(header.seconds // 30)
    n = min(n_signal_epochs, len(stages))
    y = stages[:n]
    keep = y >= 0
    return {
        "subject_id": subject_id,
        "edf": str(edf_path),
        "xml": str(xml_path),
        "channel": header.labels[channel_idx],
        "fs_original": header.samples_per_record[channel_idx] / header.record_duration,
        "fs_final": float(target_fs),
        "n_epochs_signal": n_signal_epochs,
        "n_epochs_xml": int(len(stages)),
        "n_epochs_total": int(n),
        "n_epochs_kept": int(keep.sum()),
    }


def convert_subject(subject_id: str, edf_path: Path, xml_path: Path, preferred=CHANNEL_PRIORITY, target_fs: float = 100.0):
    header = EdfHeader(edf_path)
    channel_idx = header.channel_index(preferred)
    raw, fs, header = read_edf_channel(edf_path, channel_idx)
    if abs(fs - target_fs) > 1e-6:
        raw = signal.resample(raw, int(round(len(raw) * target_fs / fs))).astype(np.float32)
    raw = bandpass_filter(raw[None, :], fs=target_fs)[0]
    epoch_samples = int(round(target_fs * 30.0))
    n_signal_epochs = len(raw) // epoch_samples
    x = raw[: n_signal_epochs * epoch_samples].reshape(n_signal_epochs, epoch_samples)
    stages = parse_sleep_stages(xml_path)
    n = min(len(x), len(stages))
    x = x[:n]
    y = stages[:n]
    keep = y >= 0
    return {
        "subject_id": subject_id,
        "x": x[keep, None, :].astype(np.float32),
        "y": y[keep].astype(np.int64),
        "channel": header.labels[channel_idx],
        "fs_original": fs,
        "fs_final": float(target_fs),
        "n_epochs_signal": int(n_signal_epochs),
        "n_epochs_xml": int(len(stages)),
        "n_epochs_total": int(n),
        "n_epochs_kept": int(keep.sum()),
    }


def save_npz_streaming(output: Path, arrays: dict[str, np.ndarray], compressed: bool):
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(output, mode="w", compression=compression, allowZip64=True) as zf:
        for name, arr in arrays.items():
            with zf.open(name + ".npy", "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


def write_manifest(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_id",
        "channel",
        "fs_original",
        "fs_final",
        "n_epochs_signal",
        "n_epochs_xml",
        "n_epochs_total",
        "n_epochs_kept",
        "edf",
        "xml",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Convert SHHS Profusion EDF/XML pairs to TaskSQI NPZ format.")
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--xml-dir", required=True)
    ap.add_argument("--output", default="data/shhs_epochs.npz")
    ap.add_argument("--manifest", default="outputs/logs/shhs_npz_manifest.csv")
    ap.add_argument("--summary", default="outputs/logs/shhs_npz_summary.json")
    ap.add_argument("--subjects", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None, help="Shuffle paired subjects before taking --subjects.")
    ap.add_argument("--target-fs", type=float, default=100.0, help="Final sampling rate. Use 125 for official-like SHHS AttnSleep.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--compressed", action="store_true", help="Use compressed NPZ. Slower, but smaller on disk.")
    args = ap.parse_args()

    edf_dir = Path(args.edf_dir)
    xml_dir = Path(args.xml_dir)
    pairs = pair_files(edf_dir, xml_dir)
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        pairs = [pairs[i] for i in rng.permutation(len(pairs))]
    selected = pairs[: args.subjects]

    ok_rows = []
    failed_rows = []
    for sid, edf, xml in selected:
        try:
            row = plan_subject(sid, edf, xml, target_fs=args.target_fs)
            row["error"] = ""
            ok_rows.append(row)
        except Exception as exc:
            failed_rows.append({"subject_id": sid, "edf": str(edf), "xml": str(xml), "error": str(exc)})

    n_epochs = sum(row["n_epochs_kept"] for row in ok_rows)
    summary = {
        "edf_dir": str(edf_dir),
        "xml_dir": str(xml_dir),
        "output": args.output,
        "n_pairs_available": len(pairs),
        "n_subjects_requested": args.subjects,
        "n_subjects_planned": len(selected),
        "n_subjects_ok": len(ok_rows),
        "n_subjects_failed": len(failed_rows),
        "n_epochs": int(n_epochs),
        "target_fs": float(args.target_fs),
        "epoch_samples": int(round(args.target_fs * 30.0)),
    }
    if args.dry_run:
        write_manifest(Path(args.manifest), ok_rows + failed_rows)
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    temp_dir = Path(args.output).parent / "tmp_shhs_npz"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    epoch_samples = int(round(args.target_fs * 30.0))
    x_mm = np.memmap(temp_dir / "x.dat", mode="w+", dtype=np.float32, shape=(n_epochs, 1, epoch_samples))
    y_mm = np.memmap(temp_dir / "y.dat", mode="w+", dtype=np.int64, shape=(n_epochs,))
    sid_mm = np.memmap(temp_dir / "subject_ids.dat", mode="w+", dtype="U32", shape=(n_epochs,))

    converted_rows = []
    failed_convert_rows = list(failed_rows)
    write_at = 0
    for row in ok_rows:
        sid = row["subject_id"]
        try:
            result = convert_subject(sid, Path(row["edf"]), Path(row["xml"]), target_fs=args.target_fs)
            n = len(result["y"])
            x_mm[write_at : write_at + n] = result["x"]
            y_mm[write_at : write_at + n] = result["y"]
            sid_mm[write_at : write_at + n] = sid
            write_at += n
            converted = {
                **row,
                **{k: result[k] for k in ["channel", "fs_original", "fs_final", "n_epochs_signal", "n_epochs_xml", "n_epochs_total", "n_epochs_kept"]},
            }
            converted["error"] = ""
            converted_rows.append(converted)
            print(f"{sid}: {n} epochs")
        except Exception as exc:
            failed_convert_rows.append({"subject_id": sid, "edf": row["edf"], "xml": row["xml"], "error": str(exc)})

    x = np.asarray(x_mm[:write_at])
    y = np.asarray(y_mm[:write_at])
    subject_ids = np.asarray(sid_mm[:write_at])
    save_npz_streaming(Path(args.output), {"x": x, "y": y, "subject_ids": subject_ids}, args.compressed)
    del x, y, subject_ids, x_mm, y_mm, sid_mm
    shutil.rmtree(temp_dir)

    saved = np.load(args.output)
    label_values, label_counts = np.unique(saved["y"], return_counts=True)
    summary.update(
        {
            "n_subjects_ok": len(converted_rows),
            "n_subjects_failed": len(failed_convert_rows),
            "n_epochs": int(write_at),
            "label_counts": {int(k): int(v) for k, v in zip(label_values, label_counts)},
        }
    )
    write_manifest(Path(args.manifest), converted_rows + failed_convert_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
