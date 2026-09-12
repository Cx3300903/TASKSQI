from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def predict_teacher(model, dataset, device, batch_size: int = 256):
    model.eval()
    probs, preds, labels, subjects, xs = [], [], [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x, y, sid in tqdm(loader, desc="predict teacher", unit="batch", ascii=True):
        x = x.to(device)
        p = torch.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(axis=1))
        labels.append(y.numpy())
        subjects.extend(list(sid))
        xs.append(x.cpu().numpy())
    return {
        "x": np.concatenate(xs),
        "prob": np.concatenate(probs),
        "pred": np.concatenate(preds),
        "y": np.concatenate(labels),
        "subject_ids": np.asarray(subjects).astype(str),
    }


@torch.no_grad()
def predict_tasksqi(model, x, device, batch_size: int = 256):
    model.eval()
    outs = []
    starts = range(0, len(x), batch_size)
    for start in tqdm(starts, desc="predict tasksqi", unit="batch", total=(len(x) + batch_size - 1) // batch_size, ascii=True):
        xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs)


def class_weights_from_dataset(dataset, num_classes: int = 5):
    y = dataset.y.numpy()
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
