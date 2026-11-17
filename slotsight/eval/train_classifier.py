"""Train the per-slot classifier (classifier.py) on hand-labeled frames.

Fine-tune on this lot (default): starts from models/slot_classifier_base.pt and
trains on the lot's labeled TUNE sessions only (eval/labels.csv, frames in
data/eval_frames/tune/...), so the test sessions that produce the R-01/R-02/R-05
verdicts are never trained on. Writes models/slot_classifier.pt (gitignored),
which the dashboard then uses instead of the base weights.

    python slotsight/eval/train_classifier.py
    python slotsight/eval/train_classifier.py --only "tune/*" --epochs 6

Base model: --from-scratch with one or more --dataset folders, each holding
slots.json, labels.csv (frame, slot_id, true_status, ...) and frames/. The
committed base weights were trained this way on a PKLot subset (CC BY 4.0, UFPR;
603 frames from three views, see models/README.md):

    python slotsight/eval/train_classifier.py --from-scratch --dataset D:/pklot/pucpr --dataset D:/pklot/ufpr04 --dataset D:/pklot/ufpr05 --out slotsight/models/slot_classifier_base.pt

About 15 % of the frames (whole frames, seeded) are held out to report a
validation accuracy; it is a sanity check, not R-01 evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as slotsight  # noqa: E402
import classifier  # noqa: E402

SEED = 0
VALIDATION_SHARE = 0.15


def load_set(slots: list[dict], labels_path: Path, frames_dir: Path, only: list[str], exclude: list[str]) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(patches, targets)] per labeled frame; targets 1 = occupied."""
    points = {str(slot["id"]): slot["points"] for slot in slots}
    by_frame: dict[str, dict[str, str]] = defaultdict(dict)
    with labels_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            frame = row["frame"].replace("\\", "/")
            if only and not any(fnmatchcase(frame, pattern) for pattern in only):
                continue
            if any(fnmatchcase(frame, pattern) for pattern in exclude):
                continue
            if row["slot_id"] in points and row["true_status"] in ("occupied", "vacant"):
                by_frame[frame][row["slot_id"]] = row["true_status"]
    frames = []
    for frame, statuses in sorted(by_frame.items()):
        image = cv2.imdecode(np.fromfile(str(frames_dir / frame), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skipping unreadable frame {frame}", file=sys.stderr)
            continue
        patches = np.stack([classifier.warp_slot(image, points[slot_id]) for slot_id in statuses])
        targets = np.array([1 if status == "occupied" else 0 for status in statuses.values()])
        frames.append((patches, targets))
    return frames


def augment(x):
    import torch

    batch = x.shape[0]
    # Brightness and contrast jitter (lighting), flips (a slot looks alike mirrored), noise.
    x = x * (0.6 + 0.8 * torch.rand(batch, 1, 1, 1)) + (torch.rand(batch, 1, 1, 1) - 0.5) * 0.3
    for dim in (3, 2):
        flip = torch.rand(batch) < 0.5
        x[flip] = x[flip].flip(dim)
    return (x + 0.02 * torch.randn_like(x)).clamp(0, 1)


def train(frames, start_weights: Path | None, epochs: int, learning_rate: float):
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    x = classifier.patches_tensor(np.concatenate([patches for patches, _ in frames]))
    y = torch.from_numpy(np.concatenate([targets for _, targets in frames])).long()
    net = classifier.build_network()
    if start_weights is not None:
        net.load_state_dict(torch.load(start_weights, map_location="cpu", weights_only=True))
    optimiser = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    steps = epochs * ((len(x) + 255) // 256)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=learning_rate * 1.5, total_steps=max(1, steps))
    share = torch.stack([(y == 0).float().mean(), (y == 1).float().mean()]).clamp(min=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=(1 / share) / (1 / share).sum())  # balance occupied/vacant
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(x))
        for start in range(0, len(x), 256):
            index = order[start : start + 256]
            optimiser.zero_grad()
            loss_fn(net(augment(x[index].clone())), y[index]).backward()
            optimiser.step()
            schedule.step()
    return net.eval()


def accuracy(net, frames) -> float | None:
    import torch

    if not frames:
        return None
    x = classifier.patches_tensor(np.concatenate([patches for patches, _ in frames]))
    y = torch.from_numpy(np.concatenate([targets for _, targets in frames]))
    with torch.inference_mode():
        return float((net(x).argmax(1) == y).float().mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, action="append", default=[], help="folder with slots.json, labels.csv and frames/ (repeatable)")
    parser.add_argument("--only", action="append", default=[], help="frame patterns to train on (default for this lot: 'tune/*')")
    parser.add_argument("--exclude", action="append", default=[], help="frame patterns to leave out")
    parser.add_argument("--from-scratch", action="store_true", help="don't start from the base weights")
    parser.add_argument("--epochs", type=int, help="default 8 from scratch, 5 when fine-tuning")
    parser.add_argument("--out", type=Path, default=slotsight.MODELS_DIR / classifier.LOCAL_WEIGHTS)
    args = parser.parse_args()

    sources = []
    if args.dataset:
        for folder in args.dataset:
            slots = json.loads((folder / "slots.json").read_text(encoding="utf-8"))
            # Only the folder name is recorded: full paths would leak local user names into the repo.
            sources.append((folder.name, load_set(slots, folder / "labels.csv", folder / "frames", args.only, args.exclude)))
    else:
        only = args.only or ["tune/*"]  # never train on the test sessions
        frames = load_set(slotsight.load_slots(), slotsight.LABELS_PATH, slotsight.EVAL_FRAMES_DIR, only, args.exclude)
        sources.append((f"eval/labels.csv ({', '.join(only)})", frames))
    frames = [frame for _, items in sources for frame in items]
    if not frames:
        print("No labeled frames found. Label some tune sessions first (see docs/recording-guide.md).", file=sys.stderr)
        return 1
    random.Random(SEED).shuffle(frames)
    held = max(1, int(len(frames) * VALIDATION_SHARE)) if len(frames) >= 5 else 0
    validation, training = frames[:held], frames[held:]
    base = slotsight.MODELS_DIR / classifier.BASE_WEIGHTS
    start = None if args.from_scratch or not base.exists() else base
    epochs = args.epochs or (5 if start else 8)
    learning_rate = 5e-4 if start else 2e-3
    patches = sum(len(targets) for _, targets in training)
    print(f"Training on {len(training)} frames ({patches} slot patches), {'fine-tuning ' + base.name if start else 'from scratch'}, {epochs} epochs...", flush=True)
    started = time.perf_counter()
    net = train(training, start, epochs, learning_rate)
    validation_accuracy = accuracy(net, validation)
    import torch

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    info = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [name for name, _ in sources],
        "start": start.name if start else None,
        "frames": len(training),
        "patches": patches,
        "epochs": epochs,
        "validation_frames": len(validation),
        "validation_accuracy": None if validation_accuracy is None else round(validation_accuracy, 4),
    }
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    shown = "--" if validation_accuracy is None else f"{validation_accuracy:.1%}"
    print(f"Saved {args.out} in {time.perf_counter() - started:.0f} s; validation accuracy {shown} on {len(validation)} held-out frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
