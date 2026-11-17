"""Measure SlotSight's accuracy against hand-labeled frames (R-01, R-02, R-05, R-07 evidence).

Runs every labeled frame through the same pipeline as the dashboard (vehicle
detection with the configured model, slot overlap check, review threshold) and compares what the
dashboard would show with the labels.

Labels file (default slotsight/eval/labels.csv), one row per slot per frame:

    frame,slot_id,true_status,condition,occluded
    day/0001.jpg,A01,occupied,daytime,0
    day/0001.jpg,A02,vacant,daytime,1

- frame: image path, relative to --frames-dir (default slotsight/data/eval_frames/)
- slot_id: an id from the slot config
- true_status: occupied or vacant
- condition: daytime, evening or rain (other names are reported but not used by R-05)
- occluded: 1 if the slot is partly hidden (by a vehicle, pillar, tree...), else 0 or blank

How results are scored (decided 2026-11-17, see docs/PLAN.md):

- R-01: per-slot accuracy over every labeled row. "unknown" counts as wrong;
  coverage (the share of rows not shown as unknown) is reported next to it.
- R-02: in frames where every configured slot is labeled, the free count the
  dashboard shows (unknowns are not free) must equal the true free count.
- R-05: R-01 accuracy per condition; evening and rain must each reach 90% of daytime.
- R-07: no occluded case may be shown as a wrong confident status. Each must be
  correct or "unknown". Wrong "vacant" readings are reported separately.

Usage (from the repository root):
    python slotsight/eval/evaluate.py
    python slotsight/eval/evaluate.py --labels my_labels.csv --frames-dir D:/lot-frames
    python slotsight/eval/evaluate.py --overlap-threshold 0.5 --review-threshold 0.6
    python slotsight/eval/evaluate.py --note "first labeled set" --write
    python slotsight/eval/evaluate.py --exclude "tune/*" --write        # report on test sessions only
    python slotsight/eval/evaluate.py --only "tune/*" --overlap-threshold 0.5   # tune on tune sessions

Detections are cached in slotsight/data/eval_cache.json (keyed by model, file
and file modification time), so re-running with other thresholds does not
repeat inference. Per-row results go to slotsight/data/eval_predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as slotsight  # noqa: E402
import labeling  # noqa: E402

LABEL_COLUMNS = labeling.LABEL_COLUMNS
TRUE_STATUSES = labeling.TRUE_STATUSES
BASELINE_CONDITION = "daytime"
COMPARED_CONDITIONS = ("evening", "rain")
ACCURACY_TARGET = 0.90  # R-01, R-02
CONDITION_RATIO_TARGET = 0.90  # R-05
SMALL_SAMPLE = 30  # below this many rows, flag the result as a small sample

DEFAULT_LABELS = ROOT / "eval" / "labels.csv"
DEFAULT_FRAMES_DIR = slotsight.DATA_DIR / "eval_frames"
DEFAULT_CACHE = slotsight.DATA_DIR / "eval_cache.json"
DEFAULT_PREDICTIONS = slotsight.DATA_DIR / "eval_predictions.csv"
EVALUATION_DOC = ROOT / "docs" / "evaluation.md"
ACCURACY_SECTION = "## R-01, R-02, R-05, R-07: Accuracy"


class LabelError(ValueError):
    """The labels file cannot be evaluated as written."""


@dataclass
class Label:
    frame: str
    slot_id: str
    true_status: str
    condition: str
    occluded: bool


@dataclass
class Outcome:
    """One labeled slot in one frame, with what the dashboard would show."""

    label: Label
    shown: str  # after the review threshold and the empty-look check: occupied, vacant or unknown
    raw: str  # before the review threshold: occupied or vacant
    confidence: float
    flagged: bool = False  # the "doesn't look empty" check turned a vacant reading into unknown

    @property
    def correct(self) -> bool:
        return self.shown == self.label.true_status

    @property
    def false_free(self) -> bool:
        """The costly error: an occupied slot shown as free (R-07)."""
        return self.label.true_status == "occupied" and self.shown == "vacant"


# --- Loading -----------------------------------------------------------------


def read_labels(path: Path, slot_ids: list[str]) -> list[Label]:
    """Read and check the labels file, raising LabelError with line numbers on problems."""
    if not path.exists():
        raise LabelError(f"Labels file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = [column for column in LABEL_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise LabelError(f"{path.name} is missing columns: {', '.join(missing)}. Expected: {','.join(LABEL_COLUMNS)}")
        labels, problems = [], []
        seen: set[tuple[str, str]] = set()
        frame_conditions: dict[str, str] = {}
        known = set(slot_ids)
        for line, row in enumerate(reader, start=2):
            frame = (row["frame"] or "").strip().replace("\\", "/")
            slot_id = (row["slot_id"] or "").strip()
            status = (row["true_status"] or "").strip().lower()
            condition = (row["condition"] or "").strip().lower()
            occluded = (row["occluded"] or "").strip()
            if not any([frame, slot_id, status, condition, occluded]):
                continue  # blank line
            if not frame:
                problems.append(f"line {line}: frame is empty")
            if slot_id not in known:
                problems.append(f"line {line}: slot_id {slot_id!r} is not in the slot config")
            if status not in TRUE_STATUSES:
                problems.append(f"line {line}: true_status must be occupied or vacant, not {status!r}")
            if not condition:
                problems.append(f"line {line}: condition is empty")
            if occluded not in ("", "0", "1"):
                problems.append(f"line {line}: occluded must be 0, 1 or blank, not {occluded!r}")
            if (frame, slot_id) in seen:
                problems.append(f"line {line}: {frame} / {slot_id} is labeled twice")
            seen.add((frame, slot_id))
            if frame in frame_conditions and frame_conditions[frame] != condition:
                problems.append(f"line {line}: {frame} is labeled both {frame_conditions[frame]!r} and {condition!r}")
            frame_conditions.setdefault(frame, condition)
            labels.append(Label(frame, slot_id, status, condition, occluded == "1"))
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ...and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise LabelError(f"{path.name} has {len(problems)} problem(s):\n  {shown}{more}")
    if not labels:
        raise LabelError(f"{path.name} has no labeled rows yet. See the docstring of this script for the format.")
    return labels


def select_frames(labels: list[Label], only: list[str], exclude: list[str]) -> list[Label]:
    """Keep labels whose frame path matches any `only` pattern (if given) and no `exclude` pattern.

    Patterns are shell-style (fnmatch) on the frame path, e.g. "tune/*" or "test/2026-11-20-*".
    """
    from fnmatch import fnmatchcase

    kept = [
        label
        for label in labels
        if (not only or any(fnmatchcase(label.frame, pattern) for pattern in only))
        and not any(fnmatchcase(label.frame, pattern) for pattern in exclude)
    ]
    if not kept:
        raise LabelError("No labeled frames match --only/--exclude.")
    return kept


def frame_path(frames_dir: Path, frame: str) -> Path:
    path = Path(frame)
    return path if path.is_absolute() else frames_dir / path


def load_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def cache_key(model_id: str, image_path: Path, size: int | None = None) -> str:
    stat = image_path.stat()
    return "|".join(
        [
            model_id,
            f"size={size or slotsight.load_settings()['inference_size']}",
            f"conf>={slotsight.DETECTION_CONFIDENCE_FLOOR}",
            str(image_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
        ]
    )


# --- Running the pipeline ----------------------------------------------------


def run_pipeline(
    labels: list[Label],
    slots: list[dict[str, Any]],
    frames_dir: Path,
    overlap_threshold: float,
    review_threshold: float,
    cache_path: Path | None,
    empty_check_threshold: float = 0.0,
) -> list[Outcome]:
    """Classify every labeled frame with the dashboard's own code and pair it with its labels.

    Frames run in name order, which is time order within a session folder (see
    extract_frames.py). With `empty_check_threshold` > 0 the "doesn't look empty"
    check runs as it does live, learning each session's empty slots as it goes
    (and starting afresh in each session folder).
    """
    import appearance
    import cv2

    by_frame: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        by_frame[label.frame].append(label)
    missing = [frame for frame in by_frame if not frame_path(frames_dir, frame).exists()]
    if missing:
        listed = "\n  ".join(missing[:10])
        raise LabelError(f"{len(missing)} labeled frame(s) not found under {frames_dir}:\n  {listed}")

    model_id = slotsight.configured_model_id()
    cache = load_cache(cache_path) if cache_path else {}
    cache_changed = False
    outcomes: list[Outcome] = []
    reference = appearance.EmptyReference()
    session = None
    for index, (frame_name, frame_labels) in enumerate(sorted(by_frame.items()), start=1):
        folder = frame_name.rpartition("/")[0]  # "" for frames directly in frames_dir
        if folder != session:
            session = folder
            reference.reset()
        image_path = frame_path(frames_dir, frame_name)
        image = cv2.imread(str(image_path))
        if image is None:
            raise LabelError(f"Could not read {image_path} as an image.")
        key = cache_key(model_id, image_path)  # includes the inference size
        if key in cache:
            detections = [tuple(box) for box in cache[key]]
        else:
            detections = slotsight.detect_vehicles(image)
            cache[key] = [list(box) for box in detections]
            cache_changed = True
        settings = {**slotsight.load_settings(), "overlap_threshold": overlap_threshold, "review_threshold": review_threshold}
        probabilities = slotsight.slot_probabilities(image, slots, settings)
        shown = slotsight.decide(image, slots, detections, settings, probabilities)
        if probabilities is None:
            raw_results = slotsight.classify_frame(image, slots, detections, overlap_threshold)
            raw_status = {result["id"]: result["status"] for result in raw_results}
        else:  # the classifier's own call, before the review threshold and the detector's vote
            raw_status = {str(slot["id"]): ("occupied" if p_occ >= p_vac else "vacant") for slot, (p_vac, p_occ) in zip(slots, probabilities)}
        shown = reference.check(image, slots, shown, empty_check_threshold)
        shown_by_id = {result["id"]: result for result in shown}
        for label in frame_labels:
            result = shown_by_id[label.slot_id]
            outcomes.append(Outcome(label, result["status"], raw_status[label.slot_id], result["confidence"], result.get("reason") == appearance.REASON))
        if index % 25 == 0:
            print(f"  {index}/{len(by_frame)} frames", file=sys.stderr)
    if cache_path and cache_changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return outcomes


# --- Scoring -----------------------------------------------------------------


def ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def accuracy_block(outcomes: list[Outcome]) -> dict[str, Any]:
    n = len(outcomes)
    correct = sum(outcome.correct for outcome in outcomes)
    committed = [outcome for outcome in outcomes if outcome.shown != "unknown"]
    return {
        "n": n,
        "correct": correct,
        "accuracy": ratio(correct, n),
        "coverage": ratio(len(committed), n),
        "committed_accuracy": ratio(sum(outcome.correct for outcome in committed), len(committed)),
        "false_free": sum(outcome.false_free for outcome in outcomes),
    }


def verdict(passed: bool | None) -> str:
    return "NO DATA" if passed is None else ("PASS" if passed else "FAIL")


def score(outcomes: list[Outcome], slot_ids: list[str]) -> dict[str, Any]:
    """Compute every metric from the paired outcomes. Pure, so it can be checked without a model."""
    overall = accuracy_block(outcomes)
    confusion = Counter((outcome.label.true_status, outcome.shown) for outcome in outcomes)

    per_slot = {}
    for slot_id in slot_ids:
        rows = [outcome for outcome in outcomes if outcome.label.slot_id == slot_id]
        if rows:
            per_slot[slot_id] = {**accuracy_block(rows), "unknown": sum(row.shown == "unknown" for row in rows)}

    # R-02: only frames where every configured slot has a label.
    frames: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        frames[outcome.label.frame].append(outcome)
    all_slots = set(slot_ids)
    complete = {name: rows for name, rows in frames.items() if {row.label.slot_id for row in rows} == all_slots}
    count_errors = []
    for rows in complete.values():
        shown_free = sum(row.shown == "vacant" for row in rows)
        true_free = sum(row.label.true_status == "vacant" for row in rows)
        count_errors.append(shown_free - true_free)
    free_count = {
        "frames": len(complete),
        "skipped_frames": len(frames) - len(complete),
        "matches": sum(error == 0 for error in count_errors),
        "match_rate": ratio(sum(error == 0 for error in count_errors), len(count_errors)),
        "mean_abs_error": (sum(abs(error) for error in count_errors) / len(count_errors)) if count_errors else None,
        "over_reported": sum(error > 0 for error in count_errors),  # showed more free slots than exist
    }

    # R-05: per condition.
    conditions = {}
    for condition in sorted({outcome.label.condition for outcome in outcomes}):
        conditions[condition] = accuracy_block([o for o in outcomes if o.label.condition == condition])
    baseline = conditions.get(BASELINE_CONDITION, {}).get("accuracy")
    condition_ratios = {}
    for condition in COMPARED_CONDITIONS:
        accuracy = conditions.get(condition, {}).get("accuracy")
        value = accuracy / baseline if accuracy is not None and baseline else None
        condition_ratios[condition] = {
            "ratio": value,
            "verdict": verdict(None if value is None else value >= CONDITION_RATIO_TARGET),
        }
    tested = [result for result in condition_ratios.values() if result["verdict"] != "NO DATA"]
    if baseline is None or not tested:
        r05 = "NO DATA"
    elif any(result["verdict"] == "FAIL" for result in tested):
        r05 = "FAIL"
    elif len(tested) < len(COMPARED_CONDITIONS):
        r05 = "INCOMPLETE"  # every tested condition passes, but one has no labeled frames yet
    else:
        r05 = "PASS"

    # R-07: occluded cases must be correct or unknown, never a wrong confident answer.
    occluded = [outcome for outcome in outcomes if outcome.label.occluded]
    wrong = [outcome for outcome in occluded if not outcome.correct and outcome.shown != "unknown"]
    occluded_block = {
        "n": len(occluded),
        "correct": sum(outcome.correct for outcome in occluded),
        "unknown": sum(outcome.shown == "unknown" for outcome in occluded),
        "wrong": len(wrong),
        "false_free": sum(outcome.false_free for outcome in occluded),
        "verdict": verdict(None if not occluded else not wrong),
    }
    # What the review threshold did: rows it turned into "unknown", and how many of
    # those raw guesses would have been wrong (errors it prevented).
    held = [outcome for outcome in outcomes if outcome.shown == "unknown"]
    low_confidence = {
        "n": len(held),
        "raw_wrong": sum(outcome.raw != outcome.label.true_status for outcome in held),
        "raw_false_free": sum(outcome.raw == "vacant" and outcome.label.true_status == "occupied" for outcome in held),
    }

    # The "doesn't look empty" check: what it changed, and the scores without it.
    flagged = [outcome for outcome in outcomes if outcome.flagged]
    without = [Outcome(o.label, "vacant" if o.flagged else o.shown, o.raw, o.confidence) for o in outcomes]
    without_block = accuracy_block(without)
    empty_check = {
        "flagged": len(flagged),
        "caught": sum(outcome.label.true_status == "occupied" for outcome in flagged),
        "cost": sum(outcome.label.true_status == "vacant" for outcome in flagged),
        "accuracy_without": without_block["accuracy"],
        "false_free_without": without_block["false_free"],
    }

    return {
        "empty_check": empty_check,
        "overall": overall,
        "r01": verdict(None if overall["accuracy"] is None else overall["accuracy"] >= ACCURACY_TARGET),
        "confusion": confusion,
        "per_slot": per_slot,
        "free_count": free_count,
        "r02": verdict(None if free_count["match_rate"] is None else free_count["match_rate"] >= ACCURACY_TARGET),
        "conditions": conditions,
        "condition_ratios": condition_ratios,
        "r05": r05,
        "occluded": occluded_block,
        "r07": occluded_block["verdict"],
        "low_confidence": low_confidence,
        "frames": len(frames),
    }


# --- Reporting ---------------------------------------------------------------


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def sample_note(n: int) -> str:
    return f" (small sample: {n} rows)" if 0 < n < SMALL_SAMPLE else ""


def render_report(results: dict[str, Any], context: dict[str, Any]) -> str:
    overall, free, occluded, held = results["overall"], results["free_count"], results["occluded"], results["low_confidence"]
    # Built outside the f-strings below: reusing a quote inside an f-string needs Python 3.12+.
    note = f" - {context['note']}" if context["note"] else ""
    mean_error = "--" if free["mean_abs_error"] is None else f"{free['mean_abs_error']:.2f}"
    lines = [
        "",
        f"### Accuracy run: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- Labels: {context['labels']}{note}",
        f"- {results['frames']} frames, {overall['n']} labeled slot readings, {len(context['slot_ids'])} configured slots",
        f"- Model: {context['model']}; inference size {context.get('size', '--')}; overlap threshold {context['overlap']:g}; review threshold {context['review']:g}",
        "- Scoring: unknown counts as wrong for R-01, R-02 and R-05; R-07 allows correct or unknown on occluded slots",
        "",
        "| Requirement | Measure | Result | Target | Verdict |",
        "| --- | --- | --- | --- | --- |",
        f"| R-01 | Per-slot accuracy | {pct(overall['accuracy'])} ({overall['correct']}/{overall['n']}) | >= 90% | **{results['r01']}**{sample_note(overall['n'])} |",
        f"| R-02 | Free-count match rate | {pct(free['match_rate'])} ({free['matches']}/{free['frames']} frames) | >= 90% | **{results['r02']}** |",
    ]
    for condition, result in results["condition_ratios"].items():
        lines.append(f"| R-05 | {condition.capitalize()} / {BASELINE_CONDITION} accuracy | {pct(result['ratio'])} | >= 90% | {result['verdict']} |")
    lines += [
        f"| R-05 | Overall | | | **{results['r05']}** |",
        f"| R-07 | Occluded slots shown wrong (not unknown) | {occluded['wrong']} of {occluded['n']} | 0 | **{results['r07']}**{sample_note(occluded['n'])} |",
        "",
        f"- Coverage (slots not shown as unknown): {pct(overall['coverage'])}; accuracy on those: {pct(overall['committed_accuracy'])}",
        f"- Occupied slots shown as free (the costly error): {overall['false_free']}",
        f"- Free count: mean absolute error {mean_error} slots; "
        f"over-reported in {free['over_reported']} frames; {free['skipped_frames']} frames skipped (not every slot labeled)",
        f"- Occluded slots: {occluded['correct']} correct, {occluded['unknown']} unknown, {occluded['wrong']} wrong "
        f"({occluded['false_free']} shown free while occupied)",
        f"- \"Doesn't look empty\" check (threshold {context.get('empty_check', 0):g}): changed {results['empty_check']['flagged']} vacant readings to unknown, "
        f"{results['empty_check']['caught']} of them occupied (missed cars caught) and {results['empty_check']['cost']} truly vacant. "
        f"Without it: accuracy {pct(results['empty_check']['accuracy_without'])}, occupied shown as free {results['empty_check']['false_free_without']}",
        f"- Shown as unknown by the review threshold: {held['n']}; of those, the raw guess would have been wrong {held['raw_wrong']} times "
        f"({held['raw_false_free']} false free)",
        "",
        "| Condition | Rows | Accuracy | Coverage | False free |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition, block in results["conditions"].items():
        lines.append(f"| {condition} | {block['n']} | {pct(block['accuracy'])} | {pct(block['coverage'])} | {block['false_free']} |")
    lines += ["", "| True \\ shown | occupied | vacant | unknown |", "| --- | --- | --- | --- |"]
    for true_status in TRUE_STATUSES:
        cells = [str(results["confusion"].get((true_status, shown), 0)) for shown in ("occupied", "vacant", "unknown")]
        lines.append(f"| {true_status} | {' | '.join(cells)} |")
    lines += ["", "| Slot | Rows | Accuracy | Unknown | False free |", "| --- | --- | --- | --- | --- |"]
    for slot_id, block in results["per_slot"].items():
        lines.append(f"| {slot_id} | {block['n']} | {pct(block['accuracy'])} | {block['unknown']} | {block['false_free']} |")
    return "\n".join(lines) + "\n"


def write_predictions(path: Path, outcomes: list[Outcome]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "slot_id", "condition", "occluded", "true_status", "shown_status", "raw_status", "confidence", "correct", "empty_check_flagged"])
        for outcome in outcomes:
            label = outcome.label
            writer.writerow(
                [label.frame, label.slot_id, label.condition, int(label.occluded), label.true_status,
                 outcome.shown, outcome.raw, outcome.confidence, int(outcome.correct), int(outcome.flagged)]
            )


def append_to_evaluation(report: str, path: Path = EVALUATION_DOC) -> None:
    """Add the run to the accuracy section of evaluation.md, placed before the R-04 section."""
    text = path.read_text(encoding="utf-8") if path.exists() else "# SlotSight Evaluation\n\nMeasured evidence for the requirements in `requirements.md`.\n"
    if ACCURACY_SECTION not in text:
        intro = f"{ACCURACY_SECTION}\n\nRuns of `slotsight/eval/evaluate.py` against hand-labeled frames of the target lot.\n"
        marker = "\n## R-04"
        text = text.replace(marker, f"\n{intro}{marker}", 1) if marker in text else f"{text.rstrip()}\n\n{intro}"
    start = text.index(ACCURACY_SECTION)
    following = text.find("\n## ", start + len(ACCURACY_SECTION))
    end = following if following != -1 else len(text)
    text = f"{text[:end].rstrip()}\n{report}{text[end:]}"
    path.write_text(text, encoding="utf-8")


# --- Entry point -------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="labels CSV (default slotsight/eval/labels.csv)")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR, help="folder the frame paths are relative to")
    parser.add_argument("--slots", type=Path, default=slotsight.CONFIG_PATH, help="slot polygons (default config/slots.json)")
    parser.add_argument("--overlap-threshold", type=float, help="override overlap_threshold from settings.txt")
    parser.add_argument("--review-threshold", type=float, help="override review_threshold from settings.txt")
    parser.add_argument("--inference-size", type=int, choices=(640, 960, 1280), help="override inference_size from settings.txt (Ultralytics models)")
    parser.add_argument("--empty-check-threshold", type=float, help="override empty_check_threshold from settings.txt (0 = off)")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS, help="per-row results CSV")
    parser.add_argument("--only", action="append", default=[], metavar="PATTERN", help="score only frames matching this pattern, e.g. 'tune/*' (repeatable)")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN", help="leave out frames matching this pattern, e.g. 'tune/*' (repeatable)")
    parser.add_argument("--no-cache", action="store_true", help="re-run detection on every frame")
    parser.add_argument("--note", default="", help="free-text note for the report, e.g. 'week 1 labels'")
    parser.add_argument("--write", action="store_true", help="add the report to slotsight/docs/evaluation.md")
    args = parser.parse_args()

    slotsight.CONFIG_PATH = args.slots
    if args.inference_size:
        # Applies to this run only: detect_vehicles reads it through load_settings().
        real_load_settings = slotsight.load_settings
        slotsight.load_settings = lambda: {**real_load_settings(), "inference_size": args.inference_size}
    slots = slotsight.load_slots()
    slot_ids = [str(slot["id"]) for slot in slots]
    settings = slotsight.load_settings()
    overlap = settings["overlap_threshold"] if args.overlap_threshold is None else args.overlap_threshold
    review = settings["review_threshold"] if args.review_threshold is None else args.review_threshold
    empty_check = settings["empty_check_threshold"] if args.empty_check_threshold is None else args.empty_check_threshold
    try:
        labels = select_frames(read_labels(args.labels, slot_ids), args.only, args.exclude)
        # A cache miss with no model loaded raises DetectionError below.
        outcomes = run_pipeline(labels, slots, args.frames_dir, overlap, review, None if args.no_cache else DEFAULT_CACHE, empty_check)
    except LabelError as error:
        print(error, file=sys.stderr)
        return 1
    except slotsight.DetectionError as error:
        print(error, file=sys.stderr)
        return 1

    results = score(outcomes, slot_ids)
    report = render_report(
        results,
        {
            "labels": args.labels.name + "".join(f" --only {p}" for p in args.only) + "".join(f" --exclude {p}" for p in args.exclude),
            "note": args.note,
            "slot_ids": slot_ids,
            "model": slotsight.model_status()["name"]
            + (f" + slot classifier ({slotsight.slot_classifier.status().get('weights')})" if slotsight.load_settings()["slot_classifier"] and slotsight.slot_classifier.weights_path() else " (detector alone)"),
            "size": slotsight.load_settings()["inference_size"],
            "empty_check": empty_check,
            "overlap": overlap,
            "review": review,
        },
    )
    print(report)
    write_predictions(args.predictions, outcomes)
    print(f"Per-row results: {args.predictions}")
    if args.write:
        append_to_evaluation(report)
        print(f"Added to {EVALUATION_DOC}")
    verdicts = [results["r01"], results["r02"], results["r05"], results["r07"]]
    return 2 if "FAIL" in verdicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
