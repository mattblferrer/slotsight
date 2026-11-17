"""Vehicle detectors from several providers behind one interface.

Every model in the registry is a COCO-trained object detector. SlotSight keeps
only its vehicle classes (car, motorcycle, bus, truck), so any of them can feed
the slot classifier in app.py.

- A model is identified by "<provider>/<name>", e.g. "ultralytics/yolo11s".
  "ultralytics/file:<name>.pt" uses a custom Ultralytics weights file from models/.
- Installing a model downloads its weight files into models/ from the provider's
  official release, checking each file's SHA-256 or MD5 hash.
- A provider's Python package is imported only when one of its models is loaded,
  so the app runs with just the providers you have installed.

ModelManager loads at most one model at a time. Its status calls never block,
so dashboard polls stay fast while a model downloads or loads.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2
import numpy as np

log = logging.getLogger("slotsight.detectors")

Box = tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence (pixels)
VEHICLE_NAMES = frozenset({"car", "motorcycle", "bus", "truck"})
# Keep weak detections: they should push a slot to "unknown" rather than read as free (R-07).
DETECTION_CONFIDENCE_FLOOR = 0.1
CUSTOM_PREFIX = "ultralytics/file:"
USER_AGENT = "SlotSight-model-installer"


class DetectorError(RuntimeError):
    """A model could not be installed, loaded or run."""


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Box]:
        """Vehicle boxes in a BGR frame, with confidence >= DETECTION_CONFIDENCE_FLOOR."""


# --- Registry ----------------------------------------------------------------


@dataclass(frozen=True)
class RemoteFile:
    url: str
    path: str  # where it goes, relative to models/
    size: int  # bytes, for progress and a sanity check
    sha256: str | None = None
    md5: str | None = None


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    package: str  # module to import
    pip: str  # what to pip install for it
    licence: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    name: str
    licence: str
    files: tuple[RemoteFile, ...]
    note: str = ""

    @property
    def size_mb(self) -> float:
        return round(sum(file.size for file in self.files) / 1e6, 1)


PROVIDERS = {
    "ultralytics": Provider("ultralytics", "Ultralytics YOLO", "ultralytics", "ultralytics", "AGPL-3.0"),
    "huggingface": Provider("huggingface", "Hugging Face Transformers", "transformers", "transformers", "Apache-2.0"),
    "rfdetr": Provider("rfdetr", "Roboflow RF-DETR", "rfdetr", "rfdetr", "Apache-2.0"),
    "opencv": Provider("opencv", "OpenCV DNN", "cv2", "opencv-python-headless (already required)", "Apache-2.0"),
}

_ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
_RTDETR_REPO = "https://huggingface.co/PekingU/rtdetr_v2_r18vd/resolve/5650961749fa93567c0d46fc7f43ea4f9e914107"
_YOLOX_REPO = "https://huggingface.co/opencv/object_detection_yolox/resolve/78c368f74ce73ee28fc7a1be418a598c71b58b52"


def _ultralytics(name: str, size: int, sha256: str | None, title: str, note: str) -> ModelSpec:
    file = RemoteFile(f"{_ULTRALYTICS_RELEASE}/{name}.pt", f"{name}.pt", size, sha256)
    return ModelSpec(f"ultralytics/{name}", "ultralytics", title, "AGPL-3.0", (file,), note)


MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in [
        _ultralytics("yolo11n", 5613764, "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1", "YOLO11 nano", "Fastest; less accurate on small or distant vehicles."),
        _ultralytics("yolo11s", 19313732, "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5", "YOLO11 small", "Default. Used for the R-04 timing evidence."),
        _ultralytics("yolo11m", 40684120, "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95", "YOLO11 medium", "More accurate, about 2-3x slower on CPU."),
        _ultralytics("yolo26n", 5544453, "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef", "YOLO26 nano", "Newest Ultralytics generation."),
        _ultralytics("yolo26s", 20422725, "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b", "YOLO26 small", "Newest Ultralytics generation."),
        ModelSpec(
            "huggingface/rtdetr-v2-r18",
            "huggingface",
            "RT-DETRv2 R18",
            "Apache-2.0",
            (
                RemoteFile(f"{_RTDETR_REPO}/config.json", "huggingface/rtdetr_v2_r18vd/config.json", 5437),
                RemoteFile(f"{_RTDETR_REPO}/preprocessor_config.json", "huggingface/rtdetr_v2_r18vd/preprocessor_config.json", 444),
                RemoteFile(
                    f"{_RTDETR_REPO}/model.safetensors",
                    "huggingface/rtdetr_v2_r18vd/model.safetensors",
                    80904640,
                    "d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115",
                ),
            ),
            "Transformer detector (PekingU/rtdetr_v2_r18vd). Slower than YOLO on CPU.",
        ),
        ModelSpec(
            "rfdetr/nano",
            "rfdetr",
            "RF-DETR nano",
            "Apache-2.0",
            (RemoteFile("https://repo.roboflow.com/rfdetr/nano_coco/checkpoint_best_regular.pth", "rfdetr/rf-detr-nano.pth", 366287238, md5="fb6504cce7fbdc783f7a46991f07639f"),),
            "Transformer detector. Large download (the checkpoint includes training state).",
        ),
        ModelSpec(
            "rfdetr/small",
            "rfdetr",
            "RF-DETR small",
            "Apache-2.0",
            (RemoteFile("https://repo.roboflow.com/rfdetr/small_coco/checkpoint_best_regular.pth", "rfdetr/rf-detr-small.pth", 386045550, md5="fb37061c1af7bace359c91b723a8d5c1"),),
            "More accurate than nano, slower on CPU. Large download.",
        ),
        ModelSpec(
            "opencv/yolox-s",
            "opencv",
            "YOLOX-S (OpenCV zoo)",
            "Apache-2.0",
            (
                RemoteFile(
                    f"{_YOLOX_REPO}/object_detection_yolox_2022nov.onnx",
                    "opencv/object_detection_yolox_2022nov.onnx",
                    35858002,
                    "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
                ),
            ),
            "Runs on OpenCV alone, with no PyTorch. Older and less accurate.",
        ),
    ]
}
DEFAULT_MODEL_ID = "ultralytics/yolo11s"


def legacy_model_id(value: str) -> str:
    """Map an old `model_path` value (e.g. "yolo11s.pt", "models/x.pt") to a model id."""
    path = Path(value.replace("\\", "/"))
    if f"ultralytics/{path.stem}" in MODELS and path.suffix == ".pt":
        return f"ultralytics/{path.stem}"
    if not path.is_absolute() and path.parts[:1] == ("models",):
        path = Path(*path.parts[1:])  # the old resolver also accepted "models/<file>"
    return CUSTOM_PREFIX + path.as_posix()


def normalize_model_id(value: str | None) -> str:
    """Accept a registry id, a custom id, or a legacy .pt path."""
    value = (value or "").strip()
    if not value:
        return DEFAULT_MODEL_ID
    if value in MODELS or value.startswith(CUSTOM_PREFIX):
        return value
    if value.lower().endswith(".pt"):
        return legacy_model_id(value)
    return value  # unknown ids are reported as such, not silently replaced


def spec_for(model_id: str, models_dir: Path) -> ModelSpec:
    if model_id in MODELS:
        return MODELS[model_id]
    if model_id.startswith(CUSTOM_PREFIX):
        relative = model_id[len(CUSTOM_PREFIX):]
        path = Path(relative)
        size = path.stat().st_size if path.is_absolute() and path.exists() else 0
        # A custom file has no download source; it must already be in models/ (or be an absolute path).
        return ModelSpec(model_id, "ultralytics", f"Custom: {path.name}", "depends on the weights", (RemoteFile("", relative, size),))
    raise DetectorError(f"Unknown model {model_id!r}. Run `python slotsight/scripts/models.py list` to see the choices.")


def file_path(models_dir: Path, file: RemoteFile) -> Path:
    path = Path(file.path)
    return path if path.is_absolute() else models_dir / path


def package_available(provider: str) -> bool:
    return importlib.util.find_spec(PROVIDERS[provider].package) is not None


def is_installed(spec: ModelSpec, models_dir: Path) -> bool:
    return all(file_path(models_dir, file).is_file() for file in spec.files)


# --- Downloading -------------------------------------------------------------

Progress = Callable[[int, int], None]  # bytes done, bytes total


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, file: RemoteFile, name: str) -> None:
    for algorithm, expected in (("sha256", file.sha256), ("md5", file.md5)):
        if expected and _digest(path, algorithm) != expected:
            raise DetectorError(f"{name} failed its {algorithm.upper()} check; the download may be corrupt or changed upstream.")


def install(spec: ModelSpec, models_dir: Path, progress: Progress | None = None) -> None:
    """Download any missing files of `spec` into models_dir, verifying each one."""
    if any(not file.url for file in spec.files):
        raise DetectorError(f"{spec.name} has no download source. Put the file in {models_dir} yourself.")
    total = sum(file.size for file in spec.files)
    done = 0
    for file in spec.files:
        target = file_path(models_dir, file)
        if target.is_file():
            done += file.size
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{target.name}.{os.getpid()}.part")
        try:
            request = urllib.request.Request(file.url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
            _verify(partial, file, target.name)
            partial.replace(target)
        except DetectorError:
            raise
        except Exception as error:
            raise DetectorError(f"Could not download {target.name}: {error}") from error
        finally:
            partial.unlink(missing_ok=True)
    if progress:
        progress(total, total)


# --- Provider adapters ---------------------------------------------------------


def _vehicle_boxes(xyxy, scores, names) -> list[Box]:
    return [
        (float(x1), float(y1), float(x2), float(y2), float(score))
        for (x1, y1, x2, y2), score, name in zip(xyxy, scores, names)
        if name in VEHICLE_NAMES and score >= DETECTION_CONFIDENCE_FLOOR
    ]


class UltralyticsDetector:
    def __init__(self, weights: Path) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        names = self.model.names  # class id -> name; custom models may differ from COCO
        self.classes = [int(index) for index, name in names.items() if name in VEHICLE_NAMES]
        if not self.classes:
            raise DetectorError(f"{weights.name} has none of the vehicle classes {sorted(VEHICLE_NAMES)}.")

    def detect(self, frame: np.ndarray) -> list[Box]:
        result = self.model(frame, verbose=False, conf=DETECTION_CONFIDENCE_FLOOR, classes=self.classes)[0]
        names = [self.model.names[int(index)] for index in result.boxes.cls.tolist()]
        return _vehicle_boxes(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), names)


class TransformersDetector:
    def __init__(self, folder: Path) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(folder), local_files_only=True)
        self.model = AutoModelForObjectDetection.from_pretrained(str(folder), local_files_only=True).eval()
        self.labels = self.model.config.id2label

    def detect(self, frame: np.ndarray) -> list[Box]:
        height, width = frame.shape[:2]
        inputs = self.processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors="pt")
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=DETECTION_CONFIDENCE_FLOOR
        )[0]
        names = [self.labels[int(index)] for index in result["labels"].tolist()]
        return _vehicle_boxes(result["boxes"].tolist(), result["scores"].tolist(), names)


class RFDETRDetector:
    def __init__(self, weights: Path, size: str) -> None:
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES

        model_class = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall}[size]
        self.model = model_class(pretrain_weights=str(weights))
        self.names = COCO_CLASSES  # 1-based COCO category ids

    def detect(self, frame: np.ndarray) -> list[Box]:
        detections = self.model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=DETECTION_CONFIDENCE_FLOOR)
        names = [self.names.get(int(index), "") for index in detections.class_id]
        return _vehicle_boxes(detections.xyxy.tolist(), detections.confidence.tolist(), names)


class YoloxDetector:
    """YOLOX-S from the OpenCV model zoo, run with cv2.dnn.

    Pre- and post-processing follow the zoo's Apache-2.0 reference code
    (opencv/object_detection_yolox: demo.py and yolox.py): RGB, letterbox to
    640x640 padded with 114, decode grid offsets per stride, then NMS.
    """

    INPUT = 640
    STRIDES = (8, 16, 32)
    NMS_THRESHOLD = 0.5
    COCO80 = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # 0-based COCO-80 ids

    def __init__(self, weights: Path) -> None:
        self.net = cv2.dnn.readNet(str(weights))
        grids, strides = [], []
        for stride in self.STRIDES:
            cells = self.INPUT // stride
            xs, ys = np.meshgrid(np.arange(cells), np.arange(cells))
            grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
            strides.append(np.full((cells * cells, 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, frame: np.ndarray) -> list[Box]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = min(self.INPUT / rgb.shape[0], self.INPUT / rgb.shape[1])
        resized = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.INPUT, self.INPUT, 3), 114.0, dtype=np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(np.transpose(padded, (2, 0, 1))[np.newaxis])
        output = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]

        centres = (output[:, :2] + self.grids) * self.strides
        sizes = np.exp(output[:, 2:4]) * self.strides
        boxes = np.concatenate([centres - sizes / 2, centres + sizes / 2], axis=1) / scale
        class_scores = output[:, 4:5] * output[:, 5:]
        classes = class_scores.argmax(axis=1)
        scores = class_scores.max(axis=1)
        mask = np.isin(classes, list(self.COCO80)) & (scores >= DETECTION_CONFIDENCE_FLOOR)
        boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
        if not len(boxes):
            return []
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
        keep = cv2.dnn.NMSBoxesBatched(rects, scores.tolist(), classes.tolist(), DETECTION_CONFIDENCE_FLOOR, self.NMS_THRESHOLD)
        keep = np.array(keep).reshape(-1)
        return _vehicle_boxes(boxes[keep].tolist(), scores[keep].tolist(), [self.COCO80[int(c)] for c in classes[keep]])


def load_detector(spec: ModelSpec, models_dir: Path) -> Detector:
    """Load an installed model and run one warm-up inference."""
    if not package_available(spec.provider):
        provider = PROVIDERS[spec.provider]
        raise DetectorError(f"{spec.name} needs the {provider.package} package: pip install {provider.pip}")
    if not is_installed(spec, models_dir):
        raise DetectorError(f"{spec.name} is not installed.")
    first = file_path(models_dir, spec.files[0])
    try:
        if spec.provider == "ultralytics":
            detector: Detector = UltralyticsDetector(first)
        elif spec.provider == "huggingface":
            detector = TransformersDetector(first.parent)
        elif spec.provider == "rfdetr":
            detector = RFDETRDetector(first, spec.id.split("/", 1)[1])
        elif spec.provider == "opencv":
            detector = YoloxDetector(first)
        else:
            raise DetectorError(f"No loader for provider {spec.provider!r}")
        # The first inference is several times slower (lazy setup); pay it at load
        # time rather than on the first real frame, so it never delays an update (R-04).
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        return detector
    except DetectorError:
        raise
    except Exception as error:
        raise DetectorError(f"Could not load {spec.name}: {error}") from error


# --- Manager -----------------------------------------------------------------


@dataclass
class _Job:
    model_id: str
    action: str  # "install" or "load"
    state: str = "running"  # running, finished, error
    done: int = 0
    total: int = 0
    error: str | None = None
    started: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.model_id,
            "action": self.action,
            "state": self.state,
            "progress": round(self.done / self.total, 3) if self.total else None,
            "error": self.error,
        }


class ModelManager:
    """Holds the one loaded detector and runs installs and loads in the background.

    `status()` and `loaded()` never block. `require()` blocks until the model is
    loaded and must not be called while holding app.state_lock.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = threading.Lock()  # guards the fields below (held only briefly)
        self._load_lock = threading.Lock()  # one load at a time; held while loading
        self._detector: Detector | None = None
        self._detector_id: str | None = None
        self._job: _Job | None = None
        self._errors: dict[str, str] = {}

    # Non-blocking views

    def loaded(self, model_id: str) -> Detector | None:
        with self._lock:
            return self._detector if self._detector_id == model_id else None

    def job(self) -> dict[str, Any] | None:
        with self._lock:
            return self._job.as_dict() if self._job else None

    def busy(self) -> bool:
        with self._lock:
            return self._job is not None and self._job.state == "running"

    def status(self, model_id: str) -> dict[str, Any]:
        """state: ready, loading, downloading, missing, needs_package, error or unknown."""
        try:
            spec = spec_for(model_id, self.models_dir)
        except DetectorError as error:
            return {"id": model_id, "name": model_id, "state": "unknown", "message": str(error), "progress": None}
        with self._lock:
            job = self._job if self._job and self._job.model_id == model_id and self._job.state == "running" else None
            ready = self._detector_id == model_id and self._detector is not None
            error = self._errors.get(model_id)
        base = {"id": model_id, "name": spec.name, "progress": None}
        if ready:
            return {**base, "state": "ready", "message": f"{spec.name} loaded"}
        if job:
            state = "downloading" if job.action == "install" else "loading"
            verb = "Downloading" if job.action == "install" else "Loading"
            return {**base, "state": state, "message": f"{verb} {spec.name}", "progress": job.as_dict()["progress"]}
        if error:
            return {**base, "state": "error", "message": error}
        if not package_available(spec.provider):
            return {**base, "state": "needs_package", "message": f"pip install {PROVIDERS[spec.provider].pip}"}
        if not is_installed(spec, self.models_dir):
            return {**base, "state": "missing", "message": f"{spec.name} is not installed"}
        return {**base, "state": "loading", "message": f"{spec.name} loads on first use"}

    def catalogue(self, active_id: str) -> list[dict[str, Any]]:
        ids = list(MODELS)
        if active_id not in MODELS:
            ids.append(active_id)
        rows = []
        for model_id in ids:
            try:
                spec = spec_for(model_id, self.models_dir)
            except DetectorError:
                continue
            provider = PROVIDERS[spec.provider]
            rows.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "provider": provider.name,
                    "licence": spec.licence,
                    "size_mb": spec.size_mb,
                    "note": spec.note,
                    "installed": is_installed(spec, self.models_dir),
                    "package_available": package_available(spec.provider),
                    "pip": provider.pip,
                    "active": spec.id == active_id,
                    "status": self.status(spec.id),
                }
            )
        return rows

    # Blocking work

    def require(self, model_id: str) -> Detector:
        """Return the loaded detector for model_id, loading it first if needed (blocking)."""
        detector = self.loaded(model_id)
        if detector is not None:
            return detector
        spec = spec_for(model_id, self.models_dir)
        with self._load_lock:
            detector = self.loaded(model_id)
            if detector is not None:
                return detector
            try:
                detector = load_detector(spec, self.models_dir)
            except DetectorError as error:
                with self._lock:
                    self._errors[model_id] = str(error)
                raise
            with self._lock:
                # Only one model stays in memory.
                self._detector, self._detector_id = detector, model_id
                self._errors.pop(model_id, None)
            log.info("Loaded %s", spec.name)
            return detector

    def install(self, model_id: str, progress: Progress | None = None) -> None:
        spec = spec_for(model_id, self.models_dir)
        install(spec, self.models_dir, progress)
        with self._lock:
            self._errors.pop(model_id, None)

    # Background jobs

    def start(self, model_id: str, download: bool, load: bool) -> dict[str, Any]:
        """Install (if `download` and missing) and/or load a model on a background thread.

        Raises DetectorError if another job is running or the id is unknown.
        """
        spec = spec_for(model_id, self.models_dir)
        needs_download = download and not is_installed(spec, self.models_dir)
        with self._lock:
            if self._job and self._job.state == "running":
                raise DetectorError(f"Busy with {self._job.model_id}; try again when it finishes.")
            job = _Job(model_id, "install" if needs_download else "load", total=sum(f.size for f in spec.files))
            self._job = job
            self._errors.pop(model_id, None)

        def progress(done: int, total: int) -> None:
            job.done, job.total = done, total

        def run() -> None:
            try:
                if needs_download:
                    log.info("Downloading %s (%.0f MB)", spec.name, spec.size_mb)
                    self.install(model_id, progress)
                if load:
                    job.action = "load"
                    self.require(model_id)
                job.state = "finished"
            except Exception as error:  # reported through status(), never swallowed
                log.error("%s: %s", spec.name, error)
                job.state, job.error = "error", str(error)
                with self._lock:
                    self._errors[model_id] = str(error)

        threading.Thread(target=run, name=f"model-{model_id}", daemon=True).start()
        return job.as_dict()
