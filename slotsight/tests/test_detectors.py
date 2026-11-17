"""Model downloads (hash checks, failures) and ModelManager states, without the network."""

import hashlib
import io

import pytest
from conftest import wait_until

import detectors

PAYLOAD = b"model weights" * 1000


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.fixture
def fake_download(monkeypatch):
    """Serve PAYLOAD for every URL and record what was requested."""
    requested = []

    def urlopen(request, timeout=None):
        requested.append(request.full_url)
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(detectors.urllib.request, "urlopen", urlopen)
    return requested


def spec_with(sha256=None, md5=None, url="https://example.invalid/weights.bin"):
    file = detectors.RemoteFile(url, "test/weights.bin", len(PAYLOAD), sha256=sha256, md5=md5)
    return detectors.ModelSpec("opencv/test", "opencv", "Test model", "-", (file,))


def test_install_verifies_sha256_and_reports_progress(tmp_path, fake_download):
    progress = []
    spec = spec_with(sha256=hashlib.sha256(PAYLOAD).hexdigest())
    detectors.install(spec, tmp_path, lambda done, total: progress.append((done, total)))
    assert (tmp_path / "test" / "weights.bin").read_bytes() == PAYLOAD
    assert progress[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert detectors.is_installed(spec, tmp_path)


def test_install_verifies_md5(tmp_path, fake_download):
    detectors.install(spec_with(md5=hashlib.md5(PAYLOAD).hexdigest()), tmp_path)
    assert (tmp_path / "test" / "weights.bin").exists()


def test_hash_mismatch_is_rejected_and_nothing_is_kept(tmp_path, fake_download):
    with pytest.raises(detectors.DetectorError, match="SHA256"):
        detectors.install(spec_with(sha256="0" * 64), tmp_path)
    assert not any((tmp_path / "test").iterdir())


def test_network_failure_leaves_no_partial_file(tmp_path, monkeypatch):
    def urlopen(request, timeout=None):
        raise OSError("network unreachable")

    monkeypatch.setattr(detectors.urllib.request, "urlopen", urlopen)
    with pytest.raises(detectors.DetectorError, match="network unreachable"):
        detectors.install(spec_with(sha256="0" * 64), tmp_path)
    assert not list(tmp_path.rglob("*.part"))
    assert not (tmp_path / "test" / "weights.bin").exists()


def test_installed_files_are_not_downloaded_again(tmp_path, fake_download):
    spec = spec_with(sha256=hashlib.sha256(PAYLOAD).hexdigest())
    detectors.install(spec, tmp_path)
    detectors.install(spec, tmp_path)
    assert len(fake_download) == 1


def test_custom_file_has_no_download_source(tmp_path):
    spec = detectors.spec_for("ultralytics/file:mine.pt", tmp_path)
    with pytest.raises(detectors.DetectorError, match="no download source"):
        detectors.install(spec, tmp_path)


def test_unknown_model_id_is_reported(tmp_path):
    manager = detectors.ModelManager(tmp_path)
    assert manager.status("nope/x")["state"] == "unknown"


def test_manager_states_missing_then_installed(tmp_path, fake_download, monkeypatch):
    spec = spec_with(sha256=hashlib.sha256(PAYLOAD).hexdigest())
    monkeypatch.setitem(detectors.MODELS, spec.id, spec)
    manager = detectors.ModelManager(tmp_path)
    assert manager.status(spec.id)["state"] == "missing"
    manager.start(spec.id, download=True, load=False)
    assert wait_until(lambda: manager.job()["state"] == "finished")
    # Installed but not loaded yet: it loads on first use.
    assert manager.status(spec.id)["state"] == "loading"


def test_manager_reports_a_load_failure(tmp_path, fake_download, monkeypatch):
    spec = spec_with(sha256=hashlib.sha256(PAYLOAD).hexdigest())  # not a real ONNX file
    monkeypatch.setitem(detectors.MODELS, spec.id, spec)
    manager = detectors.ModelManager(tmp_path)
    manager.start(spec.id, download=True, load=True)
    assert wait_until(lambda: manager.job()["state"] == "error")
    status = manager.status(spec.id)
    assert status["state"] == "error" and "Could not load" in status["message"]
    with pytest.raises(detectors.DetectorError):
        manager.require(spec.id)


def test_manager_runs_one_job_at_a_time(tmp_path, monkeypatch):
    import threading

    release = threading.Event()

    def slow_urlopen(request, timeout=None):
        release.wait(5)
        return FakeResponse(PAYLOAD)

    monkeypatch.setattr(detectors.urllib.request, "urlopen", slow_urlopen)
    spec = spec_with(sha256=hashlib.sha256(PAYLOAD).hexdigest())
    monkeypatch.setitem(detectors.MODELS, spec.id, spec)
    manager = detectors.ModelManager(tmp_path)
    manager.start(spec.id, download=True, load=False)
    with pytest.raises(detectors.DetectorError, match="Busy"):
        manager.start(spec.id, download=True, load=False)
    assert manager.status(spec.id)["state"] == "downloading"
    release.set()
    assert wait_until(lambda: manager.job()["state"] == "finished")


def test_status_does_not_block_while_a_model_loads(tmp_path, monkeypatch):
    import threading
    import time

    started, release = threading.Event(), threading.Event()

    def slow_load(spec, models_dir):
        started.set()
        release.wait(5)
        return object()

    monkeypatch.setattr(detectors, "load_detector", slow_load)
    spec = spec_with()
    monkeypatch.setitem(detectors.MODELS, spec.id, spec)
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "weights.bin").write_bytes(PAYLOAD)
    manager = detectors.ModelManager(tmp_path)
    manager.start(spec.id, download=False, load=True)
    assert started.wait(5)
    began = time.perf_counter()
    assert manager.status(spec.id)["state"] == "loading"
    assert time.perf_counter() - began < 0.1
    release.set()
    assert wait_until(lambda: manager.status(spec.id)["state"] == "ready")
