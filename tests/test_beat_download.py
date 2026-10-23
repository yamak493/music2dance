import io
import time

import numpy as np
import pytest
import requests
import torch

from m2d import audio_analysis as aa


class _Resp:
    def __init__(self, status, body=b""):
        self.status_code, self._body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP Error {self.status_code}")

    def iter_content(self, n):
        yield self._body


def _ckpt_bytes():
    buf = io.BytesIO()
    torch.save({"state_dict": {"w": torch.zeros(400_000)}, "hyper_parameters": {}}, buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _fake_hash(monkeypatch):
    import hashlib
    monkeypatch.setitem(aa.BEAT_THIS_SHA256, "final0", hashlib.sha256(_ckpt_bytes()).hexdigest())


def test_falls_back_to_mirror_and_checks_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if "cloud.cp.jku.at" in url:   # authors' host down
            return _Resp(503)
        if "github.com" in url:        # first mirror serves a corrupted file -> rejected by SHA-256
            return _Resp(200, b"x" * 2_000_000)
        return _Resp(200, _ckpt_bytes())  # second mirror OK

    monkeypatch.setattr(requests, "get", fake_get)
    path = aa.ensure_beat_this_checkpoint("final0", log=lambda *a: None)
    assert path.endswith("checkpoints/beat_this-final0.ckpt")
    assert [u.split("/")[2] for u in calls] == ["cloud.cp.jku.at", "cloud.cp.jku.at", "github.com", "huggingface.co"]
    calls.clear()
    assert aa.ensure_beat_this_checkpoint("final0") == path and not calls  # cached: no network


def test_gives_up_with_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(503))
    with pytest.raises(RuntimeError, match="BEAT_TRACKER"):
        aa.ensure_beat_this_checkpoint("final0", retries=2, log=lambda *a: None)


def test_librosa_fallback(tmp_path):
    import soundfile as sf
    sr = 22050
    y = np.zeros(sr * 12)
    for k, b in enumerate(np.arange(0.5, 11.5, 0.5)):   # 120 BPM, kick on every 4th beat
        i = int(b * sr)
        n = 2000
        tone = np.sin(2 * np.pi * (60 if k % 4 == 1 else 2000) * np.arange(n) / sr) * np.exp(-np.arange(n) / 400)
        y[i:i + n] += tone * (1.0 if k % 4 == 1 else 0.3)
    sf.write(tmp_path / "s.wav", y, sr)
    beats, down = aa.detect_beats_librosa(str(tmp_path / "s.wav"))
    assert abs(60 / np.median(np.diff(beats)) - 120) < 3
    assert len(down) >= 4 and np.allclose(np.diff(down), 2.0, atol=0.1)
