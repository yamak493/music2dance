"""Steps 1, 3, 4: vocal separation (Demucs), beats/downbeats (beat_this) and the 8-count grid."""
import gc
import json
import os

import numpy as np

COUNT_NAMES = [f"{i}{s}" for i in range(1, 9) for s in ("", "&")]  # '1', '1&', ..., '8&'


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_audio(path, sr=None, mono=False):
    import librosa
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return y, sr


# ---------------------------------------------------------------------------- step 1: Demucs
def separate_vocals(audio_path, out_dir, model_name="htdemucs", device="cuda", shifts=1, overlap=0.25):
    """Returns paths of vocals.wav and accompaniment.wav (44.1 kHz stereo)."""
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    os.makedirs(out_dir, exist_ok=True)
    model = get_model(model_name)
    model.eval()
    y, _ = load_audio(audio_path, sr=model.samplerate, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    if y.shape[0] == 1:
        y = np.concatenate([y, y])
    y = y[: model.audio_channels]
    wav = torch.as_tensor(y, dtype=torch.float32)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        sources = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=shifts, split=True,
                              overlap=overlap, progress=True)[0]
    sources = sources * std + mean
    names = list(model.sources)
    vocals = sources[names.index("vocals")].cpu().numpy()
    accomp = sum(sources[i] for i, n in enumerate(names) if n != "vocals").cpu().numpy()
    vp = os.path.join(out_dir, "vocals.wav")
    ap = os.path.join(out_dir, "accompaniment.wav")
    sf.write(vp, vocals.T, model.samplerate)
    sf.write(ap, accomp.T, model.samplerate)
    del model, sources, wav
    _free()
    return vp, ap


# ---------------------------------------------------------------------------- step 3: beat_this
def detect_beats(audio_path, checkpoint="final0", device="cuda", dbn=False):
    from beat_this.inference import Audio2Beats
    y, sr = load_audio(audio_path, sr=None, mono=True)
    a2b = Audio2Beats(checkpoint_path=checkpoint, device=device, dbn=dbn)
    beats, downbeats = a2b(y, sr)
    del a2b
    _free()
    return np.asarray(beats, dtype=np.float64), np.asarray(downbeats, dtype=np.float64)


def tempo_from_beats(beats):
    if len(beats) < 2:
        return float("nan")
    return float(60.0 / np.median(np.diff(beats)))


# ---------------------------------------------------------------------------- step 4: 8 counts
def _nearest(arr, x):
    i = int(np.searchsorted(arr, x))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(arr) and (best is None or abs(arr[j] - x) < abs(arr[best] - x)):
            best = j
    return best


def _build_eights(beats, is_down, start_idx):
    """Group beats into eights. A downbeat that would fall on a count other than 1 or 5 (an irregular
    bar or a tempo/meter change) closes the current eight early and restarts counting on that downbeat."""
    eights = []
    cur = None
    for i in range(start_idx, len(beats)):
        if cur is None:
            cur = {"beats": [i], "irregular": False}
            continue
        k = len(cur["beats"])  # the next beat would be count k+1
        if k == 8:
            eights.append(cur)
            cur = {"beats": [i], "irregular": False}
        elif is_down[i] and k != 4:
            cur["irregular"] = True
            eights.append(cur)
            cur = {"beats": [i], "irregular": False}
        else:
            cur["beats"].append(i)
    if cur is not None and cur["beats"]:
        eights.append(cur)
    return eights


def build_grid(beats, downbeats, duration, line_starts=None, phase="auto", tol=0.07):
    """8-count grid (two 4/4 bars per unit, starting on a downbeat).

    phase: 'auto' chooses whether eights start on even or odd bars so that lyric lines begin on (or
    just before) count 1; or an int bar offset (0/1) to force it.
    Returns dict with 'eights' (index, start, end, beat_times, counts{name: time}), 'beats', 'downbeats'.
    """
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    if len(beats) < 8:
        raise ValueError("Too few beats were detected to build 8 counts")
    is_down = np.zeros(len(beats), dtype=bool)
    for d in downbeats:
        j = _nearest(beats, d)
        if j is not None and abs(beats[j] - d) <= tol:
            is_down[j] = True
    down_idx = np.flatnonzero(is_down)
    if len(down_idx) == 0:
        down_idx = np.array([0])
        is_down[0] = True
    period = float(np.median(np.diff(beats)))

    def make(start_bar):
        start = int(down_idx[min(start_bar, len(down_idx) - 1)])
        ev = _build_eights(beats, is_down, start)
        return start, ev

    def score(ev):
        if not line_starts:
            return 0.0
        ones = np.array([beats[e["beats"][0]] for e in ev])
        fives = np.array([beats[e["beats"][4]] for e in ev if len(e["beats"]) > 4])
        sc = 0.0
        for ls in line_starts:
            d1 = np.min(np.abs(ones - ls)) if len(ones) else 9e9
            dpk = np.min(np.abs(ones - period - ls)) if len(ones) else 9e9  # pickup on count 8
            dph = np.min(np.abs(ones - 0.5 * period - ls)) if len(ones) else 9e9  # pickup on 8&
            d5 = np.min(np.abs(fives - ls)) if len(fives) else 9e9
            sc += 1.0 * (d1 < 0.35 * period) + 0.6 * (min(dpk, dph) < 0.3 * period) + 0.3 * (d5 < 0.35 * period)
        return sc

    if phase == "auto":
        options = [make(0), make(1)]
        scores = [float(score(ev)) for _, ev in options]
        choice = int(np.argmax(scores)) if scores[1] > scores[0] else 0
    else:
        choice = int(phase)
        scores = None
    start_idx, ev = make(choice)
    eights = []
    for n, e in enumerate(ev):
        bi = e["beats"]
        times = [float(beats[i]) for i in bi]
        nxt = float(beats[bi[-1] + 1]) if bi[-1] + 1 < len(beats) else float(times[-1] + period)
        local_p = float(np.median(np.diff(times + [nxt]))) if len(times) > 0 else period
        counts = {}
        for k, t in enumerate(times):
            t_next = float(beats[bi[k] + 1]) if bi[k] + 1 < len(beats) else t + local_p
            counts[f"{k + 1}"] = t
            counts[f"{k + 1}&"] = 0.5 * (t + t_next)
        end = nxt if len(times) == 8 else min(duration, times[-1] + local_p)
        eights.append({"index": n, "start": times[0], "end": float(end), "beat_times": times, "counts": counts,
                       "n_beats": len(times), "irregular": bool(e["irregular"]), "beat_period": local_p})
    return {"eights": eights, "beats": beats.tolist(), "downbeats": downbeats.tolist(), "bpm": tempo_from_beats(beats),
            "beat_period": period, "phase": choice, "phase_scores": scores, "pickup_beats": int(start_idx)}


def count_time(grid, eight, count):
    e = grid["eights"][int(eight)]
    count = str(count).strip()
    if count not in e["counts"]:
        raise KeyError(f"count {count!r} does not exist in eight {eight} (has {list(e['counts'])})")
    return e["counts"][count]


def nearest_slot(grid, t):
    """(eight index, count name, slot time) of the 8-count slot closest to time t."""
    best = None
    for e in grid["eights"]:
        if t < e["start"] - 2.0 or t > e["end"] + 2.0:
            continue
        for name, ct in e["counts"].items():
            d = abs(ct - t)
            if best is None or d < best[0]:
                best = (d, e["index"], name, ct)
    if best is None:
        return None
    return best[1], best[2], best[3]


def attach_lyrics(grid, lines, words):
    """Annotate eights with lyric words on their nearest count and lines with the eights they cover."""
    for e in grid["eights"]:
        e["words"] = []
        e["lines"] = []
    for w in words:
        if w.get("start") is None:
            continue
        slot = nearest_slot(grid, w["start"])
        if slot is None:
            continue
        ei, cname, _ = slot
        grid["eights"][ei]["words"].append({"count": cname, "text": w["text"], "line": w["line"]})
    for ln in lines:
        if ln.get("start") is None:
            ln["eights"] = []
            continue
        ln["eights"] = [e["index"] for e in grid["eights"] if e["end"] > ln["start"] and e["start"] < ln["end"]]
        for ei in ln["eights"]:
            grid["eights"][ei]["lines"].append(ln["index"])
    return grid


def grid_table(grid, lines=None):
    """Rows for display: one per eight."""
    rows = []
    for e in grid["eights"]:
        wtxt = " ".join(f"{w['count']}:{w['text']}" for w in e.get("words", []))
        rows.append({"eight": e["index"], "start": round(e["start"], 2), "end": round(e["end"], 2),
                     "beats": e["n_beats"], "irregular": e["irregular"], "lines": ",".join(map(str, e.get("lines", []))),
                     "lyrics_on_counts": wtxt})
    return rows


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
