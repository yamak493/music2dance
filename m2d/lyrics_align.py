"""Step 2: forced alignment of the given lyrics to the separated vocals (wav2vec2 via WhisperX).

Two stages, so that whole songs fit in memory:
  A. global pass: wav2vec2 CTC log-probabilities over the whole vocal track (computed window by window)
     and one Viterbi CTC forced alignment of the complete lyrics -> coarse line timings;
  B. WhisperX's ``align`` per lyric line inside its coarse window (+ padding) -> refined timings.
     Lines where stage B fails or looks implausible keep the stage-A result.

Japanese: the wav2vec2 vocabulary does not contain every kanji, so each pykakasi segment whose
characters are not all in the vocabulary is aligned through its hiragana reading and mapped back to
the original characters. Words = pykakasi segments (Japanese) or whitespace tokens (other languages).
"""
import re
import unicodedata

import numpy as np

SECTION_RE = re.compile(r"^\s*([\[【(（<].*[\]】)）>])\s*$")
SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------- lyrics parsing
def parse_lyrics(text):
    lines, section = [], None
    for raw in text.splitlines():
        s = unicodedata.normalize("NFKC", raw).strip()
        if not s:
            continue
        m = SECTION_RE.match(s)
        if m:
            section = m.group(1).strip("[]【】()（）<>").strip()
            continue
        lines.append({"index": len(lines), "text": s, "section": section})
    if not lines:
        raise ValueError("The lyrics are empty")
    return lines


def _kakasi():
    import pykakasi
    return pykakasi.kakasi()


def _katakana_to_hiragana(s):
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def build_tokens(lines, dictionary, language="ja"):
    """Alignment units. Returns (tokens, words): tokens carry (line, char_start, char_end, label_id, char)."""
    tokens, words = [], []
    dic = dictionary
    kks = _kakasi() if language == "ja" else None
    for ln in lines:
        text = ln["text"]
        if language == "ja":
            pos = 0
            for seg in kks.convert(text):
                orig = seg["orig"]
                start = text.find(orig, pos)
                if start < 0:
                    start = pos
                end = start + len(orig)
                pos = end
                if not orig.strip():
                    continue
                lead = len(orig) - len(orig.lstrip())
                orig = orig.strip()
                start = start + lead
                end = start + len(orig)
                w = {"line": ln["index"], "text": orig, "char_start": start, "char_end": end}
                content = [(k, c) for k, c in enumerate(orig)
                           if not unicodedata.category(c).startswith(("P", "Z", "S", "C"))]
                if not content:
                    continue  # punctuation only
                if all(c.lower() in dic for _, c in content):
                    for k, c in content:
                        tokens.append({"line": ln["index"], "c0": start + k, "c1": start + k + 1,
                                       "id": dic[c.lower()], "char": c})
                else:
                    reading = _katakana_to_hiragana(seg.get("hira", "") or "")
                    rd = [c for c in reading if c.lower() in dic]
                    if not rd:  # try katakana reading
                        rd = [c for c in (seg.get("kana", "") or "") if c.lower() in dic]
                    for c in rd:
                        tokens.append({"line": ln["index"], "c0": start, "c1": end, "id": dic[c.lower()], "char": c})
                words.append(w)
        else:
            sep = dic.get("|")
            for m in re.finditer(r"\S+", text):
                w = {"line": ln["index"], "text": m.group(0), "char_start": m.start(), "char_end": m.end()}
                words.append(w)
                if sep is not None and tokens and tokens[-1]["line"] == ln["index"]:
                    tokens.append({"line": ln["index"], "c0": m.start(), "c1": m.start(), "id": sep, "char": "|",
                                   "is_sep": True})
                for k, c in enumerate(m.group(0)):
                    if c.lower() in dic:
                        tokens.append({"line": ln["index"], "c0": m.start() + k, "c1": m.start() + k + 1,
                                       "id": dic[c.lower()], "char": c.lower()})
    return tokens, words


# ---------------------------------------------------------------------------- emissions
def load_align_model(language, device):
    """WhisperX's alignment model loader (falls back to transformers if WhisperX cannot be imported)."""
    try:
        import whisperx
        model, meta = whisperx.load_align_model(language_code=language, device=device)
        meta = dict(meta)
        meta["loader"] = "whisperx"
        return model, meta
    except Exception as e:  # pragma: no cover - depends on the environment
        print(f"[warn] whisperx.load_align_model failed ({e!r}); loading the model with transformers")
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        name = {"ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
                "en": "facebook/wav2vec2-base-960h"}.get(language)
        if name is None:
            raise
        proc = Wav2Vec2Processor.from_pretrained(name)
        model = Wav2Vec2ForCTC.from_pretrained(name).to(device)
        vocab = proc.tokenizer.get_vocab()
        return model, {"language": language, "dictionary": {k.lower(): v for k, v in vocab.items()},
                       "type": "huggingface", "loader": "transformers"}


def blank_id(meta):
    d = meta["dictionary"]
    for k in ("[pad]", "<pad>"):
        if k in d:
            return d[k]
    return 0


def compute_emissions(model, meta, audio16k, device, window_s=20.0, pad_s=1.0):
    """Log-probabilities (T_frames, vocab) for the whole signal; returns (logp, seconds_per_frame)."""
    import torch
    n = len(audio16k)
    win, pad = int(window_s * SAMPLE_RATE), int(pad_s * SAMPLE_RATE)
    outs = []
    frame_dur = None
    model.eval()
    for s in range(0, n, win):
        a, b = max(0, s - pad), min(n, s + win + pad)
        seg = np.asarray(audio16k[a:b], dtype=np.float32)
        if len(seg) < 400:
            seg = np.pad(seg, (0, 400 - len(seg)))
        x = torch.as_tensor(seg)[None].to(device)
        with torch.inference_mode():
            if meta.get("type") == "torchaudio":
                em, _ = model(x)
            else:
                xn = (x - x.mean()) / (x.std() + 1e-7)
                em = model(xn).logits
            em = torch.log_softmax(em.float(), dim=-1)[0].cpu().numpy()
        fd = (b - a) / SAMPLE_RATE / em.shape[0]
        frame_dur = fd if frame_dur is None else frame_dur
        k0 = int(round((s - a) / SAMPLE_RATE / fd))
        k1 = k0 + int(round((min(n, s + win) - s) / SAMPLE_RATE / fd))
        outs.append(em[k0:min(k1, em.shape[0])])
    return np.concatenate(outs, axis=0), frame_dur


# ---------------------------------------------------------------------------- Viterbi CTC
def ctc_force_align(logp, labels, blank, free_start=False, free_end=False):
    """Viterbi forced alignment. logp: (T, V); labels: (N,). Returns per-label (start, end, score) frames.

    free_start / free_end: the audio may begin after the first / stop before the last label (a cut-out
    section of the song). Labels outside the audio are left unaligned (start = end = -1).
    """
    T, N = logp.shape[0], len(labels)
    S = 2 * N + 1
    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = labels
    allow2 = np.zeros(S, dtype=bool)
    allow2[3::2] = np.asarray(labels[1:]) != np.asarray(labels[:-1])
    if T < N and not (free_start or free_end):
        raise ValueError(f"audio too short for the lyrics ({T} frames < {N} tokens)")
    neg = -1e30
    dp = np.full(S, neg)
    if free_start:
        dp[:] = logp[0, ext]  # may enter the label sequence anywhere
    else:
        dp[0] = logp[0, blank]
        dp[1] = logp[0, ext[1]]
    bp = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        c0 = dp
        c1 = np.concatenate([[neg], dp[:-1]])
        c2 = np.concatenate([[neg, neg], dp[:-2]])
        c2 = np.where(allow2, c2, neg)
        stack = np.stack([c0, c1, c2])
        arg = stack.argmax(0)
        dp = stack[arg, np.arange(S)] + logp[t, ext]
        bp[t] = arg
    if free_end:
        s = int(np.argmax(dp))  # may leave the label sequence anywhere
    else:
        s = int(S - 1 if dp[S - 1] >= dp[S - 2] else S - 2)
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    starts = np.full(N, -1)
    ends = np.full(N, -1)
    scores = np.zeros(N)
    counts = np.zeros(N)
    for t in range(T):
        st = path[t]
        if st % 2 == 1:
            k = (st - 1) // 2
            if starts[k] < 0:
                starts[k] = t
            ends[k] = t + 1
            scores[k] += logp[t, ext[st]]
            counts[k] += 1
    scores = scores / np.maximum(counts, 1)
    return starts, ends, scores


def _assign_times(tokens, words, lines, frame_dur, t_offset=0.0):
    """Derive word/line times from token times (characters without tokens are interpolated)."""
    for ln in lines:
        lt = [t for t in tokens if t["line"] == ln["index"] and t.get("start") is not None]
        ln["start"] = float(min(t["start"] for t in lt) + t_offset) if lt else None
        ln["end"] = float(max(t["end"] for t in lt) + t_offset) if lt else None
        ln["score"] = float(np.mean([t["score"] for t in lt])) if lt else float("nan")
    for w in words:
        wt = [t for t in tokens if t["line"] == w["line"] and t.get("start") is not None
              and t["c0"] < w["char_end"] and t["c1"] > w["char_start"] and not t.get("is_sep")]
        w["start"] = float(min(t["start"] for t in wt) + t_offset) if wt else None
        w["end"] = float(max(t["end"] for t in wt) + t_offset) if wt else None
    # interpolate missing word times inside each line
    for ln in lines:
        ws = [w for w in words if w["line"] == ln["index"]]
        known = [i for i, w in enumerate(ws) if w["start"] is not None]
        if not known:
            continue
        for i, w in enumerate(ws):
            if w["start"] is None:
                prev = max([k for k in known if k < i], default=None)
                nxt = min([k for k in known if k > i], default=None)
                if prev is not None and nxt is not None:
                    w["start"] = ws[prev]["end"]
                    w["end"] = ws[nxt]["start"]
                elif prev is not None:
                    w["start"] = w["end"] = ws[prev]["end"]
                else:
                    w["start"] = w["end"] = ws[nxt]["start"]
    return words, lines


def global_align(tokens, words, lines, logp, frame_dur, blank, free_start=False, free_end=False):
    if not tokens:
        raise ValueError("No lyric character exists in the alignment model's vocabulary")
    labels = np.array([t["id"] for t in tokens], dtype=np.int64)
    st, en, sc = ctc_force_align(logp, labels, blank, free_start, free_end)
    for t, a, b, s in zip(tokens, st, en, sc):
        t["start"], t["end"], t["score"] = (float(a * frame_dur), float(b * frame_dur), float(s)) if a >= 0 else (None, None, -99.0)
    return _assign_times(tokens, words, lines, frame_dur)


# ---------------------------------------------------------------------------- WhisperX refinement
def whisperx_refine(model, meta, audio16k, tokens, words, lines, device, language="ja", pad=0.4, max_shift=1.0,
                    log=print):
    """Re-align each line with whisperx.align inside its coarse window. Returns number of refined lines."""
    import whisperx
    dur = len(audio16k) / SAMPLE_RATE
    refined = 0
    for ln in lines:
        if ln.get("start") is None:
            continue
        ltoks = [t for t in tokens if t["line"] == ln["index"] and not t.get("is_sep")]
        if not ltoks or any(t.get("start") is None for t in ltoks):
            continue  # partly outside the processed section: keep the global alignment
        text = "".join(t["char"] for t in ltoks) if language == "ja" else ln["text"]
        seg = [{"start": max(0.0, ln["start"] - pad), "end": min(dur, ln["end"] + pad), "text": text}]
        try:
            res = whisperx.align(seg, model, meta, audio16k, device, return_char_alignments=False)
        except Exception as e:  # keep the global result
            log(f"  line {ln['index']}: whisperx.align failed ({e!r}); keeping the global alignment")
            continue
        wlist = [w for s in res.get("segments", []) for w in s.get("words", [])]
        backup = [(t["start"], t["end"]) for t in ltoks]
        if language == "ja":
            chars = [w for w in wlist if w.get("word", "").strip()]
            if len(chars) != len(ltoks) or any(c["word"].strip() != t["char"] for c, t in zip(chars, ltoks)):
                log(f"  line {ln['index']}: whisperx output does not match the tokens; keeping the global alignment")
                continue
            ok = [c for c in chars if c.get("start") is not None]
            if len(ok) < max(1, len(chars) // 2):
                continue
            for c, t in zip(chars, ltoks):
                if c.get("start") is not None:
                    t["start"], t["end"] = float(c["start"]), float(c["end"])
        else:
            lw = [w for w in words if w["line"] == ln["index"]]
            if len(wlist) != len(lw):
                log(f"  line {ln['index']}: word count mismatch; keeping the global alignment")
                continue
            for w, r in zip(lw, wlist):
                if r.get("start") is not None:
                    for t in ltoks:
                        if t["c0"] >= w["char_start"] and t["c1"] <= w["char_end"]:
                            t["start"], t["end"] = float(r["start"]), float(r["end"])
        # plausibility: keep the global result if the refined line collapsed or jumped away
        new = [(t["start"], t["end"]) for t in ltoks]
        old_span = max(e for _, e in backup) - min(s_ for s_, _ in backup)
        new_span = max(e for _, e in new) - min(s_ for s_, _ in new)
        shift = abs(min(s_ for s_, _ in new) - min(s_ for s_, _ in backup))
        if new_span < 0.3 * old_span or shift > max_shift:
            for t, (s0, e0) in zip(ltoks, backup):
                t["start"], t["end"] = s0, e0
            log(f"  line {ln['index']}: whisperx result implausible (span {new_span:.2f}s vs {old_span:.2f}s, "
                f"shift {shift:.2f}s); keeping the global alignment")
            continue
        refined += 1
    _assign_times(tokens, words, lines, None)
    return refined


# ---------------------------------------------------------------------------- manual corrections
def apply_overrides(lines, words, line_overrides=None, global_offset=0.0):
    """line_overrides: {line_index: (start, end)}; words inside a line are rescaled linearly."""
    line_overrides = line_overrides or {}
    for ln in lines:
        if ln.get("start") is None:
            continue
        s0, e0 = ln["start"], ln["end"]
        s1, e1 = line_overrides.get(ln["index"], (s0, e0))
        s1, e1 = s1 + global_offset, e1 + global_offset
        scale = (e1 - s1) / max(e0 - s0, 1e-6)
        for w in words:
            if w["line"] == ln["index"] and w.get("start") is not None:
                w["start"] = s1 + (w["start"] - s0) * scale
                w["end"] = s1 + (w["end"] - s0) * scale
        ln["start"], ln["end"] = s1, e1
        ln["manual"] = ln["index"] in line_overrides
    return lines, words


def align_lyrics(vocals_path, lyrics_text, language="ja", device="cuda", refine=True, free_start=False,
                 free_end=False, log=print):
    """Full step 2. Returns (lines, words, info).

    free_start / free_end: set when only a section of the song is processed, so that lyric lines sung
    before / after that section stay unaligned instead of being squeezed into it.
    """
    import gc

    import librosa
    import torch
    lines = parse_lyrics(lyrics_text)
    audio, _ = librosa.load(vocals_path, sr=SAMPLE_RATE, mono=True)
    model, meta = load_align_model(language, device)
    tokens, words = build_tokens(lines, meta["dictionary"], language)
    missing = [ln["index"] for ln in lines if not any(t["line"] == ln["index"] for t in tokens)]
    if missing:
        log(f"[warn] lines without alignable characters: {missing}")
    logp, frame_dur = compute_emissions(model, meta, audio, device)
    global_align(tokens, words, lines, logp, frame_dur, blank_id(meta), free_start, free_end)
    outside = [ln["index"] for ln in lines if ln.get("start") is None]
    if outside:
        log(f"lines not found in this section of the song (left unaligned): {outside}")
    n_ref = 0
    if refine:
        try:
            n_ref = whisperx_refine(model, meta, audio, tokens, words, lines, device, language, log=log)
        except ImportError as e:
            log(f"[warn] whisperx is not importable ({e!r}); using the global alignment only")
    info = {"tokens": len(tokens), "frames": int(logp.shape[0]), "frame_dur": frame_dur, "refined_lines": n_ref,
            "unaligned_lines": outside,
            "loader": meta.get("loader"), "model_type": meta.get("type")}
    del model, logp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lines, words, info
