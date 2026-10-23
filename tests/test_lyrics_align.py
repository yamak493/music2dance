import numpy as np

from m2d import lyrics_align as la


def _dict():
    chars = set("きみのえがおすだよるぞらにかやくほしずっとはろせい笑顔好夜空星") | set("abcdefghijklmnopqrstuvwxyz")
    d = {"<pad>": 0, "|": 1}
    for c in sorted(chars):
        d[c] = len(d)
    return d


def test_parse_sections():
    lines = la.parse_lyrics("[Verse]\n君の笑顔\n\n【サビ】\nHello 世界!\n")
    assert [l["text"] for l in lines] == ["君の笑顔", "Hello 世界!"]
    assert [l["section"] for l in lines] == ["Verse", "サビ"]


def test_kana_fallback_and_viterbi():
    lines = la.parse_lyrics("君の笑顔が 好きだよ\n夜空に輝く星、ずっと")
    dic = _dict()
    toks, words = la.build_tokens(lines, dic, "ja")
    assert "".join(t["char"] for t in toks).startswith("く")  # 君 -> reading (only 'く' is in the toy vocab)
    rng = np.random.default_rng(0)
    T, V = 500, len(dic)
    logp = np.full((T, V), -8.0)
    logp[:, 0] = -0.05
    true, t = [], 20
    for k, tk in enumerate(toks):
        if k and toks[k - 1]["line"] != tk["line"]:
            t += 60
        logp[t, :] = -8.0
        logp[t, tk["id"]] = -0.05
        true.append(t)
        t += int(rng.integers(3, 8))
    st, en, _ = la.ctc_force_align(logp, np.array([t["id"] for t in toks]), 0)
    assert np.array_equal(st, true)
    la.global_align(toks, words, lines, logp, 0.02, 0)
    assert lines[1]["start"] > lines[0]["end"]
    assert all(w["start"] is not None for w in words)


def test_overrides_rescale_words():
    lines = [{"index": 0, "text": "a b", "start": 1.0, "end": 2.0}]
    words = [{"line": 0, "text": "a", "start": 1.0, "end": 1.5}, {"line": 0, "text": "b", "start": 1.5, "end": 2.0}]
    la.apply_overrides(lines, words, {0: (10.0, 12.0)}, 0.5)
    assert words[1]["start"] == 11.5 and lines[0]["end"] == 12.5


def test_free_end_leaves_unsung_lines_unaligned():
    """Only part of the song is processed: later lines must not be squeezed into the audio."""
    lines = la.parse_lyrics("君の笑顔が\n夜空に星\nずっとはろ")
    dic = _dict()
    toks, words = la.build_tokens(lines, dic, "ja")
    sung = [t for t in toks if t["line"] < 2]          # the audio ends after line 1
    T, V = 120, len(dic)
    logp = np.full((T, V), -8.0)
    logp[:, 0] = -0.05
    for k, tk in enumerate(sung):
        logp[10 + 5 * k, :] = -8.0
        logp[10 + 5 * k, tk["id"]] = -0.05
    la.global_align(toks, words, lines, logp, 0.02, 0, free_end=True)
    assert lines[0]["start"] is not None and lines[1]["start"] is not None
    assert lines[2]["start"] is None and all(w["start"] is None for w in words if w["line"] == 2)
    assert abs(lines[1]["end"] - (10 + 5 * (len(sung) - 1) + 1) * 0.02) < 1e-6
