import json

import numpy as np

from m2d import audio_analysis as aa
from m2d import choreo_llm as cl


def test_grid_pickup_and_irregular_bar():
    beats = list(np.arange(0.5, 20.0, 0.5))
    downbeats = beats[1::4]
    g = aa.build_grid(beats, downbeats, 20.0, line_starts=[1.0, 5.0], phase="auto")
    assert g["pickup_beats"] == 1 and g["eights"][0]["start"] == 1.0
    assert g["eights"][0]["counts"]["1&"] == 1.25 and g["eights"][0]["n_beats"] == 8
    # a 3-beat bar starts a new eight early
    b2 = beats[:10] + [b + 0.0 for b in beats[11:]]
    d2 = [b2[1], b2[5], b2[8]] + [b for b in b2[12::4]]
    g2 = aa.build_grid(b2, d2, 20.0, phase=0)
    assert any(e["irregular"] for e in g2["eights"])


def test_script_validation_drops_bad_and_close_gestures():
    beats = list(np.arange(0.5, 40.0, 0.5))
    grid = aa.build_grid(beats, beats[0::4], 40.0, phase=0)
    obj = {"gestures": [
        {"line_index": 0, "lyric": "a", "motion_description": "raises both arms", "eight": 1, "start_count": "1",
         "hit_count": "3", "end_count": "4", "scope": "upper", "priority": 1},
        {"line_index": 0, "lyric": "b", "motion_description": "a person jumps", "eight": 1, "start_count": "5",
         "hit_count": "5&", "end_count": "6", "scope": "full", "priority": 2},
        {"line_index": 1, "lyric": "c", "motion_description": "a person points forward", "eight": 3,
         "start_count": "2", "hit_count": "1", "end_count": "3", "scope": "weird", "priority": 1},
        {"line_index": 1, "lyric": "d", "motion_description": "x", "eight": 99, "start_count": "1",
         "hit_count": "1", "end_count": "1", "scope": "upper", "priority": 3}]}
    gestures, warns = cl.validate_script(obj, grid, [], max_gestures=10, min_gap_beats=8, count_time=aa.count_time)
    assert [g["lyric"] for g in gestures] == ["a", "c"]
    assert gestures[0]["motion_description"] == "a person raises both arms"
    assert gestures[1]["scope"] == "upper" and gestures[1]["start_count"] == "1"
    assert any("eight 99" in w for w in warns)


def test_extract_json_from_markdown():
    assert cl.extract_json('ok ```json\n{"a": {"b": 1}}\n``` bye') == {"a": {"b": 1}}
    assert json.dumps(cl.extract_json('text {"x": [1, 2]} tail')) == '{"x": [1, 2]}'
