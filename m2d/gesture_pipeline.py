"""Step 7: choose the best MotionGPT3 candidate, find its key pose, cut it around the key and convert it."""
import json
import os

import numpy as np
import torch

from . import motion_ik as ik
from . import smpl_utils as su

EXAGGERATE_JOINTS_DEFAULT = [3, 6, 9, 13, 14, 16, 17, 18, 19]


def make_requests(gestures, n_candidates=4, seconds=4.0):
    return [{"id": int(g["id"]), "text": g["motion_description"], "n": int(n_candidates), "seconds": float(seconds)}
            for g in gestures]


def write_requests(gestures, path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_requests(gestures, **kw), f, ensure_ascii=False, indent=1)


def load_candidates(out_dir, gid):
    d = np.load(os.path.join(out_dir, f"gesture_{gid}.npz"))
    return d["joints"], int(d["fps"])


def score_candidate(P30, scope, pre, post):
    """Best key pose of one candidate (positions at 30 fps, SMPL scale)."""
    cands, speed = ik.detect_keyframes(P30, scope)
    T = len(P30)
    best = None
    for c in cands[:5]:
        k = c["frame"]
        context_ok = (k - pre >= 0) and (k + post <= T - 1)
        s = c["score"] * (1.0 if context_ok else 0.7)
        # gestures limited to the upper body should not walk away
        if scope == "upper":
            travel = np.linalg.norm(P30[-1, 0, [0, 2]] - P30[0, 0, [0, 2]])
            s *= 1.0 / (1.0 + 2.0 * max(0.0, travel - 0.3))
        if best is None or s > best["score"]:
            best = dict(c, score=float(s), context_ok=bool(context_ok))
    return best, speed


def process_gesture(g, joints20, beat_period, pre_beats=1.0, post_beats=1.0, exaggeration=1.15,
                    exaggerate_joints=None, ik_iters=300, device="cpu", fps=su.FPS):
    """g: script gesture (needs id, scope, hit_time). joints20: (n, L, 22, 3) MotionGPT3 candidates."""
    pre = int(round(pre_beats * beat_period * fps))
    post = int(round(post_beats * beat_period * fps))
    per_cand = []
    for ci, J in enumerate(joints20):
        P, mirrored = ik.fix_chirality(J)
        P30 = ik.resample(P) * ik.body_scale(ik.resample(P))
        best, speed = score_candidate(P30, g["scope"], pre, post)
        per_cand.append({"candidate": ci, "key": best["frame"], "score": best["score"], "context_ok": best["context_ok"],
                         "speed": speed, "mirrored": mirrored})
    chosen = max(per_cand, key=lambda c: c["score"])
    conv = ik.hml_to_smpl(joints20[chosen["candidate"]], refine=True, iters=ik_iters, device=device)
    R = conv["rotmat"]
    joints = list(exaggerate_joints if exaggerate_joints is not None else EXAGGERATE_JOINTS_DEFAULT)
    R = ik.exaggerate(R, joints, exaggeration, ref_frame=0)
    key = chosen["key"]
    rot_clip = torch.as_tensor(ik.cut_clip(R.numpy(), key, pre, post))
    root_clip = torch.as_tensor(ik.cut_clip(conv["root_pos"].numpy(), key, pre, post))
    return {
        "id": int(g["id"]), "scope": g["scope"], "hit_frame": int(round(g["hit_time"] * fps)),
        "rotmat": rot_clip.float(), "root_pos": root_clip.float(), "key": pre,
        "ref_height": float(conv["root_pos"][0, 1]),  # standing pelvis height (y-up == z-up after rotation)
        "info": {"candidates": [{k: v for k, v in c.items() if k != "speed"} for c in per_cand],
                 "chosen": chosen["candidate"], "key_frame_in_candidate": key, "pre": pre, "post": post,
                 "mirrored": conv["mirrored"], "scale": conv["scale"],
                 "ik_bone_angle_deg_mean": float(conv["errors"]["bone_angle_deg"].mean()),
                 "ik_bone_angle_deg_max": float(conv["errors"]["bone_angle_deg"].max()),
                 "ik_pos_err_cm_mean": float(conv["errors"]["pos_err_m"].mean() * 100),
                 "ik_pos_err_cm_max": float(conv["errors"]["pos_err_m"].max() * 100)},
        "debug": {"conv": conv, "speeds": [c["speed"] for c in per_cand], "rotmat_full": R},
    }
