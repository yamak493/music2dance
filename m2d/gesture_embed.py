"""Step 8/9: put gesture clips on the beat inside the base dance and build the masks for re-generation.

Per frame / channel, the mask passed to ``FlowerDance.inpaint`` means
  1 = fixed (gesture channels inside a gesture, or the base dance elsewhere), 0 = re-generated.
Transitions of ``trans_pre`` / ``trans_post`` frames before/after every gesture are re-generated
(preparation + recovery). ``decay`` selects FlowerDance's time-decayed soft mask (True) or a strict
mask (False) per element; window borders are always strict so that the re-generated windows can be
pasted back into the base dance seamlessly.
"""
import numpy as np
import torch

from . import smpl_utils as su


def _region_channels(scope, joints):
    ch = su.joint_channels(joints)
    if scope == "full":
        ch = ch + list(range(4, 7)) + su.joint_channels([0])
    return sorted(set(ch))


def place_root_full(base_rot0, base_pos, clip_rot0_y, clip_pos_y, key_idx, ref_height, mode="hybrid", floor_z=None):
    """Root rotation/position (z-up) for a `full` gesture.

    mode 'base'  : root position and orientation of the base dance (the plan's literal rule).
    mode 'hybrid': standing position (x, y) and facing (yaw) follow the base dance; the gesture adds
                   its own height change (jump / crouch), its tilt and its yaw change relative to the
                   key frame (turns). Keeps the dancer where the base dance is, without losing the motion.
    """
    if mode == "base":
        return base_rot0.clone(), base_pos.clone()
    g_rot, g_pos = su.yup_to_zup(clip_rot0_y, clip_pos_y)
    yaw_g, tilt_g = su.split_yaw_tilt_zup(g_rot)
    yaw_g = su.unwrap(yaw_g)
    yaw_b, _ = su.split_yaw_tilt_zup(base_rot0)
    yaw_b = su.unwrap(yaw_b)
    yaw = yaw_b + (yaw_g - yaw_g[key_idx])
    rot0 = su.rot_z(yaw) @ tilt_g
    pos = base_pos.clone()
    pos[:, 2] = base_pos[:, 2] + (g_pos[:, 2] - ref_height)
    return rot0, pos


def assemble(base, gestures, trans_frames, upper_joints=None, full_joints=None, full_root_mode="hybrid",
             gesture_mask_mode="soft", base_mask_mode="soft", transition_regen="replaced", min_gap_frames=0,
             log=print):
    """Build the motion with gestures pasted in, plus mask/decay tensors.

    base: dict root_pos (T,3), rotmat (T,24,3,3), contacts (T,4) -- z-up, un-normalised.
    gestures: list of dicts with keys
        id, scope ('upper'|'full'), hit_frame, rotmat (L,24,3,3) y-up local, root_pos (L,3) y-up (SMPL scale),
        key (index of the key frame inside the clip), ref_height (standing pelvis height of the clip, z-up)
        and optional trans_pre / trans_post (frames).
    """
    upper_joints = list(upper_joints or su.UPPER_JOINTS_DEFAULT)
    full_joints = list(full_joints or su.FULL_JOINTS_DEFAULT)
    T = base["rotmat"].shape[0]
    rot = base["rotmat"].clone()
    pos = base["root_pos"].clone()
    cont = base["contacts"].clone()
    mask = torch.ones(T, su.MOTION_DIM)
    changed = torch.zeros(T, su.MOTION_DIM, dtype=torch.bool)  # channels whose values come from a gesture
    decay = torch.full((T, su.MOTION_DIM), base_mask_mode == "soft", dtype=torch.bool)
    occupied = np.zeros(T, dtype=bool)
    floor_z = su.foot_floor_level(su.fk(base["rotmat"], base["root_pos"]))
    regions = []
    for g in sorted(gestures, key=lambda d: d["hit_frame"]):
        L = g["rotmat"].shape[0]
        a = int(g["hit_frame"] - g["key"])
        b = a + L
        lo, hi = max(0, a), min(T, b)
        if hi - lo < 2:
            log(f"[skip] gesture {g['id']}: outside the song (frames {a}-{b})")
            continue
        guard_lo, guard_hi = max(0, lo - min_gap_frames), min(T, hi + min_gap_frames)
        if occupied[guard_lo:guard_hi].any():
            log(f"[skip] gesture {g['id']}: overlaps an earlier gesture (frames {lo}-{hi})")
            continue
        ci = torch.arange(lo - a, hi - a)
        scope = g["scope"]
        joints = full_joints if scope == "full" else upper_joints
        rot[lo:hi][:, joints] = g["rotmat"][ci][:, joints]
        if scope == "full":
            r0, p0 = place_root_full(base["rotmat"][lo:hi, 0], base["root_pos"][lo:hi], g["rotmat"][ci, 0],
                                     g["root_pos"][ci], int(g["key"] - (lo - a)), g["ref_height"], full_root_mode)
            rot[lo:hi, 0] = r0
            pos[lo:hi] = p0
            # keep the feet above the floor of the base dance
            X = su.fk(rot[lo:hi], pos[lo:hi])
            low = X[:, list(su.FOOT_JOINTS), 2].min(-1).values
            lift = torch.clamp(floor_z - low, min=0.0)
            pos[lo:hi, 2] += lift
            cont[lo:hi] = su.compute_contacts(su.fk(rot[lo:hi], pos[lo:hi]))
        ch = _region_channels(scope, joints)
        mask[lo:hi, ch] = 1.0
        changed[lo:hi, ch] = True
        decay[lo:hi, ch] = gesture_mask_mode == "soft"
        if scope == "full":
            mask[lo:hi, 0:4] = 0.0  # foot contacts are re-generated for whole-body gestures
        occupied[lo:hi] = True
        tp = int(g.get("trans_pre", trans_frames))
        tq = int(g.get("trans_post", trans_frames))
        regions.append({"id": g["id"], "scope": scope, "start": lo, "end": hi, "hit": int(g["hit_frame"]),
                        "pre": (max(0, lo - tp), lo), "post": (hi, min(T, hi + tq)), "channels": ch})
    # transitions (never over a gesture region)
    all_ch = list(range(su.MOTION_DIM))
    for r in regions:
        regen = all_ch if (transition_regen == "all" or r["scope"] == "full") else r["channels"]
        for s, e in (r["pre"], r["post"]):
            if e <= s:
                continue
            frames = np.arange(s, e)
            frames = frames[~occupied[s:e]]
            if len(frames):
                idx = torch.as_tensor(frames)
                mask[idx[:, None], torch.as_tensor(regen)[None, :]] = 0.0
    return {"root_pos": pos, "rotmat": rot, "contacts": cont, "changed": changed}, mask, decay, regions


def edited_vector(fd, base_norm, assembled):
    """Normalised vector for masked generation: the base dance's own values everywhere, except the
    channels replaced by gestures (so the untouched parts are bit-identical to the base generation)."""
    enc = fd.encode(assembled["root_pos"], assembled["rotmat"], assembled["contacts"])
    ch = assembled["changed"]
    clipped = (enc.abs() >= 0.999) & ch  # values at the edge of the training range (clipped by the normaliser)
    frac = float(clipped.sum()) / max(1, int(ch.sum()))
    return torch.where(ch, enc, base_norm), frac


def edit_windows(regions, T, context_frames, min_len, edge_lock):
    """Merge gesture neighbourhoods into windows for masked generation. Returns [(s, e)]."""
    spans = []
    for r in regions:
        s = r["pre"][0] - context_frames
        e = r["post"][1] + context_frames
        spans.append([max(0, s), min(T, e)])
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s < min_len:
            c = (s + e) // 2
            s, e = max(0, c - min_len // 2), min(T, c - min_len // 2 + min_len)
            s = max(0, e - min_len)
        out.append((int(s), int(e)))
    # merging may be needed again after enlarging
    out.sort()
    final = []
    for s, e in out:
        if final and s <= final[-1][1]:
            final[-1] = (final[-1][0], max(final[-1][1], e))
        else:
            final.append((s, e))
    return final


def lock_window_edges(mask, decay, windows, edge_lock, T):
    """Strictly fix `edge_lock` frames at each inner window border (they hold the base dance)."""
    mask = mask.clone()
    decay = decay.clone()
    for s, e in windows:
        if s > 0:
            mask[s:s + edge_lock] = 1.0
            decay[s:s + edge_lock] = False
        if e < T:
            mask[e - edge_lock:e] = 1.0
            decay[e - edge_lock:e] = False
    return mask, decay


def run_edit(fd, base_norm, edited_norm, mask, decay, windows, cond, genre, n_steps=101, seed=0, log=print):
    """Masked generation on every window; everything outside the windows stays the base dance."""
    out = base_norm.clone()
    for k, (s, e) in enumerate(windows):
        log(f"  window {k + 1}/{len(windows)}: frames {s}-{e} ({(e - s) / su.FPS:.1f}s), "
            f"fixed {float(mask[s:e].mean()) * 100:.0f}% of channels")
        out[s:e] = fd.inpaint(edited_norm[s:e], mask[s:e], cond[s:e], genre, decay=decay[s:e], n_steps=n_steps,
                              seed=seed + k)
    return out
