"""HumanML3D joints -> FlowerDance motion (the hardest conversion of the pipeline).

Input : MotionGPT3 output, 22 joint positions, y-up, 20 fps (HumanML3D skeleton/proportions).
Output: SMPL 24 local rotations + root position at 30 fps on FlowerDance's skeleton (y-up here;
        the z-up conversion happens when the clip is embedded).

Steps
  1. chirality check (left x up must point forward, using the toe direction) and 20 -> 30 fps
     cubic-spline resampling of the joint positions;
  2. analytic IK: Kabsch fit of the pelvis (hips + spine) and of spine3 (neck + collars), minimal
     ("swing") rotations along every single-child bone;
  3. optimisation refinement (Adam on 6D rotations): bone-direction error + anatomical hinge priors
     for elbows/knees (no sideways bending / hyper-extension) + temporal smoothness + small prior to
     the analytic solution. Bone directions are scale-free, so the different body proportions of
     HumanML3D and SMPL do not matter; the root translation is rescaled by the leg-length ratio.
Joints without positional information (head, wrists, hands, toes) get identity local rotations and
are never copied into the dance (see smpl_utils.UPPER_JOINTS_DEFAULT / FULL_JOINTS_DEFAULT).
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_rotation_6d, rotation_6d_to_matrix)

from . import smpl_utils as su

HML_FPS = 20
N22 = 22
OPT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 16, 17, 18, 19]
TARGET_BONES = [(su.SMPL_PARENTS[c], c) for c in range(1, N22)]
KNEES = (4, 5)
ELBOWS = (18, 19)


# ---------------------------------------------------------------------------- preprocessing
def chirality_sign(P):
    """+1 if (left x up) points to the toes (SMPL convention), -1 if the skeleton is mirrored."""
    P = np.asarray(P, dtype=np.float64)
    left = P[:, 1] - P[:, 2]
    up = np.array([0.0, 1.0, 0.0])
    fwd = (P[:, 10] - P[:, 7]) + (P[:, 11] - P[:, 8])
    fwd[:, 1] = 0.0
    s = np.einsum("ti,ti->t", np.cross(left, up), fwd)
    return 1.0 if np.median(s) >= 0 else -1.0


def fix_chirality(P):
    """Mirror x if needed so that the joint labels match SMPL's handedness. Returns (P, mirrored)."""
    if chirality_sign(P) < 0:
        P = np.array(P, copy=True)
        P[..., 0] *= -1.0
        return P, True
    return np.asarray(P), False


def resample(P, src_fps=HML_FPS, dst_fps=su.FPS):
    from scipy.interpolate import CubicSpline
    P = np.asarray(P, dtype=np.float64)
    T = P.shape[0]
    if T < 4:
        idx = np.round(np.linspace(0, T - 1, max(1, int(round(T * dst_fps / src_fps))))).astype(int)
        return P[idx]
    t_src = np.arange(T) / src_fps
    t_dst = np.arange(0.0, t_src[-1] + 1e-9, 1.0 / dst_fps)
    return CubicSpline(t_src, P, axis=0)(t_dst)


def body_scale(P):
    """SMPL leg length / HumanML3D leg length (both legs, median over time)."""
    O = su.SMPL_OFFSETS
    smpl_leg = 0.5 * (np.linalg.norm(O[4]) + np.linalg.norm(O[7]) + np.linalg.norm(O[5]) + np.linalg.norm(O[8]))
    P = np.asarray(P)
    hml = 0.5 * (np.linalg.norm(P[:, 4] - P[:, 1], axis=-1) + np.linalg.norm(P[:, 7] - P[:, 4], axis=-1)
                 + np.linalg.norm(P[:, 5] - P[:, 2], axis=-1) + np.linalg.norm(P[:, 8] - P[:, 5], axis=-1))
    if not np.isfinite(np.median(hml)) or np.median(hml) < 1e-3:
        return 1.0  # degenerate skeleton (e.g. collapsed legs): keep metres as they are
    return float(np.clip(smpl_leg / np.median(hml), 0.5, 2.0))


# ---------------------------------------------------------------------------- analytic IK
def analytic_ik(P):
    """P: (T,22,3) torch y-up -> local rotations (T,24,3,3)."""
    T = P.shape[0]
    dt, dev = P.dtype, P.device
    O = su.offsets_tensor(P)
    R = torch.eye(3, dtype=dt, device=dev).expand(T, su.N_JOINTS, 3, 3).clone()
    G = [None] * su.N_JOINTS
    src = F.normalize(O[[1, 2, 3]], dim=-1).expand(T, 3, 3)
    dst = F.normalize(P[:, [1, 2, 3]] - P[:, [0]], dim=-1)
    w = torch.tensor([1.0, 1.0, 1.5], dtype=dt, device=dev).expand(T, 3)
    G[0] = su.kabsch(src, dst, w)
    R[:, 0] = G[0]
    for j in range(1, su.N_JOINTS):
        Gp = G[su.SMPL_PARENTS[j]]
        ch = su.children_of(j, N22) if j < N22 else []
        if not ch:
            G[j] = Gp
            continue
        if len(ch) == 1:
            c = ch[0]
            a = (Gp @ O[c].unsqueeze(-1)).squeeze(-1)
            b = P[:, c] - P[:, j]
            Gj = su.minimal_rotation(a, b) @ Gp
        else:
            a = F.normalize((Gp.unsqueeze(1) @ O[ch].unsqueeze(-1)).squeeze(-1), dim=-1)
            b = F.normalize(P[:, ch] - P[:, j:j + 1], dim=-1)
            Gj = su.kabsch(a, b) @ Gp
        R[:, j] = Gp.transpose(-1, -2) @ Gj
        G[j] = Gj
    return R


# ---------------------------------------------------------------------------- refinement
def _bone_dirs(X):
    idx_p = [p for p, _ in TARGET_BONES]
    idx_c = [c for _, c in TARGET_BONES]
    return F.normalize(X[:, idx_c] - X[:, idx_p], dim=-1)


def refine_ik(P, R_init, iters=300, lr=0.02, w_dir=1.0, w_hinge=0.05, w_hyper=0.5, w_smooth=0.5, w_prior=0.01,
              verbose=False):
    """Optimise local rotations of OPT_JOINTS (see module docstring). P: (T,22,3), R_init: (T,24,3,3)."""
    T = P.shape[0]
    dev, dt = P.device, P.dtype
    target = _bone_dirs(P)
    r6_init = matrix_to_rotation_6d(R_init[:, OPT_JOINTS]).detach()
    r6 = r6_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([r6], lr=lr)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=dt, device=dev)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=dt, device=dev)
    O = su.offsets_tensor(P)
    shin = {4: F.normalize(O[7], dim=0), 5: F.normalize(O[8], dim=0)}
    fore = {18: F.normalize(O[20], dim=0), 19: F.normalize(O[21], dim=0)}
    pos_in_opt = {j: i for i, j in enumerate(OPT_JOINTS)}
    zero_root = torch.zeros(T, 3, dtype=dt, device=dev)
    history = []
    for it in range(iters):
        Rm = rotation_6d_to_matrix(r6)
        R_full = R_init.detach().clone()
        R_full[:, OPT_JOINTS] = Rm
        X = su.fk(R_full, zero_root)
        l_dir = (1.0 - (_bone_dirs(X) * target).sum(-1)).mean()
        l_hinge = 0.0
        l_hyper = 0.0
        for k in KNEES:
            Rk = Rm[:, pos_in_opt[k]]
            l_hinge = l_hinge + ((Rk @ ex) - ex).pow(2).sum(-1).mean()
            l_hyper = l_hyper + F.relu((Rk @ shin[k])[:, 2] - shin[k][2]).mean()
        for k in ELBOWS:
            Re = Rm[:, pos_in_opt[k]]
            l_hinge = l_hinge + ((Re @ ey) - ey).pow(2).sum(-1).mean()
            l_hyper = l_hyper + F.relu(fore[k][2] - (Re @ fore[k])[:, 2]).mean()
        l_smooth = (r6[2:] - 2 * r6[1:-1] + r6[:-2]).pow(2).sum(-1).mean() if T > 2 else torch.zeros((), device=dev)
        l_prior = (r6 - r6_init).pow(2).sum(-1).mean()
        loss = w_dir * l_dir + w_hinge * l_hinge + w_hyper * l_hyper + w_smooth * l_smooth + w_prior * l_prior
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (it % 50 == 0 or it == iters - 1):
            history.append((it, float(l_dir), float(l_hinge), float(l_hyper), float(l_smooth)))
    with torch.no_grad():
        R_out = R_init.clone()
        R_out[:, OPT_JOINTS] = rotation_6d_to_matrix(r6)
    return R_out, history


# ---------------------------------------------------------------------------- evaluation
def retarget_positions(P, root_pos):
    """Ideal SMPL-proportioned joints: the target bone directions laid out with SMPL bone lengths."""
    O = su.offsets_tensor(P)
    out = torch.zeros(P.shape[0], N22, 3, dtype=P.dtype, device=P.device)
    out[:, 0] = root_pos
    for c in range(1, N22):
        p = su.SMPL_PARENTS[c]
        out[:, c] = out[:, p] + F.normalize(P[:, c] - P[:, p], dim=-1) * O[c].norm()
    return out


def ik_errors(P, R, root_pos):
    """Per-joint errors between the IK result and the (retargeted) target. Returns numpy dict."""
    with torch.no_grad():
        X = su.fk(R, root_pos)[:, :N22]
        Y = retarget_positions(P, root_pos)
        pos_err = (X - Y).norm(dim=-1)  # metres
        cosv = (_bone_dirs(X) * _bone_dirs(P)).sum(-1).clamp(-1, 1)
        ang = torch.rad2deg(torch.acos(cosv))
    return {"pos_err_m": pos_err.cpu().numpy(), "bone_angle_deg": ang.cpu().numpy(),
            "fk_positions": X.cpu().numpy(), "target_positions": Y.cpu().numpy()}


# ---------------------------------------------------------------------------- full conversion
def hml_to_smpl(joints20, refine=True, iters=300, device="cpu"):
    """joints20: (T,22,3) numpy, y-up, 20 fps -> dict with 30 fps SMPL motion (y-up)."""
    P, mirrored = fix_chirality(np.asarray(joints20, dtype=np.float64))
    P30 = resample(P, HML_FPS, su.FPS)
    scale = body_scale(P30)
    Pt = torch.as_tensor(P30, dtype=torch.float32, device=device)
    R0 = analytic_ik(Pt)
    root = Pt[:, 0] * scale
    if refine:
        R, hist = refine_ik(Pt, R0, iters=iters, verbose=True)
    else:
        R, hist = R0, []
    err0 = ik_errors(Pt, R0, root)
    err = ik_errors(Pt, R, root)
    return {"rotmat": R.cpu(), "root_pos": root.cpu(), "target30": P30 * scale, "scale": scale,
            "mirrored": mirrored, "errors_analytic": err0, "errors": err, "history": hist}


# ---------------------------------------------------------------------------- keyframes & editing
def _smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(3 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, (r, r), mode="edge")
    return np.convolve(xp, k, mode="valid")


def detect_keyframes(P30, scope="upper", fps=su.FPS, peak_window_s=0.8, sigma=1.5, edge_s=0.2):
    """'決め' candidates: local minima of end-effector speed that follow a strong movement.

    P30: (T,22,3) y-up positions at 30 fps. Returns list of dicts sorted by score (best first).
    """
    P = np.asarray(P30, dtype=np.float64)
    T = len(P)
    rel = P - P[:, :1]
    if scope == "full":
        eff, w = [20, 21, 18, 19, 7, 8], np.array([1.0, 1.0, 0.5, 0.5, 0.7, 0.7])
    else:
        eff, w = [20, 21, 18, 19], np.array([1.0, 1.0, 0.5, 0.5])
    v = np.linalg.norm(np.diff(rel[:, eff], axis=0), axis=-1) * fps  # (T-1, k)
    s = (v * w).sum(-1) / w.sum()
    if scope == "full":
        s = s + 0.7 * np.linalg.norm(np.diff(P[:, 0], axis=0), axis=-1) * fps
    s = np.concatenate([s[:1], s])
    s = _smooth(s, sigma)
    arm = np.linalg.norm(su.SMPL_OFFSETS[18]) + np.linalg.norm(su.SMPL_OFFSETS[20])
    scale = body_scale(P) if T > 0 else 1.0
    W = max(2, int(peak_window_s * fps))
    lo, hi = int(edge_s * fps), T - 1 - int(edge_s * fps)
    cands = []
    for i in range(max(1, lo), min(T - 1, hi + 1)):
        # local minimum; on a plateau only its first frame counts
        if s[i] < s[i - 1] and s[i] <= s[i + 1]:
            peak = s[max(0, i - W):i + 1].max()
            drop = peak - s[i]
            # the pose is "set" as soon as the deceleration is (almost) finished
            j = i
            while j - 1 >= lo and s[j - 1] <= s[i] + 0.1 * drop and s[j - 1] < peak:
                j -= 1
            i = j
            ext = np.linalg.norm(rel[i, [20, 21]] - rel[i, [9, 9]], axis=-1).mean() * scale / arm
            height = 0.0
            if scope == "full":
                height = abs(P[i, 0, 1] - P[0, 0, 1]) * scale
            score = drop * (0.5 + ext + 2.0 * height)
            cands.append({"frame": i, "score": float(score), "speed": float(s[i]), "peak": float(peak),
                          "extension": float(ext)})
    if not cands:  # no clear stop: take the most extended pose
        ext_all = np.linalg.norm(rel[:, [20, 21]] - rel[:, [9, 9]], axis=-1).mean(-1)
        i = int(np.argmax(ext_all[lo:hi + 1]) + lo) if hi > lo else T // 2
        cands.append({"frame": i, "score": 0.0, "speed": float(s[i]), "peak": float(s[i]),
                      "extension": float(ext_all[i] * scale / arm)})
    cands.sort(key=lambda c: -c["score"])
    return cands, s


def cut_clip(arr, key, pre, post):
    """Frames [key-pre, key+post] of arr (time on axis 0); edges are padded by repeating frames."""
    T = arr.shape[0]
    idx = np.clip(np.arange(key - pre, key + post + 1), 0, T - 1)
    return arr[idx]


def exaggerate(R, joints, factor, ref_frame=0, max_angle=np.pi * 0.95):
    """Scale each joint's rotation *relative to a reference frame* (the neutral start of the clip)."""
    if abs(factor - 1.0) < 1e-6:
        return R
    R = R.clone()
    ref = R[ref_frame:ref_frame + 1, joints]
    rel = ref.transpose(-1, -2) @ R[:, joints]
    aa = matrix_to_axis_angle(rel) * factor
    ang = aa.norm(dim=-1, keepdim=True)
    aa = torch.where(ang > max_angle, aa / ang.clamp_min(1e-8) * max_angle, aa)
    R[:, joints] = ref @ axis_angle_to_matrix(aa)
    return R
