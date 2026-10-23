import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from m2d import motion_ik as ik
from m2d import smpl_utils as su


def _synthetic_motion(T=60):
    t = torch.linspace(0, 3, T)

    def osc(a, f, ph=0.0):
        return a * torch.sin(2 * np.pi * f * t + ph)

    aa = torch.zeros(T, 24, 3)
    aa[:, 0] = torch.stack([osc(0.2, 0.5), osc(1.0, 0.2), osc(0.1, 0.7)], -1)
    aa[:, 3, 0] = osc(0.3, 0.5)
    aa[:, 16] = torch.stack([osc(0.4, 0.6), osc(0.5, 0.3, 1), -1.0 + osc(0.6, 0.5)], -1)
    aa[:, 17] = torch.stack([osc(0.3, 0.4), osc(-0.4, 0.6), 1.0 + osc(0.5, 0.7)], -1)
    aa[:, 18, 1] = -(0.8 + osc(0.6, 0.8))   # elbows: pure hinge, anatomical sign
    aa[:, 19, 1] = 0.9 + osc(0.5, 0.6)
    aa[:, 1] = torch.stack([-0.3 + osc(0.4, 0.5), osc(0.2, 0.3), osc(0.1, 0.3)], -1)
    aa[:, 4, 0] = 0.6 + osc(0.4, 0.5)      # knees: pure hinge
    aa[:, 5, 0] = 0.5 + osc(0.3, 0.7)
    root = torch.stack([osc(0.3, 0.2), 0.9 + osc(0.05, 1.0), osc(0.2, 0.3)], -1)
    return axis_angle_to_matrix(aa), root


def test_ik_round_trip_with_other_proportions():
    R_gt, root = _synthetic_motion()
    rng = np.random.default_rng(0)
    orig = su.SMPL_OFFSETS.copy()
    try:
        su.SMPL_OFFSETS[:] = orig * rng.uniform(0.85, 1.15, (24, 1)).astype(np.float32)  # "HumanML3D" body
        P = su.fk(R_gt, root * 1.1)[:, :22].numpy()
    finally:
        su.SMPL_OFFSETS[:] = orig
    res = ik.hml_to_smpl(P, refine=True, iters=300)
    assert not res["mirrored"]
    assert res["errors"]["bone_angle_deg"].mean() < 1.0
    assert res["errors"]["pos_err_m"].max() < 0.02
    R = res["rotmat"]
    idx30, idx20 = np.arange(0, len(R), 3), np.arange(0, len(range(0, len(R), 3))) * 2
    for j in (4, 5, 18, 19):  # hinge joints are recovered including twist
        rel = R_gt[idx20, j].transpose(-1, -2) @ R[idx30, j]
        assert torch.rad2deg(matrix_to_axis_angle(rel).norm(dim=-1)).mean() < 3.0


def test_chirality_detection():
    R_gt, root = _synthetic_motion()
    P = su.fk(R_gt, root)[:, :22].numpy()
    Pm = P.copy()
    Pm[..., 0] *= -1
    assert ik.chirality_sign(P) > 0 and ik.chirality_sign(Pm) < 0


def test_keyframe_and_cut():
    T = 90
    P = np.zeros((T, 22, 3))
    P[:, :, 1] = 1.0
    x = np.concatenate([np.linspace(0, 0.6, 40), np.full(50, 0.6)])  # wrists move, then hold
    P[:, 20, 0] = x
    P[:, 21, 0] = -x
    cands, _ = ik.detect_keyframes(P, "upper")
    assert abs(cands[0]["frame"] - 40) <= 3
    clip = ik.cut_clip(np.arange(T), 2, 5, 5)
    assert clip.tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
