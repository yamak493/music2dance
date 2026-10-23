import pickle

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix

from m2d import export_render as er
from m2d import smpl_utils as su


def test_pkl_frames_and_bvh(tmp_path):
    T = 10
    R = axis_angle_to_matrix(torch.randn(T, 24, 3) * 0.2)
    root_y = torch.tensor([0.0, 0.9, 0.0]).repeat(T, 1)
    R[:, 0], pos = su.yup_to_zup(R[:, 0], root_y)
    motion = {"rotmat": R, "root_pos": pos}
    er.export_smpl_pkl(motion, tmp_path / "z.pkl", "zup")
    er.export_smpl_pkl(motion, tmp_path / "y.pkl", "yup")
    z = pickle.load(open(tmp_path / "z.pkl", "rb"))
    y = pickle.load(open(tmp_path / "y.pkl", "rb"))
    assert z["smpl_poses"].shape == (T, 72) and z["full_pose"].shape == (T, 24, 3)
    assert np.allclose(y["smpl_trans"], root_y.numpy(), atol=1e-6)            # back to y-up
    assert np.allclose(z["full_pose"][:, :, 2], y["full_pose"][:, :, 1], atol=1e-5)  # z-up height == y-up height
    er.export_bvh(motion, tmp_path / "m.bvh")
    txt = open(tmp_path / "m.bvh").read()
    assert txt.startswith("HIERARCHY") and f"Frames: {T}" in txt


def test_encode_decode_roundtrip():
    T = 8
    R = axis_angle_to_matrix(torch.randn(T, 24, 3) * 0.5)
    pos = torch.randn(T, 3)
    m = su.decode(su.encode(pos, R))
    assert torch.allclose(m["rotmat"], R, atol=1e-5) and torch.allclose(m["root_pos"], pos)
