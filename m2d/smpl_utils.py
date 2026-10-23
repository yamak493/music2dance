"""SMPL skeleton helpers shared by every stage.

Conventions
-----------
* Joint order / parents / rest offsets are exactly FlowerDance's ``vis.py`` (SMPL 24 joints).
  ``check_against_flowerdance`` verifies this against the cloned repository at runtime.
* Rotation matrices act on column vectors. 6D rotations follow PyTorch3D
  (``matrix_to_rotation_6d`` = first two *rows* of the matrix).
* "y-up" is the native SMPL / HumanML3D frame (x = body left, y = up, z = forward).
  "z-up" is FlowerDance's model frame: FlowerDance rotates the data by +90 deg about X
  ((x, y, z) -> (x, -z, y)) before training, so the root rotation is left-multiplied by
  ``R_ZUP_FROM_YUP`` and the root position is rotated by it. Local joint rotations are unchanged.
* FlowerDance motion vector (151 dims): [foot contacts (4) | root position (3) | 24 x 6D rotation].
"""
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_rotation_6d, rotation_6d_to_matrix)

FPS = 30
N_JOINTS = 24
MOTION_DIM = 151
CONTACT_SLICE = slice(0, 4)
ROOT_POS_SLICE = slice(4, 7)
ROT_OFFSET = 7

JOINT_NAMES = [
    "pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle",
    "r_ankle", "spine3", "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hand", "r_hand",
]

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

# Copied verbatim from FlowerDance/vis.py (smpl_offsets), y-up SMPL rest pose, metres.
SMPL_OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
], dtype=np.float32)

FOOT_JOINTS = (7, 8, 10, 11)  # FlowerDance contact order: l_ankle, r_ankle, l_foot, r_foot
CONTACT_VEL_THRESHOLD = 0.01  # metres / frame, as in FlowerDance's dataset preprocessing

# Joints whose rotations are replaced for an `upper` gesture (wrists/hands/head keep the base dance:
# joint positions carry no information about them).
UPPER_JOINTS_DEFAULT = [3, 6, 9, 12, 13, 14, 16, 17, 18, 19]
# Joints whose rotations are replaced for a `full` gesture (root handled separately).
FULL_JOINTS_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19]

BONES = [(SMPL_PARENTS[j], j) for j in range(1, N_JOINTS)]
LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}

R_ZUP_FROM_YUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def children_of(j, limit=N_JOINTS):
    return [c for c in range(limit) if SMPL_PARENTS[c] == j]


def check_against_flowerdance(repo_dir):
    """Raise if the skeleton constants differ from the cloned FlowerDance repository."""
    import ast
    import os
    src = open(os.path.join(repo_dir, "vis.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("smpl_parents", "smpl_offsets"):
                found[node.targets[0].id] = ast.literal_eval(node.value)
    if found.get("smpl_parents") != SMPL_PARENTS:
        raise RuntimeError("smpl_parents in FlowerDance/vis.py differ from m2d.smpl_utils")
    if not np.allclose(np.asarray(found.get("smpl_offsets"), dtype=np.float32), SMPL_OFFSETS, atol=1e-7):
        raise RuntimeError("smpl_offsets in FlowerDance/vis.py differ from m2d.smpl_utils")
    return True


def _t(x, device=None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device if device is not None else x.device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def offsets_tensor(like):
    return torch.as_tensor(SMPL_OFFSETS, dtype=like.dtype, device=like.device)


def fk(rotmat, root_pos, return_global=False):
    """Forward kinematics identical to FlowerDance's SMPLSkeleton (matrix form).

    rotmat: (..., 24, 3, 3) local rotations (joint 0 = global root orientation)
    root_pos: (..., 3)
    returns positions (..., 24, 3) [and global rotations (..., 24, 3, 3)]
    """
    off = offsets_tensor(rotmat)
    pos = [None] * N_JOINTS
    glob = [None] * N_JOINTS
    for j in range(N_JOINTS):
        p = SMPL_PARENTS[j]
        if p < 0:
            pos[j] = root_pos
            glob[j] = rotmat[..., 0, :, :]
        else:
            pos[j] = pos[p] + (glob[p] @ off[j].unsqueeze(-1)).squeeze(-1)
            glob[j] = glob[p] @ rotmat[..., j, :, :]
    pos = torch.stack(pos, dim=-2)
    if return_global:
        return pos, torch.stack(glob, dim=-3)
    return pos


def compute_contacts(positions):
    """FlowerDance contact labels: foot speed < 1 cm/frame (last frame copies the previous rule: v=0)."""
    feet = positions[..., FOOT_JOINTS, :]
    v = torch.zeros(feet.shape[:-1], dtype=positions.dtype, device=positions.device)
    v[..., :-1, :] = (feet[..., 1:, :, :] - feet[..., :-1, :, :]).norm(dim=-1)
    return (v < CONTACT_VEL_THRESHOLD).to(positions.dtype)


def encode(root_pos, rotmat, contacts=None):
    """(T,3), (T,24,3,3), (T,4) -> (T,151) un-normalised FlowerDance vector."""
    if contacts is None:
        contacts = compute_contacts(fk(rotmat, root_pos))
    r6 = matrix_to_rotation_6d(rotmat).reshape(*rotmat.shape[:-3], N_JOINTS * 6)
    return torch.cat([contacts, root_pos, r6], dim=-1)


def decode(vec):
    """(..., 151) -> dict(root_pos, rotmat, contacts)."""
    contacts = vec[..., CONTACT_SLICE]
    root_pos = vec[..., ROOT_POS_SLICE]
    r6 = vec[..., ROT_OFFSET:].reshape(*vec.shape[:-1], N_JOINTS, 6)
    return {"root_pos": root_pos, "rotmat": rotation_6d_to_matrix(r6), "contacts": contacts}


def joint_channels(joints):
    """Channel indices of the 6D rotation of the given joints inside the 151-dim vector."""
    out = []
    for j in joints:
        out += list(range(ROT_OFFSET + 6 * j, ROT_OFFSET + 6 * j + 6))
    return out


def rotmat_to_aa(rotmat):
    return matrix_to_axis_angle(rotmat)


def aa_to_rotmat(aa):
    return axis_angle_to_matrix(aa)


def yup_to_zup(root_rotmat, root_pos):
    R = torch.as_tensor(R_ZUP_FROM_YUP, dtype=root_rotmat.dtype, device=root_rotmat.device)
    return R @ root_rotmat, root_pos @ R.T


def zup_to_yup(root_rotmat, root_pos):
    R = torch.as_tensor(R_ZUP_FROM_YUP, dtype=root_rotmat.dtype, device=root_rotmat.device)
    return R.T @ root_rotmat, root_pos @ R


def rot_z(angle):
    """Rotation about +z (z-up world). angle: tensor (...)."""
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def heading_zup(root_rotmat):
    """Yaw of the body's forward axis (local +z) projected on the ground plane of the z-up world."""
    fwd = root_rotmat[..., :, 2]
    return torch.atan2(fwd[..., 1], fwd[..., 0])


def split_yaw_tilt_zup(root_rotmat):
    """root = Rz(yaw) @ tilt, where tilt keeps the body's forward axis in the x-z plane rotated by yaw=0."""
    yaw = heading_zup(root_rotmat)
    tilt = rot_z(-yaw) @ root_rotmat
    return yaw, tilt


def unwrap(angle):
    a = angle.detach().cpu().numpy()
    return torch.as_tensor(np.unwrap(a, axis=0), dtype=angle.dtype, device=angle.device)


def minimal_rotation(a, b, eps=1e-8):
    """Smallest rotation matrices R with R a_hat = b_hat. a, b: (..., 3)."""
    a = F.normalize(a, dim=-1, eps=eps)
    b = F.normalize(b, dim=-1, eps=eps)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1).clamp(-1.0, 1.0)
    s = v.norm(dim=-1)
    # Antiparallel case: pick an axis orthogonal to a.
    helper = torch.where((a[..., :1].abs() < 0.9), torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a),
                         torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype).expand_as(a))
    ortho = F.normalize(torch.cross(a, helper, dim=-1), dim=-1, eps=eps)
    axis = torch.where((s > 1e-6).unsqueeze(-1), v / s.clamp_min(1e-12).unsqueeze(-1), ortho)
    angle = torch.atan2(s, c)
    return axis_angle_to_matrix(axis * angle.unsqueeze(-1))


def kabsch(src, dst, weights=None):
    """Rotation R minimising sum_i w_i |R src_i - dst_i|^2. src, dst: (..., N, 3)."""
    if weights is None:
        weights = torch.ones(src.shape[:-1], dtype=src.dtype, device=src.device)
    H = (src * weights.unsqueeze(-1)).transpose(-1, -2) @ dst  # (...,3,3) = sum w src dst^T
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.det(V @ U.transpose(-1, -2)))
    d = torch.where(d == 0, torch.ones_like(d), d)
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return V @ D @ U.transpose(-1, -2)


def slerp_rotmat(R0, R1, w):
    """Geodesic interpolation between rotation matrices with weight w in [0,1] (broadcastable)."""
    rel = R0.transpose(-1, -2) @ R1
    aa = matrix_to_axis_angle(rel)
    return R0 @ axis_angle_to_matrix(aa * w)


def foot_floor_level(positions, q=5.0):
    """Robust floor height (z-up) estimated from the lowest foot joint per frame."""
    low = positions[..., list(FOOT_JOINTS), 2].min(dim=-1).values
    return float(np.percentile(low.detach().cpu().numpy(), q))
