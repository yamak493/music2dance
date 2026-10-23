"""Step 10/11: exports (SMPL .pkl, joints .npz, BVH) and preview videos (skeleton / mannequin)."""
import math
import os
import pickle
import subprocess

import numpy as np
import torch

from . import smpl_utils as su

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# ---------------------------------------------------------------------------- exports
def export_smpl_pkl(motion, path, frame="zup"):
    """motion: dict root_pos (T,3), rotmat (T,24,3,3) in FlowerDance's z-up frame.

    frame='zup': identical layout to FlowerDance's inference output
                 {smpl_poses (T,72) axis-angle, smpl_trans (T,3), full_pose (T,24,3)}.
    frame='yup': rotated back to the SMPL/AIST++ y-up frame, AIST++-style keys
                 {smpl_poses, smpl_trans, smpl_scaling=[1.0]} (+ full_pose in y-up).
    """
    rot = motion["rotmat"].float().clone()
    pos = motion["root_pos"].float().clone()
    if frame == "yup":
        rot[:, 0], pos = su.zup_to_yup(rot[:, 0], pos)
    joints = su.fk(rot, pos)
    data = {"smpl_poses": su.rotmat_to_aa(rot).reshape(-1, 72).numpy().astype(np.float32),
            "smpl_trans": pos.numpy().astype(np.float32), "full_pose": joints.numpy().astype(np.float32)}
    if frame == "yup":
        data["smpl_scaling"] = np.array([1.0], dtype=np.float32)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return path


def export_bvh(motion, path, fps=su.FPS, scale=100.0):
    """BVH (y-up, centimetres, ZXY Euler) of the SMPL skeleton for Blender/MotionBuilder."""
    from scipy.spatial.transform import Rotation
    rot = motion["rotmat"].float().clone()
    pos = motion["root_pos"].float().clone()
    rot[:, 0], pos = su.zup_to_yup(rot[:, 0], pos)
    T = rot.shape[0]
    children = {j: su.children_of(j) for j in range(su.N_JOINTS)}
    lines = ["HIERARCHY"]

    def write(j, depth):
        ind = "  " * depth
        off = su.SMPL_OFFSETS[j] * scale
        if j == 0:
            lines.append(f"ROOT {su.JOINT_NAMES[j]}")
        else:
            lines.append(f"{ind}JOINT {su.JOINT_NAMES[j]}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
        if j == 0:
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{ind}  CHANNELS 3 Zrotation Xrotation Yrotation")
        if children[j]:
            for c in children[j]:
                write(c, depth + 1)
        else:
            lines.append(f"{ind}  End Site")
            lines.append(f"{ind}  {{")
            lines.append(f"{ind}    OFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")

    write(0, 0)
    order = []

    def dfs(j):
        order.append(j)
        for c in children[j]:
            dfs(c)

    dfs(0)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]
    eul = {j: Rotation.from_matrix(rot[:, j].numpy()).as_euler("ZXY", degrees=True) for j in order}
    p = pos.numpy() * scale
    for t in range(T):
        vals = [f"{v:.4f}" for v in p[t]]
        for j in order:
            vals += [f"{v:.4f}" for v in eul[j][t]]
        lines.append(" ".join(vals))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------- camera / overlays
class Camera:
    def __init__(self, positions, root_rot, width=640, height=640, fov_deg=40.0, distance=4.2, elevation_deg=12.0,
                 azimuth_offset_deg=0.0, follow=True, smooth_s=1.5, fps=su.FPS, floor_z=None):
        pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
        fwd = root_rot[:, :, 2].detach().cpu().numpy() if torch.is_tensor(root_rot) else root_rot[:, :, 2]
        self.W, self.H = width, height
        self.f = 0.5 * height / math.tan(math.radians(fov_deg) / 2)
        self.floor_z = float(floor_z if floor_z is not None else np.percentile(pos[:, list(su.FOOT_JOINTS), 2].min(-1), 5))
        az = math.atan2(fwd[:, 1].mean(), fwd[:, 0].mean()) + math.radians(azimuth_offset_deg)
        self.view_dir = np.array([math.cos(az), math.sin(az), 0.0])  # camera sits in front of the dancer
        root = pos[:, 0].copy()
        if follow:
            k = max(1, int(smooth_s * fps))
            ker = np.ones(k) / k
            for d in range(2):
                root[:, d] = np.convolve(np.pad(root[:, d], (k // 2, k - 1 - k // 2), mode="edge"), ker, mode="valid")
        else:
            root[:, :2] = root[:, :2].mean(0)
        self.target = np.stack([root[:, 0], root[:, 1], np.full(len(root), self.floor_z + 0.95)], -1)
        el = math.radians(elevation_deg)
        self.offset = distance * np.array([math.cos(el) * self.view_dir[0], math.cos(el) * self.view_dir[1], math.sin(el)])

    def pose(self, t):
        eye = self.target[t] + self.offset
        z = self.target[t] - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, np.array([0.0, 0.0, 1.0]))
        x /= np.linalg.norm(x)
        y = np.cross(x, z)  # up in image
        return eye, np.stack([x, y, z])  # rows: right, up, forward

    def project(self, pts, t):
        eye, R = self.pose(t)
        c = (np.asarray(pts) - eye) @ R.T
        zc = np.maximum(c[..., 2], 1e-3)
        u = self.W / 2 + self.f * c[..., 0] / zc
        v = self.H / 2 - self.f * c[..., 1] / zc
        return np.stack([u, v], -1), c[..., 2]


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def make_overlay_fn(grid=None, lines=None, regions=None, fps=su.FPS, start_frame=0):
    eights = grid["eights"] if grid else []
    regs = regions or []

    def info(frame):
        t = (frame + start_frame) / fps
        out = [f"{t:6.2f}s"]
        for e in eights:
            if e["start"] <= t < e["end"]:
                cnt = "-"
                for name, ct in sorted(e["counts"].items(), key=lambda kv: kv[1]):
                    if ct <= t + 1e-6:
                        cnt = name
                out[0] += f"   8-count E{e['index']}  [{cnt}]"
                break
        lyr = ""
        for ln in lines or []:
            if ln.get("start") is not None and ln["start"] - 0.2 <= t <= ln["end"] + 0.2:
                lyr = ln["text"]
        if lyr:
            out.append(lyr)
        for r in regs:
            if r["start"] <= frame + start_frame < r["end"]:
                hit = " *HIT*" if abs(frame + start_frame - r["hit"]) <= 2 else ""
                out.append(f"gesture #{r['id']} ({r['scope']}){hit}: {r.get('label', '')}")
        return out

    return info


def _draw_text(img, texts, font_path, size=20):
    from PIL import Image, ImageDraw, ImageFont
    if not texts:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    y = 8
    for s in texts:
        dr.text((10, y), s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += size + 6
    return np.asarray(im)


class FFmpegWriter:
    def __init__(self, path, width, height, fps, audio=None, audio_start=0.0):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-ss", f"{audio_start:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k",
                    "-shortest"]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast", path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


# ---------------------------------------------------------------------------- skeleton video
BONE_COLORS = {"left": (70, 140, 255), "right": (255, 90, 90), "center": (230, 230, 230)}


def _bone_color(j):
    if j in su.LEFT_JOINTS:
        return BONE_COLORS["left"]
    if j in su.RIGHT_JOINTS:
        return BONE_COLORS["right"]
    return BONE_COLORS["center"]


def draw_skeleton_frame(cam, t, positions_list, labels=None, floor_extent=3.0):
    import cv2
    W, H = cam.W, cam.H
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = np.linspace(40, 15, H, dtype=np.uint8)[:, None, None]
    c = cam.target[t]
    g = np.arange(-floor_extent, floor_extent + 1e-6, 0.5)
    for k in g:
        for a, b in (((c[0] + k, c[1] - floor_extent), (c[0] + k, c[1] + floor_extent)),
                     ((c[0] - floor_extent, c[1] + k), (c[0] + floor_extent, c[1] + k))):
            pts = np.array([[a[0], a[1], cam.floor_z], [b[0], b[1], cam.floor_z]])
            uv, z = cam.project(pts, t)
            if (z > 0.05).all():
                cv2.line(img, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), (70, 70, 70), 1,
                         cv2.LINE_AA)
    for pi, P in enumerate(positions_list):
        uv, z = cam.project(P[t], t)
        shift = np.zeros(2)
        order = sorted(range(1, su.N_JOINTS), key=lambda j: -z[j])
        for j in order:
            p = su.SMPL_PARENTS[j]
            col = _bone_color(j)
            if pi > 0:
                col = tuple(int(0.55 * x) for x in col)
            cv2.line(img, tuple(np.round(uv[p] + shift).astype(int)), tuple(np.round(uv[j] + shift).astype(int)), col,
                     4 if pi == 0 else 2, cv2.LINE_AA)
        for j in range(su.N_JOINTS):
            cv2.circle(img, tuple(np.round(uv[j]).astype(int)), 4 if j != 15 else 9, (250, 250, 250) if pi == 0 else (140, 140, 140),
                       -1 if j != 15 else 2, cv2.LINE_AA)
    return img


def render_skeleton_video(out_path, positions, root_rot, audio=None, start_frame=0, n_frames=None, overlay=None,
                          ghost_positions=None, width=640, height=640, fps=su.FPS, font_path=None, progress=None):
    """positions (T,24,3) z-up; ghost_positions (optional, e.g. the base dance) is drawn dimmed behind."""
    pos = positions.detach().cpu().numpy() if torch.is_tensor(positions) else np.asarray(positions)
    T = pos.shape[0]
    n = T - start_frame if n_frames is None else min(n_frames, T - start_frame)
    sl = slice(start_frame, start_frame + n)
    plist = [pos[sl]]
    if ghost_positions is not None:
        gp = ghost_positions.detach().cpu().numpy() if torch.is_tensor(ghost_positions) else np.asarray(ghost_positions)
        plist.append(gp[sl])
    cam = Camera(pos[sl], root_rot[sl], width, height, fps=fps)
    font_path = font_path or find_font()
    wr = FFmpegWriter(out_path, width, height, fps, audio, start_frame / fps)
    try:
        for t in range(n):
            img = draw_skeleton_frame(cam, t, plist)
            if overlay is not None:
                img = _draw_text(img, overlay(t), font_path)
            wr.write(img)
            if progress and t % 300 == 0:
                progress(t, n)
    finally:
        wr.close()
    return out_path


# ---------------------------------------------------------------------------- mannequin (GPU ray marching)
# (joint a, joint b, radius a, radius b, material) ; material 0 = body, 1 = joint accent, 2 = head
MANNEQUIN_CONES = [
    (0, 3, 0.115, 0.105, 0), (1, 2, 0.095, 0.095, 0), (3, 6, 0.105, 0.11, 0), (6, 9, 0.115, 0.125, 0),
    (13, 14, 0.075, 0.075, 0), (9, 12, 0.07, 0.05, 0), (12, 15, 0.048, 0.045, 0),
    (9, 16, 0.075, 0.06, 0), (9, 17, 0.075, 0.06, 0),
    (16, 18, 0.052, 0.041, 0), (17, 19, 0.052, 0.041, 0), (18, 20, 0.04, 0.031, 0), (19, 21, 0.04, 0.031, 0),
    (20, 22, 0.03, 0.034, 0), (21, 23, 0.03, 0.034, 0),
    (1, 4, 0.085, 0.058, 0), (2, 5, 0.085, 0.058, 0), (4, 7, 0.054, 0.04, 0), (5, 8, 0.054, 0.04, 0),
    (7, 10, 0.042, 0.03, 0), (8, 11, 0.042, 0.03, 0),
]
MANNEQUIN_SPHERES = [(16, 0.058, 1), (17, 0.058, 1), (18, 0.046, 1), (19, 0.046, 1), (20, 0.034, 1), (21, 0.034, 1),
                     (4, 0.062, 1), (5, 0.062, 1), (7, 0.045, 1), (8, 0.045, 1)]
MATERIAL_COLORS = torch.tensor([[0.86, 0.72, 0.56], [0.55, 0.42, 0.32], [0.88, 0.75, 0.6]])


def _round_cone(p, a, b, r1, r2):
    """Signed distance of points p (N,1,3) to round cones (1,M,...) -- Inigo Quilez's sdRoundCone."""
    ba = b - a
    l2 = (ba * ba).sum(-1).clamp_min(1e-10)
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = (pa * ba).sum(-1)
    z = y - l2
    q = pa * l2.unsqueeze(-1) - ba * y.unsqueeze(-1)
    x2 = (q * q).sum(-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = torch.sign(rr) * rr * rr * x2
    d1 = torch.sqrt(x2 + z2) * il2 - r2
    d2 = torch.sqrt(x2 + y2) * il2 - r1
    d3 = (torch.sqrt((x2 * a2 * il2).clamp_min(0)) + y * rr) * il2 - r1
    return torch.where(torch.sign(z) * a2 * z2 > k, d1, torch.where(torch.sign(y) * a2 * y2 < k, d2, d3))


class MannequinRenderer:
    def __init__(self, width=640, height=640, device="cuda", steps=64, shadow_steps=28, smooth_k=0.012):
        self.W, self.H, self.dev = width, height, torch.device(device)
        self.steps, self.shadow_steps, self.k = steps, shadow_steps, smooth_k
        self.light = torch.tensor([0.45, -0.35, 0.82], device=self.dev)
        self.light = self.light / self.light.norm()

    def _prims(self, P, G):
        """P: (24,3) positions, G: (24,3,3) global rotations (z-up). Returns cone/sphere tensors."""
        dev = self.dev
        ia = torch.tensor([c[0] for c in MANNEQUIN_CONES], device=dev)
        ib = torch.tensor([c[1] for c in MANNEQUIN_CONES], device=dev)
        A, B = P[ia], P[ib]
        r1 = torch.tensor([c[2] for c in MANNEQUIN_CONES], device=dev)
        r2 = torch.tensor([c[3] for c in MANNEQUIN_CONES], device=dev)
        cm = torch.tensor([c[4] for c in MANNEQUIN_CONES], device=dev)
        sc = [P[s[0]] for s in MANNEQUIN_SPHERES]
        sr = [s[1] for s in MANNEQUIN_SPHERES]
        sm = [s[2] for s in MANNEQUIN_SPHERES]
        head = P[15] + G[15] @ torch.tensor([0.0, 0.085, 0.02], device=dev)  # SMPL local y = up
        nose = head + G[15] @ torch.tensor([0.0, -0.005, 0.1], device=dev)   # local +z = face direction
        sc += [head, nose]
        sr += [0.105, 0.028]
        sm += [2, 1]
        return (A, B, r1, r2, cm, torch.stack(sc), torch.tensor(sr, device=dev), torch.tensor(sm, device=dev))

    def _sdf(self, p, prims, need_mat=False):
        A, B, r1, r2, cm, S, sr, sm = prims
        dc = _round_cone(p[:, None, :], A[None], B[None], r1[None], r2[None])
        ds = (p[:, None, :] - S[None]).norm(dim=-1) - sr[None]
        d = torch.cat([dc, ds], 1)
        k = self.k
        smin = -k * torch.logsumexp(-d / k, dim=1)
        if need_mat:
            mats = torch.cat([cm, sm])[d.argmin(1)]
            return smin, mats
        return smin

    @torch.no_grad()
    def render(self, P, G, eye, R, f, floor_z):
        """P (24,3), G (24,3,3) torch on device; eye (3,), R (3,3) rows = right, up, forward."""
        dev, H, W = self.dev, self.H, self.W
        eye = torch.as_tensor(eye, dtype=torch.float32, device=dev)
        R = torch.as_tensor(R, dtype=torch.float32, device=dev)
        vs, us = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        dcam = torch.stack([(us + 0.5 - W / 2) / f, -(vs + 0.5 - H / 2) / f, torch.ones_like(us)], -1)
        d = torch.nn.functional.normalize(dcam @ R, dim=-1).reshape(-1, 3)
        N = d.shape[0]
        img = torch.zeros(N, 3, device=dev)
        sky = torch.linspace(0.30, 0.12, H, device=dev).repeat_interleave(W)
        img[:] = torch.stack([sky * 0.9, sky * 0.95, sky * 1.15], -1)
        prims = self._prims(P, G)
        center = P.mean(0)
        radius = float((P - center).norm(dim=-1).max()) + 0.2
        # floor
        t_floor = torch.where(d[:, 2] < -1e-4, (floor_z - eye[2]) / d[:, 2], torch.full_like(d[:, 2], float("inf")))
        # body: sphere tracing inside the bounding sphere
        oc = eye - center
        bq = (d * oc).sum(-1)
        cq = (oc * oc).sum() - radius * radius
        disc = bq * bq - cq
        active = torch.nonzero(disc > 0).squeeze(-1)
        t_body = torch.full((N,), float("inf"), device=dev)
        normal = torch.zeros(N, 3, device=dev)
        mat = torch.zeros(N, dtype=torch.long, device=dev)
        if active.numel():
            sq = torch.sqrt(disc[active])
            t = (-bq[active] - sq).clamp_min(0.0)
            tmax = torch.minimum(-bq[active] + sq, t_floor[active])
            da = d[active]
            hit = torch.zeros_like(t, dtype=torch.bool)
            alive = torch.ones_like(hit)
            for _ in range(self.steps):
                idx = torch.nonzero(alive).squeeze(-1)
                if idx.numel() == 0:
                    break
                p = eye + da[idx] * t[idx, None]
                dist = self._sdf(p, prims)
                t[idx] += dist
                h = dist < 1e-3
                hit[idx[h]] = True
                alive[idx[h | (t[idx] > tmax[idx])]] = False
            hidx = torch.nonzero(hit).squeeze(-1)
            if hidx.numel():
                gi = active[hidx]
                p = eye + d[gi] * t[hidx, None]
                e = 1e-3
                ks = torch.tensor([[1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, 1, 1]], dtype=torch.float32, device=dev)
                n = sum(k * self._sdf(p + e * k, prims)[:, None] for k in ks)
                normal[gi] = torch.nn.functional.normalize(n, dim=-1)
                _, m = self._sdf(p, prims, need_mat=True)
                mat[gi] = m
                t_body[gi] = t[hidx]
        body = torch.isfinite(t_body) & (t_body <= t_floor)
        floor = torch.isfinite(t_floor) & ~body
        # floor shading with checkerboard + soft shadow
        fidx = torch.nonzero(floor).squeeze(-1)
        if fidx.numel():
            fp = eye + d[fidx] * t_floor[fidx, None]
            chk = ((torch.floor(fp[:, 0] / 0.5) + torch.floor(fp[:, 1] / 0.5)) % 2 == 0).float()
            base = 0.42 + 0.12 * chk
            shadow = torch.ones_like(base)
            near = torch.nonzero((fp[:, :2] - center[:2]).norm(dim=-1) < radius + 1.8).squeeze(-1)
            if near.numel():
                sp = fp[near] + self.light * 0.02
                tt = torch.full((near.numel(),), 0.02, device=dev)
                res = torch.ones_like(tt)
                for _ in range(self.shadow_steps):
                    dist = self._sdf(sp + self.light * tt[:, None], prims)
                    res = torch.minimum(res, (10.0 * dist / tt).clamp(0, 1))
                    tt = tt + dist.clamp(0.01, 0.25)
                shadow[near] = 0.35 + 0.65 * res
            dist_cam = (fp[:, :2] - center[:2]).norm(dim=-1)
            fade = torch.exp(-(dist_cam / 6.0) ** 2)
            col = (base * shadow)[:, None] * torch.tensor([1.0, 1.0, 1.03], device=dev)
            img[fidx] = col * fade[:, None] + img[fidx] * (1 - fade[:, None])
        bidx = torch.nonzero(body).squeeze(-1)
        if bidx.numel():
            n = normal[bidx]
            alb = MATERIAL_COLORS.to(dev)[mat[bidx]]
            diff = (n @ self.light).clamp_min(0)
            hemi = 0.5 + 0.5 * n[:, 2]
            v = -d[bidx]
            hvec = torch.nn.functional.normalize(v + self.light, dim=-1)
            spec = ((n * hvec).sum(-1).clamp_min(0) ** 32) * 0.25
            rim = (1 - (n * v).sum(-1).clamp(0, 1)) ** 3 * 0.25
            col = alb * (0.25 * hemi + 0.85 * diff)[:, None] + (spec + rim)[:, None]
            img[bidx] = col
        img = img.clamp(0, 1) ** (1 / 1.1)
        return (img.reshape(H, W, 3) * 255).byte().cpu().numpy()


def render_mannequin_video(out_path, motion, audio=None, start_frame=0, n_frames=None, overlay=None, width=640,
                           height=640, fps=su.FPS, device="cuda", font_path=None, progress=None, steps=64):
    rot = motion["rotmat"].float()
    pos = motion["root_pos"].float()
    T = rot.shape[0]
    n = T - start_frame if n_frames is None else min(n_frames, T - start_frame)
    sl = slice(start_frame, start_frame + n)
    P_all, G_all = su.fk(rot[sl], pos[sl], return_global=True)
    cam = Camera(P_all, G_all[:, 0], width, height, fps=fps)
    floor_z = cam.floor_z - 0.035  # feet are joint centres; lower the floor to the sole
    rend = MannequinRenderer(width, height, device, steps=steps)
    font_path = font_path or find_font()
    wr = FFmpegWriter(out_path, width, height, fps, audio, start_frame / fps)
    try:
        for t in range(n):
            eye, R = cam.pose(t)
            img = rend.render(P_all[t].to(rend.dev), G_all[t].to(rend.dev), eye, R, cam.f, floor_z)
            if overlay is not None:
                img = _draw_text(img, overlay(t), font_path)
            wr.write(img)
            if progress and t % 150 == 0:
                progress(t, n)
    finally:
        wr.close()
    return out_path
