"""FlowerDance inference wrapper: model loading, music features, whole/chunked sampling and masked editing.

The sampler and the editing loop reproduce FlowerDance's own code
(``model/flowmatching.py``: ``euler_sample`` and ``inpaint``) on top of ``DanceDecoder.infer_pred``;
the music features are computed with the functions of FlowerDance's ``inference.py``.

Masked generation (``inpaint``):
  * ``mask``  (T, 151) in [0, 1]: 1 = keep the given motion, 0 = generate.
  * ``decay`` (T, 151) bool: True  -> FlowerDance's time-decayed soft mask (weight = mask * t),
                             False -> strict mask (weight = mask, never released).
"""
import importlib
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import smpl_utils as su

GENRES = ("Dai", "ShenYun", "Wei", "Korean", "Urban", "Hiphop", "Popping", "Miao", "HanTang",
          "Breaking", "Kun", "Locking", "Jazz", "Choreography", "Chinese", "DunHuang")
HF_REPO = "xlt99/FlowerDance"
HF_FILE = "train-3700.pt"
TRAIN_HORIZON = 1200  # 40 s at 30 fps: sequence length used by the released model configuration


def patch_transformers_for_mamba():
    """mamba_ssm.utils.generation imports names that newer transformers releases removed."""
    try:
        import transformers.generation as tg
    except Exception:
        return
    aliases = {
        "GreedySearchDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
        "SampleDecoderOnlyOutput": "GenerateDecoderOnlyOutput",
    }
    for old, new in aliases.items():
        try:
            getattr(tg, old)
        except Exception:
            try:
                setattr(tg, old, getattr(tg, new))
            except Exception:
                pass


def genre_id(name):
    if isinstance(name, int):
        return name
    for i, g in enumerate(GENRES):
        if g.casefold() == str(name).strip().casefold():
            return i
    raise ValueError(f"Unknown FlowerDance genre {name!r}; choose one of {GENRES}")


class FlowerDance:
    def __init__(self, repo_dir, ckpt_path, device="cuda", mamba_backend="cuda", load_model=True):
        """mamba_backend: 'cuda' (mamba_ssm kernels) or 'torch' (pure-PyTorch fallback).
        load_model=False only loads the data normaliser (encode/decode without using the GPU)."""
        self.repo_dir = os.path.abspath(repo_dir)
        self.device = torch.device(device)
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        su.check_against_flowerdance(self.repo_dir)
        if not load_model:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.normalizer = ckpt["normalizer"]
            self.model = None
            self.mamba_backend = None
            return
        if mamba_backend == "torch":
            from . import mamba_fallback
            mamba_fallback.install()
        else:
            patch_transformers_for_mamba()
        for mod in ("model.model", "model"):
            sys.modules.pop(mod, None)
        DanceDecoder = importlib.import_module("model.model").DanceDecoder
        self.mamba_backend = mamba_backend
        # Same hyper-parameters as FlowerDance/EDGE.py
        self.model = DanceDecoder(nfeats=su.MOTION_DIM, seq_len=TRAIN_HORIZON, latent_dim=512, ff_size=1024,
                                  num_layers=8, num_heads=8, dropout=0.1, cond_feature_dim=35, activation=F.gelu)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt or "normalizer" not in ckpt:
            raise KeyError("FlowerDance checkpoint must contain 'model_state_dict' and 'normalizer'")
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.normalizer = ckpt["normalizer"]
        self.model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------ normalisation
    def normalize(self, vec):
        x = vec.detach().float().cpu().clone()
        squeeze = x.dim() == 2
        if squeeze:
            x = x[None]
        x = self.normalizer.normalize(x)  # MinMaxScaler to [-1, 1] with clipping (in-place on the clone)
        return x[0] if squeeze else x

    def unnormalize(self, xn):
        x = xn.detach().float().cpu().clone()
        squeeze = x.dim() == 2
        if squeeze:
            x = x[None]
        x = self.normalizer.unnormalize(x)
        return x[0] if squeeze else x

    def decode(self, xn):
        """normalised (T,151) -> dict(root_pos, rotmat, contacts, positions) un-normalised, z-up."""
        m = su.decode(self.unnormalize(xn))
        m["positions"] = su.fk(m["rotmat"], m["root_pos"])
        return m

    def encode(self, root_pos, rotmat, contacts=None):
        return self.normalize(su.encode(root_pos.float().cpu(), rotmat.float().cpu(),
                                        None if contacts is None else contacts.float().cpu()))

    # ------------------------------------------------------------------ features
    def music_features(self, audio_path):
        """35-dim normalised features for the whole song, exactly as FlowerDance's inference.py."""
        inf = importlib.import_module("inference")
        import librosa
        if not hasattr(librosa.beat, "tempo"):  # removed in librosa 1.0 (FlowerDance pins 0.11)
            from librosa.feature.rhythm import tempo as _tempo
            librosa.beat.tempo = _tempo
        duration_total = librosa.get_duration(path=audio_path)
        frames = int(math.floor(duration_total * inf.FPS)) - 1
        frames = max(frames, 1)
        duration = frames / inf.FPS
        audio, tempo_audio, tempo_sr, frame_count = inf.load_audio_clip(audio_path, start=0.0, duration=duration)
        start_bpm = inf.estimate_tempo(tempo_audio, tempo_sr)
        raw = inf.extract_baseline_features(audio, frame_count, start_bpm)
        feats = inf.normalize_features(raw)
        return {"features": feats, "frames": frame_count, "fps": inf.FPS, "bpm_librosa": start_bpm}

    # ------------------------------------------------------------------ sampling
    def _pred(self, x, cond, genre, t, h):
        return self.model.infer_pred(x, cond, genre, t, h)

    @torch.no_grad()
    def sample(self, cond, genre, n_steps=21, seed=None, noise=None):
        """Plain generation (FlowerDance euler_sample). cond: (T,35) np/tensor. Returns normalised (T,151) cpu."""
        cond_t = torch.as_tensor(cond, dtype=torch.float32, device=self.device)[None]
        g = torch.tensor([genre_id(genre)], device=self.device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) if seed is not None else int(time.time()))
        x = noise.to(self.device)[None] if noise is not None else torch.randn(
            (1, cond_t.shape[1], su.MOTION_DIM), device=self.device, generator=gen)
        t_values = torch.linspace(1, 0, n_steps, device=self.device)
        dt = 1.0 / (n_steps - 1)
        for i in range(n_steps - 1):
            t = torch.full((1,), float(t_values[i]), device=self.device)
            h = torch.full((1,), float(t_values[i] - t_values[i + 1]), device=self.device)
            v = self._pred(x, cond_t, g, t, h)
            x = x - v * dt
        return x[0].float().cpu()

    @torch.no_grad()
    def inpaint(self, x_known, mask, cond, genre, decay=None, n_steps=101, seed=None):
        """Masked generation (FlowerDance ``FlowMatching.inpaint`` with per-element soft/strict mask).

        x_known: normalised (T,151); mask: (T,151) in [0,1]; decay: (T,151) bool (True = soft, time-decayed).
        """
        dev = self.device
        x = torch.as_tensor(x_known, dtype=torch.float32, device=dev)[None]
        m = torch.as_tensor(mask, dtype=torch.float32, device=dev)[None]
        if decay is None:
            decay = torch.ones_like(m, dtype=torch.bool)
        dcy = torch.as_tensor(decay, dtype=torch.bool, device=dev).reshape(m.shape)
        cond_t = torch.as_tensor(cond, dtype=torch.float32, device=dev)[None]
        g = torch.tensor([genre_id(genre)], device=dev)
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) if seed is not None else int(time.time()))
        e = torch.randn(x.shape, device=dev, generator=gen)
        x_t = e.clone()
        t_values = torch.linspace(1.0, 0.0, n_steps, device=dev)
        for i in range(n_steps - 1):
            t_cur, t_next = t_values[i], t_values[i + 1]
            tb = torch.full((1,), float(t_cur), device=dev)
            hb = torch.full((1,), float(t_cur - t_next), device=dev)
            v = self._pred(x_t, cond_t, g, tb, hb)
            x_t = x_t + v * (t_next - t_cur)
            w = torch.where(dcy, m * t_next, m)
            x_t = w * ((1 - t_next) * x + t_next * e) + (1 - w) * x_t
        return x_t[0].float().cpu()

    def sample_long(self, cond, genre, mode="auto", n_steps=21, seed=0, window=960, overlap=150,
                    inpaint_steps=51, progress=print):
        """Generate a whole song.

        mode='whole'   : one pass over all frames (Mamba has no fixed horizon).
        mode='chunked' : windows of `window` frames; every window after the first is generated by masked
                         generation whose first `overlap` frames are strictly fixed to the previous window.
        mode='auto'    : 'whole' if the song fits in the training horizon (40 s), else 'chunked'.
        """
        T = len(cond)
        if mode == "auto":
            mode = "whole" if T <= TRAIN_HORIZON else "chunked"
        info = {"mode": mode, "frames": T, "windows": []}
        t0 = time.time()
        if mode == "whole" or T <= window:
            out = self.sample(cond, genre, n_steps=n_steps, seed=seed)
            info["windows"].append((0, T))
        elif mode == "chunked":
            if overlap >= window:
                raise ValueError("overlap must be smaller than window")
            out = torch.zeros(T, su.MOTION_DIM)
            out[:window] = self.sample(cond[:window], genre, n_steps=n_steps, seed=seed)
            info["windows"].append((0, window))
            filled = window
            k = 1
            while filled < T:
                e_ = min(T, filled - overlap + window)
                s_ = max(0, e_ - window)  # keep a full-length window at the end of the song
                known = torch.zeros(e_ - s_, su.MOTION_DIM)
                known[: filled - s_] = out[s_:filled]
                mask = torch.zeros_like(known)
                mask[: filled - s_] = 1.0
                seg = self.inpaint(known, mask, cond[s_:e_], genre, decay=torch.zeros_like(mask, dtype=torch.bool),
                                   n_steps=inpaint_steps, seed=seed + 1000 * k)
                out[filled:e_] = seg[filled - s_:]
                info["windows"].append((s_, e_))
                progress(f"  window {k}: frames {s_}-{e_} (fixed {filled - s_} frames of overlap)")
                filled = e_
                k += 1
        else:
            raise ValueError(f"unknown mode {mode}")
        info["seconds"] = time.time() - t0
        return out, info

    def free(self):
        if self.model is not None:
            self.model.cpu()
        self.model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def motion_stats(positions, fps=30, window_s=5.0):
    """Per-window mean joint speed (m/s) and foot-skating, to judge long-sequence quality."""
    pos = positions.detach().cpu()
    vel = (pos[1:] - pos[:-1]).norm(dim=-1) * fps  # (T-1, 24)
    speed = vel.mean(-1).numpy()
    win = int(window_s * fps)
    n = max(1, len(speed) // win)
    per_win = [float(speed[i * win:(i + 1) * win].mean()) for i in range(n)]
    contacts = su.compute_contacts(pos)
    feet = pos[:, list(su.FOOT_JOINTS)]
    fv = torch.zeros(feet.shape[:2])
    fv[:-1] = (feet[1:, :, :2] - feet[:-1, :, :2]).norm(dim=-1)
    skate = float((fv * contacts).sum() / contacts.sum().clamp_min(1)) * fps
    up = (pos[:, 12] - pos[:, 0]).mean(0)
    up_cos = float(up[2] / up.norm().clamp_min(1e-9))  # body axis vs +z: should be close to 1 in FlowerDance's frame
    return {"speed_per_window": per_win, "mean_speed": float(speed.mean()), "foot_skate_mps": skate,
            "root_travel_m": float((pos[1:, 0, :2] - pos[:-1, 0, :2]).norm(dim=-1).sum()), "up_axis_cos": up_cos}
