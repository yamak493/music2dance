"""Plots and diagnostics shown in the notebook."""
import numpy as np
import torch

from . import smpl_utils as su


def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


# ---------------------------------------------------------------------------- audio / alignment
def rms_envelope(path, sr=16000, hop=0.02):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    h = int(hop * sr)
    rms = librosa.feature.rms(y=y, frame_length=2 * h, hop_length=h)[0]
    return np.arange(len(rms)) * hop, rms


def plot_alignment(vocals_path, lines, words=None, figsize=(16, 3.2)):
    plt = _plt()
    t, rms = rms_envelope(vocals_path)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, rms / (rms.max() + 1e-9), color="0.6", lw=0.6, label="vocal energy")
    for ln in lines:
        if ln.get("start") is None:
            continue
        bad = ln.get("score", 0) < -2.5
        ax.axvspan(ln["start"], ln["end"], color="tab:red" if bad else "tab:blue", alpha=0.25)
        ax.text(ln["start"], 1.02, str(ln["index"]), fontsize=8, color="tab:red" if bad else "tab:blue")
    if words:
        for w in words:
            if w.get("start") is not None:
                ax.axvline(w["start"], color="tab:green", lw=0.4, alpha=0.6)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("seconds")
    ax.set_title("lyric lines on the vocal track (number = line index, red = low confidence; green = word starts)")
    plt.tight_layout()
    plt.show()


def plot_beats(song_path, beats, downbeats, grid=None, start=0.0, duration=30.0, figsize=(16, 3)):
    import librosa
    plt = _plt()
    y, sr = librosa.load(song_path, sr=11025, mono=True, offset=start, duration=duration)
    t = start + np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y, color="0.75", lw=0.3)
    for b in beats:
        if start <= b <= start + duration:
            ax.axvline(b, color="tab:blue", lw=0.8, alpha=0.6)
    for d in downbeats:
        if start <= d <= start + duration:
            ax.axvline(d, color="tab:red", lw=1.6)
    if grid:
        for e in grid["eights"]:
            if start <= e["start"] <= start + duration:
                ax.text(e["start"], ax.get_ylim()[1] * 0.9, f"E{e['index']}", color="tab:purple", fontsize=9)
    ax.set_xlim(start, start + duration)
    ax.set_title("beats (blue), downbeats (red), start of each eight (E#)")
    plt.tight_layout()
    plt.show()


def click_track(song_path, grid, start=0.0, duration=30.0, sr=22050, gain=0.6):
    """Song + clicks on every count (count 1 of each eight high and loud, '&' soft)."""
    import librosa
    y, _ = librosa.load(song_path, sr=sr, mono=True, offset=start, duration=duration)
    out = 0.6 * y.copy()

    def click(t, freq, amp):
        i = int((t - start) * sr)
        n = int(0.04 * sr)
        if 0 <= i < len(out) - n:
            tt = np.arange(n) / sr
            out[i:i + n] += amp * np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 60)

    for e in grid["eights"]:
        for name, ct in e["counts"].items():
            if name == "1":
                click(ct, 1760, gain)
            elif name.endswith("&"):
                click(ct, 660, gain * 0.25)
            else:
                click(ct, 1100, gain * 0.6)
    return np.clip(out, -1, 1), sr


# ---------------------------------------------------------------------------- dance diagnostics
def plot_speed_windows(stats, labels, window_s=5.0, figsize=(14, 3)):
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)
    for st, lb in zip(stats, labels):
        x = np.arange(len(st["speed_per_window"])) * window_s
        ax.plot(x, st["speed_per_window"], marker="o", ms=3, label=f"{lb} (mean {st['mean_speed']:.2f} m/s)")
    ax.set_xlabel("seconds")
    ax.set_ylabel("mean joint speed [m/s]")
    ax.set_title("motion energy per 5 s window (a decay over time suggests the sequence is too long for one pass)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_gesture_debug(r, fps=su.FPS):
    """Candidate speed curves + IK error per joint + target/IK skeleton overlay at the key pose."""
    plt = _plt()
    conv = r["debug"]["conv"]
    fig = plt.figure(figsize=(17, 4.2))
    ax = fig.add_subplot(1, 3, 1)
    for i, sp in enumerate(r["debug"]["speeds"]):
        ax.plot(np.arange(len(sp)) / fps, sp, label=f"candidate {i}", lw=2 if i == r["info"]["chosen"] else 0.8)
    k = r["info"]["key_frame_in_candidate"]
    ax.axvline(k / fps, color="k", ls="--", label="key pose")
    ax.axvspan((k - r["info"]["pre"]) / fps, (k + r["info"]["post"]) / fps, color="orange", alpha=0.2, label="cut")
    ax.set_xlabel("seconds")
    ax.set_ylabel("end-effector speed")
    ax.set_title(f"gesture #{r['id']}: key pose detection")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(1, 3, 2)
    e0 = conv["errors_analytic"]["pos_err_m"].mean(0) * 100
    e1 = conv["errors"]["pos_err_m"].mean(0) * 100
    x = np.arange(len(e1))
    ax.bar(x - 0.2, e0, 0.4, label="analytic IK")
    ax.bar(x + 0.2, e1, 0.4, label="refined IK")
    ax.set_xticks(x)
    ax.set_xticklabels([su.JOINT_NAMES[j] for j in range(len(e1))], rotation=90, fontsize=7)
    ax.set_ylabel("mean position error [cm]")
    ax.set_title("conversion error per joint (vs. retargeted target)")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    tgt = conv["errors"]["target_positions"][k]
    fkp = conv["errors"]["fk_positions"][k]
    for c in range(1, 22):
        p = su.SMPL_PARENTS[c]
        ax.plot(*[[tgt[p, i], tgt[c, i]] for i in (0, 2, 1)], color="tab:green", lw=3, alpha=0.6)
        ax.plot(*[[fkp[p, i], fkp[c, i]] for i in (0, 2, 1)], color="tab:red", lw=1.2)
    ax.set_title("key pose: target (green) vs converted SMPL (red)")
    lim = np.abs(tgt - tgt[0]).max() + 0.1
    ax.set_xlim(tgt[0, 0] - lim, tgt[0, 0] + lim)
    ax.set_ylim(tgt[0, 2] - lim, tgt[0, 2] + lim)
    ax.set_zlim(tgt[0, 1] - lim, tgt[0, 1] + lim)
    plt.tight_layout()
    plt.show()


def plot_mask_timeline(mask, regions, windows, fps=su.FPS, figsize=(16, 3.4)):
    """Rows: contacts / root / upper-body joints / lower-body joints / other joints."""
    plt = _plt()
    groups = [("contacts", list(range(0, 4))), ("root pos+rot", list(range(4, 7)) + su.joint_channels([0])),
              ("upper body", su.joint_channels(su.UPPER_JOINTS_DEFAULT)),
              ("lower body", su.joint_channels([1, 2, 4, 5, 7, 8, 10, 11])),
              ("head/wrists/hands", su.joint_channels([15, 20, 21, 22, 23]))]
    T = mask.shape[0]
    img = np.stack([mask[:, ch].float().mean(1).numpy() for _, ch in groups])
    gest = np.zeros_like(img)
    for r in regions:
        for gi, (_, ch) in enumerate(groups):
            if set(ch) & set(r["channels"]):
                gest[gi, r["start"]:r["end"]] = 1
    rgb = np.ones((len(groups), T, 3))
    rgb[img < 0.5] = (0.95, 0.55, 0.2)          # re-generated
    rgb[(img >= 0.5) & (gest > 0)] = (0.2, 0.6, 0.25)  # fixed to the gesture
    rgb[(img >= 0.5) & (gest == 0)] = (0.8, 0.85, 0.95)  # fixed to the base dance
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgb, aspect="auto", extent=[0, T / fps, len(groups) - 0.5, -0.5], interpolation="nearest")
    for s, e in windows:
        ax.axvspan(s / fps, e / fps, ymin=0, ymax=1, fill=False, ec="k", lw=1.2)
    for r in regions:
        ax.axvline(r["hit"] / fps, color="k", lw=1)
        ax.text(r["hit"] / fps, -0.6, f"#{r['id']}", fontsize=8, ha="center")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g for g, _ in groups])
    ax.set_xlabel("seconds")
    ax.set_title("mask: green = gesture (fixed), blue = base dance (fixed), orange = re-generated; boxes = windows")
    plt.tight_layout()
    plt.show()


def rotation_angle_deg(Ra, Rb):
    rel = Ra.transpose(-1, -2) @ Rb
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.acos(cos))


def gesture_adherence(final_rot, target_rot, regions, joints_by_scope):
    """Mean angular deviation (deg) between the final dance and the pasted gesture on its joints."""
    rows = []
    for r in regions:
        js = joints_by_scope[r["scope"]]
        a = rotation_angle_deg(final_rot[r["start"]:r["end"]][:, js], target_rot[r["start"]:r["end"]][:, js])
        h = min(max(r["hit"], r["start"]), r["end"] - 1)
        ah = rotation_angle_deg(final_rot[h, js], target_rot[h, js])
        rows.append({"id": r["id"], "scope": r["scope"], "mean_dev_deg": float(a.mean()),
                     "hit_dev_deg": float(ah.mean()), "max_dev_deg": float(a.max())})
    return rows


def boundary_jumps(positions, windows, fps=su.FPS):
    """Joint speed right at each window border vs. the median speed (should be similar)."""
    v = (positions[1:] - positions[:-1]).norm(dim=-1).mean(-1) * fps
    med = float(v.median())
    out = []
    for s, e in windows:
        for b in (s, e):
            if 1 <= b < len(v):
                out.append({"frame": int(b), "speed": float(v[b - 1]), "median_speed": med})
    return out
