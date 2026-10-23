"""MotionGPT3 text-to-motion in a separate process (step 7).

MotionGPT3 needs transformers==4.47.1 internals (its MoT-GPT2 code) and a few old packages; they are
installed in an override directory that is put first on PYTHONPATH only for this process, so the main
notebook keeps its own versions. Only the generation path is used: the evaluation metrics, renderers
and datasets of the repository are replaced by stubs (they need extra downloads and are not used).

Usage:
  python mgpt_runner.py prepare --repo REPO --gpt2 GPT2_DIR
  python mgpt_runner.py generate --repo REPO --ckpt CKPT --requests REQ.json --out OUT_DIR [--seed 0]
REQ.json: [{"id": 0, "text": "a person raises both arms", "n": 4, "seconds": 4.0}, ...]
Output : OUT_DIR/gesture_<id>.npz with joints (n, L, 22, 3) y-up at 20 fps and feats (n, L, 263).
"""
import argparse
import importlib
import json
import os
import shutil
import sys
import types

PROMPT_TEMPLATE = "Generate motion: {}"


def _stub(name):
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    sys.modules[name] = mod
    return mod


def install_stubs():
    import torch.nn as nn
    # Evaluation metrics (bert_score, spacy models, TM2T evaluators...) are not needed for generation.
    metrics = _stub("motGPT.metrics")

    class BaseMetrics(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    metrics.BaseMetrics = BaseMetrics
    # Renderers need pyrender / moviepy.editor.
    render_utils = _stub("motGPT.utils.render_utils")
    render_utils.render_motion = lambda *a, **k: None
    try:
        import spacy  # noqa: F401
    except Exception:
        _stub("spacy")  # only imported (never used) by the training datasets
    try:
        import scipy.ndimage.filters  # noqa: F401
    except Exception:
        import scipy.ndimage
        sys.modules["scipy.ndimage.filters"] = scipy.ndimage
    try:
        import rich  # noqa: F401
        import rich.progress  # noqa: F401
    except Exception:
        rich = _stub("rich")
        prog = _stub("rich.progress")
        prog.track = lambda it, *a, **k: it
        rich.progress = prog


def build_cfg():
    from omegaconf import OmegaConf
    from motGPT.config import get_module_config
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    cfg_assets = OmegaConf.load("./configs/assets.yaml")
    cfg_base = OmegaConf.load(os.path.join(cfg_assets.CONFIG_FOLDER, "default.yaml"))
    cfg_exp = OmegaConf.merge(cfg_base, OmegaConf.load("./configs/webui.yaml"))
    if not cfg_exp.FULL_CONFIG:
        cfg_exp = get_module_config(cfg_exp, cfg_assets.CONFIG_FOLDER)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)
    cfg.DEBUG = False
    cfg.METRIC.TYPE = []
    return cfg


def load_model(repo, ckpt, device):
    os.chdir(repo)
    sys.path.insert(0, repo)
    install_stubs()
    import torch
    from motGPT.data.build_data import build_data
    from motGPT.models.build_model import build_model
    base = importlib.import_module("motGPT.models.base")
    base.BaseModel.configure_metrics = lambda self: None
    cfg = build_cfg()
    datamodule = build_data(cfg, phase="test")
    model = build_model(cfg, datamodule).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        res = model.load_state_dict(state, strict=False)
        ignore = ("metrics.", "_losses.")
        missing = [k for k in res.missing_keys if not k.startswith(ignore)]
        unexpected = [k for k in res.unexpected_keys if not k.startswith(ignore)]
        print(f"[MotionGPT3] loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
        core_missing = [k for k in missing if k.startswith(("lm.", "vae."))]
        if core_missing:
            raise RuntimeError(f"checkpoint does not match the model: missing {core_missing[:10]}")
        if missing or unexpected:
            print(f"[MotionGPT3] note: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    return model, datamodule


def generate(model, datamodule, text, n, seconds, seed):
    import torch
    lm, vae = model.lm, model.vae
    L = max(40, min(196, int(round(seconds * datamodule.fps / 4.0)) * 4))
    torch.manual_seed(seed)
    with torch.no_grad():
        try:
            # Official test-time path ('Generate motion: <Caption_Placeholder>' template).
            outputs = lm.generate_conditional(texts=[text] * n, lengths=[L] * n, task="t2m", stage="test")
        except Exception as e:  # pragma: no cover
            print(f"[MotionGPT3] generate_conditional failed ({e!r}); using generate_direct_motion")
            outputs = lm.generate_direct_motion([PROMPT_TEMPLATE.format(text) + " \n"] * n, motion_tokens=None)
        lat, _ = lm.sample_tokens(outputs, lm.device, temperature=1.0, cfg=model.guidance_scale,
                                  vae_mean_std_inv=vae.mean_std_inv)
        lat = lat.reshape(n, vae.latent_size, -1).permute(1, 0, 2)
        feats = vae.decode(lat, lengths=[L] * n)
        joints = datamodule.feats2joints(feats)
        feats = datamodule.denormalize(feats)
    return joints.float().cpu().numpy(), feats.float().cpu().numpy()


def cmd_prepare(args):
    """scripts/gen_mot_gpt.py of MotionGPT3, with explicit paths."""
    import torch
    from transformers import GPT2LMHeadModel
    target = os.path.join(args.repo, "deps", "mot-gpt2")
    model = GPT2LMHeadModel.from_pretrained(args.gpt2).eval()
    new_sd = {}
    for name, p in model.state_dict().items():
        name = name.replace("attn.c_attn", "c_attn.fn.0").replace("attn.c_proj", "c_proj.fn.0")
        name = name.replace("mlp", "mlp.fn.0").replace("ln_f", "ln_f.fn.0").replace("ln_1", "ln_1.fn.0")
        name = name.replace("ln_2", "ln_2.fn.0")
        new_sd[name] = p
    os.makedirs(target, exist_ok=True)
    torch.save(new_sd, os.path.join(target, "model_state_dict.pth"))
    model.config.to_json_file(os.path.join(target, "config.json"))
    model.generation_config.to_json_file(os.path.join(target, "generation_config.json"))
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if os.path.exists(os.path.join(args.gpt2, f)):
            shutil.copy(os.path.join(args.gpt2, f), os.path.join(target, f))
    print(f"[MotionGPT3] prepared {target}")


def cmd_generate(args):
    import numpy as np
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reqs = json.load(open(args.requests, encoding="utf-8"))
    model, dm = load_model(os.path.abspath(args.repo), args.ckpt, device)
    os.makedirs(args.out, exist_ok=True)
    for k, r in enumerate(reqs):
        joints, feats = generate(model, dm, r["text"], int(r.get("n", 4)), float(r.get("seconds", 4.0)),
                                 args.seed + 1000 * int(r["id"]) + k)
        np.savez(os.path.join(args.out, f"gesture_{r['id']}.npz"), joints=joints, feats=feats, text=r["text"],
                 fps=dm.fps)
        print(f"[MotionGPT3] gesture {r['id']}: {joints.shape} '{r['text']}'", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--repo", required=True)
    p.add_argument("--gpt2", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--repo", required=True)
    g.add_argument("--ckpt", default="")
    g.add_argument("--requests", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
