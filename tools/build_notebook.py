"""Build music_to_dance.ipynb from the m2d sources and the cell texts below.

    python tools/build_notebook.py            # writes ../music_to_dance.ipynb
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
OUT = os.path.join(SRC, "music_to_dance.ipynb")
PKG = "/content/m2d_pkg"

cells = []


def md(text):
    cells.append(("markdown", text.strip("\n")))


def code(text):
    cells.append(("code", text.strip("\n")))


def writefile(src_rel, dst):
    with open(os.path.join(SRC, src_rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {dst}\n" + body)


# =====================================================================================================
md(r'''
# 歌詞＋音楽 → 3Dダンスモーション生成（Google Colab）

楽曲の音声・歌詞・ジャンルから、**歌詞の意味**と**音楽のリズム**の両方を反映した 3D ダンスモーション（SMPL 形式）を作ります。
映像化（実写・CG 動画の生成）は行いません。確認用に骨格アニメーションとマネキンのプレビュー動画だけを描画します。

| # | 工程 | 使うもの | 出力 |
|---|---|---|---|
| 1 | ボーカル分離 | Demucs（htdemucs） | ボーカル音源 |
| 2 | 歌詞の時刻合わせ | wav2vec2（日本語）＋ WhisperX の整列機能 | 単語ごとのタイムスタンプ |
| 3 | ビート・ダウンビート検出 | beat_this | 拍・小節頭 |
| 4 | 8カウントのグリッド | — | 8カウント単位のスロット表 |
| 5 | 振付台本 | LLM（OpenRouter） | ジェスチャーの JSON |
| 6 | ベースダンス（1回目） | FlowerDance | 曲全体のモーション |
| 7 | 歌詞ジェスチャー | MotionGPT3 ＋ 逆運動学 | 「決め」を含むジェスチャー断片 |
| 8 | 拍に合わせて埋め込み | — | 指定済みモーションとマスク |
| 9 | つなぎの作り直し（2回目） | FlowerDance のマスク付き生成 | 完成モーション |
| 10 | 書き出しとプレビュー | — | SMPL `.pkl`・BVH・骨格アニメーション |
| 11 | マネキン動画 | PyTorch のレイマーチング | プレビュー動画 |

**方針**
- すべて学習済みモデルの**推論のみ**（学習・ファインチューニングはしません）。
- **Google Drive は使いません。** モデルと中間データは `/content` に置き、成果物はブラウザからダウンロードします（セッションごとにモデルを再ダウンロード）。
- **ソースからのビルドは行いません。** mamba-ssm / causal-conv1d はビルド済みホイール、PyTorch3D は回転変換のファイルだけを同梱します。
- モデルは同時にロードせず、工程ごとにロード → 解放します（VRAM 節約）。
''')

md(r'''
## 0. 使い方

1. **ランタイム → ランタイムのタイプを変更** で GPU（**L4 または A100** 推奨）を選びます。
2. 左の 🔑（シークレット）に次を登録し、「ノートブックからのアクセス」をオンにします。
   - `OPENROUTER_API_KEY`（必須）: [OpenRouter](https://openrouter.ai/) の API キー
   - `HF_TOKEN`（任意）: Hugging Face のトークン（ダウンロードの速度制限よけ）
3. 上から順にセルを実行します。**3. 入力** で音源・歌詞・ジャンルを指定します。
4. 途中で確認・修正できる箇所があります（歌詞のタイミング、8カウント、振付台本、マスクの方式）。
5. 最後に成果物を ZIP でダウンロードします。

> ランタイムを再起動したときは **1-1** と **1-7** を実行し直してから、続きのセルを実行してください（中間結果は `/content/work` に保存されているので、終わった工程はやり直す必要はありません）。

**所要時間の目安（L4・4分の曲）**: 環境構築 5〜8分 / モデルのダウンロード 3〜5分 / 解析 2〜3分 / ベースダンス 1分前後 / ジェスチャー 2〜4分 / マスク付き生成 1〜3分 / プレビュー描画 数分。
''')

# =====================================================================================================
md(r'''
## 1. 環境構築

FlowerDance の `requirements.txt` は mamba-ssm・causal-conv1d・PyTorch3D を Git の特定コミットからソースビルドする指定になっています。Colab では長時間かかり失敗もしやすいため、次のように置き換えます。

| 依存 | 対応 |
|---|---|
| PyTorch | `torch==2.7.1`（CUDA 12.8 版）を入れ直す。ビルド済みなのでダウンロードのみ |
| mamba-ssm / causal-conv1d | PyPI 版のインストーラーと**同じ規則**（torch・CUDA・Python・C++ABI の組み合わせ）で GitHub リリースのビルド済みホイール URL を組み立て、**存在を確認してから直接インストール**。ホイールが無い場合はビルドに進まず、その場で止まります |
| PyTorch3D | 使っているのは回転変換（`pytorch3d.transforms`）だけ。純粋な PyTorch のコードなので該当ファイル（BSD ライセンス）を同梱 |
| essentia / wandb | コード中で使われていないため入れない |
| allin1 / madmom | 使わない（beat_this と librosa で代替） |
| MotionGPT3 | `transformers==4.47.1` などの内部仕様に依存するため、専用フォルダに入れて **MotionGPT3 を動かす別プロセスだけ**に見せる（本体の環境は汚さない） |

最終手段として、Mamba の選択的スキャンを **純 PyTorch の実装**（mamba-ssm の参照実装と同じ計算をチャンク単位で行うもの）に差し替える仕組みも入れてあります。
''')

code(r'''
# @title 1-1. 作業フォルダと GPU の確認
import os, sys, subprocess, json, shutil, time

ROOT = "/content"
WORK = f"{ROOT}/work"            # 中間データ・成果物
MODELS = f"{ROOT}/models"        # 学習済みモデル（セッションごとに再ダウンロード）
REPOS = f"{ROOT}/repos"          # FlowerDance / MotionGPT3 のソースコード
PKG = f"{ROOT}/m2d_pkg"          # このノートブックのコード（m2d パッケージと PyTorch3D の代替）
MGPT_SITE = f"{ROOT}/mgpt_site"  # MotionGPT3 専用の上書きパッケージ
FD_REPO = f"{REPOS}/FlowerDance"
MGPT_REPO = f"{REPOS}/MotionGPT3"
for d in (WORK, MODELS, REPOS, f"{PKG}/m2d", f"{PKG}/pytorch3d/transforms", MGPT_SITE):
    os.makedirs(d, exist_ok=True)
os.environ["TORCH_HOME"] = f"{MODELS}/torch"
os.environ["HF_HOME"] = f"{MODELS}/huggingface"
os.environ["NLTK_DATA"] = f"{MODELS}/nltk"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    gpu_info = smi.stdout.strip() if smi.returncode == 0 else ""
except FileNotFoundError:
    gpu_info = ""
if not gpu_info:
    raise SystemExit("GPU が見つかりません。「ランタイム → ランタイムのタイプを変更」で L4 / A100 などの GPU を選んでください。")
print("GPU   :", gpu_info)
print("Python:", sys.version.split()[0])
print("disk  :", shutil.disk_usage(ROOT).free // 2**30, "GB free")
''')

code(r'''
# @title 1-2. PyTorch 2.7.1（CUDA 12.8）を入れ直す
from importlib.metadata import version, PackageNotFoundError

TORCH_PINS = {"torch": "2.7.1", "torchvision": "0.22.1", "torchaudio": "2.7.1"}

def installed_version(pkg):
    try:
        return version(pkg).split("+")[0]
    except PackageNotFoundError:
        return None

need = [p for p, v in TORCH_PINS.items() if installed_version(p) != v]
if need:
    print("installing:", need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *[f"{p}=={v}" for p, v in TORCH_PINS.items()],
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
# 以降の pip が PyTorch を上書きしないための制約ファイル
with open(f"{ROOT}/constraints.txt", "w") as f:
    f.write("".join(f"{p}=={v}\n" for p, v in TORCH_PINS.items()))

probe = subprocess.run([sys.executable, "-c",
                        "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
                       capture_output=True, text=True, check=True).stdout.split()
print("torch", probe[0], "| CUDA", probe[1], "| GPU available:", probe[2])
assert probe[0].startswith("2.7.1") and probe[2] == "True", "PyTorch 2.7.1 + CUDA の確認に失敗しました"
if need and "torch" in sys.modules:
    print("このカーネルでは以前の torch が読み込み済みです。ランタイムを再起動します。再起動後に 1-1 から実行し直してください。")
    time.sleep(2)
    os.kill(os.getpid(), 9)
''')

code(r'''
# @title 1-3. Mamba（mamba-ssm / causal-conv1d）をビルド済みホイールで入れる
import urllib.request

MAMBA_VERSION = "2.2.6.post3"    # FlowerDance の requirements.txt が指すコミット（v2.2.6.post3）
CAUSAL_CONV1D_VERSION = "1.5.4"  # 指定コミット（1.5.3.post2 と 1.5.4 の間）に最も近い、ホイールのある版
ALLOW_TORCH_MAMBA_FALLBACK = True  # ホイールが無い・読み込めない場合に純 PyTorch 実装へ切り替える（最終手段・低速）

tv, cuda, abi, py = subprocess.run(
    [sys.executable, "-c", "import sys,torch;print(torch.__version__.split('+')[0], torch.version.cuda, "
     "torch._C._GLIBCXX_USE_CXX11_ABI, f'cp{sys.version_info.major}{sys.version_info.minor}')"],
    capture_output=True, text=True, check=True).stdout.split()
torch_mm = ".".join(tv.split(".")[:2])
cuda_major = cuda.split(".")[0]
abi = str(abi == "True").upper()

def wheel_url(repo, pkg, ver):
    # mamba-ssm / causal-conv1d の setup.py の get_wheel_url() と同じ命名規則
    name = f"{pkg}-{ver}+cu{cuda_major}torch{torch_mm}cxx11abi{abi}-{py}-{py}-linux_x86_64.whl"
    return f"https://github.com/{repo}/releases/download/v{ver}/{name}"

def url_exists(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status in (200, 206)
    except Exception:
        return False

wheels = {"causal_conv1d": wheel_url("Dao-AILab/causal-conv1d", "causal_conv1d", CAUSAL_CONV1D_VERSION),
          "mamba_ssm": wheel_url("state-spaces/mamba", "mamba_ssm", MAMBA_VERSION)}
missing = [k for k, u in wheels.items() if not url_exists(u)]
MAMBA_BACKEND = "cuda"
if missing:
    msg = ("ビルド済みホイールが見つかりません: " + ", ".join(f"{k} ({wheels[k]})" for k in missing) +
           "\nソースビルドは長時間止まるため行いません。")
    if not ALLOW_TORCH_MAMBA_FALLBACK:
        raise SystemExit(msg)
    print(msg + "\n→ 最終手段として純 PyTorch 版の Mamba を使います（生成は遅くなります）。")
    MAMBA_BACKEND = "torch"
else:
    for k, u in wheels.items():
        print(k, "<-", u)
    # ホイールの URL を直接渡すので、ソースビルドは起こり得ない
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation",
                    wheels["causal_conv1d"], wheels["mamba_ssm"]], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-c", f"{ROOT}/constraints.txt",
                    "einops", "ninja", "packaging", "transformers"], check=True)
json.dump({"MAMBA_BACKEND": MAMBA_BACKEND}, open(f"{WORK}/env_state.json", "w"))
print("MAMBA_BACKEND =", MAMBA_BACKEND)
''')

code(r'''
# @title 1-4. その他のライブラリ
pip = [sys.executable, "-m", "pip", "install", "-q"]
# 本体の環境（PyTorch は制約ファイルで固定）
subprocess.run(pip + ["-c", f"{ROOT}/constraints.txt",
                      "librosa==0.11.0", "soundfile", "soxr", "einops", "scipy", "huggingface_hub", "safetensors",
                      "demucs==4.1.0", "beat-this==1.1.0", "rotary-embedding-torch",
                      "pykakasi", "nltk", "pandas", "gdown", "opencv-python-headless", "ipywidgets", "requests"],
               check=True)
# WhisperX は整列機能だけを使うので依存なしで入れる（ASR 用の faster-whisper や torch 2.8 指定を避ける）
subprocess.run(pip + ["--no-deps", "whisperx==3.8.6"], check=True)
# MotionGPT3 専用（PYTHONPATH の先頭に置いて MotionGPT3 のプロセスだけに見せる）
subprocess.run(pip + ["--no-deps", "--upgrade", "--target", MGPT_SITE,
                      "transformers==4.47.1", "tokenizers==0.21.0", "huggingface_hub==0.26.5",
                      "pytorch-lightning==2.4.0", "lightning-utilities==0.11.9", "torchmetrics==1.6.0",
                      "omegaconf==2.3.0", "antlr4-python3-runtime==4.9.3"], check=True)
# プレビューの日本語表示用フォント
subprocess.run("apt-get -qq update > /dev/null && apt-get -qq install -y fonts-noto-cjk > /dev/null", shell=True)
print("done")
''')

code(r'''
# @title 1-5. FlowerDance / MotionGPT3 のソースを取得（動作確認したコミットに固定）
REPO_PINS = {
    FD_REPO: ("https://github.com/XulongT/FlowerDance", "4431435d17fb84e04057e9cf50a8e954434ced73"),
    MGPT_REPO: ("https://github.com/OpenMotionLab/MotionGPT3", "cf5bab7daa568ce472e7201cc2bf8b83d8c1b732"),
}

def git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)

for dst, (url, commit) in REPO_PINS.items():
    if not os.path.isdir(f"{dst}/.git"):
        git("init", "-q", dst)
        git("-C", dst, "remote", "add", "origin", url)
    if git("-C", dst, "fetch", "-q", "--depth", "1", "origin", commit, check=False).returncode == 0:
        git("-C", dst, "checkout", "-q", "-f", commit)
    else:
        print(f"[warn] {url} の固定コミットを取得できないため既定ブランチを使います")
        git("-C", dst, "fetch", "-q", "--depth", "1", "origin")
        git("-C", dst, "checkout", "-q", "-f", "FETCH_HEAD")
    print(os.path.basename(dst), git("-C", dst, "rev-parse", "--short", "HEAD").stdout.strip())
''')

md(r'''
### 1-6. パイプラインのコード（`m2d` パッケージ）

以下のセルは、このノートブックで使う処理を `/content/m2d_pkg` にファイルとして書き出します（実行しても何も起きないように見えますが、ファイルが作られます）。

| ファイル | 内容 |
|---|---|
| `pytorch3d/transforms/*` | PyTorch3D の回転変換（同梱、ビルド不要） |
| `m2d/smpl_utils.py` | SMPL 骨格（FlowerDance と同一の親子関係・オフセット）、順運動学、151 次元表現の変換、y-up ↔ z-up |
| `m2d/mamba_fallback.py` | 純 PyTorch 版 Mamba（最終手段。パラメータ名が同じなので重みをそのまま読める） |
| `m2d/flowerdance_runner.py` | FlowerDance の読み込み、音楽特徴量、曲全体・区間分割の生成、マスク付き生成 |
| `m2d/audio_analysis.py` | Demucs、beat_this、8カウントのグリッド |
| `m2d/lyrics_align.py` | 歌詞の強制整列（全体 CTC 整列 → WhisperX で行ごとに精密化）と手動補正 |
| `m2d/choreo_llm.py` | OpenRouter で振付台本を作る（JSON の検証と再試行） |
| `m2d/mgpt_runner.py` | MotionGPT3 を別プロセスで動かす |
| `m2d/motion_ik.py` | HumanML3D（22 関節位置・20fps）→ SMPL 回転（30fps）の変換（逆運動学） |
| `m2d/gesture_pipeline.py` | 候補の選択、「決め」の検出と切り出し、誇張 |
| `m2d/gesture_embed.py` | 拍に合わせた埋め込み、マスクとウィンドウの作成 |
| `m2d/export_render.py` | 書き出し（pkl / BVH）、骨格アニメーション、マネキン描画 |
| `m2d/viz.py` | 確認用の図と診断 |
''')

for rel, dst in [
    ("shims/pytorch3d/__init__.py", f"{PKG}/pytorch3d/__init__.py"),
    ("shims/pytorch3d/transforms/__init__.py", f"{PKG}/pytorch3d/transforms/__init__.py"),
    ("shims/pytorch3d/transforms/rotation_conversions.py", f"{PKG}/pytorch3d/transforms/rotation_conversions.py"),
    ("m2d/__init__.py", f"{PKG}/m2d/__init__.py"),
    ("m2d/smpl_utils.py", f"{PKG}/m2d/smpl_utils.py"),
    ("m2d/mamba_fallback.py", f"{PKG}/m2d/mamba_fallback.py"),
    ("m2d/flowerdance_runner.py", f"{PKG}/m2d/flowerdance_runner.py"),
    ("m2d/audio_analysis.py", f"{PKG}/m2d/audio_analysis.py"),
    ("m2d/lyrics_align.py", f"{PKG}/m2d/lyrics_align.py"),
    ("m2d/choreo_llm.py", f"{PKG}/m2d/choreo_llm.py"),
    ("m2d/mgpt_runner.py", f"{PKG}/m2d/mgpt_runner.py"),
    ("m2d/motion_ik.py", f"{PKG}/m2d/motion_ik.py"),
    ("m2d/gesture_pipeline.py", f"{PKG}/m2d/gesture_pipeline.py"),
    ("m2d/gesture_embed.py", f"{PKG}/m2d/gesture_embed.py"),
    ("m2d/export_render.py", f"{PKG}/m2d/export_render.py"),
    ("m2d/viz.py", f"{PKG}/m2d/viz.py"),
]:
    writefile(rel, dst)

code(r'''
# @title 1-7. 共通設定と補助関数（ランタイム再起動後は 1-1 とこのセルを実行し直す）
import os, sys, json, gc, time, math, shutil, subprocess, importlib, base64
ROOT, WORK, MODELS, REPOS, PKG = "/content", "/content/work", "/content/models", "/content/repos", "/content/m2d_pkg"
MGPT_SITE, FD_REPO, MGPT_REPO = f"{ROOT}/mgpt_site", f"{REPOS}/FlowerDance", f"{REPOS}/MotionGPT3"
for k, v in {"TORCH_HOME": f"{MODELS}/torch", "HF_HOME": f"{MODELS}/huggingface", "NLTK_DATA": f"{MODELS}/nltk",
             "TOKENIZERS_PARALLELISM": "false"}.items():
    os.environ[k] = v
if PKG not in sys.path:
    sys.path.insert(0, PKG)
# サブプロセスからも m2d と PyTorch3D の代替が見えるようにする
import site
with open(os.path.join(site.getsitepackages()[0], "m2d_pkg.pth"), "w") as f:
    f.write(PKG + "\n")

import numpy as np
import pandas as pd
import librosa
import torch
from IPython.display import Audio, HTML, display

import m2d
from m2d import (smpl_utils as su, mamba_fallback as mf, flowerdance_runner as fr, audio_analysis as aa,
                 lyrics_align as la, choreo_llm as cl, motion_ik as ik, gesture_pipeline as gp,
                 gesture_embed as ge, export_render as er, viz)
for _m in (su, mf, fr, aa, la, cl, ik, gp, ge, er, viz):
    importlib.reload(_m)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_STATE_PATH = f"{WORK}/env_state.json"

def env_state():
    return json.load(open(ENV_STATE_PATH)) if os.path.exists(ENV_STATE_PATH) else {}

def update_env_state(**kw):
    st = env_state(); st.update(kw); json.dump(st, open(ENV_STATE_PATH, "w"), indent=1)

def save_json(name, obj):
    with open(f"{WORK}/{name}", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)

def load_json(name):
    with open(f"{WORK}/{name}", encoding="utf-8") as f:
        return json.load(f)

def save_pt(name, obj):
    torch.save(obj, f"{WORK}/{name}")

def load_pt(name):
    return torch.load(f"{WORK}/{name}", map_location="cpu", weights_only=False)

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def show_video(path, width=480, max_mb=60):
    size = os.path.getsize(path) / 2**20
    if size > max_mb:
        print(f"{path} ({size:.0f} MB) は大きいのでここでは表示しません。ダウンロードして確認してください。")
        return
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    display(HTML(f'<video width="{width}" controls><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))

def mgpt_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([MGPT_SITE] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env

def run_streaming(cmd, env=None, cwd=None):
    proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    if proc.wait() != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd[:4])} ...")

MAMBA_BACKEND = env_state().get("MAMBA_BACKEND", "cuda")
print("torch", torch.__version__, "| device", DEVICE, "| MAMBA_BACKEND", MAMBA_BACKEND)
print("m2d", m2d.__version__, "| modules OK")
''')

# =====================================================================================================
md(r'''
## 2. モデルのダウンロード（`/content/models` に保存）

| モデル | 取得元 | 目安 |
|---|---|---|
| FlowerDance | Hugging Face `xlt99/FlowerDance`（`train-3700.pt`） | 1GB 未満 |
| Demucs htdemucs | Hugging Face（demucs 4.1 が自動取得） | 約 0.3GB |
| wav2vec2（日本語） | `jonatasgrosman/wav2vec2-large-xlsr-53-japanese`（WhisperX の既定） | 約 1.2GB |
| beat_this `final0` | 作者の公開ストレージ（自動取得） | 0.1GB 未満 |
| MotionGPT3 | Hugging Face `OpenMotionLab/motiongpt3`（失敗時は作者の Google Drive 公開リンク） | 約 1〜2GB |
| GPT-2（MotionGPT3 の言語部分） | `openai-community/gpt2` | 約 0.5GB |

合計で約 3〜4GB です。SMPL のモデルファイル（要登録）は不要です（FlowerDance は骨格オフセットをコードに持っています）。
''')

code(r'''
# @title 2-1. FlowerDance / Demucs / beat_this / wav2vec2 / NLTK
from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files

HF_TOKEN = None
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    pass

FD_CKPT = hf_hub_download("xlt99/FlowerDance", "train-3700.pt", local_dir=f"{MODELS}/flowerdance", token=HF_TOKEN)
print("FlowerDance:", FD_CKPT)

from demucs.pretrained import get_model as demucs_get_model
_ = demucs_get_model("htdemucs"); del _
print("Demucs htdemucs: OK")

from beat_this.inference import load_checkpoint as beat_this_load
_ = beat_this_load("final0"); del _
print("beat_this final0: OK")

ALIGN_MODEL_JA = "jonatasgrosman/wav2vec2-large-xlsr-53-japanese"
snapshot_download(ALIGN_MODEL_JA, token=HF_TOKEN)
print("wav2vec2 (ja): OK")

import nltk
nltk.download("punkt_tab", download_dir=os.environ["NLTK_DATA"], quiet=True)
update_env_state(FD_CKPT=FD_CKPT)
''')

code(r'''
# @title 2-2. MotionGPT3 と GPT-2
MGPT_HF_REPO = "OpenMotionLab/motiongpt3"
MGPT_GDRIVE_ID = "1Wvx5PGJjVKPRvjcl8firChw1UVjUj36l"  # MotionGPT3 の prepare スクリプトが指す公開ファイル
MGPT_CKPT = ""
try:
    files = list_repo_files(MGPT_HF_REPO, token=HF_TOKEN)
    print("files in", MGPT_HF_REPO, ":", files)
    cands = [f for f in files if f.endswith((".ckpt", ".pt", ".pth"))]
    cands.sort(key=lambda f: (os.path.basename(f) != "motiongpt3.ckpt", "motiongpt3" not in f.lower(),
                              not f.endswith(".ckpt"), len(f)))
    if cands:
        MGPT_CKPT = hf_hub_download(MGPT_HF_REPO, cands[0], local_dir=f"{MODELS}/motiongpt3", token=HF_TOKEN)
except Exception as e:
    print("[warn] Hugging Face から取得できませんでした:", repr(e))
if not MGPT_CKPT:
    import gdown
    MGPT_CKPT = f"{MODELS}/motiongpt3/motiongpt3.ckpt"
    os.makedirs(os.path.dirname(MGPT_CKPT), exist_ok=True)
    gdown.download(id=MGPT_GDRIVE_ID, output=MGPT_CKPT, quiet=False)
assert os.path.getsize(MGPT_CKPT) > 10 * 2**20, f"MotionGPT3 のチェックポイントが不正です: {MGPT_CKPT}"
print("MotionGPT3:", MGPT_CKPT, f"({os.path.getsize(MGPT_CKPT) / 2**30:.2f} GB)")

GPT2_DIR = snapshot_download("openai-community/gpt2", local_dir=f"{MODELS}/gpt2", token=HF_TOKEN,
                             allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                                             "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
# MotionGPT3 の scripts/gen_mot_gpt.py と同じ変換（deps/mot-gpt2 を作る）
run_streaming([sys.executable, f"{PKG}/m2d/mgpt_runner.py", "prepare", "--repo", MGPT_REPO, "--gpt2", GPT2_DIR],
              env=mgpt_env())
update_env_state(MGPT_CKPT=MGPT_CKPT)
print(subprocess.run(["du", "-sh", MODELS], capture_output=True, text=True).stdout)
''')

code(r'''
# @title 2-3. インストールの確認（Mamba のカーネル、FlowerDance の重みの strict 読み込み）
FD_CKPT = env_state()["FD_CKPT"]
fr.patch_transformers_for_mamba()
if MAMBA_BACKEND == "cuda":
    try:
        from mamba_ssm import Mamba as CudaMamba
        torch.manual_seed(0)
        m_cuda = CudaMamba(d_model=512, d_state=16, d_conv=4, expand=2).to(DEVICE).eval()
        m_ref = mf.Mamba(512).to(DEVICE).eval()
        m_ref.load_state_dict(m_cuda.state_dict(), strict=True)
        x = torch.randn(1, 1200, 512, device=DEVICE)
        with torch.no_grad():
            a, b = m_cuda(x), m_ref(x)
        rel = ((a - b).abs().max() / b.abs().max()).item()
        print(f"Mamba CUDA kernel: OK (純 PyTorch 参照実装との相対誤差 {rel:.2e})")
        assert rel < 1e-2, "CUDA カーネルの結果が参照実装と一致しません"
        del m_cuda, m_ref, x, a, b
    except Exception as e:
        print("[warn] Mamba の CUDA カーネルが使えません:", repr(e))
        if not globals().get("ALLOW_TORCH_MAMBA_FALLBACK", True):
            raise
        MAMBA_BACKEND = "torch"
        print("→ 純 PyTorch 版の Mamba に切り替えます")
update_env_state(MAMBA_BACKEND=MAMBA_BACKEND)

fd = fr.FlowerDance(FD_REPO, FD_CKPT, DEVICE, MAMBA_BACKEND)  # 内部で load_state_dict(strict=True)
print(f"FlowerDance: strict 読み込み OK（{fd.n_params / 1e6:.1f}M パラメータ, Mamba={fd.mamba_backend}）")
t0 = time.time()
smoke = fd.sample(np.zeros((150, 35), np.float32), "Hiphop", n_steps=4, seed=0)
assert torch.isfinite(smoke).all()
print(f"FlowerDance: 試し生成 OK（5 秒分, {time.time() - t0:.1f}s）")
fd.free(); del fd, smoke; free_gpu()
''')

# =====================================================================================================
md(r'''
## 3. 入力

- **音源**: アップロード / Colab 内のパス / URL のいずれか。`SONG_START_SEC`・`SONG_DURATION_SEC` で一部だけを使うこともできます（試運転に便利です）。
- **ジャンル**: FlowerDance が学習した 16 ジャンルから選びます。
- **歌詞**: 1 行 = 1 フレーズで貼り付けます。`[サビ]` `【Aメロ】` のような括弧だけの行はセクション名として扱います。
- **OpenRouter**: シークレット `OPENROUTER_API_KEY` を読み込み、モデル名が存在するか確認します。
''')

code(r'''
# @title 3-1. 音源とジャンル
AUDIO_SOURCE = "upload"  # @param ["upload", "path", "url"]
AUDIO_PATH_OR_URL = ""  # @param {type:"string"}
SONG_START_SEC = 0.0  # @param {type:"number"}
SONG_DURATION_SEC = 0.0  # @param {type:"number"}
GENRE = "Korean"  # @param ["Dai", "ShenYun", "Wei", "Korean", "Urban", "Hiphop", "Popping", "Miao", "HanTang", "Breaking", "Kun", "Locking", "Jazz", "Choreography", "Chinese", "DunHuang"]
SEED = 2026  # @param {type:"integer"}

os.makedirs(f"{WORK}/input", exist_ok=True)
if AUDIO_SOURCE == "upload":
    from google.colab import files
    up = files.upload()
    if not up:
        raise SystemExit("ファイルが選ばれていません")
    src = f"{WORK}/input/{next(iter(up))}"
    with open(src, "wb") as f:
        f.write(up[next(iter(up))])
elif AUDIO_SOURCE == "url":
    import requests
    src = f"{WORK}/input/download_audio"
    with requests.get(AUDIO_PATH_OR_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(src, "wb") as f:
            shutil.copyfileobj(r.raw, f)
else:
    src = AUDIO_PATH_OR_URL
    assert os.path.exists(src), f"ファイルがありません: {src}"

SONG = f"{WORK}/input/song.wav"  # 以降はこの 44.1kHz ステレオ WAV を使う
cmd = ["ffmpeg", "-y", "-loglevel", "error"]
if SONG_START_SEC > 0:
    cmd += ["-ss", str(SONG_START_SEC)]
cmd += ["-i", src]
if SONG_DURATION_SEC > 0:
    cmd += ["-t", str(SONG_DURATION_SEC)]
cmd += ["-ac", "2", "-ar", "44100", SONG]
subprocess.run(cmd, check=True)
import librosa
DURATION = librosa.get_duration(path=SONG)
fr.genre_id(GENRE)  # ジャンル名の確認
save_json("inputs.json", {"song": SONG, "source": src, "start": SONG_START_SEC, "duration": DURATION,
                          "genre": GENRE, "seed": SEED})
print(f"song: {SONG}  ({DURATION:.1f} s), genre: {GENRE}")
display(Audio(SONG))
''')

code(r'''
# @title 3-2. 歌詞（下の LYRICS を書き換えるか、LYRICS_FILE に .txt のパスを入れる）
LYRICS_LANGUAGE = "ja"  # @param ["ja", "en"]
LYRICS_FILE = ""  # @param {type:"string"}

SAMPLE_LYRICS = """
[Aメロ]
朝の光が 窓をたたく
まだ眠い目を こすって歩く
[サビ]
手を伸ばして 空へ届け
君と踊る この瞬間を
回る世界で 笑い合おう
ずっと ずっと 輝いて
"""

LYRICS = """
[Aメロ]
朝の光が 窓をたたく
まだ眠い目を こすって歩く
[サビ]
手を伸ばして 空へ届け
君と踊る この瞬間を
回る世界で 笑い合おう
ずっと ずっと 輝いて
"""

if LYRICS_FILE:
    LYRICS = open(LYRICS_FILE, encoding="utf-8").read()
if LYRICS.strip() == SAMPLE_LYRICS.strip():
    print("⚠ サンプルの歌詞のままです。曲の歌詞に書き換えてから実行してください。")
parsed = la.parse_lyrics(LYRICS)
save_json("lyrics_input.json", {"text": LYRICS, "language": LYRICS_LANGUAGE})
display(pd.DataFrame(parsed))
''')

code(r'''
# @title 3-3. OpenRouter の API キーとモデル
OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"  # @param {type:"string"}
from google.colab import userdata
OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
models = cl.list_models(OPENROUTER_API_KEY)
if OPENROUTER_MODEL not in models:
    near = [m for m in models if any(k in m for k in ("claude", "gpt", "gemini"))][:30]
    raise SystemExit(f"OpenRouter にモデル {OPENROUTER_MODEL!r} がありません。候補: {near}")
print("OpenRouter: OK,", OPENROUTER_MODEL)
''')

# =====================================================================================================
md(r'''
## 4. 工程 1〜4：音楽と歌詞の解析

### 4-1. 工程 1：Demucs（htdemucs）でボーカルを分離
''')

code(r'''
# @title 4-1. ボーカル分離
SONG = load_json("inputs.json")["song"]
VOCALS, ACCOMP = aa.separate_vocals(SONG, f"{WORK}/stems", model_name="htdemucs", device=DEVICE)
free_gpu()
print("vocals:", VOCALS)
y_v, sr_v = librosa.load(VOCALS, sr=22050, mono=True, duration=30)
display(Audio(y_v, rate=sr_v))
''')

md(r'''
### 4-2. 工程 2：歌詞とボーカルの強制整列

1. 曲全体の wav2vec2 の出力（CTC の確率）を 20 秒ずつ計算し、歌詞全体を 1 回の Viterbi 整列で当てはめて**行のおおよその位置**を決めます。
2. 各行の区間（前後 0.4 秒の余裕つき）で **WhisperX の `align`** を実行し、文字・単語の時刻を精密化します。
3. 日本語は、整列モデルの語彙にない漢字を pykakasi でひらがなに直して整列し、元の文字に対応付け直します。単語の区切りは pykakasi の分割です。

歌唱は話し声より精度が落ちるため、結果を一覧と図で確認し、必要なら 4-3 で補正してください（`score` が低い行は赤で表示）。
''')

code(r'''
# @title 4-2. 整列の実行
lyr = load_json("lyrics_input.json")
VOCALS = f"{WORK}/stems/vocals.wav"
lines_raw, words_raw, align_info = la.align_lyrics(VOCALS, lyr["text"], lyr["language"], DEVICE, refine=True)
free_gpu()
save_json("align_raw.json", {"lines": lines_raw, "words": words_raw, "info": align_info})
print(align_info)
df_lines = pd.DataFrame([{"line": l["index"], "section": l.get("section"), "start": l.get("start"), "end": l.get("end"),
                          "score": round(l.get("score", float("nan")), 2), "text": l["text"]} for l in lines_raw])
display(df_lines.style.apply(lambda r: ["background-color: #fdd" if (r["score"] < -2.5 or pd.isna(r["start"])) else ""
                                        for _ in r], axis=1))
viz.plot_alignment(VOCALS, lines_raw, words_raw)
''')

code(r'''
# @title 4-2b. 行ごとに聴いて確認（ドロップダウンで行を選ぶ）
import ipywidgets as W
SONG, VOCALS = load_json("inputs.json")["song"], f"{WORK}/stems/vocals.wav"
_raw = load_json("align_raw.json"); lines_raw, words_raw = _raw["lines"], _raw["words"]
y_full, sr_full = librosa.load(VOCALS, sr=22050, mono=True)
y_mix, _ = librosa.load(SONG, sr=22050, mono=True)
out = W.Output()
opts = [(f"{l['index']}: {l['text'][:30]}", l["index"]) for l in lines_raw if l.get("start") is not None]
dd = W.Dropdown(options=opts, description="line")
src_sel = W.ToggleButtons(options=["vocals", "mix"], value="vocals")

def _play(*_):
    with out:
        out.clear_output()
        l = lines_raw[dd.value]
        a, b = int(max(0, l["start"] - 0.3) * sr_full), int((l["end"] + 0.3) * sr_full)
        y = y_full if src_sel.value == "vocals" else y_mix
        ws = [w for w in words_raw if w["line"] == l["index"]]
        print(f"{l['start']:.2f}-{l['end']:.2f}s  " + " ".join(f"{w['text']}@{w['start']:.2f}" for w in ws if w.get("start") is not None))
        display(Audio(y[a:b], rate=sr_full, autoplay=True))

dd.observe(_play, "value"); src_sel.observe(_play, "value")
display(W.HBox([dd, src_sel]), out)
if opts:
    _play()
''')

code(r'''
# @title 4-3. タイミングの手動補正（不要ならそのまま実行）
# 行番号: (開始秒, 終了秒)。行の中の単語は比例して伸縮します。
LINE_TIME_OVERRIDES = {
    # 3: (45.20, 48.90),
}
GLOBAL_OFFSET_SEC = 0.0  # 全体を一律にずらす（秒）
EDITED_LINES_CSV = ""    # 下で書き出した lyrics_lines.csv の start/end を編集してアップロードした場合、そのパス

raw = load_json("align_raw.json")
lines, words = raw["lines"], raw["words"]
overrides = dict(LINE_TIME_OVERRIDES)
if EDITED_LINES_CSV:
    ed = pd.read_csv(EDITED_LINES_CSV)
    for _, r in ed.iterrows():
        overrides[int(r["line"])] = (float(r["start"]), float(r["end"]))
lines, words = la.apply_overrides(lines, words, overrides, GLOBAL_OFFSET_SEC)
save_json("lyrics_lines.json", lines)
save_json("lyrics_words.json", words)
pd.DataFrame([{"line": l["index"], "start": l.get("start"), "end": l.get("end"), "text": l["text"]} for l in lines]
             ).to_csv(f"{WORK}/lyrics_lines.csv", index=False)
pd.DataFrame(words).to_csv(f"{WORK}/lyrics_words.csv", index=False)
print("saved: lyrics_lines.json / lyrics_words.json (CSV: lyrics_lines.csv, lyrics_words.csv)")
display(pd.DataFrame([{"line": l["index"], "start": l.get("start"), "end": l.get("end"), "manual": l.get("manual", False),
                       "text": l["text"]} for l in lines]))
''')

md(r'''
### 4-4. 工程 3：beat_this でビートとダウンビートを検出
''')

code(r'''
# @title 4-4. ビート検出
SONG = load_json("inputs.json")["song"]; lines = load_json("lyrics_lines.json")
beats, downbeats = aa.detect_beats(SONG, checkpoint="final0", device=DEVICE)
free_gpu()
save_json("beats.json", {"beats": beats.tolist(), "downbeats": downbeats.tolist()})
print(f"beats: {len(beats)}, downbeats: {len(downbeats)}, tempo ≈ {aa.tempo_from_beats(beats):.1f} BPM")
first_vocal = min([l["start"] for l in lines if l.get("start") is not None] or [0.0])
viz.plot_beats(SONG, beats, downbeats, start=max(0.0, first_vocal - 5), duration=30)
''')

md(r'''
### 4-5. 工程 4：8 カウントのグリッド

- 4/4 拍子の 2 小節 = 8 拍を 1 単位（エイト）とし、ダウンビート（小節の 1 拍目）から数え始めます。裏拍は「&」です。
- どちらの小節を「1」にするか（偶数小節 / 奇数小節）は、**歌詞の行の頭がカウント 1（またはその直前の 8・8&）に来る**ように自動で選びます。`GRID_PHASE` で固定もできます。
- 変拍子や検出漏れでカウント 1・5 以外にダウンビートが来た場合は、そこで新しいエイトを始めます（表の `irregular`）。
- クリック音つきで聴いて確認できます（高い音 = 各エイトの 1、中くらい = 各拍、小さい音 = 裏拍）。
''')

code(r'''
# @title 4-5. 8 カウントの作成と確認
GRID_PHASE = "auto"  # @param ["auto", "0", "1"]
CLICK_PREVIEW_START_SEC = -1.0  # @param {type:"number"}
CLICK_PREVIEW_SEC = 30.0  # @param {type:"number"}

bj = load_json("beats.json"); SONG = load_json("inputs.json")["song"]
lines = load_json("lyrics_lines.json"); words = load_json("lyrics_words.json")
DURATION = load_json("inputs.json")["duration"]
grid = aa.build_grid(bj["beats"], bj["downbeats"], DURATION,
                     line_starts=[l["start"] for l in lines if l.get("start") is not None],
                     phase=GRID_PHASE if GRID_PHASE == "auto" else int(GRID_PHASE))
grid = aa.attach_lyrics(grid, lines, words)
save_json("grid.json", grid)
save_json("lyrics_lines.json", lines)
print(f"{len(grid['eights'])} eights, {grid['bpm']:.1f} BPM, phase={grid['phase']} (scores {grid['phase_scores']}), "
      f"pickup beats before the first eight: {grid['pickup_beats']}")
display(pd.DataFrame(aa.grid_table(grid)))
start = CLICK_PREVIEW_START_SEC if CLICK_PREVIEW_START_SEC >= 0 else max(0.0, (lines[0].get("start") or 0) - 4)
y_click, sr_click = viz.click_track(SONG, grid, start=start, duration=CLICK_PREVIEW_SEC)
display(Audio(y_click, rate=sr_click))
''')

# =====================================================================================================
md(r'''
## 5. 工程 5：LLM で振付台本を作る

入力は歌詞・単語のタイミング・8 カウント表・ジャンルです。LLM は、意味の強い箇所（サビなど）を優先して**一部の行だけ**にジェスチャーを置き、各ジェスチャーについて次を返します。

- `motion_description`: MotionGPT3 向けの英語の動作記述（HumanML3D の書き方: "a person raises both arms ..."）
- `eight` / `start_count` / `hit_count` / `end_count`: 配置スロットと「決め」の拍
- `scope`: `upper`（腕・手・上体で表す）または `full`（跳ぶ・しゃがむ・回るなど全身）

JSON として読めない・スロットが存在しないなどの場合は、理由を添えて自動で再試行します。間隔が近すぎるもの・上限を超えたものは優先度の低い順に外します。
''')

code(r'''
# @title 5-1. 振付台本の生成
MAX_GESTURES = 0  # @param {type:"integer"}
# 0 = 自動（エイト数 ÷ 5、3〜12 個）
MIN_GAP_BEATS = 8  # @param {type:"integer"}
LLM_TEMPERATURE = 0.7  # @param {type:"number"}
EXTRA_INSTRUCTIONS = ""  # @param {type:"string"}

grid = load_json("grid.json"); lines = load_json("lyrics_lines.json")
GENRE = load_json("inputs.json")["genre"]
if "OPENROUTER_API_KEY" not in globals():  # ランタイム再起動後
    from google.colab import userdata
    OPENROUTER_API_KEY = userdata.get("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = globals().get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
max_g = MAX_GESTURES if MAX_GESTURES > 0 else int(np.clip(len(grid["eights"]) // 5, 3, 12))
script = cl.generate_script(lines, grid, GENRE, OPENROUTER_API_KEY, OPENROUTER_MODEL, aa.count_time,
                            max_gestures=max_g, min_gap_beats=MIN_GAP_BEATS, temperature=LLM_TEMPERATURE,
                            extra_instructions=EXTRA_INSTRUCTIONS)
save_json("script.json", script)
print("concept:", script["concept"])
print(f"{len(script['gestures'])} gestures (max {max_g}), LLM attempts: {script['attempts']}")
display(pd.DataFrame([{k: g[k] for k in ("id", "hit_time", "eight", "start_count", "hit_count", "end_count", "scope",
                                          "priority", "lyric", "meaning", "motion_description")}
                      for g in script["gestures"]]))
''')

code(r'''
# @title 5-2. 台本の手直し（任意）
# /content/work/script.json を左のファイル一覧から開いて編集・保存したあと、このセルを実行すると再検証します。
# 例: motion_description を書き換える、scope を変える、eight / hit_count を動かす、不要な要素を消す。
grid = load_json("grid.json"); lines = load_json("lyrics_lines.json")
obj = load_json("script.json")
gestures, warns = cl.validate_script(obj, grid, lines, max_gestures=100, min_gap_beats=globals().get("MIN_GAP_BEATS", 8),
                                     count_time=aa.count_time)
for w in warns:
    print("[note]", w)
obj["gestures"] = gestures
save_json("script.json", obj)
display(pd.DataFrame([{k: g[k] for k in ("id", "hit_time", "eight", "hit_count", "scope", "lyric", "motion_description")}
                      for g in gestures]))
''')

# =====================================================================================================
md(r'''
## 6. 工程 6：FlowerDance で曲全体のベースダンスを生成（1 回目）

- 音楽特徴量は FlowerDance の `inference.py` と同じ関数（librosa の 35 次元）で、**元のミックス音源**から計算します。
- FlowerDance の既定の生成長は 32 秒（学習時の系列長は 40 秒）です。Mamba には位置の上限がないので曲全体を 1 回で生成できますが、学習より長い系列で質が落ちる可能性があります。
  - `BASE_MODE="whole"`: 1 回で生成
  - `BASE_MODE="chunked"`: 32 秒ずつ生成し、各区間の先頭（前の区間との重なり）を**厳密マスク**で前の結果に固定して続きを生成（境界がつながる）
  - `BASE_MODE="auto"`: 40 秒以内なら whole、それより長ければ chunked
- `RUN_LENGTH_CHECK=True`（既定）では、曲が 40 秒より長いとき、最初に「一度に生成」と「区間分割」を同じ区間（`LENGTH_CHECK_SEC`）で比べ、時間・メモリ・動きの量（5 秒ごとの関節速度）・足滑りを表示します。確認が済んだら `False` にして構いません。
''')

code(r'''
# @title 6-1. ベースダンスの生成
BASE_MODE = "auto"  # @param ["auto", "whole", "chunked"]
BASE_STEPS = 21  # @param {type:"integer"}
CHUNK_WINDOW_SEC = 32.0  # @param {type:"number"}
CHUNK_OVERLAP_SEC = 4.0  # @param {type:"number"}
CHUNK_INPAINT_STEPS = 51  # @param {type:"integer"}
RUN_LENGTH_CHECK = True  # @param {type:"boolean"}
LENGTH_CHECK_SEC = 120.0  # @param {type:"number"}

inp = load_json("inputs.json"); SONG, GENRE, SEED = inp["song"], inp["genre"], inp["seed"]
FD_CKPT = env_state()["FD_CKPT"]; MAMBA_BACKEND = env_state().get("MAMBA_BACKEND", "cuda")
fd = fr.FlowerDance(FD_REPO, FD_CKPT, DEVICE, MAMBA_BACKEND)
feat = fd.music_features(SONG)
cond = feat["features"]
save_pt("cond.pt", {"cond": cond, "fps": feat["fps"], "bpm_librosa": feat["bpm_librosa"]})
print(f"music features: {cond.shape} ({cond.shape[0] / su.FPS:.1f} s)")
win, ovl = int(CHUNK_WINDOW_SEC * su.FPS), int(CHUNK_OVERLAP_SEC * su.FPS)

if RUN_LENGTH_CHECK and len(cond) > fr.TRAIN_HORIZON:
    seg = cond[: int(min(len(cond), LENGTH_CHECK_SEC * su.FPS))]
    res = {}
    for mode in ("whole", "chunked"):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        xn, info = fd.sample_long(seg, GENRE, mode=mode, n_steps=BASE_STEPS, seed=SEED, window=win, overlap=ovl,
                                  inpaint_steps=CHUNK_INPAINT_STEPS)
        st = fr.motion_stats(fd.decode(xn)["positions"])
        res[mode] = st
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else float("nan")
        print(f"{mode:8s}: {info['seconds']:.1f}s, peak VRAM {peak:.2f} GB, "
              f"mean speed {st['mean_speed']:.2f} m/s, foot skate {st['foot_skate_mps']:.3f} m/s")
    viz.plot_speed_windows([res["whole"], res["chunked"]], ["whole", "chunked"])
    print("→ whole の後半で動きが小さくなる・崩れる場合は BASE_MODE='chunked' を使ってください。")

base_norm, info = fd.sample_long(cond, GENRE, mode=BASE_MODE, n_steps=BASE_STEPS, seed=SEED, window=win, overlap=ovl,
                                 inpaint_steps=CHUNK_INPAINT_STEPS)
base = fd.decode(base_norm)
stats = fr.motion_stats(base["positions"])
print(f"base dance: mode={info['mode']}, {info['seconds']:.1f}s, windows={info['windows']}")
print(f"mean speed {stats['mean_speed']:.2f} m/s, foot skate {stats['foot_skate_mps']:.3f} m/s, root travel {stats['root_travel_m']:.1f} m")
if stats["up_axis_cos"] < 0.8:
    print(f"[warn] 体の上方向が +z と一致しません (cos={stats['up_axis_cos']:.2f})。ジェスチャーの埋め込みは z-up を前提にしています。")
save_pt("base.pt", {"base_norm": base_norm, "info": info})
er.export_smpl_pkl(base, f"{WORK}/base_motion_zup.pkl")
viz.plot_speed_windows([stats], ["base"])
fd.free(); del fd; free_gpu()
''')

code(r'''
# @title 6-2. ベースダンスの骨格プレビュー（任意）
PREVIEW_BASE_START_SEC = 0.0  # @param {type:"number"}
PREVIEW_BASE_SEC = 20.0  # @param {type:"number"}
SONG = load_json("inputs.json")["song"]
codec = fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], "cpu", load_model=False)  # 正規化器だけ読む
base = codec.decode(load_pt("base.pt")["base_norm"])
p = er.render_skeleton_video(f"{WORK}/preview_base.mp4", base["positions"], base["rotmat"][:, 0], audio=SONG,
                             start_frame=int(PREVIEW_BASE_START_SEC * su.FPS), n_frames=int(PREVIEW_BASE_SEC * su.FPS),
                             width=480, height=480)
show_video(p)
''')

# =====================================================================================================
md(r'''
## 7. 工程 7〜8：歌詞ジェスチャーの生成と埋め込み

### 工程 7：MotionGPT3 でジェスチャーを生成し、「決め」を切り出す
1. 同じ記述から `N_CANDIDATES` 個の候補を生成します（MotionGPT3 は別プロセス・専用ライブラリで実行）。
2. 手首・肘（`full` は足首と腰も）の速度の**極小点**のうち、直前に大きな動きがあり腕が伸びているものを「決め」とし、候補間で最も明確なものを選びます。
3. 決めの前後（`PRE_BEATS` / `POST_BEATS` 拍、既定 1 拍）を切り出します。
4. **形式変換（最大の難所）**: HumanML3D（22 関節の位置、y-up、20fps）→ FlowerDance（SMPL 24 関節の回転＋ルート位置、30fps）。
   - 20fps → 30fps は関節位置の 3 次スプライン補間
   - 逆運動学: 骨盤と胸は複数の骨方向への Kabsch 当てはめ、それ以外は骨方向への最小回転で初期解 → 骨方向誤差・肘と膝の蝶番（ヒンジ）制約・逆曲がり防止・時間方向の滑らかさで最適化
   - 骨の長さの違いは方向だけを合わせることで吸収し、ルート移動は脚の長さの比で拡大縮小
   - 手首・手・頭は位置情報がないため、ベースダンスのものを残します
5. 可動域を少し誇張（`EXAGGERATION`、候補の最初のフレーム＝自然立ちを基準に回転量を拡大）してダンスらしさを補います。

変換前後の関節位置の誤差は 7-3 で図示します。
''')

code(r'''
# @title 7-1. MotionGPT3 でジェスチャー候補を生成
N_CANDIDATES = 4  # @param {type:"integer"}
GESTURE_GEN_SEC = 4.0  # @param {type:"number"}

script = load_json("script.json"); SEED = load_json("inputs.json")["seed"]
MGPT_CKPT = env_state()["MGPT_CKPT"]
GEST_DIR = f"{WORK}/gestures_raw"
os.makedirs(GEST_DIR, exist_ok=True)
gp.write_requests(script["gestures"], f"{WORK}/gesture_requests.json", n_candidates=N_CANDIDATES,
                  seconds=GESTURE_GEN_SEC)
run_streaming([sys.executable, f"{PKG}/m2d/mgpt_runner.py", "generate", "--repo", MGPT_REPO, "--ckpt", MGPT_CKPT,
               "--requests", f"{WORK}/gesture_requests.json", "--out", GEST_DIR, "--seed", str(SEED)], env=mgpt_env())
free_gpu()
print(sorted(os.listdir(GEST_DIR)))
''')

code(r'''
# @title 7-2. 決めの検出・切り出し・SMPL への変換・誇張
PRE_BEATS = 1.0  # @param {type:"number"}
POST_BEATS = 1.0  # @param {type:"number"}
EXAGGERATION = 1.15  # @param {type:"number"}
IK_ITERS = 300  # @param {type:"integer"}

script = load_json("script.json"); grid = load_json("grid.json")
GEST_DIR = f"{WORK}/gestures_raw"
processed = []
for g in script["gestures"]:
    J, fps = gp.load_candidates(GEST_DIR, g["id"])
    bp = grid["eights"][g["eight"]]["beat_period"]
    r = gp.process_gesture(g, J, bp, pre_beats=PRE_BEATS, post_beats=POST_BEATS, exaggeration=EXAGGERATION,
                           ik_iters=IK_ITERS, device=DEVICE)
    r["label"] = g["lyric"][:20] + " / " + g["motion_description"][:40]
    processed.append(r)
save_pt("gestures.pt", [{k: v for k, v in r.items() if k != "debug"} for r in processed])
display(pd.DataFrame([{"id": r["id"], "scope": r["scope"], "hit_s": round(r["hit_frame"] / su.FPS, 2),
                       "candidate": r["info"]["chosen"], "key_in_cand": r["info"]["key_frame_in_candidate"],
                       "clip_frames": r["rotmat"].shape[0], "IK angle mean°": round(r["info"]["ik_bone_angle_deg_mean"], 2),
                       "IK angle max°": round(r["info"]["ik_bone_angle_deg_max"], 2),
                       "pos err mean cm": round(r["info"]["ik_pos_err_cm_mean"], 2),
                       "pos err max cm": round(r["info"]["ik_pos_err_cm_max"], 2),
                       "mirrored": r["info"]["mirrored"], "label": r["label"]} for r in processed]))
''')

code(r'''
# @title 7-3. 変換の確認（決めの検出・関節ごとの誤差・変換前後の骨格）
# 7-2 を同じセッションで実行した直後に使います（詳細な途中結果はメモリ上にだけあります）。
INSPECT_GESTURE_ID = 0  # @param {type:"integer"}
r = next(x for x in processed if x["id"] == INSPECT_GESTURE_ID)
viz.plot_gesture_debug(r)
conv = r["debug"]["conv"]
print(f"gesture #{r['id']}: analytic IK mean error {conv['errors_analytic']['pos_err_m'].mean() * 100:.2f} cm "
      f"→ refined {conv['errors']['pos_err_m'].mean() * 100:.2f} cm (scale {conv['scale']:.3f}, mirrored {conv['mirrored']})")
# 変換後の候補全体（切り出し前）を骨格動画で確認
R_full = r["debug"]["rotmat_full"]
rot0, pos0 = su.yup_to_zup(R_full[:, 0], conv["root_pos"])
Rz = R_full.clone(); Rz[:, 0] = rot0
p = er.render_skeleton_video(f"{WORK}/preview_gesture_{r['id']}.mp4", su.fk(Rz, pos0), Rz[:, 0], width=360, height=360)
show_video(p, width=360)
''')

md(r'''
### 工程 8：ジェスチャーを拍に合わせてベースダンスに埋め込む

- 決めのフレームが台本の `hit_count` の時刻にぴったり来るように置きます。
- `upper`: 上半身（背骨・首・鎖骨・肩・肘）の回転チャンネルだけを差し替えます。
- `full`: 全関節を差し替えます。ルートは `FULL_ROOT_MODE` で選びます。
  - `"base"`: ルートの位置と向きをベースダンスのものにする（計画書どおり）
  - `"hybrid"`（既定）: **立ち位置（水平位置）と向きはベースダンス**に合わせつつ、ジェスチャー自身の高さの変化（跳ぶ・しゃがむ）、体の傾き、決めからの回転量（回る）を上乗せする
- マスク（工程 9 で使う）:
  - ジェスチャー区間の差し替えたチャンネル → 指定（固定）
  - ジェスチャーの前後 `TRANSITION_BEATS` 拍 → 生成し直す（つなぎと予備動作）
  - それ以外 → ベースダンスを指定（固定）
- `GESTURE_MASK_MODE`: `"soft"` は FlowerDance 標準の**時間減衰ソフトマスク**（生成の終盤ほど拘束が弱まる）、`"strict"` は減衰しない厳密マスク。ジェスチャーが薄まる場合は `"strict"` に変える**1 行の変更**で切り替えられます。
- マスク付き生成は、ジェスチャーの周囲だけの「ウィンドウ」ごとに行います。ウィンドウの両端 `EDGE_LOCK_FRAMES` フレームは常に厳密に固定するので、ウィンドウの外のベースダンスと継ぎ目なくつながります。
''')

code(r'''
# @title 7-4. 埋め込みとマスクの作成（工程 8）
FULL_ROOT_MODE = "hybrid"  # @param ["hybrid", "base"]
TRANSITION_BEATS = 1.5  # @param {type:"number"}
TRANSITION_REGEN = "replaced"  # @param ["replaced", "all"]
GESTURE_MASK_MODE = "soft"  # @param ["soft", "strict"]
BASE_MASK_MODE = "soft"  # @param ["soft", "strict"]
CONTEXT_SEC = 3.0  # @param {type:"number"}
MIN_WINDOW_SEC = 8.0  # @param {type:"number"}
EDGE_LOCK_FRAMES = 10  # @param {type:"integer"}

grid = load_json("grid.json")
codec = fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], "cpu", load_model=False)
base_norm = load_pt("base.pt")["base_norm"]
base = codec.decode(base_norm)
gestures = load_pt("gestures.pt")
T = base_norm.shape[0]
trans = int(round(TRANSITION_BEATS * grid["beat_period"] * su.FPS))
assert EDGE_LOCK_FRAMES <= CONTEXT_SEC * su.FPS, "EDGE_LOCK_FRAMES は CONTEXT_SEC 以下にしてください"
assembled, mask, decay, regions = ge.assemble(base, gestures, trans, full_root_mode=FULL_ROOT_MODE,
                                              gesture_mask_mode=GESTURE_MASK_MODE, base_mask_mode=BASE_MASK_MODE,
                                              transition_regen=TRANSITION_REGEN)
labels = {g["id"]: g.get("label", "") for g in gestures}
for r in regions:
    r["label"] = labels.get(r["id"], "")
windows = ge.edit_windows(regions, T, int(CONTEXT_SEC * su.FPS), int(MIN_WINDOW_SEC * su.FPS), EDGE_LOCK_FRAMES)
mask_w, decay_w = ge.lock_window_edges(mask, decay, windows, EDGE_LOCK_FRAMES, T)
edited_norm, clip_frac = ge.edited_vector(codec, base_norm, assembled)
print(f"{len(regions)} gestures placed, transition {trans} frames, {len(windows)} windows: "
      + ", ".join(f"{s / su.FPS:.1f}-{e / su.FPS:.1f}s" for s, e in windows))
if clip_frac > 0.05:
    print(f"[note] gesture values at the edge of FlowerDance's training range: {clip_frac * 100:.1f}% "
          "(they are clipped by the normaliser; lower EXAGGERATION if the poses look flattened)")
save_pt("edit_inputs.pt", {"assembled": assembled, "mask": mask_w, "decay": decay_w, "regions": regions,
                           "windows": windows, "edited_norm": edited_norm,
                           "settings": {"FULL_ROOT_MODE": FULL_ROOT_MODE, "TRANSITION_BEATS": TRANSITION_BEATS,
                                        "GESTURE_MASK_MODE": GESTURE_MASK_MODE, "BASE_MASK_MODE": BASE_MASK_MODE,
                                        "TRANSITION_REGEN": TRANSITION_REGEN}})
viz.plot_mask_timeline(mask_w, regions, windows)
''')

# =====================================================================================================
md(r'''
## 8. 工程 9：FlowerDance のマスク付き生成でつなぎを作り直す（2 回目）

付属の `inpaint.py` はデータセットの曲用のデモなので、`inference.py` の特徴量計算と `FlowMatching.inpaint` と同じ計算を組み合わせて自前で実装しています（`m2d/flowerdance_runner.py` の `inpaint`）。
要素ごとに「時間減衰ソフトマスク / 厳密マスク」を選べるようにしてあり、ウィンドウの端は常に厳密に固定します。

生成後、各ジェスチャーが完成モーションにどれだけ残っているか（差し替えた関節の回転のずれ）を表示します。ずれが大きい（目安 20° 以上）ときは 8-1 で `GESTURE_MASK_MODE = "strict"` にして 7-4 と 8-1 をやり直してください。
''')

code(r'''
# @title 8-1. マスク付き生成
INPAINT_STEPS = 101  # @param {type:"integer"}

inp = load_json("inputs.json"); GENRE, SEED = inp["genre"], inp["seed"]
ed = load_pt("edit_inputs.pt"); cond = load_pt("cond.pt")["cond"]; base_norm = load_pt("base.pt")["base_norm"]
fd = fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], DEVICE, env_state().get("MAMBA_BACKEND", "cuda"))
t0 = time.time()
final_norm = ge.run_edit(fd, base_norm, ed["edited_norm"], ed["mask"], ed["decay"], ed["windows"], cond, GENRE,
                         n_steps=INPAINT_STEPS, seed=SEED + 7)
print(f"masked generation: {time.time() - t0:.1f}s")
final = fd.decode(final_norm)
fd.free(); del fd; free_gpu()
save_pt("final.pt", {"final_norm": final_norm})

joints_by_scope = {"upper": su.UPPER_JOINTS_DEFAULT, "full": su.FULL_JOINTS_DEFAULT}
adh = viz.gesture_adherence(final["rotmat"], ed["assembled"]["rotmat"], ed["regions"], joints_by_scope)
display(pd.DataFrame(adh))
if adh and np.mean([a["mean_dev_deg"] for a in adh]) > 20:
    print("→ ジェスチャーが薄まっています。7-4 で GESTURE_MASK_MODE = 'strict' にしてやり直すと、形がそのまま残ります。")
jumps = viz.boundary_jumps(final["positions"], ed["windows"])
print("speed at window borders (m/s):", [round(j["speed"], 2) for j in jumps], "/ median", round(jumps[0]["median_speed"], 2) if jumps else "-")
st_b = fr.motion_stats(fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], "cpu", load_model=False).decode(base_norm)["positions"])
st_f = fr.motion_stats(final["positions"])
viz.plot_speed_windows([st_b, st_f], ["base", "final"])
''')

# =====================================================================================================
md(r'''
## 9. 工程 10：書き出し・プレビュー・ダウンロード

| ファイル | 内容 |
|---|---|
| `motion_zup.pkl` | **FlowerDance の推論出力と同じ形式**: `smpl_poses` (T, 72 軸角), `smpl_trans` (T, 3), `full_pose` (T, 24, 3)。z-up、30fps |
| `motion_yup_aistpp.pkl` | SMPL 本来の y-up に戻したもの（AIST++ 形式: `smpl_poses`, `smpl_trans`, `smpl_scaling`）＋ `full_pose` |
| `motion.bvh` | SMPL 骨格の BVH（y-up、cm）。Blender などで読み込めます |
| `joints_zup.npy` | 24 関節の位置 (T, 24, 3) |
| `base_motion_zup.pkl` | 比較用：ジェスチャーを入れる前のベースダンス |
| `script.json` ほか | 振付台本、歌詞の時刻、8 カウント、ビート、設定 |
| `preview_skeleton.mp4` | 骨格アニメーション（音声つき、薄い骨格 = ベースダンス） |
''')

code(r'''
# @title 9-1. 書き出しと骨格アニメーション
PREVIEW_START_SEC = 0.0  # @param {type:"number"}
PREVIEW_DURATION_SEC = 0.0  # @param {type:"number"}
SHOW_BASE_GHOST = True  # @param {type:"boolean"}

OUT = f"{WORK}/outputs"
os.makedirs(OUT, exist_ok=True)
codec = fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], "cpu", load_model=False)
final = codec.decode(load_pt("final.pt")["final_norm"])
base = codec.decode(load_pt("base.pt")["base_norm"])
ed = load_pt("edit_inputs.pt")
er.export_smpl_pkl(final, f"{OUT}/motion_zup.pkl", "zup")
er.export_smpl_pkl(final, f"{OUT}/motion_yup_aistpp.pkl", "yup")
er.export_bvh(final, f"{OUT}/motion.bvh")
np.save(f"{OUT}/joints_zup.npy", final["positions"].numpy().astype(np.float32))
er.export_smpl_pkl(base, f"{OUT}/base_motion_zup.pkl", "zup")
for name in ("script.json", "lyrics_lines.json", "lyrics_words.json", "grid.json", "beats.json", "inputs.json",
             "lyrics_lines.csv", "lyrics_words.csv"):
    if os.path.exists(f"{WORK}/{name}"):
        shutil.copy(f"{WORK}/{name}", f"{OUT}/{name}")
json.dump({"settings": ed["settings"], "regions": [{k: v for k, v in r.items() if k != "channels"} for r in ed["regions"]],
           "windows": ed["windows"]}, open(f"{OUT}/edit_summary.json", "w"), ensure_ascii=False, indent=1, default=float)

lines = load_json("lyrics_lines.json"); grid = load_json("grid.json")
s0 = int(PREVIEW_START_SEC * su.FPS)
n = None if PREVIEW_DURATION_SEC <= 0 else int(PREVIEW_DURATION_SEC * su.FPS)
overlay = er.make_overlay_fn(grid, lines, ed["regions"], start_frame=s0)
t0 = time.time()
p = er.render_skeleton_video(f"{OUT}/preview_skeleton.mp4", final["positions"], final["rotmat"][:, 0],
                             audio=load_json("inputs.json")["song"], start_frame=s0, n_frames=n, overlay=overlay,
                             ghost_positions=base["positions"] if SHOW_BASE_GHOST else None,
                             progress=lambda i, k: print(f"  {i}/{k} frames"))
print(f"skeleton preview: {time.time() - t0:.0f}s")
show_video(p)
''')

code(r'''
# @title 9-2. 成果物を ZIP でダウンロード
zip_path = shutil.make_archive(f"{WORK}/music2dance_outputs", "zip", f"{WORK}/outputs")
print(zip_path, f"{os.path.getsize(zip_path) / 2**20:.1f} MB")
from google.colab import files
files.download(zip_path)
''')

# =====================================================================================================
md(r'''
## 10. 工程 11：プレビュー動画（マネキンでダンスを描写）

関節位置からマネキン（丸みのある円錐と球を滑らかにつないだ人形）を作り、PyTorch の GPU レイマーチングで描画します（追加のレンダラーや SMPL のモデルファイルは不要）。床の市松模様と柔らかい影、歌詞・カウント・ジェスチャー名の表示つきです。
''')

code(r'''
# @title 10-1. マネキン動画
MANNEQUIN_RES = 640  # @param {type:"integer"}
MANNEQUIN_START_SEC = 0.0  # @param {type:"number"}
MANNEQUIN_DURATION_SEC = 0.0  # @param {type:"number"}
SHOW_OVERLAY = True  # @param {type:"boolean"}

codec = fr.FlowerDance(FD_REPO, env_state()["FD_CKPT"], "cpu", load_model=False)
final = codec.decode(load_pt("final.pt")["final_norm"])
ed = load_pt("edit_inputs.pt"); lines = load_json("lyrics_lines.json"); grid = load_json("grid.json")
s0 = int(MANNEQUIN_START_SEC * su.FPS)
n = None if MANNEQUIN_DURATION_SEC <= 0 else int(MANNEQUIN_DURATION_SEC * su.FPS)
overlay = er.make_overlay_fn(grid, lines, ed["regions"], start_frame=s0) if SHOW_OVERLAY else None
t0 = time.time()
p = er.render_mannequin_video(f"{WORK}/outputs/preview_mannequin.mp4", final, audio=load_json("inputs.json")["song"],
                              start_frame=s0, n_frames=n, overlay=overlay, width=MANNEQUIN_RES, height=MANNEQUIN_RES,
                              device=DEVICE, progress=lambda i, k: print(f"  {i}/{k} frames"))
free_gpu()
print(f"mannequin preview: {time.time() - t0:.0f}s")
show_video(p)
''')

code(r'''
# @title 10-2. マネキン動画を含めて ZIP をダウンロード
zip_path = shutil.make_archive(f"{WORK}/music2dance_outputs", "zip", f"{WORK}/outputs")
print(zip_path, f"{os.path.getsize(zip_path) / 2**20:.1f} MB")
from google.colab import files
files.download(zip_path)
''')

md(r'''
## 困ったとき

| 症状 | 対処 |
|---|---|
| 1-3 で「ビルド済みホイールが見つかりません」 | Colab の Python / PyTorch の組み合わせにホイールがありません。自動で純 PyTorch 版 Mamba に切り替わります（生成が遅くなるだけで結果は同等）。 |
| 6-1 で GPU メモリ不足 | `BASE_MODE="chunked"`、または `CHUNK_WINDOW_SEC` を小さくします。 |
| 曲の後半で動きが小さくなる | `RUN_LENGTH_CHECK=True` で比較し、`BASE_MODE="chunked"` を使います。 |
| 歌詞のタイミングがずれる | 4-2b で聴いて確認し、4-3 の `LINE_TIME_OVERRIDES` で行の開始・終了を直します。 |
| 8 カウントの「1」がずれている | 4-5 で `GRID_PHASE` を `0` / `1` に固定します。 |
| ジェスチャーが薄い・見えない | 7-4 で `GESTURE_MASK_MODE="strict"`、または `EXAGGERATION` を上げます。 |
| ジェスチャーが不自然・跳ねる | 7-3 で変換結果を確認し、5-2 で `motion_description` を具体的に書き直すか、`scope` を `upper` にします。 |
| OpenRouter のモデルが無い | 3-3 に表示される候補から `OPENROUTER_MODEL` を選び直します。 |
| MotionGPT3 のダウンロードに失敗 | Hugging Face の `OpenMotionLab/motiongpt3` から `.ckpt` を手動で取得し、`/content/models/motiongpt3/` に置いてから 2-2 を再実行します。 |

**ライセンス**: FlowerDance・MotionGPT3・Demucs・beat_this・WhisperX・wav2vec2 の各モデル／コードはそれぞれのライセンスに従ってください。同梱した PyTorch3D の回転変換は BSD ライセンスです。
''')


# =====================================================================================================
def build():
    nb = {"cells": [], "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": [], "toc_visible": True, "name": "music_to_dance.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    for i, (kind, src) in enumerate(cells):
        lines = src.split("\n")
        source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {"id": f"cell{i:03d}"}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(cells)} cells")


if __name__ == "__main__":
    build()
