# music2dance

歌詞＋音楽から、歌詞の意味と音楽のリズムの両方を反映した 3D ダンスモーション（SMPL 形式）を生成する Google Colab ノートブックです。

- **ノートブック本体**: [`music_to_dance.ipynb`](music_to_dance.ipynb)（Colab で開き、GPU は L4 / A100 を推奨）
- 必要なシークレット: `OPENROUTER_API_KEY`（必須）、`HF_TOKEN`（任意）

## パイプライン

| # | 工程 | モデル |
|---|---|---|
| 1 | ボーカル分離 | Demucs（htdemucs） |
| 2 | 歌詞の時刻合わせ | wav2vec2（日本語）＋ WhisperX の整列機能 |
| 3 | ビート・ダウンビート | beat_this |
| 4 | 8 カウントのグリッド | — |
| 5 | 振付台本 | LLM（OpenRouter） |
| 6 | ベースダンス（1 回目） | FlowerDance |
| 7 | 歌詞ジェスチャー | MotionGPT3 ＋ 逆運動学（HumanML3D → SMPL） |
| 8 | 拍に合わせた埋め込み | — |
| 9 | つなぎの作り直し（2 回目） | FlowerDance のマスク付き生成 |
| 10 | 書き出し・骨格アニメーション | — |
| 11 | マネキン動画 | PyTorch の GPU レイマーチング |

すべて学習済みモデルの推論のみです。Google Drive は使わず、モデルと中間データは `/content` に置き、成果物（SMPL `.pkl`、BVH、関節位置、プレビュー動画など）は ZIP でダウンロードします。ソースからのビルドは行いません（mamba-ssm / causal-conv1d はビルド済みホイール、PyTorch3D は回転変換ファイルのみ同梱）。

## リポジトリ構成

| パス | 内容 |
|---|---|
| `music_to_dance.ipynb` | Colab ノートブック（下の `m2d/` と `shims/` を `%%writefile` セルとして同梱した自己完結版） |
| `m2d/` | パイプラインのコード（ノートブックが `/content/m2d_pkg/m2d` に書き出すものと同一） |
| `shims/pytorch3d/` | PyTorch3D の `transforms` の代替（`rotation_conversions.py` は PyTorch3D から BSD ライセンスで同梱） |
| `tools/build_notebook.py` | `m2d/`・`shims/` とセルの文面からノートブックを生成するスクリプト |
| `tests/` | CPU で動く単体テスト（`pytest tests`） |

コードを変更したら `python tools/build_notebook.py` でノートブックを再生成してください。

## 開発時のテスト

```bash
pip install torch==2.7.1 numpy scipy librosa soundfile pykakasi opencv-python-headless pillow pytest
pytest tests
```

単体テストは、純 PyTorch 版 Mamba と参照実装の一致、逆運動学の往復誤差（体型の違い・肘膝のねじれを含む）、鏡像の判定、「決め」の検出、CTC 強制整列とかな読みへの置き換え、8 カウント（弱起・変則小節）、台本の検証、書き出し形式を確認します。
