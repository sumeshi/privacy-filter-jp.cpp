# CONTRIBUTING

## ビルドとテスト

```sh
cmake --preset release && cmake --build --preset release -j
ctest --preset release            # モデル不要のテストのみ実行される
```

CI と配布バイナリは `release-portable` プリセット（全 CPU ISA バリアント + 実行時ディスパッチ）。

## データセット生成・検証・学習

手順はすべて [docs/finetuning-jp.md](docs/finetuning-jp.md) にある。要点だけ:

- 生成スクリプトは `datasets/jp-data/scripts/`（submodule）。生成後は必ず
  `validate_jsonl.py` で offset・ラベル・ベンチマークへのリークを検査する。
- 学習は追加学習ではなく、ベースモデルからフルミックスで 1 回だけ回す（one-shot）。
- `datasets/benchmark/challenge2.jsonl` はブラインド評価用。チューニング中に
  行単位のエラーを見ない。診断には `eval2.jsonl` を使う。
- 評価は `scripts/run_pf_hf.py` → `scripts/eval_spans.py`。

Python 環境は `scripts/requirements.txt`（transformers / torch など）。

## GGUF 変換

```sh
python scripts/convert.py --model <ft-checkpoint-dir> \
  --outfile privacy-filter-jp-f16.gguf --name privacy-filter-jp
python scripts/requant_q8.py --in privacy-filter-jp-f16.gguf \
  --out privacy-filter-jp-q8.gguf        # experts-only Q8_0 (~1.6GB)
```

`convert.py` は `gguf` パッケージが必要（transformers は不要）。

## モデルのアップロード（Hugging Face）

`scripts/publish_hf.py` を使う。**デフォルトは dry-run**（何もアップロードしない）。
バージョン管理されたモデルカード `model-cards/privacy-filter-jp.md` が HF 側の
README.md として一緒に同期されるので、カードを更新してからアップロードする。

```sh
# dry-run（内容・sha256 の確認だけ）
python scripts/publish_hf.py --model privacy-filter-jp \
  --gguf <path>/privacy-filter-jp-f16.gguf --repo sumeshi/privacy-filter-jp-GGUF

# 本番push（f16 → q8 の順に。ローカルのファイル名は公開名に自動でマップされる）
python scripts/publish_hf.py --model privacy-filter-jp \
  --gguf <path>/privacy-filter-jp-f16.gguf --repo sumeshi/privacy-filter-jp-GGUF --upload
python scripts/publish_hf.py --model privacy-filter-jp --quant q8 \
  --gguf <path>/privacy-filter-jp-q8.gguf --repo sumeshi/privacy-filter-jp-GGUF --upload

# カードだけ直したとき
python scripts/publish_hf.py --model privacy-filter-jp --card-only \
  --repo sumeshi/privacy-filter-jp-GGUF --upload
```

備考:

- `--repo sumeshi/privacy-filter-jp-GGUF` を必ず付ける（スクリプトのデフォルト org は
  LocalAI-io）。
- write 権限のトークンがあるのに 401 になるときは、環境変数
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` が原因。`HF_HUB_DISABLE_IMPLICIT_TOKEN=0` を
  コマンドの前に付けて上書きする。
- WSL の `/mnt/` 上に `HF_HOME` があると hf_xet が
  `I/O error: No such file or directory` で落ちる。`HF_HUB_DISABLE_XET=1` を付けて
  クラシックアップロードにする。

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=0 HF_HUB_DISABLE_XET=1 \
  python scripts/publish_hf.py --model privacy-filter-jp --gguf ... --repo sumeshi/privacy-filter-jp-GGUF --upload
```

## デモ

トレース生成（実際に pf-cli でモデルを回して scene を作る）→ 再生の 2 段階。

```sh
# 1. scene を再生成（モデルや文書を変えたら実行）
python demo/gen_scan.py --cli build/release/pf-cli \
  --model <path>/privacy-filter-jp-f16.gguf \
  --doc demo/scan_doc_ja.txt --scene demo/traces/scan_ja \
  --threshold 0.5 --label privacy-filter-jp.cpp --device "CPU · privacy-filter-jp 2026-07-03 f16"

# 2. ターミナルで再生（標準ライブラリのみで動く。--dilate は時間スケール）
python demo/pii_scan.py --scene demo/traces/scan_ja --dilate 4
```

動画化は `demo/make_scan.sh`（recorder-for-agents が必要。`RECORDER=<path>` で指定）。

## リリース（配布バイナリ）

GitHub Release を発行すると `.github/workflows/release.yml` が
Linux `.tar.gz` / Windows `.zip`（pf-cli + ggml ライブラリ同梱）をビルドして
Release に添付する。事前確認は Actions から release.yml を `workflow_dispatch` で
手動実行（Artifact にだけ上がる）。

リリース手順: モデルカード・README のベンチ数値を更新 → コミット & CI green を確認
→ HF へ GGUF をアップロード → タグを切って Release 発行。
