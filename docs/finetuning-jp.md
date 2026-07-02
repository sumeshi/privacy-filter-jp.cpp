# 日本語 fine-tuning 方針

`privacy-filter-jp.cpp` の目的は、元の `privacy-filter.cpp` が持つ 8 カテゴリの PII 検知能力を維持しつつ、日本語住所・日本語氏名・日本語業務文脈の span 境界を fine-tuning で強化することです。

## 対象ラベル

- `private_person`
- `private_address`
- `private_email`
- `private_phone`
- `private_date`
- `account_number`
- `private_url`
- `secret`

強化の中心は `private_person` と `private_address`。その他 6 ラベルは、元モデルの能力維持と日本語文脈での回帰確認が目的です。

## 方針

- 実在 PII は使わない。
- 学習データ生成・外部データ変換スクリプトは `datasets/jp-data/` submodule で管理する。
- 生成済み JSONL、外部ソースのローカルコピー、checkpoint はリポジトリへ含めない（`datasets/jp-data/generated/`, `datasets/jp-data/raw/`, `runs/` は ignore 対象）。
- `private_email` 等 6 ラベルは、base model の既存 BIOES ラベル（`EMAIL`, `PHONE`, `URL`, `DATE`, `BANKACCOUNT`, `PASSWORD` など）に alias して既存 classifier row を再利用する。新規重複ラベルは追加しない（`scripts/finetune_jp.py`）。
- 住所・氏名だけに寄せすぎず、structured PII の合成データも混ぜる。
- `challenge` split を見て調整した場合、その split は blind held-out score として扱わない。

## データソース

| 対象 | ソース | スクリプト |
|---|---|---|
| `private_person` | Stockmark `ner-wikipedia-dataset`（CC-BY-SA 3.0） | `datasets/jp-data/scripts/convert_stockmark_ner.py` |
| `private_address` | 日本郵便の郵便番号データ（実在地名）+ 架空の丁目・番地・建物名 | `datasets/jp-data/scripts/gen_synthetic_address_jp.py` |
| `private_date` / `account_number` | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_date_id_jp.py` |
| structured PII | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_structured_pii_jp.py` |
| 境界回帰ケース | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_boundary_jp.py` |

Stockmark 由来の変換済み JSONL はこのリポジトリでも submodule でも再配布しない。必要な環境でスクリプトを実行して生成する。

## データ生成

```sh
git submodule update --init datasets/jp-data
mkdir -p datasets/jp-data/generated runs/pf-jp

python3 datasets/jp-data/scripts/gen_synthetic_address_jp.py \
  --out datasets/jp-data/generated/synthetic_address_jp_4000.jsonl \
  -n 4000 --seed 3

python3 datasets/jp-data/scripts/gen_synthetic_date_id_jp.py \
  --out datasets/jp-data/generated/synthetic_date_id_jp.jsonl \
  -n 2000 --seed 1

python3 datasets/jp-data/scripts/gen_synthetic_structured_pii_jp.py \
  --out datasets/jp-data/generated/synthetic_structured_pii_jp.jsonl \
  --per-label 1200 --seed 2

python3 datasets/jp-data/scripts/gen_synthetic_boundary_jp.py \
  --out datasets/jp-data/generated/synthetic_boundary_jp.jsonl \
  -n 3000 --seed 4

python3 datasets/jp-data/scripts/convert_stockmark_ner.py \
  --out datasets/jp-data/generated/stockmark_ner_jp.jsonl \
  --require-entity
```

各生成スクリプトの `--out` は省略可能です。省略時は script 位置を基準に `datasets/jp-data/generated/` へ出力します。

## Fine-tuning

```sh
python3 scripts/finetune_jp.py --base-model <base-hf-checkpoint> \
  --data datasets/benchmark/train.jsonl datasets/jp-data/generated/*.jsonl \
  --out runs/pf-jp/model-ft \
  --epochs 1 --lr 1e-4 --batch-size 8 --max-length 96
```

## 評価

```sh
python3 scripts/run_pf_hf.py --model runs/pf-jp/model-ft \
  --input datasets/benchmark/challenge.jsonl \
  --out runs/pf-jp/challenge.pred.jsonl

python3 scripts/eval_spans.py \
  --gold datasets/benchmark/challenge.jsonl \
  --pred runs/pf-jp/challenge.pred.jsonl
```

## GGUF 変換と公開

```sh
python3 scripts/convert.py \
  --model runs/pf-jp/model-ft \
  --outfile runs/pf-jp/privacy-filter-jp-f16.gguf \
  --name privacy-filter-jp

python3 scripts/publish_hf.py \
  --model privacy-filter-jp \
  --gguf runs/pf-jp/privacy-filter-jp-f16.gguf \
  --repo <org-or-user>/privacy-filter-jp-GGUF \
  --upload
```

## サニティチェック

```sh
python3 -m json.tool label_space/jp-basic.json >/dev/null
python3 -m py_compile \
  scripts/finetune_jp.py scripts/run_pf_hf.py scripts/eval_spans.py \
  datasets/jp-data/scripts/gen_synthetic_address_jp.py \
  datasets/jp-data/scripts/gen_synthetic_date_id_jp.py \
  datasets/jp-data/scripts/gen_synthetic_structured_pii_jp.py \
  datasets/jp-data/scripts/gen_synthetic_boundary_jp.py \
  datasets/jp-data/scripts/convert_stockmark_ner.py
```

## JSONL と offset

```jsonl
{"text":"山田太郎さんの住所は東京都千代田区丸の内1丁目99番99号です。","spans":[{"start":0,"end":4,"label":"private_person"},{"start":10,"end":29,"label":"private_address"}]}
```

- offset は Unicode 文字単位（`start` は含む、`end` は含まない）。UTF-8 byte offset ではない。
- Runtime の C API / CLI が返す offset は upstream と同じ UTF-8 byte offset なので、学習・評価データの offset と混同しない。

## Span 境界方針

- 住所 span には住所本体を含める。`住所：` 等のラベル文字列・郵便番号/建物名/部屋番号は自然な場合は含める。
- 氏名 span には氏名本体のみ含める。敬称・役職・会社名・部署名は含めない。

## 評価とベンチマーク split

`datasets/benchmark/{train,eval,challenge}.jsonl` の 3 split を使う。`challenge` は最終確認用で、学習や閾値調整には使わない。今回の初期版は例外的に `challenge` の誤りを boundary 改善の診断に使ったため、現行の結果は blind held-out score ではなく smoke/回帰確認値として扱う。数値・免責事項は [README.md](../README.md) と HF 上のモデルカードを参照。

## 注意事項

- 完全な匿名化を保証しない。
- 英語や一般 multilingual の精度はこのフォークではベンチマークしていない（base model 由来）。
