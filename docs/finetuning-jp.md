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
- 純テンプレート生成・手書き架空ケースの JSONL は `datasets/jp-data/generated/` で再配布する。
- `private_email` 等 6 ラベルは、base model の既存 BIOES ラベル（`EMAIL`, `PHONE`, `URL`, `DATE`, `BANKACCOUNT`, `PASSWORD` など）に alias して既存 classifier row を再利用する。新規重複ラベルは追加しない（`scripts/finetune_jp.py`）。
- 日本特有の識別番号（マイナンバー・運転免許証番号・旅券番号・基礎年金番号・保険証記号番号・在留カード番号・口座番号）は新ラベルを作らず `account_number` に集約する。ラベル空間を変えないので、モデルサイズ・推論速度は変わらない。
- 住所・氏名だけに寄せすぎず、structured PII の合成データも混ぜる。
- 短文だけでなく、複数 PII が混在する長文ビジネス文書（メール署名・申込フォーム・議事録・対応ログ等、150〜600 文字）も学習させる。Runtime は halo-window で長文を処理するため、学習分布を実文書に近づける。
- PII を含まない負例（金額・型番・バージョン・エラーコード等の数値を含む業務文）を混ぜ、過検知を抑える。日付は base model の `DATE` 挙動と矛盾しないよう負例から除外する。
- 追加学習を重ねるのではなく、ベースモデルからフルミックスで 1 回だけ fine-tuning する（one-shot）。
- `challenge` split を見て調整した場合、その split は blind held-out score として扱わない。

## データソース

| 対象 | ソース | スクリプト |
|---|---|---|
| `private_person` | Stockmark `ner-wikipedia-dataset`（CC-BY-SA 3.0） | `datasets/jp-data/scripts/convert_stockmark_ner.py` |
| `private_person`（一般氏名・フリガナ・連名・ローマ字） | 純テンプレート生成（`pii_values.py` の氏名プール） | `datasets/jp-data/scripts/gen_synthetic_person_jp.py` |
| `private_address` | 日本郵便の郵便番号データ（実在地名）+ 架空の丁目・番地・建物名 | `datasets/jp-data/scripts/gen_synthetic_address_jp.py` |
| `private_date` / `account_number` | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_date_id_jp.py` |
| `private_phone`（日本の全番号形式） / `account_number`（日本特有 ID） | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_phone_id_jp.py` |
| structured PII | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_structured_pii_jp.py` |
| 長文複合文書（メール・フォーム・議事録等、複数 PII） | テンプレート合成（住所は生成済み住所 JSONL を再利用） | `datasets/jp-data/scripts/gen_synthetic_docs_jp.py` |
| 負例（PII なし業務文） | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_negative_jp.py` |
| ログ / .env / key=value 形式の PII | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_log_pii_jp.py` |
| 境界回帰ケース | 純テンプレート生成 | `datasets/jp-data/scripts/gen_synthetic_boundary_jp.py` |

共有の架空 PII 値ライブラリは `datasets/jp-data/scripts/pii_values.py`（氏名 79×81 プール、電話番号は携帯/固定 2〜4 桁局番/050/0120/0570/+81/全角/括弧/区切りなし、日本特有 ID 8 種）。生成 JSONL の検証は `datasets/jp-data/scripts/validate_jsonl.py`（offset・ラベル・重複・ベンチマークへのリーク検査）。

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
  -n 3000 --seed 4 \
  --exclude datasets/benchmark/train.jsonl datasets/benchmark/eval.jsonl \
    datasets/benchmark/challenge.jsonl datasets/benchmark/eval2.jsonl \
    datasets/benchmark/challenge2.jsonl

python3 datasets/jp-data/scripts/convert_stockmark_ner.py \
  --out datasets/jp-data/generated/stockmark_ner_jp.jsonl \
  --require-entity

python3 datasets/jp-data/scripts/gen_synthetic_phone_id_jp.py -n 4000 --seed 11
python3 datasets/jp-data/scripts/gen_synthetic_person_jp.py -n 4000 --seed 12
python3 datasets/jp-data/scripts/gen_synthetic_negative_jp.py -n 2000 --seed 13
python3 datasets/jp-data/scripts/gen_synthetic_docs_jp.py -n 3000 --seed 14 \
  --address-file datasets/jp-data/generated/synthetic_address_jp.jsonl
python3 datasets/jp-data/scripts/gen_synthetic_log_pii_jp.py -n 1500 --seed 15

# 生成後の検証（offset・ラベル・ベンチマークへのリーク）
python3 datasets/jp-data/scripts/validate_jsonl.py \
  datasets/jp-data/generated/*.jsonl \
  --against datasets/benchmark/eval2.jsonl datasets/benchmark/challenge2.jsonl
```

`gen_synthetic_docs_jp.py` は `--address-file` の住所プールを使うため、住所生成の後に実行する（省略時は組み込みの架空住所にフォールバック）。

純合成データは `datasets/jp-data/generated/` に同梱済みなので、fine-tuning の smoke run だけなら追加生成は不要です。住所データと Stockmark 変換データを使う場合だけ、上記スクリプトでローカル生成します。各生成スクリプトの `--out` は省略可能です。省略時は script 位置を基準に `datasets/jp-data/generated/` へ出力します。

## Fine-tuning

ベースモデルからフルミックスで 1 回だけ学習する（既存 fine-tune の継続学習はしない）。`synthetic_address_jp_4000.jsonl` は 8000 版と重複クラスなので混ぜない。`--max-length 384` は長文複合文書（最大 ~470 文字 ≒ 0.6 token/文字）をカバーする値。

```sh
python3 scripts/finetune_jp.py --base-model <base-hf-checkpoint> \
  --data datasets/benchmark/train.jsonl \
    datasets/jp-data/generated/synthetic_address_jp.jsonl \
    datasets/jp-data/generated/synthetic_boundary_jp.jsonl \
    datasets/jp-data/generated/synthetic_date_id_jp.jsonl \
    datasets/jp-data/generated/synthetic_structured_pii_jp.jsonl \
    datasets/jp-data/generated/stockmark_ner_jp.jsonl \
    datasets/jp-data/generated/synthetic_person_jp.jsonl \
    datasets/jp-data/generated/synthetic_phone_id_jp.jsonl \
    datasets/jp-data/generated/synthetic_docs_jp.jsonl \
    datasets/jp-data/generated/synthetic_negative_jp.jsonl \
    datasets/jp-data/generated/synthetic_log_pii_jp.jsonl \
  --out runs/pf-jp/model-ft \
  --epochs 3 --lr 1e-4 --batch-size 8 --max-length 384
```

12GB VRAM で OOM する場合は `--batch-size 4` に下げる。

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
  datasets/jp-data/scripts/pii_values.py \
  datasets/jp-data/scripts/gen_synthetic_address_jp.py \
  datasets/jp-data/scripts/gen_synthetic_date_id_jp.py \
  datasets/jp-data/scripts/gen_synthetic_structured_pii_jp.py \
  datasets/jp-data/scripts/gen_synthetic_boundary_jp.py \
  datasets/jp-data/scripts/gen_synthetic_person_jp.py \
  datasets/jp-data/scripts/gen_synthetic_phone_id_jp.py \
  datasets/jp-data/scripts/gen_synthetic_docs_jp.py \
  datasets/jp-data/scripts/gen_synthetic_negative_jp.py \
  datasets/jp-data/scripts/validate_jsonl.py \
  datasets/jp-data/scripts/make_benchmark_v2.py \
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

なお、初期版の `synthetic_boundary_jp.jsonl` は旧 `eval`/`challenge` と同一テキストを含んでいた（train-test 汚染）ことが後から判明した。旧 split に対する公表値はこの点でも楽観的であり、smoke 値として扱う。現行の boundary 生成器は重複排除と `--exclude` によるベンチマーク除外を行う。

v2 として手作りの `datasets/benchmark/{eval2,challenge2}.jsonl`（各 50 行超、全 8 ラベル・長文複合文書・電話フォーマット変種・日本特有 ID・フリガナ・負例・境界トラップを収録。ソースは `datasets/jp-data/scripts/make_benchmark_v2.py`）を追加した。調整・エラー分析には `eval2` を使い、`challenge2` は blind held-out として行単位の誤り確認をしない。旧 3 split は回帰確認用に残す。ベンチマークと学習データの重複は `validate_jsonl.py --against` で機械的に排除している。

## 注意事項

- 完全な匿名化を保証しない。
- 英語や一般 multilingual の精度はこのフォークではベンチマークしていない（base model 由来）。
