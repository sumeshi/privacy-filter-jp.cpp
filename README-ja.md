# privacy-filter-jp.cpp

[English](README.md) | 日本語

日本語の PII（個人識別情報）検出に特化したファインチューニングモデルと推論エンジン。

![日本語文書をスキャンして PII をリアルタイムにマスクする様子](demo/out/pfj.gif)

## Overview

[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) は、入力文字列に含まれる個人情報の範囲（span）をラベル付きで返す、PII 検出用のトークン分類モデルです。  
[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp) は、この OpenAI Privacy Filter をファインチューニングしたモデルを C++/GGML 上で手軽に動かすための推論エンジンです。

本フォークは、上記のプロダクトを日本語向けにファインチューニングしたものです。元モデルの検知能力を維持したまま、日本語の住所・氏名、および申込書・請求先・配送先といった業務文脈での検出精度を改善します。対象のラベルは氏名・住所・メールアドレス・電話番号など 8 種類です（[Labels](#labels)）。

注意: これは実験的な初期リリースであり、完全な匿名化や実運用水準の精度を保証するものではありません。

## Features

- 元モデルに加えて、日本語の住所・氏名、および業務文脈での検出精度を強化
- メール・電話番号・URL・識別番号・secret など、元モデルの検知能力を維持
- CPU / GPU（CUDA・Vulkan）でローカルに完結し、外部 API が不要
- 他言語からバインドできる C API（[`include/pf.h`](include/pf.h)）

## Installation

ソースを取得してビルドします。

```sh
git clone --recursive https://github.com/sumeshi/privacy-filter-jp.cpp
cd privacy-filter-jp.cpp
cmake --preset release && cmake --build --preset release -j
```

GPU を利用する場合は `-DPF_VULKAN=ON` または `-DPF_CUDA=ON` を付与します。

ファインチューニング済みモデル（GGUF）は Hugging Face から取得します。

```sh
pip install huggingface_hub
huggingface-cli download sumeshi/privacy-filter-jp-GGUF privacy-filter-jp-f16.gguf --local-dir .
```

## Usage

CLI で分類します。`--classify` の第 2 引数は検出のしきい値で、末尾でデバイスを指定できます。

```sh
build/release/pf-cli --info privacy-filter-jp-f16.gguf
echo "配送先：〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室" | \
  build/release/pf-cli --classify privacy-filter-jp-f16.gguf 0.5   # [cpu|cuda|vulkan]
```

プログラムから利用する場合は、[`include/pf.h`](include/pf.h) のフラット C API を使います。`pf_ctx` は不透明ハンドラで、バッファは呼び出し側が所有し、例外は API 境界を越えません。

```c
#include "pf.h"
#include <string.h>
#include <stdio.h>

pf_ctx * ctx = pf_load("privacy-filter-jp-f16.gguf", NULL, 0);
if (pf_last_error(ctx)) { fprintf(stderr, "%s\n", pf_last_error(ctx)); return 1; }

const char * text = "お問い合わせは山田太郎（taro@example.com）まで";
pf_entity * ents = NULL;
size_t n = 0;
if (pf_classify(ctx, text, strlen(text), /*threshold=*/0.5f, &ents, &n) == 0) {
    for (size_t i = 0; i < n; i++)
        printf("%-12s [%d,%d) %.2f  %.*s\n", ents[i].label, ents[i].start,
               ents[i].end, ents[i].score, ents[i].end - ents[i].start,
               text + ents[i].start);
}
pf_entities_free(ents, n);
pf_free(ctx);
```

`pf_classify` は検出した span（バイトオフセット・スコア・ラベル）を返します。しきい値未満の span は除外され、結果は `pf_entities_free` で解放します。

## Labels

対象とする 8 ラベルは次のとおりです。定義は [`label_space/jp-basic.json`](label_space/jp-basic.json) にあります。

| ラベル | 対象 |
|---|---|
| `private_person` | 個人名 |
| `private_address` | 住所 |
| `private_date` | 日付 |
| `private_email` | メールアドレス |
| `private_phone` | 電話番号 |
| `private_url` | URL |
| `account_number` | 顧客番号・契約番号などの識別子 |
| `secret` | API キー・トークンなど |

## Benchmark

小規模な回帰確認用データセット（`datasets/benchmark/`）での span 完全一致 F1 を、ファインチューニング前後で比較した結果です。

| ラベル | Before | After |
|---|---:|---:|
| `private_person` | 0.583 | 0.917 |
| `private_address` | 0.400 | 1.000 |
| その他 6 ラベル（回帰確認） | 0.400 | 0.833 |
| 全体（micro F1） | 0.475 | 0.929 |

このベンチマークは動作確認と回帰検知を目的とした小規模なもので、実運用精度を示すものではありません。詳細な数値と限界事項は、公開モデルのページ [sumeshi/privacy-filter-jp-GGUF](https://huggingface.co/sumeshi/privacy-filter-jp-GGUF) を参照してください。

## Fine-tuning

データセットの設計方針と再現手順（学習データ生成・ファインチューニング・評価・GGUF 変換・公開）は [`docs/finetuning-jp.md`](docs/finetuning-jp.md) にまとめています。日本語学習データの生成スクリプトは `datasets/jp-data` submodule で管理し、生成・変換済みの学習 JSONL は再配布しません。

## License

本リポジトリは MIT License（[LICENSE](LICENSE)）です。

## Third-Party Licenses

- フォーク元 [privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp)（推論エンジン本体）: MIT License
- [ggml](https://github.com/ggml-org/ggml)（サブモジュール、推論エンジンの依存）: MIT License
- ベースモデル [OpenMed/privacy-filter-multilingual](https://huggingface.co/OpenMed/privacy-filter-multilingual)（[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) のファインチューニング）: Apache License 2.0
- 学習データの一部に [Stockmark ner-wikipedia-dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset)（CC-BY-SA 3.0）を利用
- 住所生成に[日本郵便の公開郵便番号データ](https://www.post.japanpost.jp/service/search/zipcode/download/)を利用
