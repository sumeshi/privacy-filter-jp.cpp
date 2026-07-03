# privacy-filter-jp.cpp

English | [日本語](README-ja.md)

A fine-tuned model and inference engine specialized for Japanese PII (personally identifiable information) detection.

![privacy-filter-jp.cpp scanning a Japanese document and masking PII spans in real time](demo/out/pfj.gif)

## Overview

[OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) is a token-classification model for PII detection: instead of generating text, it returns the labeled spans of personal information contained in the input.  
[privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp) is an inference engine that runs models fine-tuned from OpenAI Privacy Filter on top of C++/GGML with minimal setup.

This fork adds a model fine-tuned for Japanese on top of that engine. It preserves the base model's detection capability while improving accuracy on Japanese addresses and names, as well as business contexts such as application forms, billing addresses, and shipping addresses. It targets eight labels including names, addresses, email addresses, and phone numbers (see [Labels](#labels)).

Note: this is an experimental initial release and does not guarantee complete anonymization or production-grade accuracy.

## Features

- Improved detection of Japanese addresses, names, and business contexts, on top of the base model
- Retains the base model's detection of emails, phone numbers, URLs, identifiers, and secrets
- Runs entirely locally on CPU or GPU (CUDA / Vulkan); no external API required
- C API bindable from other languages ([`include/pf.h`](include/pf.h))

## Installation

Prebuilt CLI binaries (Linux x86-64 `.tar.gz`, Windows x86-64 `.zip`) are attached to [GitHub Releases](https://github.com/sumeshi/privacy-filter-jp.cpp/releases): download, extract, and run `pf-cli` / `pf-cli.exe` from the extracted folder (it loads the bundled ggml libraries next to it) — no toolchain required. Then grab a GGUF from Hugging Face as shown below.

Or build from source:

```sh
git clone --recursive https://github.com/sumeshi/privacy-filter-jp.cpp
cd privacy-filter-jp.cpp
cmake --preset release && cmake --build --preset release -j
```

To enable GPU, add `-DPF_VULKAN=ON` or `-DPF_CUDA=ON`.

Download the fine-tuned model (GGUF) from Hugging Face:

```sh
pip install huggingface_hub
huggingface-cli download sumeshi/privacy-filter-jp-GGUF privacy-filter-jp-f16.gguf --local-dir .
```

## Usage

The main CLI workflow is redaction. It reads UTF-8 text from stdin and writes the
redacted text to stdout after inference finishes.

```sh
echo "配送先：〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室" | \
  ./pf-cli redact privacy-filter-jp-f16.gguf
```

Default output replaces each detected span with `***`:

```text
配送先：***
```

Use `--labels` when you want category names instead:

```sh
echo "配送先：〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室" | \
  ./pf-cli redact privacy-filter-jp-f16.gguf --labels
```

```text
配送先：[ADDRESS]
```

Useful options:

```sh
./pf-cli redact privacy-filter-jp-f16.gguf --threshold 0.6 --device cuda
cat input.txt | ./pf-cli classify privacy-filter-jp-f16.gguf --threshold 0.5
./pf-cli info privacy-filter-jp-f16.gguf
```

Source builds use `build/release/pf-cli` instead of `./pf-cli`. On Windows
PowerShell, use `.\pf-cli.exe redact privacy-filter-jp-f16.gguf`.

To use it from a program, call the flat C API in [`include/pf.h`](include/pf.h). `pf_ctx` is an opaque handle, buffers are owned by the caller, and exceptions never cross the API boundary.

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

`pf_classify` returns the detected spans (byte offsets, score, label). Spans below the threshold are dropped, and the result is freed with `pf_entities_free`.

## Labels

The eight target labels are listed below. Their definitions live in [`label_space/jp-basic.json`](label_space/jp-basic.json).

| Label | Target |
|---|---|
| `private_person` | Personal names |
| `private_address` | Addresses |
| `private_date` | Dates |
| `private_email` | Email addresses |
| `private_phone` | Phone numbers |
| `private_url` | URLs |
| `account_number` | Customer/contract numbers and similar identifiers |
| `secret` | API keys, tokens, etc. |

## Benchmark

Exact-match span micro F1 on the regression datasets (`datasets/benchmark/`).

The current benchmark (`eval2.jsonl` / `challenge2.jsonl`, 106 hand-written examples) targets realistic conditions: multi-paragraph business documents (emails with signature blocks, application forms, support logs), Japanese phone-number format variants, Japan-specific identifiers (My Number, driver's license, passport, pension, health insurance — all mapped to `account_number`), furigana name pairs, and PII-free negatives.

| Benchmark | OpenAI no-custom | Latest (2026-07-03) |
|---|---:|---:|
| `eval2` (realistic documents) | 0.366 | **0.717** |
| `challenge2` (blind held-out) | 0.400 | **0.693** |
| `challenge` (legacy regression split) | 0.561 | **0.964** |

Both columns are measured with the same span post-processing (edge trimming and person-span splitting on enumeration/furigana separators). `OpenAI no-custom` is `openai/privacy-filter` without this repository's Japanese fine-tune. The legacy splits (`train`/`eval`/`challenge`, 56 short single-sentence rows) turned out to share template-generated texts with earlier local training data, so previously published smoke numbers on those splits were optimistic; those splits are kept for regression checking only. Exact-match F1 is also strict: on `eval2`, most remaining errors in the latest model are span-boundary differences, not missed detections.

This benchmark is still small and does not represent production accuracy. The latest f16 and q8 GGUF files are published at [sumeshi/privacy-filter-jp-GGUF](https://huggingface.co/sumeshi/privacy-filter-jp-GGUF). For dataset design and reproduction steps, see [`docs/finetuning-jp.md`](docs/finetuning-jp.md).

## Fine-tuning

The dataset design and full reproduction steps (data generation, fine-tuning, evaluation, GGUF conversion, and publishing) are documented in [`docs/finetuning-jp.md`](docs/finetuning-jp.md) (in Japanese). Japanese training-data generators live in the `datasets/jp-data` submodule. Generated or converted training JSONL files are not redistributed.

## License

This repository is licensed under the MIT License ([LICENSE](LICENSE)).

## Third-Party Licenses

- Fork source [privacy-filter.cpp](https://github.com/localai-org/privacy-filter.cpp) (the inference engine): MIT License
- [ggml](https://github.com/ggml-org/ggml) (submodule, inference-engine dependency): MIT License
- Base model [OpenMed/privacy-filter-multilingual](https://huggingface.co/OpenMed/privacy-filter-multilingual) (a fine-tune of [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter)): Apache License 2.0
- Part of the training data uses [Stockmark ner-wikipedia-dataset](https://github.com/stockmarkteam/ner-wikipedia-dataset) (CC-BY-SA 3.0)
- Address generation uses [Japan Post's public postal code data](https://www.post.japanpost.jp/service/search/zipcode/download/)
