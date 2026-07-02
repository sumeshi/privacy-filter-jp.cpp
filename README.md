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

Build from source:

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

Classify from the CLI. The second argument to `--classify` is the detection threshold, and the trailing argument selects the device.

```sh
build/release/pf-cli --info privacy-filter-jp-f16.gguf
echo "配送先：〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室" | \
  build/release/pf-cli --classify privacy-filter-jp-f16.gguf 0.5   # [cpu|cuda|vulkan]
```

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

Exact-match span F1 on a small regression dataset (`datasets/benchmark/`), comparing before and after fine-tuning.

| Label | Before | After |
|---|---:|---:|
| `private_person` | 0.583 | 0.917 |
| `private_address` | 0.400 | 1.000 |
| Other 6 labels (regression check) | 0.400 | 0.833 |
| Overall (micro F1) | 0.475 | 0.929 |

This benchmark is small and intended for sanity and regression checking; it does not represent production accuracy. For detailed numbers and limitations, see the published model page [sumeshi/privacy-filter-jp-GGUF](https://huggingface.co/sumeshi/privacy-filter-jp-GGUF).

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
