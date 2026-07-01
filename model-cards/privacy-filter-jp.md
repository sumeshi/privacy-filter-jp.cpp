---
license: apache-2.0
base_model: OpenMed/privacy-filter-multilingual
base_model_relation: finetune
pipeline_tag: token-classification
library_name: gguf
tags:
  - gguf
  - privacy-filter.cpp
  - token-classification
  - pii
  - ner
  - privacy
  - redaction
  - japanese
language:
  - ja
---

# privacy-filter-jp — GGUF

Experimental Japanese fine-tune for `privacy-filter.cpp`.

This model keeps the normalized 8-label privacy-filter target space and focuses
on Japanese addresses, Japanese person names, and Japanese business-context
boundaries. English/general multilingual behavior is inherited from the base
model but is not benchmarked here.

This model is not a complete anonymization system and is not a drop-in
production guarantee.

## Labels

- `private_person`
- `private_address`
- `private_email`
- `private_phone`
- `private_date`
- `account_number`
- `private_url`
- `secret`

## Training Summary

Release candidate trained locally on 2026-07-02:

- base: multilingual privacy-filter compatible checkpoint
- final checkpoint: `runs/pf-jp/model-ft-balanced-v2`
- stage 1: balanced fine-tune, 1 epoch, learning rate `1e-4`
- stage 2: boundary refresh, 0.5 epoch, learning rate `5e-5`
- batch size: 8
- max length: 96
- GGUF: `privacy-filter-jp-f16.gguf`
- GGUF sha256: `b800ec1a756640f181c97773586673574e4146414af96b77b6f5ebbb6b02f8da`

Training data summary:

- small in-repository benchmark train split
- synthetic Japanese address examples using public Japan Post postal-code data
  as area-level material plus fabricated street/building/room details
- synthetic Japanese date and account-number examples
- synthetic structured PII examples for email, phone, URL, and secret
- synthetic Japanese boundary examples for honorifics, roles, URL prefixes, and
  multi-address text
- converted Japanese person-name NER examples

No real PII is intentionally used.

## Initial Benchmark

The bundled benchmark is small and intended as an initial smoke/regression
check. It should not be read as production accuracy.

Important caveat: during this initial iteration, `challenge` errors were used to
diagnose boundary issues before the v2 fine-tune. The challenge numbers below
are not a blind held-out score.

Challenge smoke result:

| group | baseline F1 | FT v2 F1 | FT v2 partial-fn |
| --- | ---: | ---: | ---: |
| `private_person` | 0.583 | 0.917 | 1 |
| `private_address` | 0.400 | 1.000 | 0 |
| other 6 labels | 0.400 | 0.833 | 1 |
| micro | 0.475 | 0.929 | 2 |

Label-level FT v2 result on `datasets/benchmark/challenge.jsonl`:

| label | precision | recall | F1 | partial-fn |
| --- | ---: | ---: | ---: | ---: |
| `account_number` | 1.000 | 1.000 | 1.000 | 0 |
| `private_address` | 1.000 | 1.000 | 1.000 | 0 |
| `private_date` | 1.000 | 1.000 | 1.000 | 0 |
| `private_email` | 1.000 | 1.000 | 1.000 | 0 |
| `private_person` | 0.917 | 0.917 | 0.917 | 1 |
| `private_phone` | 1.000 | 1.000 | 1.000 | 0 |
| `private_url` | 0.000 | 0.000 | 0.000 | 1 |
| `secret` | 1.000 | 1.000 | 1.000 | 0 |
| micro | 0.929 | 0.929 | 0.929 | 2 |

The `private_url` miss is an exact-boundary miss on a tiny split. Use
deterministic detectors for URLs, email addresses, phone numbers, IDs, and
secrets when those spans are high-risk.

## Limitations

- Experimental initial checkpoint.
- Primary target is Japanese text; English is not benchmarked here.
- Benchmark is small and partly diagnostic, not a blind production benchmark.
- Does not guarantee complete anonymization.
- Validate on your own data before use in any workflow that affects privacy,
  compliance, or security.

## Runtime

This GGUF uses the custom `openai-privacy-filter` architecture supported by
`privacy-filter.cpp`.

```sh
build/release/pf-cli --info privacy-filter-jp-f16.gguf
echo "配送先：〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室" | \
  build/release/pf-cli --classify privacy-filter-jp-f16.gguf 0.5
```

## Source

See the source repository README for data generation, fine-tuning, benchmark,
conversion, and upload commands.
