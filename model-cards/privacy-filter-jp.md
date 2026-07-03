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

v2 release candidate trained locally on 2026-07-03:

- base: multilingual privacy-filter compatible checkpoint
- final checkpoint: `runs/pf-jp/model-ft-v2b`
- one-shot fine-tune from the base model (no incremental stages): 3 epochs,
  learning rate `1e-4`, batch size 8, max length 384
- label space unchanged from the base model (no new classifier rows; Japanese
  identifiers are aliased into `account_number`), so model size and inference
  speed are identical to v1
- GGUF f16 sha256: `e2bafc05ef7e6beb354e78cd77c8cfb5101d55044f23b2cdf343731a882f3b1c`
- GGUF q8 (experts-only Q8_0) sha256: `de9499518ade053d65d20c2eaae2833954e114d29dd77fa7466dade782da9cd6`

Training data summary (~34,000 rows, all offsets machine-validated):

- small in-repository benchmark train split
- synthetic Japanese address examples using public Japan Post postal-code data
  as area-level material plus fabricated street/building/room details
- synthetic Japanese date and account-number examples
- synthetic structured PII examples for email, phone, URL, and secret
- synthetic Japanese phone numbers in all common formats (mobile / landline
  with 2–4 digit area codes / 0120 / 0570 / +81 / fullwidth digits /
  parentheses / no separators) and Japan-specific identifiers (My Number,
  driver's license, passport, pension, health insurance, bank and yucho
  accounts, residence card) labeled as `account_number`
- synthetic ordinary Japanese person names (kanji/kana/romaji, furigana pairs,
  joint names, honorific and title boundaries)
- synthetic long multi-PII business documents (150–600 chars: emails with
  signature blocks, application forms, support logs, delivery notes, minutes,
  incident reports)
- key=value log / .env / HTTP-header style PII lines
- PII-free negative rows (prices, model numbers, versions, error codes) to
  suppress over-detection
- synthetic Japanese boundary examples for honorifics, roles, URL prefixes, and
  multi-address text
- converted Japanese person-name NER examples

No real PII is intentionally used.

## Benchmark

Exact-match span micro F1, measured with the runtime span post-processing that
ships with `privacy-filter.cpp` (edge trimming and person-span splitting).
The v2 benchmark (`datasets/benchmark/{eval2,challenge2}.jsonl`, 106
hand-written examples) targets realistic multi-paragraph business documents,
Japanese phone-format variants, Japan-specific identifiers, furigana name
pairs, and PII-free negatives. `challenge2` is kept blind: it is never used
for tuning or per-row error analysis.

| benchmark | v1 model | v2 model |
| --- | ---: | ---: |
| `eval2` (realistic documents) | 0.400 | 0.717 |
| `challenge2` (blind held-out) | 0.453 | 0.693 |
| `challenge` (v1 split, regression) | 0.912 | 0.964 |

The v1 split numbers previously published (0.929 overall) predate this
pipeline and were optimistic: the v1 boundary training data shared
template-generated texts with the v1 `eval`/`challenge` splits.

Label-level v2 result on `eval2.jsonl`:

| label | precision | recall | F1 | partial-fn |
| --- | ---: | ---: | ---: | ---: |
| `account_number` | 0.625 | 0.625 | 0.625 | 5 |
| `private_address` | 0.700 | 0.875 | 0.778 | 1 |
| `private_date` | 0.643 | 0.643 | 0.643 | 5 |
| `private_email` | 0.333 | 0.333 | 0.333 | 4 |
| `private_person` | 0.793 | 0.821 | 0.807 | 4 |
| `private_phone` | 0.917 | 0.917 | 0.917 | 1 |
| `private_url` | 0.500 | 0.500 | 0.500 | 3 |
| `secret` | 0.667 | 1.000 | 0.800 | 0 |
| micro | 0.696 | 0.740 | 0.717 | 23 |

Most remaining false negatives are partial (boundary differences on detected
entities rather than complete misses; see the `partial-fn` column). Use
deterministic detectors for URLs, email addresses, phone numbers, IDs, and
secrets when those spans are high-risk.

## Limitations

- Experimental checkpoint.
- Primary target is Japanese text; English is not benchmarked here.
- Benchmarks are small and do not represent production accuracy.
- Known gaps: dates at the head of numbered list items ("5. 2026年7月3日、")
  can fall below threshold; identifiers in key=value log lines are sometimes
  labeled as a different PII category than expected (still redacted); email
  span boundaries in long documents are the weakest label.
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
