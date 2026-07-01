#!/usr/bin/env python3
"""Generate synthetic structured PII examples for Japanese contexts.

The examples are fully fabricated. Domains use reserved/example domains, phone
numbers use test-like 0000 blocks, and secrets are fake token strings.
"""

import argparse
import json
import random
import string
import sys


LABELS = ["private_email", "private_phone", "private_url", "secret"]

EMAIL_TEMPLATES = [
    ("メール：", ""),
    ("連絡先メールは", "です。"),
    ("担当者のメールは", "、電話は別途確認します。"),
    ("請求書の送付先：", ""),
    ("", " へ確認してください。"),
]

PHONE_TEMPLATES = [
    ("電話：", ""),
    ("携帯番号：", ""),
    ("連絡先は", "です。"),
    ("折り返し先：", ""),
    ("担当者の電話は", "、メールは別途確認します。"),
]

URL_TEMPLATES = [
    ("確認URL：", ""),
    ("本人確認URL：", ""),
    ("以下のURLを開いてください：", ""),
    ("注文詳細は", "です。"),
    ("", " を社内で共有しないでください。"),
]

SECRET_TEMPLATES = [
    ("APIキー：", ""),
    ("環境変数にPRIVATE_API_KEY=", "を保存しないでください。"),
    ("Bearer tokenは", "です。"),
    ("パスワード：", ""),
    ("認証トークン：", ""),
]


def token(rng: random.Random, n: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def random_email(rng: random.Random) -> str:
    first = rng.choice(["taro", "hanako", "yumi", "ken", "demo", "sample"])
    last = rng.choice(["yamada", "sato", "suzuki", "tanaka", "kobayashi"])
    sep = rng.choice([".", "_", "-"])
    domain = rng.choice(["example.invalid", "example.test", "example.jp"])
    return f"{first}{sep}{last}{rng.randint(1, 999):03d}@{domain}"


def random_phone(rng: random.Random) -> str:
    prefix = rng.choice(["090", "080", "070", "050", "03", "06"])
    if prefix in {"03", "06"}:
        return f"{prefix}-{rng.randint(0000, 9999):04d}-{rng.randint(0000, 9999):04d}"
    return f"{prefix}-{rng.randint(0000, 9999):04d}-{rng.randint(0000, 9999):04d}"


def random_url(rng: random.Random) -> str:
    host = rng.choice(["example.invalid", "example.test", "privacy.example.invalid"])
    path = rng.choice(["orders", "verify", "accounts", "tickets", "download"])
    ident = rng.choice([
        f"test-{rng.randint(1, 9999):04d}",
        f"user-{rng.randint(1, 9999):04d}",
        f"R2026{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
    ])
    if rng.random() < 0.35:
        return f"https://{host}/{path}/{ident}?token={token(rng, 12)}"
    return f"https://{host}/{path}/{ident}"


def random_secret(rng: random.Random) -> str:
    kind = rng.choice(["sk", "pk", "bearer", "password", "jwt", "api"])
    if kind == "sk":
        return "sk-test-" + token(rng, 24)
    if kind == "pk":
        return "pk_test_" + token(rng, 24)
    if kind == "bearer":
        return "Bearer " + token(rng, 32)
    if kind == "jwt":
        return ".".join(token(rng, n) for n in [16, 24, 16])
    if kind == "api":
        return "api_" + token(rng, 28)
    return rng.choice(["P@ssw0rd-test-001", "dummy-password-0000", "TempPass-123456"])


def apply_template(rng: random.Random, templates, value: str) -> dict:
    prefix, suffix = rng.choice(templates)
    text = f"{prefix}{value}{suffix}"
    start = len(prefix)
    end = start + len(value)
    return text, start, end


def make_example(rng: random.Random, label: str) -> dict:
    if label == "private_email":
        value = random_email(rng)
        text, start, end = apply_template(rng, EMAIL_TEMPLATES, value)
    elif label == "private_phone":
        value = random_phone(rng)
        text, start, end = apply_template(rng, PHONE_TEMPLATES, value)
    elif label == "private_url":
        value = random_url(rng)
        text, start, end = apply_template(rng, URL_TEMPLATES, value)
    elif label == "secret":
        value = random_secret(rng)
        text, start, end = apply_template(rng, SECRET_TEMPLATES, value)
        if value.startswith("Bearer ") and text.startswith("Bearer tokenは"):
            start += len("Bearer ")
            value = value[len("Bearer "):]
            end = start + len(value)
    else:
        raise ValueError(label)
    return {"text": text, "spans": [{"start": start, "end": end, "label": label}]}


def generate(per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for label in LABELS:
        for _ in range(per_label):
            rows.append(make_example(rng, label))
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/generated/synthetic_structured_pii_jp.jsonl")
    ap.add_argument("--per-label", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = generate(args.per_label, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["spans"][0]["label"]] += 1
    for label in LABELS:
        print(f"{label}: {counts[label]}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
