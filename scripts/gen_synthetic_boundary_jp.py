#!/usr/bin/env python3
"""Generate Japanese boundary-focused PII examples.

These examples target common over-span errors: prefixes like "氏名 ", particles
before names, honorifics/roles after names, URL colons, and two addresses in one
sentence that must remain separate spans. All names/addresses are fabricated.
"""

import argparse
import json
import random
import sys


NAMES = [
    "山田 太郎",
    "佐藤花子",
    "鈴木 一郎",
    "小林由美",
    "斎藤 一郎",
    "山本健一",
    "田中",
    "伊藤 美咲",
]

ADDRESSES = [
    "東京都目黒区中根1-2-3",
    "東京都目黒区中根1-99-99",
    "福岡県福岡市博多区博多駅前1-99-9",
    "大阪市北区梅田3-99-1",
    "京都府京都市中京区錦小路通烏丸東入元法然寺町99",
    "〒160-0022 東京都新宿区新宿3-99-88 サンプルマンション101号室",
]

NEGATIVES = [
    "受付番号は公開サンプルです。",
    "公開サンプルは個人名ではありません。",
    "テスト担当へ確認してください。",
    "サンプル株式会社の公開資料です。",
    "住所サンプルは架空データです。",
]


def one_span(text: str, value: str, label: str) -> dict:
    start = text.index(value)
    end = start + len(value)
    return {"text": text, "spans": [{"start": start, "end": end, "label": label}]}


def person_example(rng: random.Random) -> dict:
    name = rng.choice(NAMES)
    template = rng.choice([
        "申込者氏名 {name}",
        "契約者名：{name}様",
        "経理部 {name}様から請求先住所の連絡です。",
        "お問い合わせ本文：先日申し込んだ{name}です。",
        "サンプル株式会社 代表取締役社長 {name}",
        "{name}部長様",
        "担当の{name}さんへ共有してください。",
        "営業部の{name}課長に確認してください。",
    ])
    if "{name}部長様" in template and len(name) > 2:
        name = name.replace(" ", "")[:2]
    text = template.format(name=name)
    return one_span(text, name, "private_person")


def address_example(rng: random.Random) -> dict:
    a = rng.choice(ADDRESSES)
    b = rng.choice([x for x in ADDRESSES if x != a])
    template = rng.choice([
        "住所変更前：{a} 住所変更後：{b}",
        "旧住所：{a} / 新住所：{b}",
        "配送先：{a} 請求先：{b}",
    ])
    text = template.format(a=a, b=b)
    spans = []
    first = text.index(a)
    spans.append({"start": first, "end": first + len(a), "label": "private_address"})
    second = text.index(b, first + len(a))
    spans.append({"start": second, "end": second + len(b), "label": "private_address"})
    return {"text": text, "spans": spans}


def url_boundary_example(rng: random.Random) -> dict:
    value = rng.choice([
        "https://example.invalid/verify/user-0001?token=test",
        "https://example.invalid/orders/test-001",
        "https://example.test/accounts/R20260401",
    ])
    text = rng.choice([
        f"本人確認URL：{value}",
        f"確認URLは{value}です。",
        f"URL：{value} を開いてください。",
    ])
    return one_span(text, value, "private_url")


def generate(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    makers = [person_example, address_example, url_boundary_example]
    for i in range(count):
        if i % 7 == 0:
            rows.append({"text": rng.choice(NEGATIVES), "spans": []})
        else:
            rows.append(rng.choice(makers)(rng))
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/generated/synthetic_boundary_jp.jsonl")
    ap.add_argument("-n", "--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = generate(args.count, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {"negative": 0}
    for row in rows:
        if not row["spans"]:
            counts["negative"] += 1
        for span in row["spans"]:
            counts[span["label"]] = counts.get(span["label"], 0) + 1
    for key in sorted(counts):
        print(f"{key}: {counts[key]}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
