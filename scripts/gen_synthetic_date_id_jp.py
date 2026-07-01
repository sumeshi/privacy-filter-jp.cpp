#!/usr/bin/env python3
"""Generate synthetic Japanese date / internal-identifier examples in this
project's JSONL span format, for the `private_date` and `account_number`
labels. Pure template generation -- no external data source, no real PII.
"""
import argparse
import json
import random
import sys

# (prefix, suffix, year-range key). Span offsets come from len(prefix)/len(suffix).
DATE_TEMPLATES = [
    ("生年月日：", "", "birth"),
    ("生年月日は", "です。", "birth"),
    ("お誕生日：", "", "birth"),
    ("誕生日：", "", "birth"),
    ("", "", "any"),
    ("申込日：", "", "recent"),
    ("お申込み日：", "", "recent"),
    ("ご予約日：", "", "recent"),
    ("予約日：", "", "recent"),
    ("契約日：", "", "recent"),
    ("受付日：", "", "recent"),
    ("来店日：", "", "recent"),
    ("発送日：", "", "recent"),
    ("お届け予定日：", "", "recent"),
    ("申込日は", "です。", "recent"),
    ("契約日は", "になります。", "recent"),
]

YEAR_RANGES = {
    "birth": (1940, 2015),
    "recent": (2015, 2026),
    "any": (1940, 2026),
}

# (era name, valid era-year range). Era years are independent of the western
# year above -- we only need a plausible-looking 和暦 string, not a real
# calendar conversion.
ERAS = [
    ("令和", (1, 8)),
    ("平成", (1, 31)),
    ("昭和", (1, 64)),
    ("大正", (1, 15)),
]

# (prefix, suffix, id-style key).
ID_TEMPLATES = [
    ("顧客番号：", "", "cust"),
    ("お客様番号：", "", "cust"),
    ("会員番号：", "", "member"),
    ("契約番号：", "", "contract"),
    ("ご契約番号：", "", "contract"),
    ("申込番号：", "", "app"),
    ("お申込み番号：", "", "app"),
    ("受付番号：", "", "receipt"),
    ("受付番号は", "です。", "receipt"),
    ("お問い合わせ番号：", "", "receipt"),
    ("", "", "any"),
]


def wareki_year_str(era_year: int) -> str:
    return "元" if era_year == 1 else str(era_year)


def random_date(rng: random.Random, year_key: str) -> str:
    year = rng.randint(*YEAR_RANGES[year_key])
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    style = rng.choice(["kanji", "slash", "slash_zero", "hyphen", "dot", "wareki"])
    if style == "kanji":
        return f"{year}年{month}月{day}日"
    if style == "slash":
        return f"{year}/{month}/{day}"
    if style == "slash_zero":
        return f"{year}/{month:02d}/{day:02d}"
    if style == "hyphen":
        return f"{year}-{month:02d}-{day:02d}"
    if style == "dot":
        return f"{year}.{month:02d}.{day:02d}"
    era_name, (lo, hi) = rng.choice(ERAS)
    return f"{era_name}{wareki_year_str(rng.randint(lo, hi))}年{month}月{day}日"


def random_identifier(rng: random.Random, id_key: str) -> str:
    if id_key == "any":
        id_key = rng.choice(["cust", "member", "contract", "app", "receipt"])
    if id_key == "cust":
        return "C" + "".join(str(rng.randint(0, 9)) for _ in range(7))
    if id_key == "member":
        return "M-" + "".join(str(rng.randint(0, 9)) for _ in range(6))
    if id_key == "contract":
        return f"CNTR-{rng.randint(2015, 2026)}-{rng.randint(1, 999):03d}"
    if id_key == "app":
        return "A" + "".join(str(rng.randint(0, 9)) for _ in range(8))
    # receipt
    y, m, d = rng.randint(2015, 2026), rng.randint(1, 12), rng.randint(1, 28)
    return f"R{y}{m:02d}{d:02d}-{rng.randint(1, 9999):04d}"


def apply_template(rng: random.Random, templates: list[tuple[str, str, str]], value_fn) -> dict:
    prefix, suffix, key = rng.choice(templates)
    value = value_fn(rng, key)
    text = f"{prefix}{value}{suffix}"
    start = len(prefix)
    end = start + len(value)
    return text, start, end


def generate(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            text, start, end = apply_template(rng, DATE_TEMPLATES, random_date)
            label = "private_date"
        else:
            text, start, end = apply_template(rng, ID_TEMPLATES, random_identifier)
            label = "account_number"
        out.append({"text": text, "spans": [{"start": start, "end": end, "label": label}]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="datasets/generated/synthetic_date_id_jp.jsonl")
    ap.add_argument("-n", "--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    examples = generate(args.count, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    n_date = sum(1 for ex in examples if ex["spans"][0]["label"] == "private_date")
    n_id = sum(1 for ex in examples if ex["spans"][0]["label"] == "account_number")
    print(f"private_date: {n_date}", file=sys.stderr)
    print(f"account_number: {n_id}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
