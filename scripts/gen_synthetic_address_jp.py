#!/usr/bin/env python3
"""Generate synthetic Japanese address examples in this project's JSONL span
format.

Place names (prefecture/city/town) come from Japan Post's public postal-code
master, so they are real, valid Japanese addresses at the neighborhood level.
Everything below that level -- chome/banchi/go numbers, building names, room
numbers -- is fabricated and does not correspond to any real building or
person. No real individual's mailing address is used.

Source: https://www.post.japanpost.jp/zipcode/download.html (UTF-8 CSV,
"utf_ken_all.zip"). This script does not bundle that data; download it
yourself (or pass --download to fetch it directly) and point --ken-all at
the extracted CSV.
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
import zipfile

KEN_ALL_URL = "https://www.post.japanpost.jp/service/search/zipcode/download/utf/zip/utf_ken_all.zip"

PAREN_RE = re.compile(r"（(.*)）")

# Fabricated, generic-sounding building names -- not real buildings/brands.
BUILDING_NAMES = [
    "サンプルマンション", "グランドテストコーポ", "みらいハイツ", "さくらコート",
    "けやき荘", "第一テストビル", "パークサイドマンション", "ひまわりアパート",
    "グリーンヒルズ", "セントラルテストタワー", "テストレジデンス", "あおぞら荘",
]

# (prefix, suffix) around the address span. Offsets are computed from
# len(prefix)/len(suffix), never hand-counted, so they can't drift.
ADDRESS_TEMPLATES = [
    ("", ""),
    ("住所：", ""),
    ("ご住所：", ""),
    ("配送先：", ""),
    ("送付先：", ""),
    ("請求先住所：", ""),
    ("契約者住所：", ""),
    ("お届け先：", ""),
    ("請求先住所は", "です。"),
    ("配送先住所は", "になります。"),
    ("現在のご住所は", "です。"),
]


def download_ken_all() -> bytes:
    with urllib.request.urlopen(KEN_ALL_URL) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        return zf.read(name)


def load_ken_all(path: str | None) -> list[tuple[str, str, str, str, bool]]:
    raw = open(path, "rb").read() if path else download_ken_all()
    rows = []
    for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
        pref, city, town, zip7, chome_flag = row[6], row[7], row[8], row[2], row[11] == "1"
        if town == "以下に掲載がない場合":
            town = ""
        rows.append((pref, city, town, zip7, chome_flag))
    return rows


def clean_town(town: str) -> str:
    return PAREN_RE.sub("", town).strip()


def kyoto_street_variants(rows: list[tuple[str, str, str, str, bool]]) -> list[tuple[str, str, str, str]]:
    """Real Kyoto 通り(street)-based town names, e.g. '寺町通御池上る上本能寺前町'."""
    variants = []
    for pref, city, town, zip7, _chome_flag in rows:
        if pref != "京都府":
            continue
        m = PAREN_RE.search(town)
        if not m or "、" not in m.group(1):
            continue
        for part in m.group(1).split("、"):
            if any(k in part for k in ("通", "上る", "下る", "西入", "東入")):
                variants.append((pref, city, part, zip7))
    return variants


def random_banchi(rng: random.Random, chome_flag: bool) -> str:
    # Deliberately balanced ~50/50 between "1-1-1" and "1丁目1番1号" style,
    # independent of the town's real chome_flag: CLAUDE.md wants the model to
    # see both formats often, not in their real-world (chome-minority)
    # proportions. chome_flag is unused for now but kept for callers/tests.
    del chome_flag
    if rng.random() < 0.5:
        chome, ban, go = rng.randint(1, 9), rng.randint(1, 30), rng.randint(1, 20)
        if rng.random() < 0.85:
            return f"{chome}丁目{ban}番{go}号"
        return f"{chome}丁目{ban}番"
    a, b = rng.randint(1, 30), rng.randint(1, 20)
    if rng.random() < 0.3:
        return f"{a}-{b}"
    c = rng.randint(1, 15)
    return f"{a}-{b}-{c}"


def random_building(rng: random.Random) -> str:
    name = rng.choice(BUILDING_NAMES)
    floor, room = rng.randint(1, 12), rng.randint(1, 20)
    style = rng.choice(["room_go", "room_only", "floor_only"])
    if style == "room_go":
        return f"{name}{floor}{room:02d}号室"
    if style == "room_only":
        return f"{name}{room}"
    return f"{name}{floor}F"


def generate(
    n: int,
    rows: list[tuple[str, str, str, str, bool]],
    kyoto_variants: list[tuple[str, str, str, str]],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        if kyoto_variants and rng.random() < 0.05:
            pref, city, street, zip7 = rng.choice(kyoto_variants)
            addr = f"{pref}{city}{street}{rng.randint(1, 600)}"
        else:
            pref, city, town, zip7, chome_flag = rng.choice(rows)
            town = clean_town(town)
            include_pref = rng.random() < 0.85
            addr = (pref if include_pref else "") + city + town + random_banchi(rng, chome_flag)

        if rng.random() < 0.3:
            addr += random_building(rng)
        if zip7 and rng.random() < 0.4:
            addr = f"〒{zip7[:3]}-{zip7[3:]} " + addr

        prefix, suffix = rng.choice(ADDRESS_TEMPLATES)
        text = f"{prefix}{addr}{suffix}"
        start = len(prefix)
        end = start + len(addr)
        out.append({"text": text, "spans": [{"start": start, "end": end, "label": "private_address"}]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ken-all", help="local utf_ken_all.csv path (default: download from Japan Post)")
    ap.add_argument("--out", default="datasets/generated/synthetic_address_jp.jsonl")
    ap.add_argument("-n", "--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_ken_all(args.ken_all)
    kyoto_variants = kyoto_street_variants(rows)
    examples = generate(args.count, rows, kyoto_variants, args.seed)

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"ken_all rows: {len(rows)}", file=sys.stderr)
    print(f"kyoto street variants: {len(kyoto_variants)}", file=sys.stderr)
    print(f"generated: {len(examples)}", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
