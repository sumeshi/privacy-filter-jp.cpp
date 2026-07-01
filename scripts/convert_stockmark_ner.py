#!/usr/bin/env python3
"""Convert Stockmark's ner-wikipedia-dataset (CC-BY-SA 3.0) into this
project's JSONL span format, keeping only the entity types that map onto
the jp-basic label space.

Source: https://github.com/stockmarkteam/ner-wikipedia-dataset
License: CC-BY-SA 3.0 (see LICENSE.md in that repo). Not redistributed here;
run this script locally to regenerate the converted JSONL.

Only "人名" (person name) is mapped by default. "地名" (place name) is NOT
mapped to private_address by default: in this Wikipedia-derived dataset it
is dominated by country/foreign-city mentions ("日本", "アメリカ", "フランス",
"ブルックリン", ...), not Japanese mailing addresses. Pass
--include-place-as-address to opt in anyway, but expect to filter/review the
result before training on it.
"""
import argparse
import json
import sys
import urllib.request

SOURCE_URL = "https://raw.githubusercontent.com/stockmarkteam/ner-wikipedia-dataset/main/ner.json"

TYPE_TO_LABEL = {
    "人名": "private_person",
}
PLACE_TYPE = "地名"
PLACE_LABEL = "private_address"


def load_source(path: str | None) -> list[dict]:
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with urllib.request.urlopen(SOURCE_URL) as resp:
        return json.loads(resp.read().decode("utf-8"))


def convert(rows: list[dict], require_entity: bool, include_place: bool) -> list[dict]:
    type_to_label = dict(TYPE_TO_LABEL)
    if include_place:
        type_to_label[PLACE_TYPE] = PLACE_LABEL

    out = []
    for row in rows:
        spans = [
            {"start": e["span"][0], "end": e["span"][1], "label": type_to_label[e["type"]]}
            for e in row["entities"]
            if e["type"] in type_to_label
        ]
        if require_entity and not spans:
            continue
        spans.sort(key=lambda s: s["start"])
        out.append({"text": row["text"], "spans": spans})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", help="local ner.json path (default: download from GitHub)")
    ap.add_argument("--out", default="datasets/generated/stockmark_ner_jp.jsonl")
    ap.add_argument(
        "--require-entity",
        action="store_true",
        help="drop rows with no mapped span (default: keep as negatives)",
    )
    ap.add_argument(
        "--include-place-as-address",
        action="store_true",
        help="also map 地名 -> private_address (noisy: mostly country/foreign-city mentions, review before training)",
    )
    args = ap.parse_args()

    rows = load_source(args.input)
    converted = convert(rows, args.require_entity, args.include_place_as_address)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_person = sum(1 for r in converted for s in r["spans"] if s["label"] == "private_person")
    n_address = sum(1 for r in converted for s in r["spans"] if s["label"] == "private_address")
    print(f"rows in:  {len(rows)}", file=sys.stderr)
    print(f"rows out: {len(converted)}", file=sys.stderr)
    print(f"private_person spans:  {n_person}", file=sys.stderr)
    print(f"private_address spans: {n_address}", file=sys.stderr)
    if args.include_place_as_address:
        print(
            "warning: private_address spans came from 地名 and include many non-address "
            "place mentions (countries, foreign cities). Review before training.",
            file=sys.stderr,
        )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
