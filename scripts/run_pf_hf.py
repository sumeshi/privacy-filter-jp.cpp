#!/usr/bin/env python3
"""Run a HF token-classification checkpoint over a JSONL's "text" field and
emit predicted spans in this project's span format (see docs/datasets-jp.md).

Two cases are handled:
  - the checkpoint's own labels already ARE our 4 target labels
    (account_number / private_address / private_date / private_email / private_person / private_phone / private_url / secret);
  - the checkpoint uses the base model's fine-grained BIOES categories
    (FIRSTNAME, LASTNAME, STREET, CITY, ZIPCODE, DATE, ...), which get
    remapped and merged into our 4 target labels via CATEGORY_MAP. 
Output is fed straight into scripts/eval_spans.py.
"""
import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

CATEGORY_MAP = {
    # Normalized openai/privacy-filter labels pass through via OUR_LABELS.
    # The aliases below are for multilingual/fine-grained checkpoints.
    "FIRSTNAME": "private_person",
    "MIDDLENAME": "private_person",
    "LASTNAME": "private_person",
    "PREFIX": "private_person",
    "STREET": "private_address",
    "CITY": "private_address",
    "STATE": "private_address",
    "COUNTY": "private_address",
    "ZIPCODE": "private_address",
    "BUILDINGNUMBER": "private_address",
    "SECONDARYADDRESS": "private_address",
    "DATE": "private_date",
    "DATEOFBIRTH": "private_date",
    "TIME": "private_date",
    "EMAIL": "private_email",
    "EMAILADDRESS": "private_email",
    "E_MAIL": "private_email",
    "PHONE": "private_phone",
    "PHONENUMBER": "private_phone",
    "PHONE_NUMBER": "private_phone",
    "TELEPHONENUMBER": "private_phone",
    "URL": "private_url",
    "URI": "private_url",
    "ACCOUNT": "account_number",
    "ACCOUNTNUMBER": "account_number",
    "ACCOUNT_NUMBER": "account_number",
    "BANKACCOUNT": "account_number",
    "BANK_ACCOUNT": "account_number",
    "IBAN": "account_number",
    "SECRET": "secret",
    "APIKEY": "secret",
    "API_KEY": "secret",
    "TOKEN": "secret",
    "PASSWORD": "secret",
    "BEARER": "secret",
}
OUR_LABELS = {
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
}
MERGE_GAP = 4  # chars: merge same-target spans, but only across whitespace-only gaps
# Separators that must terminate a person span: enumeration commas, furigana
# parentheses, slashes. Spaces are NOT separators (names contain internal
# spaces: "高村　沙耶", "Taro Yamada").
PERSON_SEPARATORS = re.compile(r"[、，,・／/（）()|｜]+")


def decode_bioes(labels: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    spans = []
    cur_cat, cur_start, cur_end = None, None, None
    for lbl, (s, e) in zip(labels, offsets):
        if s == e:  # special/padding token
            continue
        tag, cat = ("O", None) if lbl == "O" else lbl.split("-", 1)
        if tag in ("B", "S"):
            if cur_cat is not None:
                spans.append((cur_start, cur_end, cur_cat))
            cur_cat, cur_start, cur_end = cat, s, e
            if tag == "S":
                spans.append((cur_start, cur_end, cur_cat))
                cur_cat = None
        elif tag in ("I", "E") and cur_cat == cat:
            cur_end = e
            if tag == "E":
                spans.append((cur_start, cur_end, cur_cat))
                cur_cat = None
        else:
            if cur_cat is not None:
                spans.append((cur_start, cur_end, cur_cat))
                cur_cat = None
            if tag in ("I", "E"):
                cur_cat, cur_start, cur_end = cat, s, e
    if cur_cat is not None:
        spans.append((cur_start, cur_end, cur_cat))
    return spans


def remap_and_merge(spans: list[tuple[int, int, str]], text: str) -> list[list]:
    mapped = []
    for s, e, cat in spans:
        target = cat if cat in OUR_LABELS else CATEGORY_MAP.get(cat)
        if target:
            mapped.append([s, e, target])
    mapped.sort(key=lambda x: x[0])
    merged: list[list] = []
    for s, e, label in mapped:
        gap_is_space = merged and not text[merged[-1][1]:s].strip()
        if merged and merged[-1][2] == label and s - merged[-1][1] <= MERGE_GAP and gap_is_space:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, label])
    return merged


def postprocess(spans: list[list], text: str) -> list[list]:
    """Trim whitespace at span edges; split person spans on separators."""
    out = []
    for s, e, label in spans:
        parts = [(s, e)]
        if label == "private_person":
            parts = []
            pos = s
            for m in PERSON_SEPARATORS.finditer(text, s, e):
                if pos < m.start():
                    parts.append((pos, m.start()))
                pos = m.end()
            if pos < e:
                parts.append((pos, e))
        for ps, pe in parts:
            # leading '=' guards against key=value delimiters folded into the
            # first token; trailing '=' is kept (base64 padding in secrets)
            while ps < pe and (text[ps].isspace() or text[ps] == "="):
                ps += 1
            while pe > ps and text[pe - 1].isspace():
                pe -= 1
            if ps < pe:
                out.append([ps, pe, label])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model dir or hub id")
    ap.add_argument("--input", required=True, help="JSONL with a 'text' field (gold spans, if any, are ignored)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    model.eval()
    id2label = model.config.id2label

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            texts = [r["text"] for r in batch]
            enc = tok(texts, return_tensors="pt", return_offsets_mapping=True, padding=True, truncation=True)
            offsets = enc.pop("offset_mapping")
            enc = {k: v.to(args.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits
            pred_ids = logits.argmax(-1).cpu()
            for row, ids, offs in zip(batch, pred_ids, offsets):
                labels = [id2label[t.item()] for t in ids]
                spans = decode_bioes(labels, offs.tolist())
                merged = postprocess(remap_and_merge(spans, row["text"]), row["text"])
                out_spans = [{"start": s, "end": e, "label": lbl} for s, e, lbl in merged]
                f.write(json.dumps({"text": row["text"], "spans": out_spans}, ensure_ascii=False) + "\n")
            print(f"{min(i + args.batch_size, len(rows))}/{len(rows)}", file=sys.stderr, end="\r")

    print(f"\nwrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
