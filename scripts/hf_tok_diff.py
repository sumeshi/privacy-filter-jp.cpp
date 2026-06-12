#!/usr/bin/env python3
"""Tokenizer differential test against HF tokenizers (the exact reference).

Curated torture corpus + optional random fuzzing. For each text, HF ids and
byte offsets are compared with the engine's (pf-cli --tok-batch). HF reports
char offsets; converted here. The HF side uses split_special_tokens semantics
implicitly: texts containing special-token literals are tokenized as plain
text by the engine BY DESIGN; HF AutoTokenizer does the same for raw text
input (no parsing of specials in plain encode unless they're added tokens —
they are, so such strings are EXCLUDED from the differential and pinned by a
dedicated expectation instead).

Usage:
  hf_tok_diff.py --model DIR --pf-cli BIN --gguf FILE [--fuzz N] [--save-corpus FILE]
"""
from __future__ import annotations

import argparse
import random
import struct
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

CURATED = [
    "",
    " ",
    "a",
    "Contact John Doe at john.doe@example.com or call +1-202-555-0173.",
    "don't they'LL we'Ve I'm you'd O'Brien's",
    "12345 1 22 333 4444 3.14159 1,000,000",
    "    leading and trailing    ",
    "tabs\tand\nnewlines\r\nmixed \n\n  \n end",
    "a   b  c d",
    "...!!!??? ---- ////  /:/ a/b/c",
    "snake_case camelCase PascalCase SCREAMING_CASE iPhone McDonald",
    "héllo wörld Đorđe Škoda façade naïve",
    "ZA̡͊͠͝LGΌ ISͮ̂҉̯͈͕̹̘̱ TO͇̹̺ͅƝ̴ȳ̳",                       # combining mark storms
    "اسمي أحمد الخطيب وأعيش في الرياض",                  # Arabic RTL
    "আমার নাম অমিতা চক্রবর্তী",                            # Bengali
    "我叫王小明，住在北京市朝阳区建国路88号。",            # CJK + fullwidth punct
    "日本語のテキストとカタカナとひらがな",
    "Меня зовут Ирина Соколова",                          # Cyrillic
    "한국어 텍스트 예시입니다",                            # Hangul
    "🙂 emoji 👍🏽 with modifiers 👨‍👩‍👧‍👦 ZWJ families 🇩🇪 flags",
    "non breaking space and thin space and　ideographic",
    "​zero​width​space",                   # ZWSP is Cf, not WS
    "mixed العربية and English و الأرقام 123 معاً",
    "Ｆｕｌｌｗｉｄｔｈ ｌａｔｉｎ",
    "ﬁ ﬂ ligatures and ǅ titlecase",
    "x́ combining acute, ä́ stacked marks",
    "១២៣ Khmer digits ٠١٢ Arabic-Indic ໐໑໒ Lao",
    "\n", "\r\n", "\n\n\n", " \n ", "\t\t",
    "@#$%^&*()_+-=[]{}|;':\",./<>?`~",
    "a" * 500,
    "." * 500,
    " " * 300,
    ("12" * 200),
    "word " * 200,
]


def byte_offsets(text: str, char_offsets):
    cum = [0]
    for ch in text:
        cum.append(cum[-1] + len(ch.encode("utf-8")))
    return [(cum[a], cum[b]) for a, b in char_offsets]


def hf_encode(tok, text: str):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], byte_offsets(text, enc["offset_mapping"])


def engine_encode_batch(pf_cli: str, gguf: str, texts: list[str]):
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.bin", Path(td) / "out.bin"
        with open(inp, "wb") as f:
            f.write(struct.pack("<I", len(texts)))
            for t in texts:
                b = t.encode("utf-8")
                f.write(struct.pack("<I", len(b)))
                f.write(b)
        subprocess.run([pf_cli, "--tok-batch", gguf, str(inp), str(outp)], check=True)
        results = []
        with open(outp, "rb") as f:
            for _ in texts:
                (n,) = struct.unpack("<I", f.read(4))
                ids = list(struct.unpack(f"<{n}i", f.read(4 * n))) if n else []
                offs = list(struct.unpack(f"<{2*n}i", f.read(8 * n))) if n else []
                results.append((ids, [(offs[2*i], offs[2*i+1]) for i in range(n)]))
        return results


ASSIGNED = None

def random_text(rng: random.Random) -> str:
    global ASSIGNED
    if ASSIGNED is None:
        ASSIGNED = [cp for cp in range(0x10000) if unicodedata.category(chr(cp)) != "Cn"
                    and not (0xD800 <= cp <= 0xDFFF)]
    pieces = []
    for _ in range(rng.randint(1, 20)):
        kind = rng.random()
        if kind < 0.35:
            pieces.append("".join(chr(rng.choice(ASSIGNED)) for _ in range(rng.randint(1, 8))))
        elif kind < 0.55:
            pieces.append("".join(rng.choice("abcdefgABCDEFG'") for _ in range(rng.randint(1, 10))))
        elif kind < 0.7:
            pieces.append("".join(rng.choice("0123456789") for _ in range(rng.randint(1, 6))))
        elif kind < 0.85:
            pieces.append("".join(rng.choice(" \t\n\r  ") for _ in range(rng.randint(1, 5))))
        else:
            pieces.append("".join(rng.choice(".,!?-/(){}@#") for _ in range(rng.randint(1, 6))))
    return "".join(pieces)


def minimize(tok, pf_cli, gguf, text):
    def differs(t):
        return hf_encode(tok, t) != engine_encode_batch(pf_cli, gguf, [t])[0]
    changed = True
    while changed and len(text) > 1:
        changed = False
        for cut in (len(text) // 2, 1):
            for cand in (text[cut:], text[:-cut] if cut <= len(text) else None):
                if cand is not None and differs(cand):
                    text = cand
                    changed = True
                    break
            if changed:
                break
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pf-cli", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--save-corpus", help="write pack file of curated texts + HF expectations")
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def run_batch(texts, tag):
        engine = engine_encode_batch(args.pf_cli, args.gguf, texts)
        bad = 0
        for t, en in zip(texts, engine):
            hf = hf_encode(tok, t)
            if hf != en:
                bad += 1
                small = minimize(tok, args.pf_cli, args.gguf, t)
                print(f"MISMATCH ({tag}): {small!r}")
                h = hf_encode(tok, small)
                e = engine_encode_batch(args.pf_cli, args.gguf, [small])[0]
                print(f"  hf    : {h[0][:12]} {h[1][:6]}")
                print(f"  engine: {e[0][:12]} {e[1][:6]}")
        return bad

    bad = run_batch(CURATED, "curated")
    print(f"curated: {len(CURATED)} texts, {bad} mismatches")

    if args.save_corpus and bad == 0:
        with open(args.save_corpus, "wb") as f:
            f.write(struct.pack("<I", len(CURATED)))
            for t in CURATED:
                b = t.encode("utf-8")
                ids, offs = hf_encode(tok, t)
                f.write(struct.pack("<I", len(b)))
                f.write(b)
                f.write(struct.pack("<I", len(ids)))
                f.write(struct.pack(f"<{len(ids)}i", *ids))
                flat = [v for p in offs for v in p]
                f.write(struct.pack(f"<{len(flat)}i", *flat))
        print(f"corpus written: {args.save_corpus}")

    if args.fuzz:
        rng = random.Random(args.seed)
        CHUNK = 2000
        total_bad = 0
        done = 0
        while done < args.fuzz:
            n = min(CHUNK, args.fuzz - done)
            texts = [random_text(rng) for _ in range(n)]
            total_bad += run_batch(texts, f"fuzz@{done}")
            done += n
            if done % 20000 == 0:
                print(f"fuzz: {done}/{args.fuzz}, mismatches {total_bad}")
        print(f"fuzz: {args.fuzz} texts, {total_bad} mismatches")
        return 1 if total_bad else 0

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
