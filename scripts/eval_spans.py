#!/usr/bin/env python3
"""Score predicted spans against gold spans: precision/recall/F1 per label.

Both --gold and --pred are JSONL files in this project's span format
(see docs/datasets-jp.md), one line per example, in the SAME order (line i
of --pred is scored against line i of --gold; their "text" must match).

This script is decoupled from how predictions were produced -- it doesn't
care whether they came from pf-cli, an HF Transformers pipeline, or anything
else. Run your model over the gold file's "text" values, write the result in
the same span format, then compare:

    python scripts/eval_spans.py --gold gold.jsonl --pred pred.jsonl
"""
import argparse
import json
import sys


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def span_set(row: dict) -> set[tuple[int, int, str]]:
    return {(s["start"], s["end"], s["label"]) for s in row["spans"]}


def overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    a_start, a_end, a_label = a
    b_start, b_end, b_label = b
    return a_label == b_label and a_start < b_end and b_start < a_end


def score(gold_rows: list[dict], pred_rows: list[dict]) -> dict[str, dict[str, int]]:
    if len(gold_rows) != len(pred_rows):
        sys.exit(f"error: gold has {len(gold_rows)} lines, pred has {len(pred_rows)} lines")

    counts: dict[str, dict[str, int]] = {}

    def bump(label: str, key: str, n: int = 1) -> None:
        counts.setdefault(label, {"tp": 0, "fp": 0, "fn": 0, "partial": 0})[key] += n

    for i, (gold, pred) in enumerate(zip(gold_rows, pred_rows)):
        if gold["text"] != pred["text"]:
            sys.exit(f"error: line {i + 1}: gold/pred text mismatch (misaligned files?)")

        gold_spans, pred_spans = span_set(gold), span_set(pred)
        tp = gold_spans & pred_spans
        fn = gold_spans - pred_spans
        fp = pred_spans - gold_spans

        for span in tp:
            bump(span[2], "tp")
        for span in fn:
            bump(span[2], "fn")
            if any(overlaps(span, p) for p in fp):
                bump(span[2], "partial")
        for span in fp:
            bump(span[2], "fp")

    return counts


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def report(counts: dict[str, dict[str, int]]) -> None:
    header = f"{'label':<22}{'tp':>6}{'fp':>6}{'fn':>6}{'precision':>11}{'recall':>9}{'f1':>9}{'partial-fn':>12}"
    print(header)
    print("-" * len(header))

    total_tp = total_fp = total_fn = 0
    macro_f1s = []
    for label in sorted(counts):
        c = counts[label]
        p, r, f1 = prf1(c["tp"], c["fp"], c["fn"])
        macro_f1s.append(f1)
        total_tp += c["tp"]
        total_fp += c["fp"]
        total_fn += c["fn"]
        print(f"{label:<22}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{p:>11.3f}{r:>9.3f}{f1:>9.3f}{c['partial']:>12}")

    print("-" * len(header))
    micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    print(f"{'micro (all labels)':<22}{total_tp:>6}{total_fp:>6}{total_fn:>6}{micro_p:>11.3f}{micro_r:>9.3f}{micro_f1:>9.3f}")
    print(f"macro F1 (unweighted mean over labels): {macro_f1:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    counts = score(load(args.gold), load(args.pred))
    if not counts:
        print("no spans in either file")
        return
    report(counts)


if __name__ == "__main__":
    main()
