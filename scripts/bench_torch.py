#!/usr/bin/env python3
"""PyTorch reference throughput, comparable to tools/bench (pf-bench).

Times the HF `openai_privacy_filter` model's forward pass at several document
lengths, one untimed warm-up + N timed iters per length, and prints a markdown
table of forward ms / tok/s -- the same shape pf-bench emits for the ggml
engine, so the two tables line up column-for-column.

  python scripts/bench_torch.py --model <hf-dir> --device cpu
  python scripts/bench_torch.py --model <hf-dir> --device cuda --dtype fp16

Only the model forward is timed (the comparable quantity): tokenization and
BIOES decode are excluded on both sides. Inputs are real token ids -- a PII
paragraph tokenized once and tiled/truncated to the exact target length -- so
both engines see identical sequence lengths. Lengths that OOM or error are
reported as such and skipped rather than aborting the run.
"""
from __future__ import annotations

import argparse
import time

# A PII-shaped paragraph, mirroring tools/bench/pf-bench.cpp make_text(), so the
# token stream is representative rather than degenerate (repeated single token).
SEED_TEXT = (
    "Case 0: Anna Kowalski reported an issue. Contact at anna.kowalski0@mail.example.com "
    "or +48 123 456 789. Ships to 12 Elm Street, Lyon. "
    "Refund to IBAN DE89 3704 0044 0532 0130 00.\n\n"
)

DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def build_ids(tok, n: int):
    import torch

    base = tok(SEED_TEXT, add_special_tokens=False)["input_ids"]
    reps = (n + len(base) - 1) // len(base)
    ids = (base * reps)[:n]
    return torch.tensor([ids], dtype=torch.long)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF checkpoint dir")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--dtype", default="auto", choices=["auto", *DTYPES],
                    help="auto: fp32 on cpu, fp16 on cuda")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--lengths", default="189,756,2898,11403,45234",
                    help="comma-separated token counts (match pf-bench output)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=5,
                    help="untimed warm-up iters (small/fast lengths need several)")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    ap.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    ap.add_argument("--compile-mode", default="default",
                    choices=["default", "reduce-overhead", "max-autotune"])
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    dtype_name = args.dtype
    if dtype_name == "auto":
        dtype_name = "fp16" if args.device.startswith("cuda") else "fp32"
    dtype = getattr(torch, DTYPES[dtype_name])
    dev = torch.device(args.device)
    cuda = dev.type == "cuda"
    lengths = [int(x) for x in args.lengths.split(",") if x]

    tok = AutoTokenizer.from_pretrained(args.model)
    t_load0 = time.perf_counter()
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, dtype=dtype, attn_implementation=args.attn).eval().to(dev)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    if cuda:
        torch.cuda.synchronize()
    t_load1 = time.perf_counter()

    def fwd(ids):
        with torch.inference_mode():
            model(input_ids=ids)

    name = torch.cuda.get_device_name(dev) if cuda else f"cpu x{torch.get_num_threads()}"
    comp = f"compile:{args.compile_mode}" if args.compile else "eager-run"
    print(f"torch {torch.__version__} | tf {transformers.__version__} | {name} | "
          f"{dtype_name} | {args.attn} | {comp} | load {t_load1 - t_load0:.2f}s | {args.iters} iters\n")
    print(f"| {'tokens':>8} | {'forward ms':>11} | {'tok/s':>8} | {'peak MiB':>8} |")
    print("|---------:|------------:|---------:|---------:|")

    for n in lengths:
        ids = build_ids(tok, n).to(dev)
        try:
            if cuda:
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize()
            for _ in range(args.warmup):    # warm-up: compile, cuBLAS algo pick,
                fwd(ids)                    # workspace alloc, allocator growth
            if cuda:
                torch.cuda.synchronize()
                ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                ev0.record()
                for _ in range(args.iters):
                    fwd(ids)
                ev1.record()
                torch.cuda.synchronize()
                fwd_ms = ev0.elapsed_time(ev1) / args.iters
                peak = torch.cuda.max_memory_allocated(dev) / 1024 / 1024
            else:
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    fwd(ids)
                fwd_ms = (time.perf_counter() - t0) * 1e3 / args.iters
                peak = _rss_mib()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if cuda:
                torch.cuda.empty_cache()
            msg = "OOM" if "out of memory" in str(e).lower() else f"err: {str(e)[:40]}"
            print(f"| {n:>8} | {msg:>11} | {'-':>8} | {'-':>8} |")
            continue
        print(f"| {n:>8} | {fwd_ms:>11.1f} | {n / (fwd_ms / 1e3):>8.0f} | {peak:>8.0f} |")
        del ids
        if cuda:
            torch.cuda.empty_cache()
    return 0


def _rss_mib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
