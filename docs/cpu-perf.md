# CPU performance

## TL;DR — the build was SSE-only

The CPU slowness traced to a build trap, not the engine. Under Nix the gcc/clang
wrapper strips `-march=native` (`NIX_ENFORCE_NO_NATIVE`), so a `GGML_NATIVE=ON`
build silently compiles ggml-cpu with **no AVX2/AVX-512/FMA** — and the CI build
(`-DGGML_NATIVE=OFF`) has no SIMD either. Confirmed by disassembly:

```
$ objdump -d libggml-cpu.so | grep -c zmm   # AVX-512   -> 0
$ objdump -d libggml-cpu.so | grep -c ymm   # AVX2      -> 0
$ objdump -d libggml-cpu.so | grep -c vfmadd# FMA       -> 0   (37k xmm/SSE only)
```

With SIMD actually enabled, ggml-f16 on CPU is **~10× faster and beats the
PyTorch/transformers reference** — no quantization needed. The fix is to build the
CPU backend for all ISAs and pick at runtime (`GGML_CPU_ALL_VARIANTS`), which also
sidesteps the Nix `-march=native` stripping.

| 512 tok, f16 | tok/s |
|---|---:|
| SSE-only (the trap: `GGML_NATIVE=OFF`, or `=ON` under Nix) | 280 |
| AVX-512 (explicit `-mavx512*`, or the zen4 runtime variant) | **~3000** |
| PyTorch CPU (fp32, MKL) | 1935 |

## The fix: GGML_CPU_ALL_VARIANTS (runtime ISA dispatch)

`-march=native` is fragile (stripped by Nix; wrong if you build on a different
host than you run on). ggml's portable answer is to compile the CPU backend once
per ISA level and score+load the best at run time:

```sh
cmake -B build -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON ...
```

This produces `libggml-cpu-{sse42,haswell,skylakex,icelake,zen4,…}.so`; on this
Ryzen 9 7900 it loads `libggml-cpu-zen4.so` (AVX-512 + VNNI + BF16):

```
load_backend: loaded CPU backend from libggml-cpu-zen4.so
```

Engine support (`src/backend.cpp`): call `ggml_backend_load_all()` before
`ggml_backend_init_by_type`, and set threads through the registry
(`ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads")`) since the
CPU-specific symbol now lives in the variant `.so`, not in linked base. Both calls
are no-ops for a static build, so one code path serves both. Use the
`release-portable` preset.

## ggml vs MKL once SIMD is real

`pf-gemm-bench` (ggml CPU `mul_mat` GFLOP/s) vs `torch.matmul` (MKL), 12 threads:

| shape (M,K,N) | ggml SSE | ggml AVX-512 | MKL f32 |
|---|---:|---:|---:|
| expert gate_up N=16 (f32) | 39 | 426 | 463 |
| expert gate_up N=512 (f32) | 41 | 528 | 1098 |
| large 4096² N=512 (f32) | 39 | 442 | 1137 |

ggml's f32 GEMM goes 40 → ~500 GFLOP/s — within ~2× of MKL, and at the actual
model level (lower per-op overhead than HF's Python expert loop) ggml-f16 **wins**:
512 tok f16 3006 vs PyTorch 1935. So there was never a missing blocked SGEMM — it
just wasn't compiled with SIMD.

## Profile (AVX-512) and the minor Q8 option

`PF_PROF=noattn|nomoe` ablation (512 tok, AVX-512): **MoE 64%, attention 34%**,
rest <1%. `PF_NTHREADS` sweep: near-linear to 12 physical cores, SMT regresses —
the default is optimal.

Quantization is now a *minor* lever, not a necessity: `scripts/requant_q8.py`
(Q8_0 experts) adds ~15% over f16+AVX-512 (512 tok: 3006 → 3567) but is a strict
precision drop — on the 3k-token case it falls below the f16 parity gate (cos
0.9972, 1 argmax flip in 3053), so it would need its own tier. Given f16+AVX-512
already beats the reference, Q8 is optional (e.g. for memory: 1.6 vs 2.8 GiB).

## Reproduce

```sh
cmake --preset release-portable && cmake --build --preset release-portable -j
build/release-portable/pf-gemm-bench 12          # GFLOP/s by dtype/shape
build/release-portable/pf-bench <f16.gguf> cpu 5 512
objdump -d build/release-portable/bin/libggml-cpu-zen4.so | grep -c zmm   # > 0
```
