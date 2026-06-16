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

## Flash attention (both backends)

With SIMD fixed, attention became the dominant cost at length on *both* backends
(`PF_PROF` ablation — CPU 8192 tok: attention 72%; Vulkan 2k–32k: ~69%), because
the engine built the full `[n,n]` score matrix and masked it to the sliding
window — O(n²) work for an O(n·256) receptive field.

`ggml_flash_attn_ext` (default; `PF_NOFLASH` selects the explicit path) fuses
QK·softmax·V with no materialized scores, carries the attention sinks
(`ggml_flash_attn_ext_add_sinks`) and the sliding-window mask, and accumulates in
F32. It is numerically exact here — passes the f32 `cos>=0.99999` gate and
window-stitch — and faster where attention dominates:

| forward tok/s | CPU 2048 | CPU 8192 | Vulkan 8192 | Vulkan 131072 |
|---|---:|---:|---:|---:|
| explicit (`PF_NOFLASH`) | 1881 | 798 | 11845 | 8992 |
| flash (default) | 3319 | 1928 | 26918 | 20631 |
| speedup | 1.8× | 2.4× | 2.3× | 2.3× |

## Memory and the processing window (W)

`PF_WINDOW` (the `pf_set_window` knob, default 4096) sets tokens per forward;
longer inputs run as overlapping halo windows. At the default, GGML's footprint
is **flat across document length** — the compute buffer is bounded by the window,
not the input:

| length | PyTorch VRAM (eager) | GGML Vulkan VRAM (flash, W=4096) |
|---:|---:|---:|
| 4 096 | 5 439 | 2 883 |
| 8 192 | 13 637 | 2 883 |
| 32 768 | OOM | 2 883 |
| 131 072 | OOM | **2 883** |

PyTorch (single-pass) grows O(n²) and OOMs by ~16k tokens; GGML holds ~2.9 GiB at
131k. So the default W=4096 is a good fit for VRAM-constrained deployments.

Raising W to cut the halo recompute is tempting but currently a **bad trade**: it
OOMs by W=16384. Flash removed the O(n²) *scores*, but the sliding-window **mask
is still a materialized `[n,n]` tensor** — the last O(n²) term.

### Banded mask (prototype, `pf-banded-proto`)

Grouping tokens into blocks of `B ≥ radius` and having each query block attend
only to blocks `{i-1, i, i+1}` makes the mask **O(n·B)** (a `[3B, B, n_blocks]`
band, constant per block) and the attention compute **O(n·band)** — while being
**bit-identical** to full masked attention (same dot products, computed locally):

```
$ pf-banded-proto 256 8192
n=8192 B=256 r=128 | max|d|=0.00e+00 | mask: full 256.0 MiB, band 24.0 MiB (10.7x)
```

Mask scaling (B=256): 21× smaller at 16k, 85× at 64k.

**On by default for sequences >= 2048 tokens** (`src/model.cpp`; `PF_BANDED`
forces it on/off): blocks of B=256, each query block flash-attends to blocks
`{i-1,i,i+1}` with the F16 band mask + sinks; GQA broadcasts over heads;
out-of-range tokens are padded and masked. Parity-exact — passes the f32
`cos>=0.99999` gate and window-stitch on CPU and Vulkan. Speedups
(flash → banded, default W):

| tok/s | CPU 8192 | Vulkan 8192 | Vulkan 32768 |
|---|---:|---:|---:|
| flash | 2068 | 42407 | 33893 |
| banded | 2325 | **105058** | **83664** |
| | 1.1× | **2.5×** | **2.5×** |

Big on Vulkan (the flash kernel computes the full window; banded only the band),
modest on CPU. The measured crossover (banded/flash): 0.9× at 256–512 tok, 1.0×
at 2048, then 1.1× (CPU) / 2.5× (Vulkan) at 4096+. Hence the 2048 default cutoff.

### Dropping the window (`PF_MOE_CHUNK`)

With banded attention the only remaining O(n) cap on a large single window was the
MoE expert matmul's activation scratch (`mul_mat_id y_sz > maxStorageBufferRange`
on Vulkan). The MoE is per-token, so `PF_MOE_CHUNK=C` runs it in C-token chunks
(exact, no halo). It defaults to the forward window (4096), so it's inert at the
default window (n <= W) but keeps a *larger* window from OOMing. Banded + chunking
lets a **131072-token document run in one window** instead of windowing at W=4096:

| 131072 tok, Vulkan | tok/s | compute buffer |
|---|---:|---:|
| banded, windowed W=4096 | 80 897 | 166 MiB |
| banded + chunk, single window | **103 539** | 2 389 MiB |

~1.28× faster (no halo recompute) for more memory -- the throughput/VRAM tradeoff
the window now exposes, capped only by total VRAM. Passes the f32 parity gate.

## Reproduce

```sh
cmake --preset release-portable && cmake --build --preset release-portable -j
build/release-portable/pf-gemm-bench 12          # GFLOP/s by dtype/shape
build/release-portable/pf-bench <f16.gguf> cpu 5 512
objdump -d build/release-portable/bin/libggml-cpu-zen4.so | grep -c zmm   # > 0
```
