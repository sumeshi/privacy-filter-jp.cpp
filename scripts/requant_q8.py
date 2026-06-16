#!/usr/bin/env python3
"""Experiment tool: requantize selected weights of an existing GGUF to Q8_0.

Copies every KV field and tensor verbatim, except tensors whose name matches
--match (default: the MoE expert weights), which are quantized to Q8_0. Used to
test the hypothesis that ggml's int8 mul_mat_id kernel beats the f16 path on CPU.

  python scripts/requant_q8.py --in f16.gguf --out q8.gguf [--match SUBSTR ...]
"""
from __future__ import annotations

import argparse

import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT, GGUFValueType as VT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", nargs="*",
                    default=["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"])
    args = ap.parse_args()

    r = gguf.GGUFReader(args.inp)
    arch = r.fields["general.architecture"].contents()
    w = gguf.GGUFWriter(args.out, arch)

    # copy metadata (the writer already set general.architecture itself)
    scalar_add = {
        VT.UINT8: w.add_uint8, VT.INT8: w.add_int8, VT.UINT16: w.add_uint16,
        VT.INT16: w.add_int16, VT.UINT32: w.add_uint32, VT.INT32: w.add_int32,
        VT.FLOAT32: w.add_float32, VT.UINT64: w.add_uint64, VT.INT64: w.add_int64,
        VT.FLOAT64: w.add_float64, VT.BOOL: w.add_bool, VT.STRING: w.add_string,
    }
    for key, field in r.fields.items():
        if key == "general.architecture":
            continue
        val = field.contents()
        if field.types and field.types[0] == VT.ARRAY:
            w.add_array(key, val)
        else:
            scalar_add[field.types[0]](key, val)

    # copy / quantize tensors
    n_q = 0
    for t in r.tensors:
        if any(m in t.name for m in args.match):
            x = t.data.astype(np.float32)                      # numpy order [.., ne0]
            q = gguf.quants.quantize(x, QT.Q8_0)               # uint8, last dim -> bytes/row
            w.add_tensor(t.name, q, raw_dtype=QT.Q8_0)         # writer derives logical shape
            n_q += 1
        else:
            w.add_tensor(t.name, t.data)                       # verbatim (preserves F16/F32)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}: {len(r.tensors)} tensors ({n_q} quantized to Q8_0), {len(r.fields)} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
