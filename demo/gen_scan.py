#!/usr/bin/env python3
"""Build the single-document NER-scan trace for pii_scan.py from REAL pf-cli output.

Unlike gen_corpus.py (which tiles a paragraph and replicates hard-coded spans),
this runs the actual engine on demo/scan_doc.txt:
  * pf-cli --tok-batch  -> exact token count (vocab only)
  * pf-cli --classify   -> real entity spans (entity_group/start/end/score/text)

so every span and category on screen is exactly what the model emits. Writes
demo/traces/scan/{content.json,engines.json}.

  python3 gen_scan.py --cli build/release/pf-cli \
      --model ~/ggufs_perf/pf-q8experts.gguf --ld build/release/ggml/src \
      --tps 366 --device "Raspberry Pi 5 · CPU · q8 @ 1.5 GHz"
"""
import argparse, json, os, struct, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_cli(cli, ld, args, stdin=None):
    env = dict(os.environ, LD_LIBRARY_PATH=ld)
    return subprocess.run([cli, *args], input=stdin, capture_output=True, env=env)


def token_count(cli, ld, model, doc_bytes):
    inp, outp = "/tmp/scan_tb_in.bin", "/tmp/scan_tb_out.bin"
    with open(inp, "wb") as f:
        f.write(struct.pack("<I", 1) + struct.pack("<I", len(doc_bytes)) + doc_bytes)
    run_cli(cli, ld, ["--tok-batch", model, inp, outp]).check_returncode()
    return struct.unpack_from("<I", open(outp, "rb").read(), 0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ld", default="", help="LD_LIBRARY_PATH for the cli (shared ggml build)")
    ap.add_argument("--doc", default=str(HERE / "scan_doc.txt"))
    ap.add_argument("--scene", default=str(HERE / "traces/scan"))
    ap.add_argument("--threshold", default="0.5")
    ap.add_argument("--tps", type=float, default=0.0, help="measured engine throughput (tok/s)")
    ap.add_argument("--label", default="privacy-filter.cpp")
    ap.add_argument("--device", default="Raspberry Pi 5 · CPU")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    doc_bytes = Path(a.doc).read_bytes()
    n_tok = token_count(a.cli, a.ld, a.model, doc_bytes)
    r = run_cli(a.cli, a.ld, ["--classify", a.model, a.threshold, "cpu"], stdin=doc_bytes)
    ents_raw = json.loads(r.stdout)
    ents = [{"type": e["entity_group"], "start": e["start"], "end": e["end"],
             "text": e.get("text", ""), "score": e.get("score", 0.0)} for e in ents_raw]

    d = Path(a.scene); d.mkdir(parents=True, exist_ok=True)
    content = {"document": doc_bytes.decode("utf-8"), "n_tokens": n_tok,
               "note": a.note, "entities": ents}
    json.dump(content, open(d / "content.json", "w"), ensure_ascii=False)
    json.dump([{"label": a.label, "device": a.device, "tps": a.tps}],
              open(d / "engines.json", "w"), indent=2, ensure_ascii=False)

    cats = {}
    for e in ents:
        cats[e["type"]] = cats.get(e["type"], 0) + 1
    print(f"{len(doc_bytes):,} chars  {n_tok:,} tokens  {len(ents)} entities  "
          f"{len(cats)} categories")
    if a.tps:
        print(f"engine: {a.tps:g} tok/s  ->  {n_tok / a.tps:.2f}s run")
    print("categories:", ", ".join(f"{k}:{v}" for k, v in
                                    sorted(cats.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
