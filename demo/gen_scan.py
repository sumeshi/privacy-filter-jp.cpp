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
      --run-device cpu --tps 366 --device "Raspberry Pi 5 · CPU · q8 @ 1.5 GHz"
"""
import argparse, json, os, struct, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_cli(cli, ld, args, stdin=None):
    env = dict(os.environ)
    if ld:
        env["LD_LIBRARY_PATH"] = ld
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
    ap.add_argument("--ld", default="", help="LD_LIBRARY_PATH for source-build shared ggml libraries; omit for release binaries")
    ap.add_argument("--doc", default=str(HERE / "scan_doc.txt"))
    ap.add_argument("--scene", default=str(HERE / "traces/scan"))
    ap.add_argument("--threshold", default="0.5")
    ap.add_argument("--run-device", default="cpu",
                    help="pf-cli classify device argument: cpu, gpu, cuda, vulkan, cuda:1, ...")
    ap.add_argument("--tps", type=float, default=0.0, help="measured engine throughput (tok/s)")
    ap.add_argument("--label", default="privacy-filter.cpp")
    ap.add_argument("--device", default="Raspberry Pi 5 · CPU",
                    help="display label written to engines.json")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    doc_bytes = Path(a.doc).read_bytes()
    n_tok = token_count(a.cli, a.ld, a.model, doc_bytes)
    classify_args = ["--classify", a.model, a.threshold]
    if a.run_device:
        classify_args.append(a.run_device)
    r = run_cli(a.cli, a.ld, classify_args, stdin=doc_bytes)
    r.check_returncode()
    ents_raw = json.loads(r.stdout)

    # pf-cli returns UTF-8 *byte* offsets; pii_scan.py indexes the document as a
    # Python str (char offsets). Convert so masking aligns on multibyte text
    # (Japanese). For ASCII this is the identity.
    def b2c(bo):
        return len(doc_bytes[:bo].decode("utf-8", errors="ignore"))

    ents = [{"type": e["entity_group"], "start": b2c(e["start"]), "end": b2c(e["end"]),
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
