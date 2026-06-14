#!/usr/bin/env python3
"""Publish privacy-filter.cpp GGUF models + model cards to the HuggingFace Hub.

The GGUFs are produced by the llama.cpp-fork converter (the ``conversion/``
package on the ``openai-privacy-filter-arch`` branch), e.g.:

    python convert_hf_to_gguf.py <hf-model-dir> --outtype f16 \\
        --outfile <name>-f16.gguf

Note: the source models' ``tokenizer_config.json`` names ``tokenizer_class:
TokenizersBackend`` (a newer-transformers concept). With a stock ``transformers``
set it to ``PreTrainedTokenizerFast`` first so ``AutoTokenizer`` can load the
o200k ``tokenizer.json`` — same vocab/merges, recognized wrapper class.

This script uploads a converted GGUF plus its **version-controlled** model card
(``model-cards/<key>.md`` -> the repo's ``README.md``) to the matching HF repo,
so the published card never drifts from the one in this repo.

**DRY-RUN BY DEFAULT** — without ``--upload`` it prints what it *would* push
(repo, files, size, sha256) and never contacts HuggingFace. Pass ``--upload``
to perform the real push. The sha256 it prints is what the LocalAI gallery
entry should pin.

Usage:
    python scripts/publish_hf.py --model privacy-filter \\
        --gguf ~/models/privacy-filter/privacy-filter-f16.gguf          # dry-run

    python scripts/publish_hf.py --model privacy-filter-multilingual \\
        --gguf .../privacy-filter-multilingual-f16.gguf --upload         # push

    python scripts/publish_hf.py --model privacy-filter-multilingual \\
        --card-only --upload                  # sync just the card (README.md)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = REPO_ROOT / "model-cards"

HF_ORG = "LocalAI-io"

# key -> (HF repo id, published GGUF filename, model-card filename under model-cards/)
MODELS: dict[str, tuple[str, str, str]] = {
    "privacy-filter-multilingual": (
        f"{HF_ORG}/privacy-filter-multilingual-GGUF",
        "privacy-filter-multilingual-f16.gguf",
        "privacy-filter-multilingual.md",
    ),
    "privacy-filter": (
        f"{HF_ORG}/privacy-filter-GGUF",
        "privacy-filter-f16.gguf",
        "privacy-filter.md",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="which model to publish")
    ap.add_argument("--gguf", type=Path, default=None,
                    help="path to the converted <name>-f16.gguf (omit with --card-only)")
    ap.add_argument("--card-only", action="store_true",
                    help="sync only the model card (README.md); leave the published GGUF untouched")
    ap.add_argument("--repo", default=None, help="override the target HF repo id")
    ap.add_argument("--upload", action="store_true",
                    help="actually push (default: dry-run, contacts nothing)")
    args = ap.parse_args()

    repo, gguf_name, card_name = MODELS[args.model]
    repo = args.repo or repo
    card = CARDS_DIR / card_name

    if not card.is_file():
        sys.exit(f"model card not found: {card}")
    if not args.card_only:
        if args.gguf is None:
            sys.exit("--gguf is required unless --card-only is given")
        if not args.gguf.is_file():
            sys.exit(f"GGUF not found: {args.gguf}")

    print(f"model:   {args.model}")
    print(f"repo:    https://huggingface.co/{repo}")
    print(f"card:    {card.relative_to(REPO_ROOT)} -> README.md")
    if args.card_only:
        print("gguf:    (card-only — published GGUF left untouched)")
    else:
        size = args.gguf.stat().st_size
        print(f"gguf:    {args.gguf}  ({size / 1e9:.2f} GB)  uploaded as {gguf_name}")
        print(f"sha256:  {sha256(args.gguf)}   <- pin this in the LocalAI gallery entry")

    if not args.upload:
        print("\n[dry-run] nothing uploaded. Re-run with --upload to push.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    print("\nuploading README.md ...")
    api.upload_file(
        path_or_fileobj=str(card), path_in_repo="README.md",
        repo_id=repo, repo_type="model",
        commit_message=f"card: sync from privacy-filter.cpp ({card_name})",
    )
    if not args.card_only:
        print(f"uploading {gguf_name} ...")
        api.upload_file(
            path_or_fileobj=str(args.gguf), path_in_repo=gguf_name,
            repo_id=repo, repo_type="model",
            commit_message=f"gguf: {gguf_name} (f16)",
        )
    print(f"done -> https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
