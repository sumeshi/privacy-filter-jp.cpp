#!/usr/bin/env python3
"""Publish the privacy-filter-jp GGUF model and model card to Hugging Face Hub.

Dry-run by default: without --upload this script prints what it would push and
does not contact Hugging Face. Pass --upload to create/update the target repo.

Examples:
  python3 scripts/publish_hf.py --model privacy-filter-jp \
      --gguf runs/pf-jp/privacy-filter-jp-f16.gguf

  python3 scripts/publish_hf.py --model privacy-filter-jp --quant q8 \
      --gguf runs/pf-jp/privacy-filter-jp-q8.gguf --upload

  python3 scripts/publish_hf.py --model privacy-filter-jp --card-only --upload
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CARDS_DIR = REPO_ROOT / "model-cards"

# key -> (HF repo id, {quant: published GGUF filename}, model-card filename).
MODELS: dict[str, tuple[str, dict[str, str], str]] = {
    "privacy-filter-jp": (
        "sumeshi/privacy-filter-jp-GGUF",
        {"f16": "privacy-filter-jp-f16.gguf", "q8": "privacy-filter-jp-q8.gguf"},
        "privacy-filter-jp.md",
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
    ap.add_argument("--model", required=True, choices=sorted(MODELS), help="model to publish")
    ap.add_argument("--gguf", type=Path, default=None, help="path to converted GGUF")
    ap.add_argument(
        "--quant",
        default="f16",
        choices=["f16", "q8"],
        help="which precision --gguf is; picks published filename",
    )
    ap.add_argument(
        "--card-only",
        action="store_true",
        help="sync only model card README.md; leave published GGUF untouched",
    )
    ap.add_argument("--repo", default=None, help="override target HF repo id")
    ap.add_argument("--upload", action="store_true", help="actually push")
    args = ap.parse_args()

    repo, gguf_names, card_name = MODELS[args.model]
    repo = args.repo or repo
    gguf_name = gguf_names[args.quant]
    card = CARDS_DIR / card_name

    if not card.is_file():
        sys.exit(f"model card not found: {card}")
    if not args.card_only:
        if args.gguf is None:
            sys.exit("--gguf is required unless --card-only is given")
        if not args.gguf.is_file():
            sys.exit(f"GGUF not found: {args.gguf}")

    print(f"model: {args.model}")
    print(f"repo: https://huggingface.co/{repo}")
    print(f"card: {card.relative_to(REPO_ROOT)} -> README.md")
    if args.card_only:
        print("(card-only)")
    else:
        size_gb = args.gguf.stat().st_size / (1024**3)
        print(f"gguf: {args.gguf} -> {gguf_name} ({size_gb:.2f} GB, {args.quant})")
        print(f"sha256: {sha256(args.gguf)}")

    if not args.upload:
        print("\n[dry-run] Re-run with --upload to push.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="Update model card",
    )
    if not args.card_only:
        api.upload_file(
            path_or_fileobj=str(args.gguf),
            path_in_repo=gguf_name,
            repo_id=repo,
            repo_type="model",
            commit_message=f"Upload {gguf_name}",
        )
    print(f"pushed: https://huggingface.co/{repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
