#!/usr/bin/env python3
"""Fine-tune the OpenMed multilingual privacy-filter checkpoint on our
Japanese PII data using the normalized 8-label privacy-filter label space.

Strategy: make sure the normalized 8 BIOES-tagged categories exist on top of
the base model's existing categories. If a checkpoint already has those labels
(as openai/privacy-filter does), they are reused. If a checkpoint uses a larger
or different multilingual label set, only missing BIOES rows are appended and
old rows are copied as-is. This avoids replacing the label space, so existing
capability (email/phone/url/secret/etc, other languages) is not deliberately
discarded.

The MoE expert weights are ~90% of the model's 1.4B params
(model.layers.*.mlp.experts.*). They are FROZEN. Only attention
projections, the MoE router, layernorms, and the (resized) classifier head
are trained -- about 90M trainable params, chosen to fit comfortably in
12GB of VRAM and to limit how much the shared representation can drift
(catastrophic forgetting of the base model's other categories/languages).
"""
import argparse
import json
import random
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

NEW_CATEGORIES = [
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
]
BIOES_TAGS = ["B", "I", "E", "S"]

# Prefer existing multilingual classifier rows for labels that the base model
# already knows. This preserves structured PII capability better than adding
# randomly initialized duplicate rows such as private_email next to EMAIL.
BASE_CATEGORY_ALIASES = {
    "account_number": ["BANKACCOUNT", "ACCOUNTNAME"],
    "private_date": ["DATE", "DATEOFBIRTH"],
    "private_email": ["EMAIL"],
    "private_phone": ["PHONE"],
    "private_url": ["URL"],
    "secret": ["PASSWORD"],
}


def has_bioes_category(category: str, label2id: dict) -> bool:
    return all(f"{tag}-{category}" in label2id for tag in BIOES_TAGS)


def resolve_training_category(category: str, label2id: dict) -> str:
    if has_bioes_category(category, label2id):
        return category
    for alias in BASE_CATEGORY_ALIASES.get(category, []):
        if has_bioes_category(alias, label2id):
            return alias
    return category

def load_training_rows(paths: list[str], skip_first_lines: int) -> list[dict]:
    rows = []
    for path in paths:
        lines = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        rows += lines[skip_first_lines:]
    return rows


def extend_label_space(model, tokenizer):
    """Ensure all normalized target BIOES labels exist.

    The original openai/privacy-filter already has the 8 normalized
    categories. Some multilingual checkpoints use a larger/finer label set.
    We therefore add only missing BIOES labels instead of blindly duplicating
    labels that are already present.
    """
    del tokenizer
    old_id2label = model.config.id2label
    old_num_labels = len(old_id2label)

    id2label = dict(old_id2label)
    label2id = {v: k for k, v in id2label.items()}

    labels_to_add = []
    for cat in NEW_CATEGORIES:
        if resolve_training_category(cat, label2id) != cat:
            continue
        for tag in BIOES_TAGS:
            lbl = f"{tag}-{cat}"
            if lbl not in label2id:
                labels_to_add.append(lbl)

    if not labels_to_add:
        model.config.id2label = id2label
        model.config.label2id = label2id
        model.config.num_labels = old_num_labels
        return model

    for i, lbl in enumerate(labels_to_add):
        id2label[old_num_labels + i] = lbl
        label2id[lbl] = old_num_labels + i

    new_num_labels = old_num_labels + len(labels_to_add)
    old_score = model.score
    new_score = torch.nn.Linear(
        old_score.in_features, new_num_labels,
        dtype=old_score.weight.dtype, device=old_score.weight.device,
    )
    with torch.no_grad():
        new_score.weight[:old_num_labels] = old_score.weight
        new_score.bias[:old_num_labels] = old_score.bias
        torch.nn.init.normal_(new_score.weight[old_num_labels:], std=0.02)
        new_score.bias[old_num_labels:].zero_()
    model.score = new_score
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.num_labels = new_num_labels
    return model


def freeze_experts(model) -> None:
    trainable, frozen = 0, 0
    for name, p in model.named_parameters():
        if ".mlp.experts." in name or name == "model.embed_tokens.weight":
            p.requires_grad_(False)
            frozen += p.numel()
        else:
            p.requires_grad_(True)
            trainable += p.numel()
    print(f"trainable params: {trainable/1e6:.1f}M, frozen: {frozen/1e6:.1f}M", file=sys.stderr)


def bioes_labels_for_row(row: dict, tokenizer, label2id: dict, max_length: int) -> dict:
    enc = tokenizer(row["text"], return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc.pop("offset_mapping")
    labels = [-100 if s == e else label2id["O"] for s, e in offsets]

    for span in row["spans"]:
        tok_idxs = [
            i for i, (s, e) in enumerate(offsets)
            if s != e and s < span["end"] and span["start"] < e
        ]
        if not tok_idxs:
            continue
        cat = resolve_training_category(span["label"], label2id)
        if len(tok_idxs) == 1:
            labels[tok_idxs[0]] = label2id.get(f"S-{cat}", labels[tok_idxs[0]])
        else:
            labels[tok_idxs[0]] = label2id.get(f"B-{cat}", labels[tok_idxs[0]])
            labels[tok_idxs[-1]] = label2id.get(f"E-{cat}", labels[tok_idxs[-1]])
            for i in tok_idxs[1:-1]:
                labels[i] = label2id.get(f"I-{cat}", labels[i])

    enc["labels"] = labels
    return enc


class SpanDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_length):
        self.encoded = [bioes_labels_for_row(r, tokenizer, label2id, max_length) for r in rows]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="/home/sumeshi/models/privacy-filter-multilingual")
    ap.add_argument("--data", nargs="+", default=[
        "datasets/generated/synthetic_address_jp.jsonl",
        "datasets/generated/synthetic_date_id_jp.jsonl",
        "datasets/generated/stockmark_ner_jp.jsonl",
    ])
    ap.add_argument("--out", default="/home/sumeshi/models/privacy-filter-jp-ft")
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument(
        "--skip-first-lines",
        type=int,
        default=0,
        help=(
            "skip the first N rows of each --data file before training. "
            "Use only if you intentionally reserved a prefix slice outside "
            "the repository benchmark splits."
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = extend_label_space(model, tokenizer)
    freeze_experts(model)

    rows = load_training_rows(args.data, args.skip_first_lines)
    random.Random(args.seed).shuffle(rows)
    n_val = int(len(rows) * args.val_fraction)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"train: {len(train_rows)}  val: {len(val_rows)}", file=sys.stderr)

    label2id = model.config.label2id
    train_ds = SpanDataset(train_rows, tokenizer, label2id, args.max_length)
    val_ds = SpanDataset(val_rows, tokenizer, label2id, args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out + "-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        bf16=True,
        report_to=[],
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved fine-tuned checkpoint to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
