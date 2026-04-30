# scripts/train_svg_tokenizer.py

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def count_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def train_tokenizer(train_jsonl, tokenizer_path, vocab_size):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # ByteLevel works well for code/XML because it preserves punctuation,
    # brackets, slashes, quotes, numbers, and whitespace patterns.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    n_train = count_jsonl(train_jsonl)

    def iterator():
        for rec in read_jsonl(train_jsonl):
            yield rec["svg"]

    print(f"Training BPE tokenizer on {n_train} training SVGs...")
    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=n_train)

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))

    print(f"Saved tokenizer to: {tokenizer_path}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer


def encode_split(tokenizer, input_jsonl, output_bin, max_tokens=None):
    eos_id = tokenizer.token_to_id("[EOS]")

    all_ids = []
    lengths = []
    kept = 0
    skipped_too_long = 0

    n = count_jsonl(input_jsonl)

    for rec in tqdm(read_jsonl(input_jsonl), total=n, desc=f"Encoding {input_jsonl.name}"):
        ids = tokenizer.encode(rec["svg"]).ids

        if max_tokens is not None and len(ids) > max_tokens:
            skipped_too_long += 1
            continue

        lengths.append(len(ids))
        ids.append(eos_id)
        all_ids.extend(ids)
        kept += 1

    output_bin.parent.mkdir(parents=True, exist_ok=True)

    # uint16 is enough because vocab_size is far below 65536.
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_bin)

    stats = {
        "input_file": str(input_jsonl),
        "output_file": str(output_bin),
        "kept_svg_count": kept,
        "skipped_too_long": skipped_too_long,
        "total_tokens_with_eos": int(len(arr)),
        "min_svg_tokens": int(min(lengths)) if lengths else 0,
        "max_svg_tokens": int(max(lengths)) if lengths else 0,
        "mean_svg_tokens": float(mean(lengths)) if lengths else 0.0,
        "median_svg_tokens": float(median(lengths)) if lengths else 0.0,
        "max_tokens_filter": max_tokens,
    }

    print(f"\nEncoded {input_jsonl.name}")
    print(f"  kept SVGs: {kept}")
    print(f"  skipped too long: {skipped_too_long}")
    print(f"  total tokens including EOS: {len(arr)}")
    print(f"  saved: {output_bin}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizer for cleaned SVG JSONL files and encode splits."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing cleaned/train.jsonl, val.jsonl, test.jsonl.",
    )

    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        default="tokenizer",
        help="Directory to save tokenizer JSON.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/tokens",
        help="Directory to save encoded .bin files.",
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=4096,
        help="BPE vocabulary size. Reasonable SVG range: 1000 to 8000.",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Drop SVGs longer than this many tokens. Use 0 to disable.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cleaned_dir = data_dir / "cleaned"
    out_dir = Path(args.out_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    stats_dir = data_dir / "stats"

    train_jsonl = cleaned_dir / "train.jsonl"
    val_jsonl = cleaned_dir / "val.jsonl"
    test_jsonl = cleaned_dir / "test.jsonl"

    tokenizer_path = tokenizer_dir / f"svg_bpe_{args.vocab_size}.json"

    max_tokens = None if args.max_tokens == 0 else args.max_tokens

    tokenizer = train_tokenizer(
        train_jsonl=train_jsonl,
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
    )

    split_stats = {
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "max_tokens": max_tokens,
        "splits": {},
    }

    split_stats["splits"]["train"] = encode_split(
        tokenizer,
        train_jsonl,
        out_dir / "train.bin",
        max_tokens=max_tokens,
    )

    split_stats["splits"]["val"] = encode_split(
        tokenizer,
        val_jsonl,
        out_dir / "val.bin",
        max_tokens=max_tokens,
    )

    split_stats["splits"]["test"] = encode_split(
        tokenizer,
        test_jsonl,
        out_dir / "test.bin",
        max_tokens=max_tokens,
    )

    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "tokenizer_stats.json"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    print("\nSaved files:")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"  train bin: {out_dir / 'train.bin'}")
    print(f"  val bin:   {out_dir / 'val.bin'}")
    print(f"  test bin:  {out_dir / 'test.bin'}")
    print(f"  stats:     {stats_path}")


if __name__ == "__main__":
    main()