# scripts/prepare_svg_data.py

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from lxml import etree


COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Safer MVP number cleanup: only round decimal numbers, not integers.
# This avoids accidentally changing hex colors like #00ff00.
DECIMAL_RE = re.compile(r"(?<![#A-Za-z0-9])[-+]?(?:\d+\.\d+|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_hf_ref(dataset_ref: str) -> str:
    """
    Allows both:
      starvector/svg-icons-simple
      https://huggingface.co/datasets/starvector/svg-icons-simple
    """
    dataset_ref = dataset_ref.strip()

    prefix = "https://huggingface.co/datasets/"
    if dataset_ref.startswith(prefix):
        return dataset_ref[len(prefix):].strip("/")

    prefix = "http://huggingface.co/datasets/"
    if dataset_ref.startswith(prefix):
        return dataset_ref[len(prefix):].strip("/")

    return dataset_ref


def find_svg_column(example: dict) -> str:
    """
    Try to find the field containing SVG XML.
    """
    preferred_names = ["svg", "Svg", "SVG", "code", "content", "text"]

    for name in preferred_names:
        if name in example and isinstance(example[name], str) and "<svg" in example[name]:
            return name

    for key, value in example.items():
        if isinstance(value, str) and "<svg" in value:
            return key

    raise ValueError(f"Could not find SVG field. Available keys: {list(example.keys())}")


def round_decimal(match):
    try:
        x = float(match.group(0))
        return f"{x:.1f}"
    except Exception:
        return match.group(0)


def clean_svg(svg_text: str, min_chars: int, round_numbers: bool):
    """
    Minimal SVG cleaning + XML validation.
    Returns:
      cleaned_svg, status
    """
    if not isinstance(svg_text, str):
        return None, "not_string"

    x = svg_text.strip()

    if not x:
        return None, "empty"

    # Remove XML comments.
    x = COMMENT_RE.sub("", x)

    # Collapse unnecessary whitespace.
    x = re.sub(r">\s+<", "><", x)
    x = re.sub(r"\s+", " ", x).strip()

    # Optional decimal rounding.
    if round_numbers:
        x = DECIMAL_RE.sub(round_decimal, x)

    if len(x) < min_chars:
        return None, "too_short"

    # Validate XML and check root is SVG.
    try:
        root = etree.fromstring(x.encode("utf-8"))
    except Exception:
        return None, "xml_invalid"

    root_tag = str(root.tag).lower()
    if not root_tag.endswith("svg"):
        return None, "root_not_svg"

    return x, "ok"


def iter_dataset_examples(dataset_name: str):
    """
    Load a HuggingFace dataset and yield examples from all splits.
    We re-split ourselves later to avoid relying on dataset-provided splits.
    """
    ds = load_dataset(dataset_name)

    if hasattr(ds, "keys"):
        for split_name in ds.keys():
            for ex in ds[split_name]:
                yield split_name, ex
    else:
        for ex in ds:
            yield "train", ex


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def process_datasets(args):
    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "cleaned"
    stats_dir = out_dir / "stats"

    cleaned_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    global_stats = Counter()

    for dataset_arg in args.datasets:
        dataset_name = normalize_hf_ref(dataset_arg)

        print(f"\nLoading dataset: {dataset_name}")
        example_iter = iter_dataset_examples(dataset_name)

        svg_col = None
        dataset_stats = Counter()
        seen = 0
        kept = 0

        for idx, (original_split, ex) in enumerate(example_iter):
            if args.limit_per_dataset is not None and seen >= args.limit_per_dataset:
                break

            seen += 1

            if svg_col is None:
                svg_col = find_svg_column(ex)
                print(f"Detected SVG column for {dataset_name}: {svg_col}")
                print(f"Available columns: {list(ex.keys())}")

            raw_svg = ex.get(svg_col)
            cleaned_svg, status = clean_svg(
                raw_svg,
                min_chars=args.min_chars,
                round_numbers=args.round_numbers,
            )

            dataset_stats[status] += 1
            global_stats[status] += 1

            if status == "ok":
                rec = {
                    "id": f"{dataset_name.replace('/', '_')}_{idx}",
                    "source": dataset_name,
                    "original_split": original_split,
                    "svg": cleaned_svg,
                    "char_len": len(cleaned_svg),
                }
                all_records.append(rec)
                kept += 1

        print(f"Finished {dataset_name}")
        print(f"Seen: {seen}")
        print(f"Kept: {kept}")
        print(f"Stats: {dict(dataset_stats)}")

    print("\nTotal cleaned records:", len(all_records))
    print("Global stats:", dict(global_stats))

    return all_records, global_stats


def split_records(records, train_frac, val_frac, seed):
    random.seed(seed)
    random.shuffle(records)

    n = len(records)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_records = records[:n_train]
    val_records = records[n_train:n_train + n_val]
    test_records = records[n_train + n_val:]

    return train_records, val_records, test_records


def main():
    parser = argparse.ArgumentParser(
        description="Download, clean, validate, and split SVG datasets."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "HuggingFace dataset names or URLs. Example: "
            "starvector/svg-icons-simple "
            "https://huggingface.co/datasets/starvector/svg-emoji-simple"
        ),
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="data",
        help="Output directory.",
    )

    parser.add_argument(
        "--min_chars",
        type=int,
        default=50,
        help="Drop SVGs shorter than this many characters.",
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.98,
        help="Train split fraction.",
    )

    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.01,
        help="Validation split fraction. Test gets the remainder.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )

    parser.add_argument(
        "--round_numbers",
        action="store_true",
        help="Round decimal numbers to 1 decimal place.",
    )

    parser.add_argument(
        "--limit_per_dataset",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )

    args = parser.parse_args()

    records, stats = process_datasets(args)

    train_records, val_records, test_records = split_records(
        records,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    out_dir = Path(args.out_dir)
    cleaned_dir = out_dir / "cleaned"
    stats_dir = out_dir / "stats"

    write_jsonl(train_records, cleaned_dir / "train.jsonl")
    write_jsonl(val_records, cleaned_dir / "val.jsonl")
    write_jsonl(test_records, cleaned_dir / "test.jsonl")

    summary = {
        "total_records": len(records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "test_records": len(test_records),
        "filter_stats": dict(stats),
        "datasets": [normalize_hf_ref(x) for x in args.datasets],
        "min_chars": args.min_chars,
        "round_numbers": args.round_numbers,
        "seed": args.seed,
    }

    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "data_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    print(f"  {cleaned_dir / 'train.jsonl'}")
    print(f"  {cleaned_dir / 'val.jsonl'}")
    print(f"  {cleaned_dir / 'test.jsonl'}")
    print(f"  {stats_dir / 'data_summary.json'}")

    print("\nSplit sizes:")
    print(f"  train: {len(train_records)}")
    print(f"  val:   {len(val_records)}")
    print(f"  test:  {len(test_records)}")


if __name__ == "__main__":
    main()