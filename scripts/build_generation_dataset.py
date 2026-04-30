# scripts/build_generation_dataset.py

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm.auto import tqdm
from lxml import etree
import cairosvg
from tokenizers import Tokenizer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_valid_xml(svg):
    try:
        root = etree.fromstring(svg.encode("utf-8"))
        return str(root.tag).lower().endswith("svg")
    except Exception:
        return False


def renders_ok(svg):
    try:
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=64, output_height=64)
        return True
    except Exception:
        return False


def filter_split(
    input_path,
    output_jsonl,
    tokenizer,
    max_tokens,
    target_tokens=None,
    check_render=True,
):
    eos_id = tokenizer.token_to_id("[EOS]")

    records = []
    all_ids = []
    stats = Counter()

    total_tokens = 0

    input_records = list(read_jsonl(input_path))

    for rec in tqdm(input_records, desc=f"Filtering {input_path.name}"):
        svg = rec["svg"]

        ids = tokenizer.encode(svg).ids
        n_tokens = len(ids)

        if n_tokens > max_tokens:
            stats["too_long"] += 1
            continue

        if not is_valid_xml(svg):
            stats["xml_invalid"] += 1
            continue

        if check_render and not renders_ok(svg):
            stats["render_invalid"] += 1
            continue

        new_rec = dict(rec)
        new_rec["token_len"] = n_tokens
        new_rec["render_valid"] = True

        records.append(new_rec)

        ids.append(eos_id)
        all_ids.extend(ids)

        total_tokens += len(ids)
        stats["kept"] += 1

        if target_tokens is not None and total_tokens >= target_tokens:
            break

    write_jsonl(records, output_jsonl)

    arr = np.array(all_ids, dtype=np.uint16)
    output_bin = output_jsonl.parent.parent / "tokens" / output_jsonl.name.replace(".jsonl", ".bin")
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(output_bin)

    return {
        "input": str(input_path),
        "output_jsonl": str(output_jsonl),
        "output_bin": str(output_bin),
        "max_tokens": max_tokens,
        "target_tokens": target_tokens,
        "check_render": check_render,
        "kept_records": len(records),
        "total_tokens_with_eos": int(len(arr)),
        "stats": dict(stats),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data_dir", required=True)
    parser.add_argument("--output_data_dir", required=True)
    parser.add_argument("--tokenizer", required=True)

    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--target_train_tokens", type=int, default=105_000_000)
    parser.add_argument("--target_val_tokens", type=int, default=1_000_000)
    parser.add_argument("--target_test_tokens", type=int, default=1_000_000)
    parser.add_argument("--no_render_check", action="store_true")

    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)

    input_cleaned = Path(args.input_data_dir) / "cleaned"
    output_dir = Path(args.output_data_dir)
    output_cleaned = output_dir / "cleaned"
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    check_render = not args.no_render_check

    summary = {}

    summary["train"] = filter_split(
        input_path=input_cleaned / "train.jsonl",
        output_jsonl=output_cleaned / "train.jsonl",
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        target_tokens=args.target_train_tokens,
        check_render=check_render,
    )

    summary["val"] = filter_split(
        input_path=input_cleaned / "val.jsonl",
        output_jsonl=output_cleaned / "val.jsonl",
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        target_tokens=args.target_val_tokens,
        check_render=check_render,
    )

    summary["test"] = filter_split(
        input_path=input_cleaned / "test.jsonl",
        output_jsonl=output_cleaned / "test.jsonl",
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
        target_tokens=args.target_test_tokens,
        check_render=check_render,
    )

    with open(stats_dir / "generation_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()