%%writefile /content/svg-llm/scripts/generate_eval_svg.py

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from lxml import etree
import cairosvg
from PIL import Image, ImageDraw

from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, config, ckpt


def crop_context(idx, block_size):
    if idx.size(1) > block_size:
        idx = idx[:, -block_size:]
    return idx


def apply_top_k_top_p(logits, top_k=0, top_p=1.0):
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        cutoff = values[:, [-1]]
        logits[logits < cutoff] = -float("inf")

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)

        remove = cum_probs > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False

        for b in range(logits.size(0)):
            indices_to_remove = sorted_indices[b][remove[b]]
            logits[b, indices_to_remove] = -float("inf")

    return logits


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    device,
    block_size,
    max_new_tokens=1024,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = crop_context(idx, block_size)

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            logits = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

        text = tokenizer.decode(idx[0].tolist())

        # Stop early when model closes the SVG.
        if "</svg>" in text:
            break

    text = tokenizer.decode(idx[0].tolist())

    # Trim anything after first closing SVG tag.
    if "</svg>" in text:
        text = text[: text.find("</svg>") + len("</svg>")]

    return text


def validate_xml(svg_text):
    try:
        root = etree.fromstring(svg_text.encode("utf-8"))
        root_is_svg = str(root.tag).lower().endswith("svg")
        return True, root_is_svg, None
    except Exception as e:
        return False, False, str(e)


def render_svg(svg_text, png_path):
    try:
        cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            write_to=str(png_path),
            output_width=256,
            output_height=256,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def make_placeholder(path, text="invalid"):
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 110), text, fill="black")
    img.save(path)


def make_grid(image_paths, labels, out_path, cols=5, cell=256, label_h=32):
    rows = math.ceil(len(image_paths) / cols)
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(grid)

    for i, img_path in enumerate(image_paths):
        r = i // cols
        c = i % cols

        try:
            img = Image.open(img_path).convert("RGB").resize((cell, cell))
        except Exception:
            img = Image.new("RGB", (cell, cell), "white")

        x = c * cell
        y = r * (cell + label_h)

        grid.paste(img, (x, y))
        draw.text((x + 5, y + cell + 5), labels[i][:35], fill="black")

    grid.save(out_path)


@torch.no_grad()
def evaluate_test_loss(model, test_bin, block_size, batch_size, device):
    data = np.memmap(test_bin, dtype=np.uint16, mode="r")

    starts = list(range(0, len(data) - block_size - 1, block_size))
    losses = []
    token_counts = []

    for i in tqdm(range(0, len(starts), batch_size), desc="Evaluating test loss"):
        batch_starts = starts[i : i + batch_size]

        x = torch.stack([
            torch.from_numpy(data[s : s + block_size].astype(np.int64))
            for s in batch_starts
        ]).to(device)

        y = torch.stack([
            torch.from_numpy(data[s + 1 : s + 1 + block_size].astype(np.int64))
            for s in batch_starts
        ]).to(device)

        _, loss = model(x, y)

        losses.append(loss.item())
        token_counts.append(x.numel())

    avg_loss = float(np.average(losses, weights=token_counts))
    perplexity = float(math.exp(avg_loss))

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--test_bin", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--eval_batch_size", type=int, default=8)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    svg_dir = out_dir / "svg"
    png_dir = out_dir / "png"

    svg_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, config, ckpt = load_model(args.ckpt, device)

    print("Loaded checkpoint:", args.ckpt)
    print("Model config:", config)
    print("Best val loss:", ckpt.get("best_val_loss"))
    print("Params:", ckpt.get("params"))

    # Test loss and perplexity.
    test_loss, test_ppl = evaluate_test_loss(
        model=model,
        test_bin=args.test_bin,
        block_size=config.block_size,
        batch_size=args.eval_batch_size,
        device=device,
    )

    print("Test loss:", test_loss)
    print("Test perplexity:", test_ppl)

    unconditional_prompt = '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'

    prefix_prompts = [
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" fill="none" stroke="black"/><circle cx="8" cy="9" r="1.5" fill="black"/>',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M4 12',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><g transform="translate(2 2)">',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2"',
        '<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="6" fill="#',
    ]

    sample_specs = []

    # 10 unconditional samples with varied temperatures.
    temps = [0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 0.7, 0.9]
    for i, temp in enumerate(temps):
        sample_specs.append({
            "kind": "unconditional",
            "name": f"uncond_{i:02d}_temp_{temp}",
            "prompt": unconditional_prompt,
            "temperature": temp,
        })

    # 5 prefix-conditioned samples.
    for i, prompt in enumerate(prefix_prompts):
        sample_specs.append({
            "kind": "prefix",
            "name": f"prefix_{i:02d}",
            "prompt": prompt,
            "temperature": 0.8,
        })

    records = []
    image_paths = []
    labels = []

    for spec in tqdm(sample_specs, desc="Generating samples"):
        text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=spec["prompt"],
            device=device,
            block_size=config.block_size,
            max_new_tokens=args.max_new_tokens,
            temperature=spec["temperature"],
            top_k=args.top_k,
            top_p=args.top_p,
        )

        svg_path = svg_dir / f"{spec['name']}.svg"
        png_path = png_dir / f"{spec['name']}.png"

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(text)

        xml_valid, root_is_svg, xml_error = validate_xml(text)

        if xml_valid:
            render_ok, render_error = render_svg(text, png_path)
        else:
            render_ok, render_error = False, "skipped because XML invalid"
            make_placeholder(png_path, "invalid XML")

        image_paths.append(png_path)
        labels.append(f"{spec['name']}")

        records.append({
            "name": spec["name"],
            "kind": spec["kind"],
            "temperature": spec["temperature"],
            "top_k": args.top_k,
            "top_p": args.top_p,
            "prompt": spec["prompt"],
            "svg_path": str(svg_path),
            "png_path": str(png_path),
            "char_len": len(text),
            "xml_valid": xml_valid,
            "root_is_svg": root_is_svg,
            "render_ok": render_ok,
            "xml_error": xml_error,
            "render_error": render_error,
        })

    grid_path = out_dir / "generated_grid.png"
    make_grid(image_paths, labels, grid_path, cols=5)

    df_path = out_dir / "generation_metrics.csv"
    json_path = out_dir / "generation_metrics.json"

    import pandas as pd

    df = pd.DataFrame(records)
    df.to_csv(df_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    summary = {
        "checkpoint": args.ckpt,
        "tokenizer": args.tokenizer,
        "params": ckpt.get("params"),
        "checkpoint_best_val_loss": ckpt.get("best_val_loss"),
        "test_loss": test_loss,
        "test_perplexity": test_ppl,
        "num_samples": len(records),
        "xml_valid_rate": float(df["xml_valid"].mean()),
        "svg_root_rate": float(df["root_is_svg"].mean()),
        "render_rate": float(df["render_ok"].mean()),
        "num_unconditional": int((df["kind"] == "unconditional").sum()),
        "num_prefix": int((df["kind"] == "prefix").sum()),
    }

    summary_path = out_dir / "generation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nGeneration/evaluation summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print("SVG samples:", svg_dir)
    print("PNG samples:", png_dir)
    print("Grid:", grid_path)
    print("Metrics CSV:", df_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()