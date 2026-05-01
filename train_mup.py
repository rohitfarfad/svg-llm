import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import mup
from tqdm.auto import tqdm

from model_mup import GPTMuPConfig, build_mup_models


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64))
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, device):
    model.eval()

    results = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = float(np.mean(losses))

    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_bin", type=str, required=True)
    parser.add_argument("--val_bin", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # Model
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    # µP base shape widths
    parser.add_argument("--base_width", type=int, default=128)
    parser.add_argument("--delta_width", type=int, default=256)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--min_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging/eval
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=50)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")

    config = GPTMuPConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )

    model, base_model, delta_model = build_mup_models(
        target_config=config,
        base_width=args.base_width,
        delta_width=args.delta_width,
    )

    model = model.to(device)

    n_params = model.count_params()

    print("µP model config:")
    print(config)
    print(f"Number of parameters: {n_params:,}")
    print(f"Base width: {args.base_width}")
    print(f"Delta width: {args.delta_width}")

    # µP optimizer
    optimizer = mup.MuAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    best_val_loss = float("inf")
    start_time = time.time()
    last_time = start_time

    log_rows = []

    pbar = tqdm(
        range(args.max_iters),
        total=args.max_iters,
        desc=f"µP train width={args.n_embd}",
        dynamic_ncols=True,
    )

    for step in pbar:
        lr = get_lr(
            step=step,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_iters,
            max_steps=args.max_iters,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.block_size, device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        if step % args.log_interval == 0:
            now = time.time()
            dt = now - last_time
            last_time = now

            tokens_per_iter = args.batch_size * args.block_size
            tokens_per_sec = tokens_per_iter * args.log_interval / max(dt, 1e-8)

            if args.device == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
            else:
                mem_gb = 0.0

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "memGB": f"{mem_gb:.2f}",
            })

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, args, device)

            print(
                f"eval iter {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            row = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "params": n_params,
                "elapsed_sec": time.time() - start_time,
            }
            log_rows.append(row)

            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                ckpt = {
                    "model": model.state_dict(),
                    "config": config.__dict__,
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                    "params": n_params,
                }

                torch.save(ckpt, out_dir / "best.pt")

    final_summary = {
        "params": n_params,
        "best_val_loss": best_val_loss,
        "final_train_loss": log_rows[-1]["train_loss"],
        "final_val_loss": log_rows[-1]["val_loss"],
        "elapsed_sec": time.time() - start_time,
        "args": vars(args),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    print("\nDone.")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()