# SVG-LLM: Scaling Laws for Decoder-Only Transformers on SVG Code

This repository contains code for training decoder-only Transformer language models on SVG code. The project studies scaling behavior across model sizes, learning-rate transfer, µP parameterization, and SVG sample generation/evaluation.

The task is framed as next-token prediction over tokenized SVG/XML code. Generated outputs are evaluated both as text and as rendered images using XML validity and CairoSVG render success.

---
<span style="color:red">## !! Please check the python notebook in the repository for detailed history of training and execution</span>


## Project Summary

This project implements:

- SVG data download, cleaning, XML validation, and JSONL export
- ByteLevel BPE tokenizer training for SVG code
- Tokenization into binary token streams
- Decoder-only Transformer training
- Learning-rate sweep on the smallest model
- Standard model scaling study across five model sizes
- µP learning-rate sweep and width-scaling experiment
- SVG generation and evaluation
- XML validity, SVG root validity, render-rate, and test perplexity metrics
- Report artifact generation: plots, sample grids, and summary CSV/JSON files

---

## Repository Structure

```text
svg-llm/
├── model.py                         # Standard decoder-only Transformer model
├── train.py                         # Standard training script
├── model_mup.py                     # µP-compatible Transformer model
├── train_mup.py                     # µP training script
├── requirements.txt                 # Python dependencies
├── tokenizer/
│   └── svg_bpe_4096.json            # Trained SVG BPE tokenizer
├── scripts/
│   ├── 01_download_and_inspect.py   # Download and inspect raw SVG datasets
│   ├── 02_clean_svg.py              # Clean and validate SVG records
│   ├── 03_split_data.py             # Create train/val/test splits
│   ├── train_svg_tokenizer.py       # Train BPE tokenizer and create token .bin files
│   ├── build_generation_dataset.py  # Build <=1024-token render-valid dataset
│   ├── generate_eval_svg.py         # Initial generation/evaluation script
│   ├── generate_eval_svg_v2.py      # Generation using real SVG prefixes
│   └── large_generation_sweep.py    # Large generation sweep and rendered sample grid
├── data/
│   └── stats/
│       ├── data_summary.json
│       └── tokenizer_stats.json
└── reports/
    └── figures/                     # Saved plots and rendered sample grids
```

Large files such as cleaned JSONL data, token `.bin` files, checkpoints, and generated outputs are not committed to GitHub. They should be stored locally or on Google Drive.

---

## Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For Colab, install requirements after cloning:

```bash
git clone https://github.com/rohitfarfad/svg-llm.git
cd svg-llm
pip install -r requirements.txt
```

---

## Data

The project uses SVG datasets from HuggingFace:

- `starvector/svg-icons-simple`
- `starvector/svg-emoji-simple`
- `starvector/svg-fonts-simple`

The cleaned dataset summary was:

```text
Total cleaned SVG records: 244,476
Train records: 239,586
Validation records: 2,444
Test records: 2,446
Minimum character length: 50
XML-valid records: 244,476
```

After tokenization and filtering to SVGs with at most 2048 tokens, the main scaling dataset contained:

```text
Train tokens: 177,418,221
Validation tokens: 1,834,317
Test tokens: 1,789,826
```

A separate generation-focused dataset was also built by filtering to SVGs with at most 1024 tokens and requiring successful CairoSVG rendering:

```text
Train tokens: 98,155,882
Validation tokens: 1,000,067
Test tokens: 1,000,158
```

---

## Data Pipeline

### 1. Download and inspect datasets

```bash
python scripts/01_download_and_inspect.py \
  --datasets starvector/svg-icons-simple starvector/svg-emoji-simple starvector/svg-fonts-simple \
  --out_dir data/raw
```

### 2. Clean SVGs

```bash
python scripts/02_clean_svg.py \
  --input_dir data/raw \
  --output_file data/cleaned/all.jsonl \
  --min_chars 50
```

### 3. Split data

```bash
python scripts/03_split_data.py \
  --input_file data/cleaned/all.jsonl \
  --output_dir data/cleaned \
  --train_ratio 0.98 \
  --val_ratio 0.01 \
  --test_ratio 0.01 \
  --seed 42
```

---

## Tokenizer

The project uses a ByteLevel BPE tokenizer trained on SVG code.

Tokenizer configuration:

```text
Tokenizer type: ByteLevel BPE
Vocabulary size: 4096
Special tokens: [PAD], [BOS], [EOS], [UNK]
Tokenizer path: tokenizer/svg_bpe_4096.json
```

Train tokenizer and create binary token streams:

```bash
python scripts/train_svg_tokenizer.py \
  --train_jsonl data/cleaned/train.jsonl \
  --val_jsonl data/cleaned/val.jsonl \
  --test_jsonl data/cleaned/test.jsonl \
  --tokenizer_out tokenizer/svg_bpe_4096.json \
  --output_dir data/tokens \
  --vocab_size 4096 \
  --max_tokens 2048
```

This creates:

```text
data/tokens/train.bin
data/tokens/val.bin
data/tokens/test.bin
data/stats/tokenizer_stats.json
```

---

## Standard Transformer Training

### Tiny sanity check

```bash
python train.py \
  --train_bin data/tokens/train.bin \
  --val_bin data/tokens/val.bin \
  --out_dir checkpoints/tiny_sanity \
  --vocab_size 4096 \
  --block_size 1024 \
  --n_layer 4 \
  --n_head 4 \
  --n_embd 128 \
  --batch_size 16 \
  --max_iters 500 \
  --lr 0.0003 \
  --min_lr 0.00003 \
  --warmup_iters 100 \
  --eval_interval 100 \
  --eval_iters 10 \
  --device cuda
```

---

## Learning Rate Sweep

The learning-rate sweep was performed on the Tiny model. The selected standard learning rate was:

```text
Best standard LR: 6e-3
```

Initial sweep values:

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

Refined sweep values:

```text
6e-3, 1e-2, 2e-2, 3e-2
```

The refined sweep selected `6e-3`, which was then used for all standard scaling runs.

---

## Standard Scaling Study

The five standard models trained were:

| Model | Params | Layers | Heads | Embedding Dim | Block Size |
|---|---:|---:|---:|---:|---:|
| Tiny | 1.45M | 4 | 4 | 128 | 1024 |
| Small | 3.65M | 6 | 6 | 192 | 1024 |
| Medium | 12.61M | 6 | 6 | 384 | 1024 |
| Medium Plus | 21.61M | 8 | 8 | 448 | 1024 |
| Large | 27.84M | 8 | 8 | 512 | 1024 |

Example command for Medium Plus:

```bash
python train.py \
  --train_bin data/tokens/train.bin \
  --val_bin data/tokens/val.bin \
  --out_dir checkpoints/scaling_standard/medium_plus \
  --vocab_size 4096 \
  --block_size 1024 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 448 \
  --batch_size 8 \
  --max_iters 21657 \
  --lr 0.006 \
  --min_lr 0.0006 \
  --warmup_iters 300 \
  --eval_interval 1000 \
  --eval_iters 30 \
  --device cuda
```

Standard scaling results:

| Model | Best Val Loss |
|---|---:|
| Tiny | 1.7532 |
| Small | 1.3640 |
| Medium | 1.2438 |
| Medium Plus | 1.1640 |
| Large | 1.2983 |

The best standard model was `medium_plus`.

---

## µP Experiments

The repository also includes an MVP µP implementation using the `mup` package.

### µP LR sweep

The µP proxy LR sweep tested:

```text
1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2
```

Best µP LR:

```text
Best µP LR: 1e-2
Best validation loss: 2.1372
```

### µP scaling

The width-scaled µP models were:

| Model | Params | Layers | Heads | Embedding Dim | Best Val Loss |
|---|---:|---:|---:|---:|---:|
| mup_w128 | 2.37M | 6 | 4 | 128 | 1.5683 |
| mup_w192 | 4.44M | 6 | 6 | 192 | 1.4044 |
| mup_w256 | 7.10M | 6 | 8 | 256 | 1.3103 |
| mup_w384 | 14.19M | 6 | 8 | 384 | 1.4044 |
| mup_w512 | 23.63M | 6 | 8 | 512 | 1.3680 |

The best µP model was `mup_w256`, but it did not outperform the best standard model.

---

## Generation-Focused Dataset

To improve generation validity, a filtered dataset was created with:

- SVG token length <= 1024
- XML-valid SVGs
- CairoSVG-renderable SVGs
- Same 4096-token BPE tokenizer

Build the dataset:

```bash
python scripts/build_generation_dataset.py \
  --input_data_dir data \
  --output_data_dir generation_dataset_1024 \
  --tokenizer tokenizer/svg_bpe_4096.json \
  --max_tokens 1024 \
  --target_train_tokens 105000000 \
  --target_val_tokens 1000000 \
  --target_test_tokens 1000000
```

---

## Final Generation Model

The final generation model used the Medium Plus architecture trained on the filtered generation dataset.

Training command:

```bash
python train.py \
  --train_bin generation_dataset_1024/tokens/train.bin \
  --val_bin generation_dataset_1024/tokens/val.bin \
  --out_dir checkpoints/generation_1024/medium_plus \
  --vocab_size 4096 \
  --block_size 1024 \
  --n_layer 8 \
  --n_head 8 \
  --n_embd 448 \
  --batch_size 16 \
  --max_iters 5981 \
  --lr 0.003 \
  --min_lr 0.0003 \
  --warmup_iters 300 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --eval_interval 1000 \
  --eval_iters 30 \
  --device cuda
```

Final generation-focused model:

```text
Params: 21,608,832
Best validation loss: 1.2539
Test loss: 1.2645
Test perplexity: 3.5412
```

---

## SVG Generation and Evaluation

Run a large generation sweep:

```bash
python scripts/large_generation_sweep.py \
  --ckpt checkpoints/generation_1024/medium_plus/best.pt \
  --tokenizer tokenizer/svg_bpe_4096.json \
  --test_bin generation_dataset_1024/tokens/test.bin \
  --val_jsonl generation_dataset_1024/cleaned/val.jsonl \
  --out_dir generation/large_sweep_generation_1024 \
  --device cuda \
  --eval_batch_size 8 \
  --max_new_tokens 1024 \
  --prefix_tokens 256 \
  --num_unconditional 200 \
  --num_real_prefix 100 \
  --max_grid_items 25
```

Final large generation sweep results:

```text
Total generated samples: 300
Unconditional samples: 200
Real-prefix samples: 100
Valid XML samples: 53
Renderable samples: 53
Overall XML validity rate: 17.67%
Overall render rate: 17.67%
Real-prefix render rate: 43.0%
Best unconditional setting: temperature 0.3, top-k 10, render rate 10.0%
```

Generated outputs are saved under:

```text
generation/large_sweep_generation_1024/
├── svg/
├── png/
├── generation_summary.json
├── generation_metrics.csv
├── generation_grouped_summary.csv
├── valid_samples.csv
└── valid_samples_grid_white_bg.png
```

---

## Important Outputs

The main files used for the report are:

```text
data/stats/data_summary.json
data/stats/tokenizer_stats.json
checkpoints/scaling_standard/scaling_standard_summary.csv
checkpoints/mup_lr_sweep/mup_lr_sweep_summary.csv
checkpoints/mup_scaling/mup_scaling_summary.csv
generation/large_sweep_generation_1024/generation_summary.json
generation/large_sweep_generation_1024/generation_grouped_summary.csv
generation/large_sweep_generation_1024/valid_samples_grid_white_bg.png
reports/figures/
```

---

## Notes on Large Files

The following files are intentionally not committed to GitHub because they are large:

```text
data/cleaned/*.jsonl
data/tokens/*.bin
generation_dataset_1024/
checkpoints/
generation/
```

These files should be stored on Google Drive or generated locally using the scripts in this repository.

Recommended `.gitignore` entries:

```gitignore
data/cleaned/
data/tokens/
generation_dataset_1024/
checkpoints/
generation/
*.pt
*.bin
```

---

## Report

The final report is organized as:

```text
1. Introduction
2. Data
3. Methods
4. Results
5. Discussion
6. Conclusion
```

The report includes:

- dataset statistics
- tokenizer summary
- standard scaling results
- power-law fit
- standard vs. µP comparison
- generation evaluation
- rendered SVG examples

---

## Dependencies

Main dependencies:

```text
torch
numpy
pandas
matplotlib
tqdm
tokenizers
datasets
lxml
cairosvg
Pillow
mup
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Acknowledgements

This project was completed for CS-GY 6923 Machine Learning at NYU Tandon. The model code is inspired by nanoGPT, with modifications for SVG tokenization, scaling experiments, µP experiments, and SVG-specific generation/evaluation.
