---
language:
- en
- kn
- hi
- ta
- te
- mr
- gu
tags:
- lora
- sft
- indic
- qwen2.5
license: other
base_model: Qwen/Qwen2.5-1.5B
datasets:
- ai4bharat/indic-align
library_name: mlx
pipeline_tag: text-generation
---

# qwen1.5B-india-finetuned

## FOR HUGGINGFACE LINK : KINDLY VISIT : https://huggingface.co/5ivatej/qwen2.5-1.5B-india-finetuned

## Overview
This repository contains **Qwen2.5-1.5B** fine-tuned with **LoRA** on small Indic instruction-following datasets.  
The LoRA adapters were merged into the base weights, producing a **standalone checkpoint** that can be used directly with `mlx_lm`.

---

## License
- The base model [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) is released under the [Qwen License](https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/LICENSE).  
- This fine-tuned checkpoint is subject to the same license. Please review the terms before use, especially for commercial scenarios.  
- Marked here as `license: other` to follow Hugging Face conventions.

---

## Training Configuration
- **Method:** LoRA-SFT (attention + MLP)
- **LoRA hyperparams:** `r=16, alpha=32, dropout=0.05`
- **Max sequence length:** 1024
- **Steps:** 1500
- **Batch size:** 1
- **Optimizer:** AdamW (default in `mlx_lm`)
- **Hardware:** Apple Silicon (MacBook Pro M4)  
- **Framework:** [`mlx_lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)

The training configuration YAML used can be found at:
[`configs/qwen2.5-3b_lora.yaml`](./configs/qwen2.5-3b_lora.yaml)

---

## Data
- Subsets from [ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align) were used: **Dolly_T** and **Anudesh**.
- Converted into completion-style prompts (`prompt`/`completion` pairs).  
- The focus is on **Indic languages** (Kannada, Hindi, Tamil, Telugu, Marathi, Gujarati) with some English instructions.  
- Preprocessed into `train.jsonl` / `valid.jsonl` (not included here).

---

## Usage

Run with `mlx_lm`:

```bash
python -m mlx_lm generate \
  --model 5ivatej/qwen25-1p5b-india-merged \
  --max-tokens 200 \
  --prompt "Reply ONLY in Kannada written in English letters. Question: kannada dalli mathadoo?\n\nAnswer:"
