import json, os
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B"  # match your YAML
MAX_TOK = 1024

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

def clip_file(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = json.loads(line)
            # MLX "completions" format: {"prompt": "...", "completion": "..."}
            prompt = ex.get("prompt","")
            comp   = ex.get("completion","")
            ids_p  = tok(prompt, add_special_tokens=False)["input_ids"]
            ids_c  = tok(comp, add_special_tokens=False)["input_ids"]

            # if too long, truncate from the END of completion first (keep the question intact)
            room = MAX_TOK - len(ids_p)
            if room <= 0:
                # prompt alone is too big -> chop prompt tail
                ids_p = ids_p[:max(1, MAX_TOK//2)]
                room  = MAX_TOK - len(ids_p)

            if len(ids_c) > room:
                ids_c = ids_c[:max(1, room)]

            new_prompt = tok.decode(ids_p, skip_special_tokens=True)
            new_comp   = tok.decode(ids_c, skip_special_tokens=True)

            if new_comp.strip():  # keep only if we still have an answer
                out.append({"prompt": new_prompt, "completion": new_comp})

    tmp = path + ".clipped"
    with open(tmp, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"Clipped {path}: kept {len(out)} examples ≤ {MAX_TOK} tokens")

for p in ["data/train.jsonl", "data/valid.jsonl"]:
    clip_file(p)
