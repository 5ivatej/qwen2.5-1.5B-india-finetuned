# scripts/make_mlx_splits.py
import json, os, random
random.seed(7)

SRC = "data/sft.jsonl"
OUT_DIR = "data"
TRAIN = os.path.join(OUT_DIR, "train.jsonl")
VALID = os.path.join(OUT_DIR, "valid.jsonl")
VAL_FRACTION = 0.1  # 10% dev set

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

rows = list(read_jsonl(SRC))
random.shuffle(rows)

n = len(rows)
n_val = max(1, int(n * VAL_FRACTION))
valid = rows[:n_val]
train = rows[n_val:]

def to_completion(rec):
    instr = (rec.get("instruction") or "").strip()
    inp   = (rec.get("input") or "").strip()
    out   = (rec.get("output") or "").strip()
    # build a simple prompt-completion pair
    if inp:
        prompt = f"{instr}\n\nInput:\n{inp}\n\nAnswer:"
    else:
        prompt = f"{instr}\n\nAnswer:"
    completion = out
    return {"prompt": prompt, "completion": completion}

os.makedirs(OUT_DIR, exist_ok=True)
with open(TRAIN, "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(to_completion(r), ensure_ascii=False) + "\n")

with open(VALID, "w", encoding="utf-8") as f:
    for r in valid:
        f.write(json.dumps(to_completion(r), ensure_ascii=False) + "\n")

print(f"Wrote {len(train)} -> {TRAIN}")
print(f"Wrote {len(valid)} -> {VALID}")
