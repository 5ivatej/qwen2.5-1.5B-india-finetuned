# scripts/make_tiny_sft.py
import os, json, random, re
from typing import List, Dict, Any, Iterable, Tuple
from datasets import load_dataset

random.seed(7)
OUT = "data/sft.jsonl"
os.makedirs("data", exist_ok=True)

# Keep this light so you can train quickly on Mac
MAX_DOCS_DOLLY   = 4000   # docs to sample from Dolly_T
MAX_DOCS_ANUDESH = 4000   # docs to sample from Anudesh
MAX_PAIRS_PER_DOC = 2     # take up to N (user->assistant) pairs per doc to keep it small

# Prefer native-script columns first, fall back to Latin if needed
LANG_PREFS = [
    # Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati (adjust as you like)
    "hin_Deva","ben_Beng","tam_Taml","tel_Telu","mar_Deva","guj_Gujr",
    # fallbacks (latin transliterations)
    "hin_Latn","ben_Latn","tam_Latn","tel_Latn","mar_Latn","guj_Latn",
    # finally English if nothing else
    "eng_Latn"
]

def clean(s: Any) -> str:
    if s is None: return ""
    s = str(s).strip()
    # normalize whitespace a bit
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def pair_from_list(items: List[Any]) -> List[Tuple[str,str]]:
    """
    Turn a list into [(user, assistant), ...].
    Supports:
      - list[str] alternating u/a/u/a...
      - list[dict] with role/from/speaker + content/value/text
      - nested lists, we'll flatten one level
    """
    pairs = []
    if not items:
        return pairs

    # Flatten one level if needed (sometimes it's [[..],[..]])
    if items and isinstance(items[0], list):
        flat = []
        for x in items:
            flat.extend(x if isinstance(x, list) else [x])
        items = flat

    # dict-style?
    if isinstance(items[0], dict):
        user_buf, asst_buf = None, None
        for m in items:
            role = (m.get("role") or m.get("from") or m.get("speaker") or m.get("author") or "").lower()
            text = (m.get("content") or m.get("value") or m.get("text") or m.get("utterance") or "")
            text = clean(text)
            if not text:
                continue
            if role in ("user","human","prompt","question"):
                if user_buf is None:
                    user_buf = text
                else:
                    # consecutive users: flush previous if no assistant yet
                    user_buf = text
            elif role in ("assistant","gpt","bot","answer","response","agent"):
                if user_buf is not None:
                    asst_buf = text
                    pairs.append((user_buf, asst_buf))
                    user_buf, asst_buf = None, None
            else:
                # Unknown role: if we have a pending user, treat as assistant
                if user_buf is not None:
                    pairs.append((user_buf, text))
                    user_buf = None
        return pairs

    # string-style alternating
    if isinstance(items[0], str):
        items = [clean(x) for x in items if clean(x)]
        for i in range(0, len(items)-1, 2):
            u = items[i].strip()
            a = items[i+1].strip()
            if u and a:
                pairs.append((u, a))
        return pairs

    # fallback: nothing recognized
    return pairs

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_from_dolly_t() -> List[Dict[str, Any]]:
    print("[Dolly_T] Loading (cached if already downloaded)...")
    ds = load_dataset("ai4bharat/indic-align", "Dolly_T", split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:MAX_DOCS_DOLLY]

    out_rows = []
    for i in idxs:
        ex = ds[i]
        # pick first available preferred language column
        col = next((c for c in LANG_PREFS if c in ex and ex[c]), None)
        if not col:
            # last ditch: any non-empty list column
            col = next((k for k,v in ex.items() if isinstance(v, list) and v), None)
        if not col:
            continue

        pairs = pair_from_list(ex[col])
        random.shuffle(pairs)
        for u,a in pairs[:MAX_PAIRS_PER_DOC]:
            out_rows.append({
                "instruction": u,
                "input": "",
                "output": a,
                "language": col
            })
    print(f"[Dolly_T] Collected {len(out_rows)} pairs")
    return out_rows

def build_from_anudesh() -> List[Dict[str, Any]]:
    print("[Anudesh] Loading (cached if already downloaded)...")
    ds = load_dataset("ai4bharat/indic-align", "Anudesh", split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[:MAX_DOCS_ANUDESH]

    out_rows = []
    for i in idxs:
        ex = ds[i]
        interactions = ex.get("interactions") or []
        pairs = pair_from_list(interactions)
        random.shuffle(pairs)
        for u,a in pairs[:MAX_PAIRS_PER_DOC]:
            out_rows.append({
                "instruction": u,
                "input": "",
                "output": a,
                "language": "xx"
            })
    print(f"[Anudesh] Collected {len(out_rows)} pairs")
    return out_rows

if __name__ == "__main__":
    all_rows = []
    all_rows += build_from_dolly_t()
    all_rows += build_from_anudesh()
    random.shuffle(all_rows)
    write_jsonl(OUT, all_rows)
    print(f"Wrote {len(all_rows)} examples to {OUT}")
