# scripts/prepare_data.py
import os, json, random
from typing import Dict, Any, Iterable
from datasets import load_dataset, get_dataset_config_names

random.seed(7)

OUT = "data/sft.jsonl"
os.makedirs("data/raw", exist_ok=True)

# ---- config you can tweak ----
langs = ["hi", "bn", "ta", "te", "mr", "gu"]  # which Indic languages to keep
max_indicalign = 60000
max_indicqa     = 20000
max_samanantar  = 20000   # set 0 to skip
# ------------------------------

def add_row(rows, instruction, inp, output, lang):
    if not instruction or not output:
        return
    rows.append({
        "instruction": instruction.strip(),
        "input": (inp or "").strip(),
        "output": output.strip(),
        "language": lang or "xx"
    })

def norm_lang(val: str) -> str:
    if not val: return "xx"
    v = val.lower().strip()
    # normalize common labels
    mapping = {
        "hin":"hi", "hi-in":"hi", "hindi":"hi",
        "ben":"bn", "bengali":"bn",
        "tam":"ta", "tamil":"ta",
        "tel":"te", "telugu":"te",
        "mar":"mr", "marathi":"mr",
        "guj":"gu", "gujarati":"gu",
        "en":"en", "eng":"en", "english":"en",
    }
    return mapping.get(v, v)

def load_indic_align() -> Iterable[Dict[str, Any]]:
    wanted = {
        "Indic_ShareLlama", "Dolly_T", "OpenAssistant_T",
        "Anudesh", "Wiki_Conv", "Wiki_Chat", "HHRLHF_T"
    }
    configs = [c for c in get_dataset_config_names("ai4bharat/indic-align") if c in wanted]
    rows = []

    for cfg in configs:
        print(f"[Indic-Align] Loading {cfg}…")
        ds = load_dataset("ai4bharat/indic-align", cfg, split="train")
        for ex in ds:
            # best-effort key fishing across subsets
            lang = norm_lang(ex.get("lang") or ex.get("language") or ex.get("lang_code") or "")
            if lang not in langs and lang != "en":
                continue

            instr = (ex.get("instruction") or ex.get("prompt") or ex.get("query")
                     or ex.get("question") or ex.get("input") or "")
            # some sets have "context"
            ctx = ex.get("context") or ex.get("passage") or ""
            if ctx and instr:
                instr = f"{instr}\n\nContext:\n{ctx}"

            out = (ex.get("output") or ex.get("response") or ex.get("answer")
                   or ex.get("target") or ex.get("text") or "")

            # for conversational subsets
            if not instr and isinstance(ex.get("conversations"), list):
                conv = ex["conversations"]
                # assume first user → assistant
                user_msgs = [c.get("value") for c in conv if c.get("from","").lower() in ("human","user")]
                asst_msgs = [c.get("value") for c in conv if c.get("from","").lower() in ("assistant","gpt","bot")]
                if user_msgs and asst_msgs:
                    instr = user_msgs[0]
                    out = asst_msgs[0]
            add_row(rows, instr, "", out, lang)

    random.shuffle(rows)
    return rows[:max_indicalign]

def load_indicqa() -> Iterable[Dict[str, Any]]:
    rows = []
    try:
        print("[IndicQA] Loading…")
        ds = load_dataset("ai4bharat/IndicQA", split="train")
        for ex in ds:
            lang = norm_lang(ex.get("language") or ex.get("lang") or "")
            if lang not in langs:
                continue
            q = ex.get("question") or ""
            ctx = ex.get("context") or ""
            ans = ex.get("answer") or ""
            instr = f"Answer the question in {lang.upper()}."
            if ctx:
                instr += f"\nContext:\n{ctx}"
            add_row(rows, instr, q, ans, lang)
    except Exception as e:
        print(f"[IndicQA] Skipped ({e})")
    random.shuffle(rows)
    return rows[:max_indicqa]

def load_samanantar() -> Iterable[Dict[str, Any]]:
    if max_samanantar <= 0:
        return []
    rows = []
    try:
        print("[Samanantar] Loading (this can be large; we will sample)…")
        ds = load_dataset("ai4bharat/samanantar", split="train")
        for ex in ds:
            src_lang = norm_lang(ex.get("src_lang") or ex.get("source_language") or "en")
            tgt_lang = norm_lang(ex.get("tgt_lang") or ex.get("target_language") or "")
            if src_lang == "en" and tgt_lang in langs:
                src = ex.get("source_sentence") or ex.get("src") or ""
                tgt = ex.get("target_sentence") or ex.get("tgt") or ""
                instr = f"Translate from English to {tgt_lang.upper()}."
                add_row(rows, instr, src, tgt, tgt_lang)
    except Exception as e:
        print(f"[Samanantar] Skipped ({e})")
    random.shuffle(rows)
    return rows[:max_samanantar]

def write_jsonl(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    all_rows = []
    all_rows += load_indic_align()
    all_rows += load_indicqa()
    all_rows += load_samanantar()
    random.shuffle(all_rows)
    write_jsonl(OUT, all_rows)
    print(f"Wrote {len(all_rows)} examples to {OUT}")
