import re
import json
from transformers import AutoTokenizer

def tokenize_len(text: str, retriever_tokenizer) -> int:
    if retriever_tokenizer is None:
        return len(text.split())
    return len(retriever_tokenizer.encode(text, truncation=False))

def extract_json_like(raw_text: str):
    try:
        text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        j = m.group(0)
        j = re.sub(r",\s*}", "}", j)
        j = re.sub(r",\s*\]", "]", j)
        return json.loads(j)
    except Exception:
        try:
            j2 = text.replace("'", '"')
            m2 = re.search(r"\{.*\}", j2, flags=re.DOTALL)
            if m2:
                j3 = m2.group(0)
                j3 = re.sub(r",\s*}", "}", j3)
                j3 = re.sub(r",\s*\]", "]", j3)
                return json.loads(j3)
        except Exception:
            return None

def clean_json_response(raw_text: str):
    parsed = extract_json_like(raw_text)
    if not parsed:
        return None
    for k, v in list(parsed.items()):
        if isinstance(v, str) and v.strip() in ["...", "is_blank", ""]:
            parsed[k] = "N/A"
    ref = parsed.get("ref_id", [])
    if ref in [None, "Unknown", "", [], "N/A"]:
        parsed["ref_id"] = []
    elif isinstance(ref, str):
        if "," in ref:
            parsed["ref_id"] = [r.strip() for r in ref.split(",") if r.strip()]
        else:
            parsed["ref_id"] = [ref]
    elif isinstance(ref, (list, tuple, set)):
        parsed["ref_id"] = list(ref)
    else:
        parsed["ref_id"] = [str(ref)]
    return parsed