"""Two helpers for agents scoring reports one at a time with few tool calls.

  python3 score_tools.py show <task_id> [part]
      print the MD&A text in parts of at most 20000 chars (the header says how many)
  python3 score_tools.py submit <out.jsonl> <<'JSON'
  {...one scoring object...}
  JSON
      validate it (all seven dimensions, levels 0-4, evidence empty iff level 0,
      quotes verbatim in the text) and append it; prints OK or the problems
"""
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMS = ["market_sensing", "brand_management", "customer_relationship", "channel_management",
        "product_development", "marketing_communication", "digital_marketing"]


def text_of(tid):
    try:
        with gzip.open(HERE / f"mdna/{tid.split('_')[1]}.jsonl.gz", "rt", encoding="utf-8") as f:
            return next((r["text"] for r in map(json.loads, f) if r["task_id"] == tid), None)
    except (IndexError, FileNotFoundError):
        return None


def show(tid, part):
    text = text_of(tid)
    if text is None:
        sys.exit("task_id not found")
    parts, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) > 20000 and cur:
            parts.append(cur)
            cur = ""
        cur += para + "\n"
    parts.append(cur)
    print(f"=== {tid} part {part} of {len(parts)} ===")
    print(parts[part - 1])


def submit(out):
    try:
        r = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        sys.exit(f"NOT APPENDED: invalid JSON ({e})")
    text = text_of(r.get("task_id", ""))
    probs = [] if text else [f"unknown task_id {r.get('task_id')}"]
    body = re.sub(r"\s+", "", text or "")
    if r.get("text_quality") not in ("ok", "incomplete", "garbled"):
        probs.append("text_quality must be ok/incomplete/garbled")
    for k in DIMS:
        d = r.get(k)
        if not isinstance(d, dict) or d.get("level") not in (0, 1, 2, 3, 4):
            probs.append(f"{k}: missing or bad level")
            continue
        ev = d.get("evidence")
        if not isinstance(ev, list) or not isinstance(d.get("reason"), str):
            probs.append(f"{k}: evidence must be a list and reason a string")
            continue
        if (d["level"] == 0) != (not ev):
            probs.append(f"{k}: level 0 needs empty evidence, levels 1-4 need evidence")
        for e in ev:
            q = re.sub(r"\s+", "", e.get("quote", ""))
            if not q or q not in body:
                probs.append(f"{k}: quote not found verbatim: {e.get('quote', '')[:40]}")
    if probs:
        print("NOT APPENDED:\n" + "\n".join(probs))
        sys.exit(1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("OK", r["task_id"])


if __name__ == "__main__":
    if sys.argv[1] == "show":
        show(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
    else:
        submit(sys.argv[2])
