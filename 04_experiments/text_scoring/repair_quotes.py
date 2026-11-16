"""Replace evidence quotes that are not verbatim in the report text (one headless Haiku call per report).

For each report in results/<year>/scores.jsonl with such quotes, the model is shown the text, the
dimension's level and reason, and the faulty quotes, and returns replacement quotes copied exactly
from the text. Levels and reasons are never changed. Replacements are validated, page numbers are
re-derived from where the quote sits, and the file is rewritten at the end. Resumable via
results/<year>/tmp/quote_repair.jsonl.

  python3 repair_quotes.py 2014 [--workers 1] [--model claude-haiku-5-5]
"""
import argparse
import gzip
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from score_tools import DIMS

HERE = Path(__file__).resolve().parent
SCHEMA = json.dumps({
    "type": "object", "additionalProperties": False, "required": ["repairs"],
    "properties": {"repairs": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["dimension", "quotes"],
        "properties": {"dimension": {"type": "string", "enum": DIMS},
                       "quotes": {"type": "array", "items": {"type": "string"}}}}}}})
SYSTEM = ("你是文本编码员的校对助手。年报评分里有些证据摘录不是原文的逐字片段。请针对每个列出的维度，"
          "从给定年报文本中重新挑选1–3条能支持该维度已定分值的摘录，每条必须逐字、连续地复制自文本中的一个片段（不超过80个汉字，"
          "不跨页标记、不改写、不拼接、不加省略号）。不要改分值或理由。")


def norm(s):
    return re.sub(r"\s+", "", s)


def page_of(text, quote):
    body, marks = "", []
    for line in text.split("\n"):
        m = re.match(r"\[第(\d+)页\]", line)
        if m:
            marks.append((len(body), int(m.group(1))))
        else:
            body += norm(line)
    i = body.find(norm(quote))
    return None if i < 0 else max([p for pos, p in marks if pos <= i], default=0)


def ask(model, msg):
    while True:
        p = subprocess.run(["claude", "-p", "--model", model, "--effort", "medium", "--system-prompt", SYSTEM,
                            "--tools", "", "--strict-mcp-config", "--no-session-persistence",
                            "--json-schema", SCHEMA, "--output-format", "json"],
                           input=msg, capture_output=True, text=True, timeout=1800)
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            return None
        if d.get("is_error") and re.search(r"limit|429|rate", str(d.get("result", "")), re.I):
            print("usage limit, waiting 45 min", flush=True)
            time.sleep(2700)
            continue
        return d.get("structured_output")


FINAL_SCHEMA = json.dumps({
    "type": "object", "additionalProperties": False, "required": ["repairs"],
    "properties": {"repairs": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["dimension", "sentences"],
        "properties": {"dimension": {"type": "string", "enum": DIMS},
                       "sentences": {"type": "array", "items": {"type": "integer"}}}}}}})
FINAL_SYSTEM = ("你是文本编码员的校对助手。下面的年报文本已按句编号。请针对每个列出的维度，挑选1–3个能支持该维度已定分值的句子，"
                "只返回句子编号。不要改分值或理由。")


def sentences(text):
    out = []
    for line in text.split("\n"):
        if re.match(r"\[第\d+页\]", line):
            continue
        for s in re.split(r"(?<=[。！？；])", line):
            if len(norm(s)) >= 8:
                out.append(s.strip())
    return out


def repair_final(row, text, model):
    """Last resort: the model only picks sentence numbers; quotes are the sentences themselves (<=80 chars)."""
    sents = sentences(text)
    bad = {k: [e["quote"] for e in row[k]["evidence"] if norm(e["quote"]) not in norm(text)]
           for k in DIMS if any(norm(e["quote"]) not in norm(text) for e in row[k]["evidence"])}
    items = "\n".join(f"- {k}（分值{row[k]['level']}，理由：{row[k]['reason']}）" for k in bad)
    numbered = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sents))
    msg = f"需要修补的维度：\n{items}\n\n<编号后的年报文本>\n{numbered}\n</编号后的年报文本>"
    p = subprocess.run(["claude", "-p", "--model", model, "--effort", "medium", "--system-prompt", FINAL_SYSTEM,
                        "--tools", "", "--strict-mcp-config", "--no-session-persistence",
                        "--json-schema", FINAL_SCHEMA, "--output-format", "json"],
                       input=msg, capture_output=True, text=True, timeout=1800)
    try:
        r = json.loads(p.stdout).get("structured_output") or {}
    except json.JSONDecodeError:
        return None
    fix = {}
    for x in r.get("repairs", []):
        qs = [sents[i - 1][:80] for i in x["sentences"] if 1 <= i <= len(sents)]
        qs = [q for q in qs if page_of(text, q) is not None]
        if x["dimension"] in bad and qs:
            fix[x["dimension"]] = [{"page": page_of(text, q), "quote": q} for q in qs[:3]]
    return fix if set(fix) == set(bad) else None


def repair(row, text, model):
    bad = {k: [e["quote"] for e in row[k]["evidence"] if norm(e["quote"]) not in norm(text)]
           for k in DIMS if any(norm(e["quote"]) not in norm(text) for e in row[k]["evidence"])}
    items = "\n".join(f"- {k}（分值{row[k]['level']}，理由：{row[k]['reason']}）有问题的摘录：{'；'.join(q)}" for k, q in bad.items())
    msg = f"需要修补的维度：\n{items}\n\n<年报文本>\n{text}\n</年报文本>"
    for _ in range(2):
        r = ask(model, msg)
        done = {}
        for x in (r or {}).get("repairs", []):
            qs = [q for q in x["quotes"] if page_of(text, q) is not None]
            if x["dimension"] in bad and qs:
                done[x["dimension"]] = qs
        if set(done) == set(bad):
            return {k: [{"page": page_of(text, q), "quote": q} for q in qs] for k, qs in done.items()}
        msg += "\n\n上一次的摘录未能全部在原文中逐字找到，请重新复制，只挑选短而确定的片段。"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--final", action="store_true", help="last resort: model picks numbered sentences")
    a = ap.parse_args()
    scores = HERE / f"results/{a.year}/scores.jsonl"
    log = HERE / f"results/{a.year}/{os.environ.get('SCORE_SUBDIR', 'tmp')}/quote_repair{'_final' if a.final else ''}.jsonl"
    log.parent.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(scores, encoding="utf-8")]
    with gzip.open(HERE / f"mdna/{a.year}.jsonl.gz", "rt", encoding="utf-8") as f:
        texts = {r["task_id"]: r["text"] for r in map(json.loads, f)}
    done = {json.loads(l)["task_id"] for l in open(log, encoding="utf-8")} if log.exists() else set()
    todo = [r for r in rows if r["task_id"] not in done and
            any(norm(e["quote"]) not in norm(texts[r["task_id"]]) for k in DIMS for e in r[k]["evidence"])]
    print(f"{len(todo)} reports to repair", flush=True)

    def work(row):
        fix = (repair_final if a.final else repair)(row, texts[row["task_id"]], a.model)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_id": row["task_id"], "fix": fix}, ensure_ascii=False) + "\n")
        print(row["task_id"], "ok" if fix else "FAILED", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    fixes = {}
    for l in (open(log, encoding="utf-8") if log.exists() else []):
        r = json.loads(l)
        if r["fix"]:
            fixes[r["task_id"]] = r["fix"]
    for row in rows:
        for k, ev in fixes.get(row["task_id"], {}).items():
            row[k]["evidence"] = ev
    with open(scores, "w", encoding="utf-8") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("rewrote", scores, flush=True)


if __name__ == "__main__":
    main()
