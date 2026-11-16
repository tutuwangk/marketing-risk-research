"""Wrap-up steps for one scored year, using headless Claude Haiku calls where a model is needed.

  python3 finalize_year.py YEAR merge     tmp/*.jsonl -> scores.jsonl (one valid row per report,
                                          pages re-derived from where each quote sits); reports gaps
  python3 finalize_year.py YEAR retest    5% sample re-scored independently -> retest.jsonl
  python3 finalize_year.py YEAR channel   re-score channel_management only for reports whose reason
                                          applied the main-business exclusion to a firm that sells its
                                          own goods or services through its own channel; patches scores.jsonl
Quote repair for Codex-scored years is `repair_quotes.py YEAR`, run between merge and the other steps.
"""
import gzip
import json
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_cli_scoring as rcs
from repair_quotes import norm, page_of
from score_tools import DIMS

HERE = Path(__file__).resolve().parent
EXCLUDED = re.compile(r"主营|本身|排除|不计|自身业务|零售业务|经营门店")
CHANNEL_SCHEMA = json.dumps({
    "type": "object", "additionalProperties": False, "required": ["channel_management"],
    "properties": {"channel_management": {"type": "object", "additionalProperties": False,
        "required": ["level", "evidence", "reason"], "properties": {
            "level": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            "evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["page", "quote"], "properties": {"page": {"type": "integer"}, "quote": {"type": "string"}}}},
            "reason": {"type": "string"}}}}})
CHANNEL_NOTE = (
    "\n\n只重评 channel_management 一个维度，并按以下说明理解细则：“主营业务本身就是渠道时不计分”只适用于"
    "向其他企业出售渠道、物流或分销服务的公司；零售商、百货、超市、汽车经销商、分销商、出版发行、院线等公司"
    "自有的门店、网点、经销网络、线上店铺是其销售自家商品或服务的渠道，其描述、扩张、调整和管理应当计入，"
    "按常规0–4分标准判断。证据逐字摘自原文（15–40字短句），page 填所在页码。")


def load(year):
    with gzip.open(HERE / f"mdna/{year}.jsonl.gz", "rt", encoding="utf-8") as f:
        return {r["task_id"]: r["text"] for r in map(json.loads, f)}


def valid_row(r):
    return isinstance(r, dict) and all(isinstance(r.get(k), dict) and r[k].get("level") in (0, 1, 2, 3, 4)
                                       and isinstance(r[k].get("evidence"), list) for k in DIMS)


def merge(year):
    texts = load(year)
    rows = {}
    for f in sorted((HERE / f"results/{year}/{os.environ.get('SCORE_SUBDIR', 'tmp')}").glob("*.jsonl")):
        if f.name.startswith(("quote_repair", "evidence_repair", "channel")):
            continue
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid_row(r) and r.get("task_id") in texts and r["task_id"] not in rows:
                r.pop("scorer_model", None)
                rows[r["task_id"]] = r
    for r in rows.values():
        for k in DIMS:
            for e in r[k]["evidence"]:
                p = page_of(texts[r["task_id"]], e["quote"])
                if p is not None:
                    e["page"] = p
    out = HERE / f"results/{year}/scores.jsonl"
    with open(out, "w", encoding="utf-8") as g:
        for t in texts:
            if t in rows:
                g.write(json.dumps(rows[t], ensure_ascii=False) + "\n")
    print(f"{year}: {len(rows)} of {len(texts)} reports merged into {out.name}; missing {len(texts) - len(rows)}")


def retest(year):
    texts = load(year)
    ids = sorted(texts)
    sample = random.Random(int(year)).sample(ids, max(1, round(len(ids) * 0.05)))
    out = HERE / f"results/{year}/retest.jsonl"
    out.unlink(missing_ok=True)
    tasks = [{"task_id": t, "text": texts[t]} for t in sample]
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(lambda t: rcs.score(t, out, "claude-haiku-5-5", "medium", "claude"), tasks))
    print(f"{year}: {sum(1 for _ in open(out, encoding='utf-8'))} of {len(sample)} re-test reports scored")


def channel(year):
    texts = load(year)
    path = HERE / f"results/{year}/scores.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    cand = [r for r in rows if EXCLUDED.search(r["channel_management"]["reason"])]
    print(f"{year}: {len(cand)} candidates")

    def one(r):
        text = texts[r["task_id"]]
        msg = rcs.TEMPLATE.format(task_id=r["task_id"], text=text) + CHANNEL_NOTE
        for _ in range(2):
            p = subprocess.run(["claude", "-p", "--model", "claude-haiku-5-5", "--effort", "medium",
                                "--system-prompt-file", str(rcs.SYSTEM), "--tools", "", "--strict-mcp-config",
                                "--no-session-persistence", "--json-schema", CHANNEL_SCHEMA, "--output-format", "json"],
                               input=msg, capture_output=True, text=True, timeout=1800)
            try:
                c = json.loads(p.stdout).get("structured_output", {}).get("channel_management")
            except json.JSONDecodeError:
                c = None
            if c and all(norm(e["quote"]) in norm(text) for e in c["evidence"]) and (c["level"] == 0) == (not c["evidence"]):
                for e in c["evidence"]:
                    e["page"] = page_of(text, e["quote"])
                return r["task_id"], c
        return r["task_id"], None

    with ThreadPoolExecutor(2) as ex:
        new = dict(ex.map(one, cand))
    changed = []
    for r in rows:
        c = new.get(r["task_id"])
        if c:
            changed.append((r["task_id"], r["channel_management"]["level"], c["level"]))
            r["channel_management"] = c
    with open(path, "w", encoding="utf-8") as g:
        for r in rows:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    up = sum(1 for _, a, b in changed if b > a)
    print(f"{year}: {len(changed)} re-scored, {up} up, {sum(1 for _, a, b in changed if b < a)} down, "
          f"{sum(1 for _, a, b in changed if b == a)} unchanged; not repaired: {len(cand) - len(changed)}")


if __name__ == "__main__":
    {"merge": merge, "retest": retest, "channel": channel}[sys.argv[2]](sys.argv[1])
