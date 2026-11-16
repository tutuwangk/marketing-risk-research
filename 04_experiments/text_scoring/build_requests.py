"""Assemble the scoring prompt and the per-report requests.

system_prompt.md = the 评分规则 section of rubric.md + the 35 calibration anchors
(anchors.csv, set=C). Each request pairs that fixed prompt with one MD&A text from
mdna/ and asks for JSON matching output_schema.json.

Usage:
  python3 build_requests.py                          # (re)write system_prompt.md and anchor_test_inputs.jsonl (no gold levels)
  python3 build_requests.py --anchors --model M      # 65 held-out anchors -> requests/anchor_test.jsonl
  python3 build_requests.py --sample 100 --model M   # random pilot -> requests/pilot.jsonl
  python3 build_requests.py --all --model M          # every report, one file per year

Request files use the OpenAI Batch API format (/v1/chat/completions, strict JSON
schema). The same system prompt and user message work in any chat interface.
"""
import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMS = {
    "market_sensing": "市场感知",
    "brand_management": "品牌管理",
    "customer_relationship": "客户关系管理",
    "channel_management": "渠道管理",
    "product_development": "产品开发与新品",
    "marketing_communication": "营销沟通",
    "digital_marketing": "数字营销",
}
USER_TEMPLATE = (HERE / "user_prompt_template.md").read_text(encoding="utf-8")


def load_anchors():
    with open(HERE / "anchors.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def system_prompt():
    rubric = (HERE / "rubric.md").read_text(encoding="utf-8")
    rules = rubric.split("## 评分规则", 1)[1].split("## 校准与使用流程", 1)[0].strip()
    rules = re.sub(r"^### ", "## ", rules, flags=re.M)
    parts = [
        "你是营销学研究的文本编码员，按下面的评分细则，对一份中国A股上市公司年报的管理层讨论与分析章节，"
        "就七个营销能力维度分别给出0–4分的叙事强度评分。严格按细则判断，不使用细则以外的标准。",
        rules,
        "## 校准锚点",
        "以下段落取自其他公司年报，标注了按本细则应得的分值。它们说明“如果全文在该维度的证据就是这样，应得几分”，"
        "用来统一尺度；不要把锚点内容当作待评年报的证据。",
    ]
    anchors = [a for a in load_anchors() if a["set"] == "C"]
    for key, name in DIMS.items():
        parts.append(f"### {name}（{key}）")
        for a in sorted((a for a in anchors if a["dimension"] == name), key=lambda a: int(a["level"])):
            parts.append(f"【{a['level']}分】{a['text']}\n理由：{a['rationale']}")
    text = "\n\n".join(parts) + "\n"
    (HERE / "system_prompt.md").write_text(text, encoding="utf-8")
    return text


def request(custom_id, task_text, sys_text, schema, model, effort):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": USER_TEMPLATE.format(task_id=custom_id, text=task_text)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {"name": "marketing_capability_scores", "strict": True, "schema": schema}},
    }
    if effort:
        body["reasoning_effort"] = effort
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}


def iter_reports():
    for f in sorted((HERE / "mdna").glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                yield json.loads(line)


def write(path, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(HERE)}: {len(rows)} requests")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="exact model id, e.g. a dated snapshot")
    ap.add_argument("--effort", help="reasoning_effort for reasoning models (low/medium/high)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--anchors", action="store_true")
    g.add_argument("--sample", type=int)
    g.add_argument("--all", action="store_true")
    a = ap.parse_args()
    sys_text = system_prompt()
    write(HERE / "anchor_test_inputs.jsonl",
          [{"task_id": x["anchor_id"], "text": x["text"]} for x in load_anchors() if x["set"] == "T"])
    print("system_prompt.md sha256:", hashlib.sha256(sys_text.encode()).hexdigest(), f"({len(sys_text)} chars)")
    if not (a.anchors or a.sample or a.all):
        return
    if not a.model:
        ap.error("--model is required when building requests")
    schema = json.loads((HERE / "output_schema.json").read_text(encoding="utf-8"))
    mk = lambda cid, text: request(cid, text, sys_text, schema, a.model, a.effort)
    out = HERE / "requests"
    if a.anchors:
        write(out / "anchor_test.jsonl", [mk(x["anchor_id"], x["text"]) for x in load_anchors() if x["set"] == "T"])
    elif a.sample:
        recs = list(iter_reports())
        random.Random(2026).shuffle(recs)
        write(out / "pilot.jsonl", [mk(r["task_id"], r["text"]) for r in recs[: a.sample]])
    else:
        by_year = {}
        for r in iter_reports():
            by_year.setdefault(r["task_id"].split("_")[1], []).append(mk(r["task_id"], r["text"]))
        for y, rows in sorted(by_year.items()):
            write(out / f"reports_{y}.jsonl", rows)


if __name__ == "__main__":
    main()
