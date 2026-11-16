"""Score one year's reports with one headless call per report: Claude Code (`claude -p`) or
Codex (`codex exec`, --engine codex; the system prompt is prepended to the message).

Each call gets system_prompt.md as its system prompt, one report as the user message
(user_prompt_template.md) and output_schema.json as the structured-output schema, with
no tools. Results are validated like score_tools.py submit; a result whose quotes are
not verbatim is retried once with the problems appended. Already-scored task_ids in the
output file are skipped, so the script can be stopped and rerun. On a usage limit it
waits and retries.

  python3 run_cli_scoring.py 2019 --workers 4 [--model claude-haiku-5-5] [--effort medium]
  python3 run_cli_scoring.py 2016 --engine codex --workers 1 [--model gpt-6-luna]
Writes results/<year>/tmp/cli.jsonl (codex: codex.jsonl). Task_ids already in any
results/<year>/tmp/*.jsonl are skipped. Prints progress and token usage.
"""
import argparse
import os
import gzip
import json
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from score_tools import DIMS

HERE = Path(__file__).resolve().parent
SYSTEM = HERE / "system_prompt.md"
SCHEMA = (HERE / "output_schema.json").read_text(encoding="utf-8")
TEMPLATE = (HERE / "user_prompt_template.md").read_text(encoding="utf-8")
LIMIT_WAIT = 2700
ORDER_NOTE = "\n对每个维度，请先摘录证据、写出理由，最后再给出分值；分值必须由证据和上述细则直接决定。"
QUOTE_NOTE = ("\n\n格式提醒：evidence 的 quote 必须是年报文本中连续的原文片段，建议只摘 15–40 个汉字的短句，"
              "宁短勿长；不要缩写、不要改写、不要用省略号，不要跨越 [第N页] 标记，标点和数字与原文完全一致。")
lock = threading.Lock()
usage = {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "done": 0}


def problems(r, text, check_quotes=True):
    body = re.sub(r"\s+", "", text)
    out = []
    if r.get("text_quality") not in ("ok", "incomplete", "garbled"):
        out.append("text_quality must be ok/incomplete/garbled")
    for k in DIMS:
        d = r.get(k) or {}
        if d.get("level") not in (0, 1, 2, 3, 4):
            out.append(f"{k}: bad level")
            continue
        ev = d.get("evidence") or []
        if (d["level"] == 0) != (not ev):
            out.append(f"{k}: level 0 needs empty evidence, levels 1-4 need evidence")
        for e in ev if check_quotes else []:
            q = re.sub(r"\s+", "", e.get("quote", ""))
            if not q or q not in body:
                out.append(f"{k}: quote not found verbatim: {e.get('quote', '')[:60]}")
    return out


def call(msg, model, effort):
    cmd = ["claude", "-p", "--model", model, "--effort", effort, "--system-prompt-file", str(SYSTEM),
           "--tools", "", "--strict-mcp-config", "--no-session-persistence",
           "--json-schema", SCHEMA, "--output-format", "json"]
    while True:
        try:
            p = subprocess.run(cmd, input=msg, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            return None
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            d = {"is_error": True, "result": p.stdout[-300:] + p.stderr[-300:]}
        err = str(d.get("result", "")) + p.stderr
        if d.get("is_error") and re.search(r"limit|429|rate", err, re.I):
            print(f"usage limit, waiting {LIMIT_WAIT // 60} min", flush=True)
            time.sleep(LIMIT_WAIT)
            continue
        u = d.get("usage") or {}
        with lock:
            usage["input"] += u.get("input_tokens", 0)
            usage["cache_write"] += u.get("cache_creation_input_tokens", 0)
            usage["cache_read"] += u.get("cache_read_input_tokens", 0)
            usage["output"] += u.get("output_tokens", 0)
        return d.get("structured_output") if not d.get("is_error") else None


def call_codex(msg, model, effort):
    prompt = SYSTEM.read_text(encoding="utf-8") + "\n\n---\n\n" + msg + QUOTE_NOTE + (ORDER_NOTE if os.environ.get("CODEX_SCHEMA") else "")
    with tempfile.TemporaryDirectory() as d:
        last = Path(d) / "last.json"
        cmd = ["codex", "exec", "-m", model, "-c", f"model_reasoning_effort={effort}", "--ephemeral",
               "--ignore-user-config", "--ignore-rules", "-s", "read-only", "--skip-git-repo-check",
               "-C", d, "--output-schema", os.environ.get("CODEX_SCHEMA", str(HERE / "output_schema.json")), "-o", str(last), "-"]
        while True:
            try:
                p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                return None
            tail = (p.stdout + p.stderr)[-2000:]
            if p.returncode != 0 and re.search(r"usage limit|rate limit|429|quota", tail, re.I):
                print(f"usage limit, waiting {LIMIT_WAIT // 60} min", flush=True)
                time.sleep(LIMIT_WAIT)
                continue
            m = re.search(r"tokens used\s*\n?\s*([\d,]+)", tail)
            with lock:
                usage["output"] += int(m.group(1).replace(",", "")) if m else 0
            try:
                return json.loads(last.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None


def score(task, out, model, effort, engine):
    call_fn = call_codex if engine == "codex" else call
    tid, text = task["task_id"], task["text"]
    msg = TEMPLATE.format(task_id=tid, text=text)
    check = engine != "codex"  # codex quotes are repaired afterwards by repair_quotes.py (Haiku)
    r = call_fn(msg, model, effort)
    probs = problems(r, text, check) if r else ["no structured output"]
    if probs:
        retry = msg + "\n\n上一次输出未通过校验，请修正以下问题后重新给出完整评分（证据必须逐字摘自原文）：\n" + "\n".join(probs)
        r = call_fn(retry, model, effort)
        probs = problems(r, text, check) if r else ["no structured output"]
    if probs:
        print(f"FAILED {tid}: {probs[:2]}", flush=True)
        return
    r["task_id"] = tid
    with lock:
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        usage["done"] += 1
        if usage["done"] % 10 == 0:
            print(f"{usage['done']} done; tokens {usage}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--engine", choices=["claude", "codex"], default="claude")
    ap.add_argument("--model")
    ap.add_argument("--effort", default="medium")
    a = ap.parse_args()
    a.model = a.model or ("gpt-6-luna" if a.engine == "codex" else "claude-haiku-5-5")
    out = HERE / f"results/{a.year}/{os.environ.get('SCORE_SUBDIR', 'tmp')}/{'codex' if a.engine == 'codex' else 'cli'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in out.parent.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
                if all(isinstance(r.get(k), dict) for k in DIMS):
                    done.add(r["task_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    with gzip.open(HERE / f"mdna/{a.year}.jsonl.gz", "rt", encoding="utf-8") as f:
        tasks = [t for t in map(json.loads, f) if t["task_id"] not in done]
    print(f"{len(done)} already scored, {len(tasks)} to go", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda t: score(t, out, a.model, a.effort, a.engine), tasks))
    print(f"finished; tokens {usage}", flush=True)


if __name__ == "__main__":
    main()
