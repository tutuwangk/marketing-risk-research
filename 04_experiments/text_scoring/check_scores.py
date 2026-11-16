"""Check GPT scoring output and assemble the final score table.

Accepts JSONL in either form: OpenAI Batch API output lines, or one scoring JSON
object per line. Several files may be passed; a task scored more than once is
treated as a re-test pair.

  python3 check_scores.py --anchors results/anchor_test_out.jsonl
      agreement with the 65 held-out anchors (exact, within 1, weighted kappa)
  python3 check_scores.py results/pilot_out.jsonl
      schema, coverage, verbatim evidence, level distribution, re-test agreement
  python3 check_scores.py results/*.jsonl --write
      also writes 02_data/final/annual_report_text_scores.csv (first run of each task)
"""
import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_requests import DIMS, HERE

ROOT = HERE.parents[1]
FINAL = ROOT / "02_data/final/annual_report_text_scores.csv"
VAR = {k: f"{v}叙事强度" for k, v in DIMS.items()}
SPACE = re.compile(r"\s+")


def parse(path):
    out = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        if "response" in rec:
            body = (rec.get("response") or {}).get("body") or {}
            try:
                rec = json.loads(body["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                out.append({"_error": f"{path.name}:{n} no parsable content", "task_id": cid})
                continue
        if cid and rec.get("task_id") != cid:
            rec["_id_mismatch"] = rec.get("task_id")
            rec["task_id"] = cid
        out.append(rec)
    return out


def valid(rec):
    if "_error" in rec:
        return rec["_error"]
    for k in DIMS:
        d = rec.get(k)
        if not isinstance(d, dict) or d.get("level") not in (0, 1, 2, 3, 4):
            return f"bad {k}"
        if not isinstance(d.get("evidence"), list):
            return f"bad {k}.evidence"
    return None


def qwk(a, b, k=5):
    """Quadratic weighted kappa for integer ratings 0..k-1."""
    n = len(a)
    if n == 0:
        return float("nan")
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ra, rb = Counter(a), Counter(b)
    num = sum(((i - j) ** 2) * obs[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) * ra[i] * rb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else 1.0


def agreement(pairs, label):
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    if not pairs:
        return
    exact = sum(x == y for x, y in pairs) / len(pairs)
    near = sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs)
    print(f"{label:<14} n={len(pairs):>4}  exact={exact:.1%}  within1={near:.1%}  qwk={qwk(a, b):.3f}")


def check_anchors(recs):
    with open(HERE / "anchors.csv", encoding="utf-8-sig") as f:
        gold = {a["anchor_id"]: a for a in csv.DictReader(f) if a["set"] == "T"}
    name2key = {v: k for k, v in DIMS.items()}
    pairs, misses = defaultdict(list), []
    for r in recs:
        g = gold.get(r.get("task_id"))
        if g is None or valid(r):
            continue
        targets = (
            {name2key[k]: int(v) for k, v in (x.split("=") for x in g["levels"].split(";"))}
            if g["levels"] else {name2key[g["dimension"]]: int(g["level"])}
        )
        for k, lv in targets.items():
            got = r[k]["level"]
            pairs[k].append((lv, got))
            pairs["all"].append((lv, got))
            if got != lv:
                misses.append((g["anchor_id"], DIMS[k], lv, got, r[k].get("reason", "")))
    print(f"anchors scored: {len({r.get('task_id') for r in recs} & set(gold))}/{len(gold)}")
    for k in list(DIMS) + ["all"]:
        agreement(pairs[k], DIMS.get(k, "合计"))
    if misses:
        print("\nmismatches (anchor, dimension, gold, model, model reason):")
        for m in misses:
            print("  ", *m)


def load_texts(ids):
    texts = {}
    for f in sorted((HERE / "mdna").glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                r = json.loads(line)
                if r["task_id"] in ids:
                    texts[r["task_id"]] = SPACE.sub("", r["text"])
    return texts


def check_reports(recs, write):
    bad = [(r.get("task_id"), valid(r)) for r in recs if valid(r)]
    ok = [r for r in recs if not valid(r)]
    print(f"records: {len(recs)}  valid: {len(ok)}  invalid: {len(bad)}")
    for b in bad[:20]:
        print("   invalid:", *b)
    mism = [r for r in ok if "_id_mismatch" in r]
    if mism:
        print(f"task_id echoed wrongly in {len(mism)} records (custom_id used)")

    runs = defaultdict(list)
    for r in ok:
        runs[r["task_id"]].append(r)
    texts = load_texts(set(runs))
    index = pd.read_csv(HERE / "mdna_index.csv", dtype={"股票代码": str})
    expected = set(index.loc[index["chars"].notna(), "task_id"])
    print(f"tasks scored: {len(runs)} of {len(expected)} with text; unknown ids: {len(set(runs) - expected)}")

    quotes = found = zero_with_ev = 0
    unverified = Counter()
    for tid, rs in runs.items():
        body = texts.get(tid, "")
        for r in rs:
            for k in DIMS:
                d = r[k]
                if d["level"] == 0 and d["evidence"]:
                    zero_with_ev += 1
                for e in d["evidence"]:
                    quotes += 1
                    if SPACE.sub("", e.get("quote", "")) in body:
                        found += 1
                    else:
                        unverified[tid] += 1
    print(f"evidence quotes found verbatim: {found}/{quotes} ({found / max(quotes, 1):.1%}); "
          f"tasks with any unverified quote: {len(unverified)}; level-0 with evidence: {zero_with_ev}")

    first = {tid: rs[0] for tid, rs in runs.items()}
    dist = pd.DataFrame({DIMS[k]: Counter(r[k]["level"] for r in first.values()) for k in DIMS}).fillna(0).astype(int).sort_index()
    print("\nlevel distribution (first run of each task):")
    print(dist.to_string())
    print("text_quality:", dict(Counter(r.get("text_quality") for r in first.values())))

    retest = defaultdict(list)
    for rs in runs.values():
        if len(rs) > 1:
            for k in DIMS:
                retest[k].append((rs[0][k]["level"], rs[1][k]["level"]))
                retest["all"].append((rs[0][k]["level"], rs[1][k]["level"]))
    if retest:
        print("\nre-test agreement (first vs second run):")
        for k in list(DIMS) + ["all"]:
            agreement(retest[k], DIMS.get(k, "合计"))

    if write:
        rows = []
        meta = index.set_index("task_id")
        for tid, r in sorted(first.items()):
            code, year = tid.split("_")
            row = {"股票代码": code, "报告年度": int(year), "预测年度": int(year) + 1}
            row.update({VAR[k]: r[k]["level"] for k in DIMS})
            row["年报文本质量"] = r.get("text_quality")
            row["年报文本字数"] = int(meta.loc[tid, "chars"]) if tid in meta.index else None
            row["证据未逐字核验条数"] = unverified.get(tid, 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(FINAL, index=False, encoding="utf-8-sig")
        print(f"\nwrote {FINAL.relative_to(ROOT)} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--anchors", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    recs = [r for f in a.files for r in parse(f)]
    if a.anchors:
        check_anchors(recs)
    else:
        check_reports(recs, a.write)


if __name__ == "__main__":
    main()
