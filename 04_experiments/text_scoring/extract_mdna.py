"""Extract the narrative MD&A text of each annual report for LLM scoring.

Scope is fixed for every firm-year: the mandated sections 公司业务概要 / 董事会报告 /
经营情况讨论与分析 / 管理层讨论与分析, from the first of them in the body to the next
section heading outside this set. Letters to shareholders and other optional sections
are left out so that every firm is scored on the same disclosure scope.

Cleaning keeps prose only: page headers, page numbers, table cells and numeric lines
are dropped, and hard-wrapped lines are re-joined into paragraphs. Output is
deterministic; each record carries the SHA-256 of the text sent to the model.

Usage: python3 extract_mdna.py [--limit N] [--rows ROWS.csv]
Writes 04_experiments/text_scoring/mdna/{year}.jsonl.gz and mdna_index.csv.
"""
import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "02_data"
OUT = Path(__file__).resolve().parent / "mdna"

MDNA_NAMES = ("公司业务概要", "董事会报告", "董事局报告", "董事会工作报告",
              "经营情况讨论与分析", "管理层讨论与分析")
CN_NUM = "一二三四五六七八九十"
# A heading starts a line: 第N节, optional repeats of that token, then a name ending the
# line. Cross-references (详见第四节“…”之…) fail the line-start or line-end test.
HEAD_RE = re.compile(rf"(?m)^[ \t　]*第([{CN_NUM}]{{1,3}})[节章](?:[\s　]*第[{CN_NUM}]{{1,3}}[节章])*[ \t　]*\n?[ \t　]*([^\s　”“\"「」《》…\.、，,：:（(之]{{0,16}})[ \t　]*(?:[（(][^\n]{{0,12}}[）)])?[ \t　]*(\.{{2}}|…|\n|$)")
CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\u2000-\u200b\xa0]")
# used only when no 第N节 heading survives in the text layer (headings drawn as images,
# names split across lines); markers must start a line, span capped at 40 pages
FALLBACK_START = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、)?[ \t]*(经营情况讨论与分析|管理层讨论与分析|报告期内公司从事的主要业务|报告期内公司所处行业情况|报告期内公司经营情况)")
FALLBACK_END = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、)?[ \t]*(公司治理相关情况说明|公司治理的基本状况|公司治理及内幕知情人|承诺事项履行情况|重大诉讼、仲裁)")
# second-level fallback for older layouts ("四、董事会报告", "§4 董事会报告", 董事会关于…的讨论与分析)
FALLBACK2_START = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+节[、 ]*|§\d+[ \t]*|\d+\.[ \t]*)?[ \t]*(董事会报告[ \t]*$|董事局报告[ \t]*$|董事会关于[^\n]{0,12}经营情况的讨论与分析|管理层讨论与分析[ \t]*$)")
FALLBACK2_END = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+节[、 ]*|§\d+[ \t]*)[ \t]*(重要事项|监事会报告|公司治理|股份变动及股东情况|财务报告|内部控制)")
# third level for traditional-character and H-share layouts (董事會報告, 董事会报告书, 管理層討論及分析)
FALLBACK3_START = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[节節章][、 ]*|§\d+[ \t]*|\d+\.[ \t]*)?[ \t]*(公司業務概要[ \t]*$|經營情況討論與分析[ \t]*$|管理層討論與分析[ \t]*$|管理层讨论及分析[ \t]*$|管理層討論及分析[ \t]*$|董事會報告[ \t]*$|董事会报告书[ \t]*$)")
FALLBACK3_END = re.compile(r"(?m)^[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[节節章][、 ]*|§\d+[ \t]*)[ \t]*(重要事項|重要事项|監事會報告|公司治理|股份變動|財務報告|內部控制)")
FALLBACK_PAGES = 40
DIGITISH = re.compile(r"[\d\.,%\-－—\s　()（）元万亿/:：;；√□]")
CJK = re.compile(r"[一-鿿]")
MAX_CHARS = 60000


def cn2int(s):
    d = {c: i + 1 for i, c in enumerate(CN_NUM)}
    if s == "十":
        return 10
    if s.startswith("十"):
        return 10 + d[s[1]]
    if len(s) == 3:
        return d[s[0]] * 10 + d[s[2]]
    if len(s) == 2 and s[1] == "十":
        return d[s[0]] * 10
    return d[s[0]]


def mdna_name(name):
    return next((n for n in MDNA_NAMES if name.startswith(n)), None)


def find_span(pages):
    """Locate the MD&A by section number, which survives running headers and OCR noise."""
    hits, mdna_nums = [], set()
    for i, p in enumerate(pages):
        found = [(m.start(), cn2int(m.group(1)), m.group(2), m.group(3)) for m in HEAD_RE.finditer(p)]
        for _, num, name, _ in found:
            if mdna_name(name):
                mdna_nums.add(num)
        if len({f[1] for f in found}) >= 4:  # table of contents page
            continue
        hits += [(i, pos, num, name) for pos, num, name, tail in found if tail not in ("..", "…")]
    if not mdna_nums:
        return fallback_span(pages)
    # the first body heading carrying an MD&A section number opens the span; its name may
    # be split across lines or replaced by an English label
    start = next((h for h in hits if h[2] in mdna_nums), None)
    if start is None:
        return fallback_span(pages)
    top = max(mdna_nums)
    end = next((h for h in hits if (h[0], h[1]) > (start[0], start[1]) and h[2] > top), None)
    names = sorted({mdna_name(h[3]) for h in hits if mdna_name(h[3])
                    and (end is None or (h[0], h[1]) < (end[0], end[1]))})
    return (start[0], start[1]), (end[0], end[1]) if end else None, names


def fallback_span(pages):
    def first(pattern, lo, hi):
        for i in range(lo, min(hi, len(pages))):
            m = pattern.search(pages[i])
            if m:
                return i, m.start()
        return None
    for fs, fe in ((FALLBACK_START, FALLBACK_END), (FALLBACK2_START, FALLBACK2_END), (FALLBACK3_START, FALLBACK3_END)):
        start = first(fs, 3, len(pages))
        if start is None:
            continue
        end = first(fe, start[0] + 1, start[0] + FALLBACK_PAGES) or (min(start[0] + FALLBACK_PAGES, len(pages) - 1), 0)
        return start, end, ["fallback"]
    return None


def is_prose(line):
    s = line.strip()
    if len(CJK.findall(s)) < 6:
        return False
    if re.search(r"年(年度)?报告(全文)?\s*$", s) and len(s) < 40:
        return False
    if len(DIGITISH.sub("", s)) < 0.4 * len(s):
        return False
    if re.fullmatch(r"[□√\s　]*(适用|不适用)[□√\s　适用不]*", s):
        return False
    return True


def clean(text):
    paras, buf = [], ""
    for line in text.split("\n"):
        if not is_prose(line):
            if buf:
                paras.append(buf)
                buf = ""
            continue
        s = line.strip()
        buf += s
        if re.search(r"[。！？；：]$", s) or len(s) < 30:
            paras.append(buf)
            buf = ""
    if buf:
        paras.append(buf)
    return "\n".join(p for p in paras if len(CJK.findall(p)) >= 10)


CJ = "[\u4e00-\u9fff，。、；：“”（）《》！？]"


def pages_from_pdf(path):
    """Re-read the PDF with reading-order sorted text (the stored text layer is scrambled for a few
    reports) and close the spaces between CJK characters."""
    import fitz
    pdf = DATA / str(path).replace("/text/", "/pdf/").replace(".jsonl.gz", ".pdf")
    out = []
    for i, pg in enumerate(fitz.open(pdf)):
        s = pg.get_text("text", sort=True)
        for _ in range(2):
            s = re.sub(rf"({CJ})[ \t]+(?={CJ})", r"\1", s)
        out.append({"PDF页序号": i + 1, "正文": s})
    return out


# manual_texts/<task_id>.txt holds text recovered by OCR or by hand from the PDF for reports whose stored
# text layer cannot be read (same format as the extractor output); manual_spans.csv holds page ranges.
def manual_spans():
    f = Path(__file__).resolve().parent / "manual_spans.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype=str)
    return {r.task_id: [tuple(map(int, x.split("-"))) for x in r.page_ranges.split(";")] for r in d.itertuples()}


def extract(path, task_id=None):
    rows = [json.loads(l) for l in gzip.open(DATA / path, "rt", encoding="utf-8")]
    rows.sort(key=lambda r: r["PDF页序号"])
    mt = Path(__file__).resolve().parent / "manual_texts" / f"{task_id}.txt"
    if task_id and mt.exists():
        text = mt.read_text(encoding="utf-8").strip()
        marks = [int(x) for x in re.findall(r"(?m)^\[第(\d+)页\]", text)]
        return text[:MAX_CHARS], {"status": "manual_text", "sections": "manual_texts", "page_from": marks[0],
                                  "page_to": marks[-1], "chars_full": len(text), "truncated": len(text) > MAX_CHARS}
    manual = manual_spans().get(task_id)
    if manual:
        chunks = []
        for lo, hi in manual:
            for r in rows:
                if lo <= r["PDF页序号"] <= hi:
                    chunks.append(f"[第{r['PDF页序号']}页]\n" + clean(CTRL.sub(" ", r["正文"] or "")))
        text = "\n".join(c for c in chunks if c.count("\n") or len(c) > 12)
        return text[:MAX_CHARS], {"status": "manual", "sections": "manual_spans.csv", "page_from": manual[0][0],
                                  "page_to": manual[-1][1], "chars_full": len(text), "truncated": len(text) > MAX_CHARS}
    pages = [re.sub(r"(?i)chapter[ \t]*\d+", "", CTRL.sub(" ", r["正文"] or "")) for r in rows]
    span = find_span(pages)
    if span is None:
        try:
            rows = pages_from_pdf(path)
        except Exception:
            return None, {"status": "no_mdna_heading"}
        pages = [re.sub(r"(?i)chapter[ \t]*\d+", "", CTRL.sub(" ", r["正文"])) for r in rows]
        span = find_span(pages)
    if span is None:
        return None, {"status": "no_mdna_heading"}
    (si, sp), end, names = span
    ei, ep = end if end else (len(pages) - 1, len(pages[-1]))
    chunks = []
    for i in range(si, ei + 1):
        p = pages[i]
        a = sp if i == si else 0
        b = ep if i == ei else len(p)
        chunks.append(f"[第{rows[i]['PDF页序号']}页]\n" + clean(p[a:b]))
    text = "\n".join(c for c in chunks if c.count("\n") or len(c) > 12)
    meta = {
        "status": "fallback" if names == ["fallback"] else "ok",
        "sections": "|".join(names),
        "page_from": rows[si]["PDF页序号"],
        "page_to": rows[ei]["PDF页序号"],
        "chars_full": len(text),
        "truncated": len(text) > MAX_CHARS,
    }
    return text[:MAX_CHARS], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rows")
    a = ap.parse_args()
    tasks = pd.read_csv(DATA / "final/text_extraction_tasks.csv", dtype={"股票代码": str})
    if a.rows:
        keep = pd.read_csv(a.rows, dtype={"股票代码": str})[["股票代码", "报告年度"]]
        tasks = tasks.merge(keep, on=["股票代码", "报告年度"])
    if a.limit:
        tasks = tasks.sample(a.limit, random_state=0)
    OUT.mkdir(exist_ok=True)
    index, by_year = [], {}
    for r in tasks.itertuples():
        text, meta = extract(r.所选压缩全文, f"{r.股票代码}_{r.报告年度}")
        key = f"{r.股票代码}_{r.报告年度}"
        meta.update(task_id=key, 股票代码=r.股票代码, 报告年度=r.报告年度, 所选压缩全文=r.所选压缩全文)
        if text is not None:
            meta["chars"] = len(text)
            meta["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
            by_year.setdefault(r.报告年度, []).append({"task_id": key, "text": text})
        index.append(meta)
    for y, recs in by_year.items():
        with gzip.open(OUT / f"{y}.jsonl.gz", "wt", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pd.DataFrame(index).to_csv(OUT.parent / "mdna_index.csv", index=False, encoding="utf-8-sig")
    df = pd.DataFrame(index)
    print(df["status"].value_counts().to_dict())
    if "chars" in df:
        print(df.groupby("报告年度")["chars"].describe().round(0).to_string())
        print("truncated:", int(df["truncated"].eq(True).sum()))


if __name__ == "__main__":
    main()
