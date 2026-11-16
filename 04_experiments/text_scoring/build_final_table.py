"""Build 02_data/final/annual_report_text_scores.csv from results/<year>/scores.jsonl (2013-2024).

One row per scored report: 股票代码, 报告年度, 预测年度, the seven 0-4 scores, their sum, the text length and
披露晚于4月30日 (1 when the scored report was disclosed after 30 April of the prediction year).
Run: python3 build_final_table.py
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DIMS = ["market_sensing", "brand_management", "customer_relationship", "channel_management",
        "product_development", "marketing_communication", "digital_marketing"]
ZH = ["市场感知叙事强度", "品牌管理叙事强度", "客户关系管理叙事强度", "渠道管理叙事强度",
      "产品开发与新品叙事强度", "营销沟通叙事强度", "数字营销叙事强度"]

idx = pd.read_csv(HERE / "mdna_index.csv", dtype={"股票代码": str}).set_index("task_id")
DATA = HERE.parents[1] / "02_data"
tasks = pd.read_csv(DATA / "final/text_extraction_tasks.csv", dtype=str, encoding="utf-8-sig")
reports = pd.read_csv(DATA / "raw/annual_reports/index.csv", dtype=str, encoding="utf-8-sig", low_memory=False)
disclosed = {f"{r.股票代码}_{r.年度}": r.披露日期 for r in reports.itertuples()}
disclosed.update({f"{r.股票代码}_{r.报告年度}": r.所选披露日期 for r in tasks.itertuples() if isinstance(r.所选披露日期, str)})
rows = []
for year in range(2013, 2025):
    for line in open(HERE / f"results/{year}/scores.jsonl", encoding="utf-8"):
        r = json.loads(line)
        code, y = r["task_id"].split("_")
        row = {"股票代码": code, "报告年度": int(y), "预测年度": int(y) + 1}
        row.update({z: r[d]["level"] for z, d in zip(ZH, DIMS)})
        row["叙事强度合计"] = sum(r[d]["level"] for d in DIMS)
        row["年报文本字数"] = int(idx.loc[r["task_id"], "chars"])
        row["披露晚于4月30日"] = int(disclosed[r["task_id"]] > f"{int(y) + 1}-04-30")
        rows.append(row)
df = pd.DataFrame(rows).sort_values(["股票代码", "报告年度"])
assert not df.duplicated(["股票代码", "报告年度"]).any()
out = HERE.parents[1] / "02_data/final/annual_report_text_scores.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"{len(df)} rows -> {out}")
