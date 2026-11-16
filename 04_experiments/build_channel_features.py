"""Customer and supplier concentration (HHI) from CSMAR FN_FN010, written into model_features_X.csv.

HHI = sum over ranks 1-5 of the squared share of annual sales (purchases), as in the proposal. A firm-year whose
report gives only the top-5 total has no HHI (NaN,口径 仅披露合计); the model then sees the top-5 share and a
missing flag. When some ranks are given with the total, the undisclosed remainder is spread evenly over the missing
ranks, the least concentrated split consistent with the disclosure (口径 部分名次补齐). Ranks that contradict the stated
total (sum off by more than one percentage point) give no HHI either (口径 名次与合计不符).

Replaces 客户集中度_按开题报告 / 供应商集中度_按开题报告 (the squared top-5 share) and updates data_dictionary.csv.

  python3 build_channel_features.py
"""
import csv
import re
import sys

import numpy as np
import pandas as pd

from mpdr.config import DICT_PATH, PROJECT_DIR, X_PATH

RAW = PROJECT_DIR / "02_data" / "raw" / "csmar" / "top5_customers_suppliers" / "FN_FN010.csv"
RELATIONS = {"1": "客户", "2": "供应商"}
RANKS = ["1", "2", "3", "4", "5"]
SUM_TOL = 0.005  # disclosed ranks within this of the total count as all counterparties
MISMATCH_TOL = 0.01  # ranks that miss the stated total by more than this give no HHI (名次与合计不符)
OLD = {p: f"{p}集中度_按开题报告" for p in RELATIONS.values()}


def read_fn010(path=RAW):
    """FN_FN010 rows, recovered by their code/date prefix (the download has unescaped quotes in names)."""
    raw = path.read_text(encoding="utf-8-sig")
    header = next(csv.reader([raw.splitlines()[0]]))
    rows = []
    for s in re.split(r'(?m)(?=^"\d{6}","\d{4}-\d{2}-\d{2}",)', raw)[1:]:
        values = s.rstrip("\r\n")[1:-1].split('","', 11)
        assert len(values) == len(header) == 12, values[:3]
        rows.append(values)
    return pd.DataFrame(rows, columns=header)


def rank_shares(raw):
    """One row per 股票代码 + 年度 + relation with shares (0-1) for ranks 1-5 and the total ("6").

    Year-end consolidated statements only; a rank reported with conflicting shares, or a share outside [0, 1], is NaN.
    """
    d = raw[raw["accper"].str.endswith("12-31") & raw["typrep"].eq("1")].copy()
    d["股票代码"] = d["stkcd"]
    d["年度"] = d["accper"].str[:4].astype(int)
    d["share"] = pd.to_numeric(d["FN_Fn01006"], errors="coerce") / 100
    d.loc[~d["share"].between(0, 1), "share"] = np.nan
    g = d.dropna(subset=["share"]).groupby(["股票代码", "年度", "FN_Fn01003", "FN_Fn01001"])["share"]
    one = g.agg(lambda v: v.iloc[0] if v.nunique() == 1 else np.nan)
    return one.unstack("FN_Fn01001").reindex(columns=RANKS + ["6"]).reset_index()


def concentration(w):
    """Top-5 share, HHI and its 口径 from rank_shares output."""
    s = w[RANKS]
    known = s.notna().sum(axis=1)
    total = w["6"].where(w["6"].notna(), s.sum(axis=1).where(known == 5))
    total = total.where(total.between(0, 1))
    sq = (s ** 2).sum(axis=1)
    rest = (total - s.sum(axis=1)).clip(lower=0)
    partial = (known.between(1, 4)) & total.notna()
    fill = np.where(rest > SUM_TOL, rest ** 2 / (5 - known).clip(lower=1), 0.0)
    gap = s.sum(axis=1) - total
    mismatch = total.notna() & (((known == 5) & (gap.abs() > MISMATCH_TOL)) | (partial & (gap > MISMATCH_TOL)))
    full = (known == 5) & total.notna() & ~mismatch
    partial = partial & ~mismatch
    hhi = pd.Series(np.nan, index=w.index)
    hhi[full] = sq[full]
    hhi[partial] = (sq + fill)[partial]
    how = np.select([full, partial, mismatch, total.notna()], ["完整名次", "部分名次补齐", "名次与合计不符", "仅披露合计"],
                    "无数据")
    return pd.DataFrame({"top5": total, "hhi": hhi, "how": how}, index=w.index)


def build(X, w):
    """X with the HHI, its 口径 and the recomputed top-5 share for customers and suppliers."""
    X = X.drop(columns=[c for c in OLD.values() if c in X])
    for rel, p in RELATIONS.items():
        c = w[w["FN_Fn01003"] == rel]
        c = pd.concat([c[["股票代码", "年度"]], concentration(c)], axis=1)
        c.columns = ["股票代码", "报告年度", f"前五大{p}占比_重算", f"{p}集中度", f"{p}集中度口径"]
        X = X.drop(columns=[f"{p}集中度", f"{p}集中度口径"], errors="ignore")
        X = X.merge(c, on=["股票代码", "报告年度"], how="left", validate="one_to_one")
        X[f"{p}集中度口径"] = X[f"{p}集中度口径"].fillna("无数据")
        diff = (X[f"前五大{p}占比_重算"] - X[f"前五大{p}占比"]).abs()
        assert (diff.fillna(0) < 1e-9).all() and (X[f"前五大{p}占比_重算"].isna() == X[f"前五大{p}占比"].isna()).all(), \
            f"recomputed top-5 {p} share disagrees with model_features_X.csv"
        X = X.drop(columns=f"前五大{p}占比_重算")
        cols = list(X.columns)
        for name in (f"{p}集中度口径", f"{p}集中度"):  # place after the source column, as the old variable was
            cols.remove(name)
            cols.insert(cols.index(f"{p}合计来源") + 1, name)
        X = X[cols]
    return X


def update_dictionary(D):
    """New definitions; the top-5 shares become numeric model features."""
    D = D.copy()
    for p, flow in (("客户", "销售"), ("供应商", "采购")):
        old = D.index[D["字段"] == OLD[p]]
        if len(old):
            D.loc[old, ["字段", "定义或公式"]] = [f"{p}集中度", f"排名1—5的{flow}比例/100的平方和(HHI)；仅部分名次有比例时，"
                                              f"合计减已披露部分平均分配给缺失名次；仅披露合计或名次与合计相差超过1个百分点时留缺"]
            row = D.loc[old[0]].copy()
            row[["字段", "角色", "模型候选", "数据类型", "定义或公式", "单位或格式", "来源"]] = [
                f"{p}集中度口径", "标识、来源或样本元数据", np.nan, "object", "完整名次/部分名次补齐/名次与合计不符/仅披露合计/无数据",
                "按字段含义", "CSMAR FN_FN010，见04_experiments/build_channel_features.py"]
            D = pd.concat([D.loc[:old[0]], row.to_frame().T, D.loc[old[0] + 1:]], ignore_index=True)
        D.loc[D["字段"] == f"{p}集中度", "来源"] = "CSMAR FN_FN010，见04_experiments/build_channel_features.py"
        D.loc[D["字段"] == f"前五大{p}占比", ["角色", "模型候选"]] = ["数值模型特征", "数值特征"]
    return D


def main():
    X = pd.read_csv(X_PATH, dtype={"股票代码": str}, encoding="utf-8-sig", low_memory=False)
    out = build(X, rank_shares(read_fn010()))
    assert len(out) == len(X) and out[["股票代码", "报告年度"]].equals(X[["股票代码", "报告年度"]])
    D = update_dictionary(pd.read_csv(DICT_PATH, encoding="utf-8-sig"))
    assert set(D.loc[D["所在文件"] == "model_features_X.csv", "字段"]) == set(out.columns), "dictionary and X columns differ"
    for p in RELATIONS.values():
        print(out[f"{p}集中度口径"].value_counts().to_string(), "\n", out[f"{p}集中度"].describe().round(4).to_string())
    if "--dry-run" in sys.argv:
        return
    out.to_csv(X_PATH, index=False, encoding="utf-8-sig")
    D.to_csv(DICT_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
