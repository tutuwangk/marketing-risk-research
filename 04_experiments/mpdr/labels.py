"""MPDR label: actual ln GP(t+1) minus a yearly cross-sectional benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import (DATA_DIR, CONSUMPTION, INDUSTRY_RAW, KEY, MIN_INDUSTRY_OBS, ST, TARGET, WINSOR_Q,
                     X_PATH, Y_PATH)

CONTINUOUS = ["ln_se", "size", "age", "lev", "cfoa", "capexa"]
DUMMIES = ["se_zero", "capex_missing", "gp_nonpos"]
RAW = {"size": "企业规模", "age": "成立年限", "lev": "资产负债率",
       "cfoa": "经营现金流资产比", "capexa": "资本支出强度"}


def load_xy(x_path=X_PATH, y_path=Y_PATH):
    """Read X and Y with 6-digit text stock codes."""
    read = lambda p: pd.read_csv(p, dtype={"股票代码": str}, encoding="utf-8-sig", low_memory=False)
    return read(x_path), read(y_path)


def industry_groups(industry, cons_type, fit, min_obs=MIN_INDUSTRY_OBS):
    """Industry label, with industries under min_obs fit rows replaced by their 消费类型."""
    counts = industry[fit].value_counts()
    return industry.where(industry.isin(counts.index[counts >= min_obs]), cons_type)


def regressors(d, fit, include_lagged_mp=False):
    """Winsorized, filled benchmark regressors for one report year."""
    se = d["销售费用"]
    r = pd.DataFrame({k: d[v] for k, v in RAW.items()}, index=d.index)
    # ln_se is NaN for SE missing or 0 so those rows stay out of the cutoffs; they get 0 plus the dummy.
    r["ln_se"] = np.log1p(se.where(se > 0))
    r["se_zero"] = r["ln_se"].isna().astype(float)
    r["capex_missing"] = r["capexa"].isna().astype(float)
    continuous, dummies = list(CONTINUOUS), ["se_zero", "capex_missing"]
    if include_lagged_mp:
        gp = d["毛利润"]
        r["mp_lag"] = np.log(gp.where(gp > 0))
        r["gp_nonpos"] = r["mp_lag"].isna().astype(float)
        continuous.append("mp_lag")
        dummies.append("gp_nonpos")
    for c in continuous:
        lo, hi = r.loc[fit, c].quantile(WINSOR_Q)
        r[c] = r[c].clip(lo, hi)
    r["ln_se"] = r["ln_se"].fillna(0.0)
    r["capexa"] = r["capexa"].fillna(r.loc[fit, "capexa"].median())
    if include_lagged_mp:
        # The non-ST minimum of ln GP, after clipping, is the lower winsor cutoff.
        r["mp_lag"] = r["mp_lag"].fillna(r.loc[fit, "mp_lag"].min())
    return r[continuous + dummies]


def fit_year(d, include_lagged_mp=False):
    """Fit the benchmark on non-ST rows of one report year and predict every row."""
    fit = ~d[ST].to_numpy(bool)
    r = regressors(d, fit, include_lagged_mp)
    # A missing-value dummy with no fit-sample cases cannot be estimated; drop it for that year.
    r = r.drop(columns=[c for c in DUMMIES if c in r and r.loc[fit, c].sum() == 0])
    groups = industry_groups(d[INDUSTRY_RAW], d[CONSUMPTION], fit)
    levels = sorted(groups[fit].unique())
    assert groups.isin(levels).all(), "an ST row falls in a group with no fit rows"
    fe = pd.get_dummies(pd.Categorical(groups, categories=levels), drop_first=True,
                        dtype=float).set_axis(d.index)
    design = sm.add_constant(pd.concat([r, fe], axis=1))
    assert np.linalg.matrix_rank(design[fit].to_numpy()) == design.shape[1]
    res = sm.OLS(d.loc[fit, TARGET], design[fit]).fit()
    stats = {"n_fit": int(fit.sum()), "n_scored": len(d), "r2": res.rsquared,
             "adj_r2": res.rsquared_adj, "n_industry_groups": len(levels)}
    return res.predict(design), stats


def build_mpdr(X, Y, include_lagged_mp=False):
    """Return (labels_df, fit_stats_df); one label per X row, one stats row per prediction year.

    include_lagged_mp adds ln GP(t) (gp_nonpos dummy for GP <= 0) to the benchmark.
    """
    cols = KEY + [ST, INDUSTRY_RAW, CONSUMPTION, "销售费用", "毛利润"] + list(RAW.values())
    d = X[cols].merge(Y[KEY + [TARGET]], on=KEY, how="left", validate="one_to_one")
    assert d[TARGET].notna().all()
    expected = pd.Series(np.nan, index=d.index)
    stats = []
    for year, g in d.groupby("报告年度"):
        expected[g.index], s = fit_year(g, include_lagged_mp)
        stats.append({"报告年度": year, "预测年度": year + 1, **s})
    labels = pd.DataFrame({"股票代码": d["股票代码"], "报告年度": d["报告年度"],
                           "mp_actual": d[TARGET], "mp_expected": expected,
                           "mpdr": d[TARGET] - expected, "st": d[ST].astype(bool)})
    return labels, pd.DataFrame(stats)


def write_outputs(labels, fit_stats, out_dir, suffix=""):
    """Write mpdr_label{suffix}.csv and mpdr_benchmark_fit{suffix}.csv as UTF-8 with BOM."""
    out_dir = Path(out_dir)
    labels.to_csv(out_dir / f"mpdr_label{suffix}.csv", index=False, encoding="utf-8-sig")
    fit_stats.to_csv(out_dir / f"mpdr_benchmark_fit{suffix}.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    X, Y = load_xy()
    write_outputs(*build_mpdr(X, Y), DATA_DIR)
    write_outputs(*build_mpdr(X, Y, include_lagged_mp=True), DATA_DIR, suffix="_lagged")
