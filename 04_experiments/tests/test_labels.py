import numpy as np
import pandas as pd
import pytest

from mpdr import labels as lb

COEF = {"ln_se": 0.6, "se_zero": 1.5, "size": 0.3, "age": -0.01, "lev": -0.5,
        "cfoa": 2.0, "capexa": 3.0, "capex_missing": 0.4}
LAG_COEF = {"mp_lag": 0.7, "gp_nonpos": -1.2}
GROUP_EFFECT = {"A": 0.0, "B": 0.3, "T2": -0.2}
# industry -> (consumption type, non-ST rows per year); C, D and the ST-only E fall into T2
INDUSTRIES = {"A": ("T1", 50), "B": ("T1", 40), "C": ("T2", 12), "D": ("T2", 10), "E": ("T2", 0)}


def _year_panel(rng, year, missing_in_fit):
    rows = []
    for ind, (ctype, n) in INDUSTRIES.items():
        rows += [(ind, ctype, False)] * n + [(ind, ctype, True)] * 2
    d = pd.DataFrame(rows, columns=[lb.INDUSTRY_RAW, lb.CONSUMPTION, lb.ST])
    n = len(d)
    # Discrete regressors with heavy mass at the extremes, so 1/99 winsorizing is a no-op on fit rows.
    d["销售费用"] = np.expm1(rng.choice([14.0, 15.0, 16.0, 17.0], n))
    d["企业规模"] = rng.choice([20.0, 21.0, 22.0, 23.0, 24.0], n)
    d["成立年限"] = rng.choice([5.0, 10.0, 15.0, 20.0], n)
    d["资产负债率"] = rng.choice([0.2, 0.4, 0.6], n)
    d["经营现金流资产比"] = rng.choice([-0.05, 0.0, 0.05, 0.1], n)
    d["资本支出强度"] = rng.choice([0.01, 0.03, 0.05], n)
    d["毛利润"] = np.exp(rng.choice([16.0, 17.0, 18.0, 19.0], n))
    fit_idx = np.flatnonzero(~d[lb.ST])
    d.loc[fit_idx[:3], "销售费用"] = 0.0
    d.loc[fit_idx[3:5], "销售费用"] = np.nan
    if missing_in_fit:
        d.loc[fit_idx[5:8], "资本支出强度"] = np.nan
        d.loc[fit_idx[8:11], "毛利润"] = [0.0, -1e6, -5e7]
    d["报告年度"] = year
    d["股票代码"] = [f"{i:06d}" for i in range(n)]
    return d


def _true_design(d, lagged=False):
    """The benchmark design implied by the DGP (no clipping needed on these values)."""
    se = d["销售费用"].fillna(0)
    cols = {
        "const": np.ones(len(d)),
        "ln_se": np.log1p(se).where(se > 0, 0.0),
        "se_zero": (se == 0).astype(float),
        "size": d["企业规模"].clip(upper=24.0),
        "age": d["成立年限"], "lev": d["资产负债率"], "cfoa": d["经营现金流资产比"],
        "capexa": d["资本支出强度"].fillna(0.0),
        "capex_missing": d["资本支出强度"].isna().astype(float),
    }
    if lagged:
        gp = d["毛利润"]
        cols["mp_lag"] = np.log(gp.where(gp > 0)).fillna(0.0)
        cols["gp_nonpos"] = (gp <= 0).astype(float)
    group = d[lb.INDUSTRY_RAW].map(lambda i: i if i in ("A", "B") else "T2")
    for g, eff in GROUP_EFFECT.items():
        cols[f"g_{g}"] = (group == g).astype(float)
    return pd.DataFrame(cols, index=d.index)


def make_panel(seed=0, lagged=False):
    """Two report years with a known residual. Returns (X, Y, true_residual).

    lagged=True puts ln GP(t) and the gp_nonpos dummy into the true benchmark.
    """
    rng = np.random.default_rng(seed)
    coef = pd.Series({"const": 2.0, **COEF, **(LAG_COEF if lagged else {}),
                      **{f"g_{g}": e for g, e in GROUP_EFFECT.items()}})
    X, Y, resid = [], [], []
    for year, missing_in_fit in [(2015, True), (2016, False)]:
        d = _year_panel(rng, year, missing_in_fit)
        st = d[lb.ST].to_numpy()
        # One ST row far outside the fit range: its size must be clipped to the fit 99% cutoff (24).
        d.loc[np.flatnonzero(st)[0], "企业规模"] = 40.0
        D = _true_design(d, lagged)
        eps = rng.normal(0, 0.5, len(d))
        # Make the fit-sample residual orthogonal to the design so OLS recovers it exactly.
        Df = D[~st].drop(columns="g_A").to_numpy()
        proj = Df @ np.linalg.lstsq(Df, eps[~st], rcond=None)[0]
        eps[~st] -= proj
        y = D.to_numpy() @ coef[D.columns].to_numpy() + eps
        X.append(d)
        Y.append(pd.DataFrame({"股票代码": d["股票代码"], "报告年度": year, lb.TARGET: y}))
        resid.append(pd.Series(eps, index=d.index))
    return (pd.concat(X, ignore_index=True), pd.concat(Y, ignore_index=True),
            pd.concat(resid, ignore_index=True))


def test_label_recovers_known_residual():
    X, Y, resid = make_panel()
    labels, stats = lb.build_mpdr(X, Y)
    np.testing.assert_allclose(labels["mpdr"], resid, atol=1e-8)
    np.testing.assert_allclose(labels["mp_actual"] - labels["mp_expected"], labels["mpdr"])
    assert labels["st"].equals(X[lb.ST])
    assert stats["r2"].between(0, 1).all()


def test_st_rows_scored_but_not_fitted():
    X, Y, _ = make_panel()
    labels, stats = lb.build_mpdr(X, Y)
    st = X[lb.ST].to_numpy()
    assert labels.loc[st, "mpdr"].notna().all()
    assert stats["n_fit"].tolist() == X[~st].groupby("报告年度").size().tolist()
    assert stats["n_scored"].tolist() == X.groupby("报告年度").size().tolist()
    # Moving ST outcomes moves only their own labels.
    Y2 = Y.copy()
    Y2.loc[st, lb.TARGET] += 100.0
    labels2, stats2 = lb.build_mpdr(X, Y2)
    np.testing.assert_allclose(labels2.loc[~st, "mpdr"], labels.loc[~st, "mpdr"], atol=1e-10)
    np.testing.assert_allclose(labels2.loc[st, "mpdr"], labels.loc[st, "mpdr"] + 100.0)
    np.testing.assert_allclose(stats2["r2"], stats["r2"])


def test_non_st_mean_zero_each_year():
    X, Y, _ = make_panel()
    labels, _ = lb.build_mpdr(X, Y)
    means = labels[~labels["st"]].groupby("报告年度")["mpdr"].mean()
    np.testing.assert_allclose(means, 0, atol=1e-10)


def test_small_industries_merge_into_consumption_type():
    X, Y, _ = make_panel()
    d = X[X["报告年度"] == 2015]
    groups = lb.industry_groups(d[lb.INDUSTRY_RAW], d[lb.CONSUMPTION], ~d[lb.ST])
    assert set(groups[d[lb.INDUSTRY_RAW].isin(["C", "D", "E"])]) == {"T2"}
    assert set(groups[d[lb.INDUSTRY_RAW].isin(["A", "B"])]) == {"A", "B"}
    _, stats = lb.build_mpdr(X, Y)
    assert stats["n_industry_groups"].tolist() == [3, 3]


def test_st_outlier_clipped_to_fit_cutoff():
    X, Y, _ = make_panel()
    labels, _ = lb.build_mpdr(X, Y)
    i = X.index[X["企业规模"] == 40.0][0]
    X2 = X.copy()
    X2.loc[i, "企业规模"] = 24.0
    labels2, _ = lb.build_mpdr(X2, Y)
    assert labels.loc[i, "mp_expected"] == pytest.approx(labels2.loc[i, "mp_expected"])


def test_missing_fill_rules_for_st_rows():
    """In 2016 no fit row lacks capex: ST rows with missing capex get the fit median and no dummy.
    SE missing and SE == 0 are the same case."""
    X, Y, _ = make_panel()
    st_2016 = X.index[(X["报告年度"] == 2016) & X[lb.ST]]
    i, j = st_2016[:2]
    fit_median = X.loc[(X["报告年度"] == 2016) & ~X[lb.ST], "资本支出强度"].median()
    X.loc[j] = X.loc[i]
    X.loc[j, "股票代码"] = "999999"
    Y.loc[j, "股票代码"] = "999999"
    X.loc[i, "资本支出强度"] = np.nan
    X.loc[j, "资本支出强度"] = fit_median
    X.loc[i, "销售费用"] = np.nan
    X.loc[j, "销售费用"] = 0.0
    labels, _ = lb.build_mpdr(X, Y)
    assert labels.loc[i, "mp_expected"] == pytest.approx(labels.loc[j, "mp_expected"])


def test_lagged_variant_recovers_known_residual():
    X, Y, resid = make_panel(lagged=True)
    labels, stats = lb.build_mpdr(X, Y, include_lagged_mp=True)
    np.testing.assert_allclose(labels["mpdr"], resid, atol=1e-8)
    # The default benchmark omits ln GP(t), so it cannot recover this residual.
    default, default_stats = lb.build_mpdr(X, Y)
    assert np.abs(default["mpdr"] - resid).max() > 0.1
    assert (stats["r2"] > default_stats["r2"]).all()


def test_lagged_variant_gp_nonpos_fill():
    """In 2016 no fit row has GP <= 0: an ST row with GP <= 0 gets the fit minimum of ln GP."""
    X, Y, _ = make_panel(lagged=True)
    i, j = X.index[(X["报告年度"] == 2016) & X[lb.ST]][:2]
    X.loc[j] = X.loc[i]
    X.loc[j, "股票代码"] = "999999"
    Y.loc[j, "股票代码"] = "999999"
    X.loc[i, "毛利润"] = -3e6
    X.loc[j, "毛利润"] = np.exp(16.0)
    labels, _ = lb.build_mpdr(X, Y, include_lagged_mp=True)
    assert labels.loc[i, "mp_expected"] == pytest.approx(labels.loc[j, "mp_expected"])


@pytest.fixture(scope="module")
def real():
    X, Y = lb.load_xy()
    labels, stats = lb.build_mpdr(X, Y)
    return X, labels, stats


def test_real_data_one_label_per_row(real):
    X, labels, stats = real
    assert len(labels) == len(X) == 12004
    assert labels[lb.KEY].equals(X[lb.KEY])
    assert not labels.duplicated(lb.KEY).any()
    assert labels[["mp_actual", "mp_expected", "mpdr"]].notna().all().all()
    assert stats["n_scored"].sum() == len(X)
    assert stats["n_fit"].sum() == (~X[lb.ST]).sum()


def test_real_data_non_st_mean_zero(real):
    _, labels, _ = real
    means = labels[~labels["st"]].groupby("报告年度")["mpdr"].mean()
    assert len(means) == 12
    np.testing.assert_allclose(means, 0, atol=1e-8)
