import numpy as np
import pandas as pd
import pytest

from mpdr.config import (
    CATEGORICAL, CONTINUOUS, DICT_PATH, FLAGS, PRE2021_INDUSTRIES, RAW_NUMERIC, YEAR_LEVEL,
)
from mpdr.features import WINSORIZED, FeaturePreprocessor, harmonize_industry, load_x
from mpdr.split import split_masks


@pytest.fixture(scope="module")
def X():
    return load_x()


@pytest.fixture(scope="module")
def masks(X):
    return split_masks(X)


@pytest.fixture(scope="module")
def fitted(X, masks):
    train = X[masks["train"]]
    return {m: FeaturePreprocessor(m).fit(train) for m in ("lasso", "tabnet")}


def fitted_params(p):
    return {k: v for k, v in vars(p).items() if k.endswith("_")}


def assert_params_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], pd.Series):
            pd.testing.assert_series_equal(a[k], b[k])
        else:
            assert a[k] == b[k], k


def test_candidates_match_dictionary():
    d = pd.read_csv(DICT_PATH, encoding="utf-8-sig")
    assert RAW_NUMERIC == d.loc[d["模型候选"] == "数值特征", "字段"].tolist()
    assert d.loc[d["模型候选"] == "分类特征", "字段"].tolist()[0] == "消费类型"


def test_split_masks(X, masks):
    year = X["报告年度"]
    assert year[masks["train"]].between(2013, 2020).all()
    assert year[masks["valid"]].between(2021, 2022).all()
    assert year[masks["test"]].between(2023, 2024).all()
    assert (masks["train"].astype(int) + masks["valid"] + masks["test"] == 1).all()
    assert (masks["refit"] == (masks["train"] | masks["valid"])).all()


def perturb_outside_train(X, masks):
    """Corrupt every valid/test row: scale numbers, inject NaN, invent categories."""
    Xp = X.copy()
    rows = ~masks["train"]
    rng = np.random.default_rng(0)
    num_cols = [c for c in Xp.columns if pd.api.types.is_float_dtype(Xp[c])]
    for c in num_cols:
        v = Xp.loc[rows, c] * 1000 + 7
        Xp.loc[rows, c] = v.where(rng.random(rows.sum()) > 0.3)
    Xp.loc[rows, "消费类型"] = "新消费"
    Xp.loc[rows, "对应目标行业"] = "新行业"
    Xp.loc[rows, "标记_预测时点ST或*ST"] = True
    return Xp


@pytest.mark.parametrize("mode", ["lasso", "tabnet"])
def test_nothing_fitted_on_valid_or_test(X, masks, fitted, mode):
    Xp = perturb_outside_train(X, masks)
    mp = split_masks(Xp)
    assert (mp["train"] == masks["train"]).all()
    p2 = FeaturePreprocessor(mode).fit(Xp[mp["train"]])
    assert_params_equal(fitted_params(fitted[mode]), fitted_params(p2))
    # transform is row-wise: corrupting other rows in the batch leaves train rows unchanged
    a = fitted[mode].transform(X)[masks["train"]]
    b = fitted[mode].transform(Xp)[masks["train"]]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("mode", ["lasso", "tabnet"])
def test_shapes_nan_and_order(X, masks, fitted, mode):
    p = fitted[mode]
    n_num = len(CONTINUOUS) + len(FLAGS) + len(p.missing_groups_)
    assert p.numeric_names[: len(CONTINUOUS)] == CONTINUOUS
    for split in ("train", "valid", "test"):
        T = p.transform(X[masks[split]])
        assert T.shape[0] == masks[split].sum()
        assert list(T.columns) == p.feature_names
        assert not T.isna().any().any()
        if mode == "tabnet":
            assert T.shape[1] == n_num + len(CATEGORICAL)
            assert [T.columns[i] for i in p.cat_idxs] == CATEGORICAL
            for c, dim in zip(CATEGORICAL, p.cat_dims):
                assert T[c].between(0, dim - 1).all()
        else:
            assert T.shape[1] == n_num + sum(len(v) for v in p.levels_.values())
            assert T.iloc[:, n_num:].isin([0.0, 1.0]).all().all()
    if mode == "lasso":
        T = p.transform(X[masks["train"]]).iloc[:, :n_num]
        np.testing.assert_allclose(T.mean(), 0, atol=1e-9)
        np.testing.assert_allclose(T.std(ddof=0), 1, atol=1e-9)
    else:
        T = p.transform(X[masks["train"]])
        assert (T[WINSORIZED] >= p.lower_ - 1e-12).all().all()
        assert (T[WINSORIZED] <= p.upper_ + 1e-12).all().all()


def test_year_level_variables_pass_through(X, fitted):
    assert set(YEAR_LEVEL) <= set(CONTINUOUS) and not set(YEAR_LEVEL) & set(WINSORIZED)
    T = fitted["tabnet"].transform(X)
    pd.testing.assert_frame_equal(T[YEAR_LEVEL], X[YEAR_LEVEL])


def test_flags_match_dataset_markers(X, fitted):
    T = fitted["tabnet"].transform(X)
    assert (T["gp_nonpos"] == X["标记_本年毛利润非正"]).all()
    assert (T["se_missing_or_zero"] == X["标记_销售费用对数无法计算"]).all()
    assert (T["capex_missing"] == X["标记_资本支出强度无法计算"]).all()
    assert (T["st"] == X["标记_预测时点ST或*ST"]).all()


def test_flag_logic_on_real_rows(X, fitted):
    p = fitted["tabnet"]
    T = p.transform(X)

    i = X.index[X["毛利润"] <= 0][0]
    assert T.at[i, "gp_nonpos"] == 1
    # filled with the train minimum of ln(GP), then winsorized at the train 1% cutoff
    assert T.at[i, "mp_t"] == max(p.gp_log_min_, p.lower_["mp_t"])

    i = X.index[X["毛利润"] > 0][0]
    assert T.at[i, "gp_nonpos"] == 0
    assert T.at[i, "mp_t"] == np.clip(np.log(X.at[i, "毛利润"]), p.lower_["mp_t"], p.upper_["mp_t"])

    i = X.index[X["销售费用"].isna()][0]
    assert T.at[i, "se_missing_or_zero"] == 1
    assert T.at[i, "ln1p_se"] == p.lower_["ln1p_se"]
    i = X.index[X["销售费用"] == 0][0]
    assert T.at[i, "se_missing_or_zero"] == 1

    i = X.index[X["存货周转率"].isna()][0]
    assert T.at[i, "存货周转率_missing"] == 1
    assert T.at[i, "存货周转率"] == p.medians_["存货周转率"]
    i = X.index[X["存货周转率"] < 0][0]
    assert T.at[i, "存货周转率_missing"] == 0
    assert T.at[i, "存货周转率"] == np.clip(0.0, p.lower_["存货周转率"], p.upper_["存货周转率"])

    i = X.index[X["资本支出强度"].isna()][0]
    assert T.at[i, "capex_missing"] == 1
    assert T.at[i, "资本支出强度"] == p.medians_["资本支出强度"]


def test_missing_flags_follow_train_rate(X, masks, fitted):
    p = fitted["tabnet"]
    members = [c for cols in p.missing_groups_.values() for c in cols]
    assert sorted(members) == sorted(c for c in CONTINUOUS if p.missing_rate_[c] > 0.01)
    assert "第一大客户占比" in members and "资本支出强度" not in members
    M = X.loc[masks["train"], members].isna()
    leads = list(p.missing_groups_)
    for lead, cols in p.missing_groups_.items():
        assert lead == cols[0]
        assert all((M[c] == M[lead]).mean() > 0.999 for c in cols)
    for i, a in enumerate(leads):
        assert all((M[a] == M[b]).mean() <= 0.999 for b in leads[i + 1:])
    T = p.transform(X)
    for lead, cols in p.missing_groups_.items():
        assert (T[f"{lead}_missing"] == p._continuous(X)[cols].isna().any(axis=1)).all()


def test_unseen_categories_map_to_other(X, masks, fitted):
    row = X[masks["test"]].head(3).copy()
    row["消费类型"] = ["新消费", "居住消费", "居住消费"]
    row["对应目标行业"] = ["新行业", "新行业", "房地产开发"]
    tab = fitted["tabnet"]
    T = tab.transform(row)
    other_cons, other_ind = (d - 1 for d in tab.cat_dims)
    assert T["消费类型"].tolist()[0] == other_cons
    assert T["消费类型"].tolist()[1] != other_cons
    # an unknown industry falls back to its consumption-type group; "other" only if that type is unseen
    lv = tab.levels_["industry"]
    assert T["industry"].tolist() == [other_ind, lv.index("居住消费(合并)"), lv.index("房地产开发")]
    assert tab.category_label("industry", other_ind) == "other"

    L = fitted["lasso"].transform(row)
    cons_cols = [c for c in L.columns if c.startswith("消费类型=")]
    ind_cols = [c for c in L.columns if c.startswith("industry=")]
    assert L[cons_cols].sum(axis=1).tolist() == [0, 1, 1]
    assert L[ind_cols].sum(axis=1).tolist() == [0, 1, 1]


def test_small_industry_merges_into_consumption_type(X, masks):
    train = X[masks["train"]]
    small = "酒店餐饮"
    drop = train.index[train["对应目标行业"] == small][29:]  # keep 29 obs (< 30)
    p = FeaturePreprocessor("tabnet").fit(train.drop(drop))
    cons = train.loc[train["对应目标行业"] == small, "消费类型"].iloc[0]
    assert small not in p.levels_["industry"]
    assert f"{cons}(合并)" in p.levels_["industry"]
    T = p.transform(X[X["对应目标行业"] == small].head(5))
    assert (T["industry"] == p.levels_["industry"].index(f"{cons}(合并)")).all()


def test_harmonize_industry(X, masks):
    assert PRE2021_INDUSTRIES == sorted(X.loc[masks["train"], "对应目标行业"].unique())
    df = pd.DataFrame({
        "对应目标行业": ["个护用品", "化妆品", "食品加工", "休闲食品", "饰品", "房地产服务", "医疗美容", "一般零售"],
        "消费类型": ["食品与日常消费"] * 5 + ["居住消费", "健康与生活服务消费", "零售与流通消费"],
    })
    assert harmonize_industry(df).tolist() == [
        "个护用品|化妆品", "个护用品|化妆品", "食品加工", "食品加工|休闲食品", "饰品",
        "居住消费(合并)", "健康与生活服务消费(合并)", "一般零售",
    ]
    h = harmonize_industry(X)
    assert h[masks["train"]].equals(X.loc[masks["train"], "对应目标行业"])
    assert h.notna().all()


def test_no_valid_or_test_row_in_other(X, masks, fitted):
    p = fitted["tabnet"]
    for split in ("valid", "test"):
        T = p.transform(X[masks[split]])
        for c, dim in zip(CATEGORICAL, p.cat_dims):
            assert (T[c] != dim - 1).all(), (split, c)
