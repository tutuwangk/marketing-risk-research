import numpy as np
import pandas as pd
import pytest

from mpdr import metrics as m
from mpdr.config import CATEGORICAL, KEY, PERSISTENCE, TEXT_COLUMNS, YEAR
from mpdr.features import FeaturePreprocessor
from mpdr.modules import TextBlock, module_columns, module_of
from mpdr.split import split_masks
from mpdr.stages import PersistenceStage
from run_two_stage import build, load_data


@pytest.fixture(scope="module")
def X():
    return load_data()


@pytest.fixture(scope="module")
def built(X):
    return build(X, split_masks(X), False)


def test_r2_vs_benchmark():
    y = np.array([1.0, 2.0, 3.0])
    # SSE(model) = 0.25 + 0 + 0.25, SSE(bench) = 1 + 1 + 1
    assert m.r2_vs(y, np.array([1.5, 2.0, 2.5]), np.array([2.0, 3.0, 2.0])) == pytest.approx(1 - 0.5 / 3)


def test_clark_west_hand_computed():
    y = np.array([1.0, 0.0, 2.0, 1.0])
    bench = np.zeros(4)
    model = np.array([0.5, 0.0, 1.0, 1.0])
    # d = (y-b)^2 - [(y-f)^2 - (b-f)^2] = [1-0, 0-0, 4-0, 1-(-1)] = [1, 0, 4, 2]
    d = np.array([1.0, 0.0, 4.0, 2.0])
    t, p = m.clark_west(y, bench, model)
    assert t == pytest.approx(d.mean() / (d.std(ddof=1) / 2))
    assert 0 < p < 0.5


def test_diebold_mariano_sign():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    good, bad = y + np.array([0.1, -0.1, 0.1, -0.2, 0.1]), y + np.array([1.0, -1.5, 1.0, -1.0, 2.0])
    t, _ = m.diebold_mariano(y, bad, good)
    assert t > 0  # positive when the second forecast is more accurate


def test_yearly_tail_is_within_year():
    y = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
    year = np.array([1] * 5 + [2] * 5)
    low = m.yearly_tail(y, year, q=0.2)
    assert low.tolist() == [True] + [False] * 4 + [True] + [False] * 4
    high = m.yearly_tail(y, year, q=0.2, tail="positive")
    assert np.flatnonzero(high).tolist() == [4, 9]


def test_tail_auc_direction():
    y = np.arange(10, dtype=float)
    year = np.zeros(10)
    assert m.tail_auc(y, y, year, "negative", q=0.2) == 1.0
    assert m.tail_auc(y, y, year, "positive", q=0.2) == 1.0
    assert m.tail_auc(y, -y, year, "negative", q=0.2) == 0.0


def test_new_entry_mask_treats_missing_lag_as_not_in_tail():
    assert m.new_entry_mask(pd.Series([True, False, None])).tolist() == [False, True, True]


def test_cluster_bootstrap_brackets_full_sample():
    rng = np.random.default_rng(1)
    v = rng.normal(size=300)
    firms = np.repeat(np.arange(100), 3)
    lo, hi = m.cluster_bootstrap(lambda i: v[i].mean(), firms, reps=300)
    assert lo < v.mean() < hi


def test_persistence_stage_recovers_coefficients():
    rng = np.random.default_rng(0)
    D = pd.DataFrame(rng.normal(size=(500, len(PERSISTENCE))), columns=PERSISTENCE)
    D["other"] = rng.normal(size=500)
    beta = np.linspace(-1, 1, len(PERSISTENCE))
    y = 0.3 + D[PERSISTENCE].to_numpy() @ beta + 5 * D["other"]
    s = PersistenceStage().fit(D, y)
    np.testing.assert_allclose(s.model_.coef_, beta, atol=0.6)  # 'other' is noise to stage 1
    np.testing.assert_allclose(s.residual(D, y), y - s.predict(D))


def test_text_block_fits_on_given_rows_only():
    fit = pd.DataFrame({c: [1.0, 3.0, np.nan] for c in TEXT_COLUMNS})
    other = pd.DataFrame({c: [np.nan, 4.0] for c in TEXT_COLUMNS})
    tb = TextBlock("tabnet").fit(fit)
    out = tb.transform(other)
    assert out[TEXT_COLUMNS[0]].tolist() == [2.0, 4.0]  # median of the fit rows
    assert out["text_missing"].tolist() == [1.0, 0.0]
    assert out["text_late"].tolist() == [0.0, 0.0]  # no flag column given
    flagged = other.assign(**{"披露晚于4月30日": [np.nan, 1.0]})
    assert tb.transform(flagged)["text_late"].tolist() == [0.0, 1.0]


def test_every_design_column_has_a_module(X):
    pre = FeaturePreprocessor("lasso").fit(X[split_masks(X)["train"]])
    names = pre.feature_names + TEXT_COLUMNS + ["text_missing", "text_late"]
    mods = module_columns(names)
    assert sorted(sum(mods.values(), [])) == sorted(names)
    assert module_of("capex_missing") == "persistence"
    assert module_of("第一大客户占比_missing") == "channel_supply"
    assert module_of("industry=白酒II") == module_of("industry") == "macro_industry"
    assert module_of("text_missing") == module_of("text_late") == "text"
    assert all(len(v) == len(set(v)) for v in mods.values())


def test_lag_tail_flags_come_from_previous_year(X):
    lab = X[KEY + ["mpdr"]].copy()
    lab["down"] = m.yearly_tail(lab["mpdr"], lab[YEAR])
    prev = lab.assign(**{YEAR: lab[YEAR] + 1})[KEY + ["down"]]
    got = X[KEY + ["lag_down_tail"]].merge(prev, on=KEY, how="inner")
    assert len(got) > 9000
    assert (got["lag_down_tail"].astype(bool) == got["down"]).all()
    first_years = X.groupby("股票代码")[YEAR].transform("min") == X[YEAR]
    assert X.loc[first_years, "lag_down_tail"].isna().all()


def test_test_labels_do_not_reach_stage1_or_residuals(X, built):
    folds, _, base, _ = built
    Xp = X.copy()
    test = split_masks(Xp)["test"]
    Xp.loc[test, "mpdr"] = Xp.loc[test, "mpdr"] + 10.0
    folds2, _, base2, y2 = build(Xp, split_masks(Xp), False)
    np.testing.assert_allclose(base, base2)
    np.testing.assert_allclose(folds["lasso"].r_refit, folds2["lasso"].r_refit)


def test_tuning_residuals_use_train_fit_only(X, built):
    folds, _, _, _ = built
    Xp = X.copy()
    mk = split_masks(Xp)
    Xp.loc[mk["valid"], "mpdr"] = Xp.loc[mk["valid"], "mpdr"] + 10.0
    folds2, _, _, _ = build(Xp, split_masks(Xp), False)
    np.testing.assert_allclose(folds["lasso"].r_train, folds2["lasso"].r_train)
    np.testing.assert_allclose(folds2["lasso"].r_valid - folds["lasso"].r_valid, 10.0)


def test_tabnet_design_matches_lasso_numeric_block(built):
    folds, cat_dims, _, _ = built
    fl, ft = folds["lasso"], folds["tabnet"]
    numeric = [c for c in ft.test.columns if c not in CATEGORICAL]
    pd.testing.assert_frame_equal(ft.test[numeric], fl.test[numeric])
    assert list(ft.test.columns[-2:]) == CATEGORICAL
    for c in CATEGORICAL:
        assert ft.refit[c].max() < cat_dims[c] and ft.train[c].max() < cat_dims[c]


def test_text_scores_join_design_and_ablation(X, tmp_path):
    from mpdr.modules import attach_text, load_text
    rng = np.random.default_rng(0)
    T = X[KEY].sample(frac=0.9, random_state=0)
    for c in TEXT_COLUMNS:
        T[c] = rng.integers(0, 5, len(T))
    path = tmp_path / "annual_report_text_scores.csv"
    T.assign(年报文本质量="ok").to_csv(path, index=False, encoding="utf-8-sig")
    assert load_text(tmp_path / "missing.csv") is None
    Xt = attach_text(X.drop(columns=[c for c in TEXT_COLUMNS if c in X]), load_text(path))
    folds, _, _, _ = build(Xt, split_masks(Xt), True)
    cols = list(folds["tabnet"].test.columns)
    assert set(TEXT_COLUMNS + ["text_missing", "text_late"]) <= set(cols)
    assert module_columns(cols)["text"] == TEXT_COLUMNS + ["text_missing", "text_late"]
    # standardized flag: positive exactly on rows without a score
    assert (folds["lasso"].train["text_missing"] > 0).mean() == pytest.approx(
        Xt.loc[split_masks(Xt)["train"], TEXT_COLUMNS[0]].isna().mean())
