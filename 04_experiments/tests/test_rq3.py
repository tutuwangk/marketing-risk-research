import numpy as np
import pandas as pd
import pytest

from mpdr import metrics as m
from mpdr.attribution import group_shapley, top_drivers
from mpdr.config import YEAR_LEVEL
from mpdr.modules import attribution_groups
from run_evaluation import deviation_group
from run_rq3_diagnosis import band_of, independent_columns


def test_sign_detection_hand_computed():
    y = np.array([-1.0, -2.0, 1.0, 2.0, -0.5])
    p = np.array([-1.0, 0.5, -0.2, 1.0, -0.1])
    # actual negatives: 0, 1, 4; flagged: 0, 2, 4 -> hits 2
    r = m.sign_detection(y, p)
    assert r["neg_recall"] == pytest.approx(2 / 3)
    assert r["neg_precision"] == pytest.approx(2 / 3)
    assert r["neg_f1"] == pytest.approx(2 / 3)
    # negatives hit 2/3, non-negatives hit 1/2
    assert m.balanced_direction_accuracy(y, p) == pytest.approx((2 / 3 + 1 / 2) / 2)


def test_yearly_topk_hit_is_within_year():
    y = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)
    year = np.array([1] * 5 + [2] * 5)
    p = np.array([2, 1, 3, 4, 5, 10, 20, 30, 40, 50], dtype=float)  # year 1 lowest prediction is the 2nd row
    assert m.yearly_topk_hit(y, p, year, q=0.2) == pytest.approx(0.5)
    assert m.yearly_topk_hit(y, p, year, q=0.2, tail="positive") == pytest.approx(1.0)


def test_yearly_spearman_ignores_level_shift_between_years():
    y = np.array([1, 2, 3, 100, 200, 300], dtype=float)
    p = np.array([3, 2, 1, 1, 2, 3], dtype=float)
    year = np.array([1, 1, 1, 2, 2, 2])
    assert m.yearly_spearman(y, p, year) == pytest.approx(0.0)


def test_cluster_resamples_keep_firms_whole():
    firms = np.array(["a", "a", "b", "c", "c", "c"])
    for idx in m.cluster_resamples(firms, reps=20):
        counts = pd.Series(firms[idx]).value_counts()
        assert all(counts[f] % (firms == f).sum() == 0 for f in counts.index)


def test_group_shapley_additive_model_recovers_group_effects():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=list("abcd"))
    bg = pd.DataFrame(rng.normal(size=(10, 4)), columns=list("abcd"))
    f = lambda D: 2 * D["a"] - D["b"] ** 2 + np.sin(D["c"]) * D["d"]
    phi, base = group_shapley(f, X, bg, {"g1": ["a"], "g2": ["b"], "g3": ["c", "d"]}, chunk=7)
    np.testing.assert_allclose(phi["g1"], 2 * (X["a"] - bg["a"].mean()))
    np.testing.assert_allclose(phi["g2"], -(X["b"] ** 2 - (bg["b"] ** 2).mean()))
    assert base == pytest.approx(f(bg).mean())


def test_group_shapley_efficiency_with_interactions():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(12, 3)), columns=list("abc"))
    bg = pd.DataFrame(rng.normal(size=(6, 3)), columns=list("abc"))
    f = lambda D: D["a"] * D["b"] + np.maximum(D["c"], 0) * D["a"]
    phi, base = group_shapley(f, X, bg, {"a": ["a"], "b": ["b"], "c": ["c"]})
    np.testing.assert_allclose(phi.sum(axis=1) + base, f(X))
    # symmetric interaction a*b with b's and a's roles swapped gives equal shares when x_a = x_b
    Y = pd.DataFrame({"a": [1.0], "b": [1.0], "c": [0.0]})
    zero = pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]})
    phi1, _ = group_shapley(lambda D: D["a"] * D["b"], Y, zero, {"a": ["a"], "b": ["b"], "c": ["c"]})
    assert phi1.iloc[0].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_attribution_groups_split_macro_and_industry():
    g = attribution_groups(["mp_t", YEAR_LEVEL[0], "industry=白酒II", "消费类型=必需消费", "industry", "ROA"])
    assert list(g) == ["persistence", "operating_base", "macro", "industry"]
    assert g["industry"] == ["industry=白酒II", "消费类型=必需消费", "industry"]


def test_top_drivers_signed_and_formatted():
    contrib = pd.DataFrame({"a": [-0.3, 0.1], "b": [0.2, -0.05], "c": [-0.1, 0.0]})
    values = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    assert top_drivers(contrib, values, k=2, sign=-1) == ["a=1(-0.300); c=5(-0.100)", "b=4(-0.050)"]
    assert top_drivers(contrib, values, k=1, sign=1) == ["b=3(+0.200)", "a=2(+0.100)"]


def test_deviation_group_and_band_are_within_year():
    y = np.array([-5, -1, 1, 2, 9, -50, -10, 10, 20, 90], dtype=float)
    year = np.array([1] * 5 + [2] * 5)
    g = deviation_group(y, year)
    assert g.tolist() == ["severe_negative", "moderate_negative", "moderate_positive", "moderate_positive",
                          "severe_positive", "severe_negative", "moderate_negative", "moderate_positive",
                          "moderate_positive", "severe_positive"]
    b = band_of(y, year)
    assert (b == "downside_warning").sum() == 2 and (b == "upside_opportunity").sum() == 2


def test_independent_columns_drops_exact_collinearity():
    D = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 4, 6, 8], "d1": [1.0, 0, 1, 0], "d2": [0.0, 1, 0, 1]})
    assert independent_columns(D) == ["a", "d1"]
