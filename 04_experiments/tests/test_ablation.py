"""Cumulative ablation steps, Shapley decomposition and the expanding-window split."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_cumulative_ablation import available_steps, columns_for, shapley
from run_robustness import window_masks


def fold(columns):
    return SimpleNamespace(test=pd.DataFrame(columns=columns))


DESIGN = ["mp_t", "gp_to_assets", "销售费用率", "毛利率", "GDP实际增长率", "industry", "客户集中度", "存货周转率_missing"]


def test_steps_skip_modules_without_columns():
    steps = available_steps(fold(DESIGN))
    assert [(s, ok) for s, _, _, ok in steps] == [
        ("model_1", True), ("model_2", False), ("model_3", False), ("model_4", True), ("model_5", True)]
    assert steps[-1][2] == ["marketing_input", "operating_base", "macro_industry", "channel_supply"]


def test_steps_include_text_once_scored():
    steps = available_steps(fold(DESIGN + ["市场感知叙事强度", "text_missing"]))
    assert steps[1][3] and steps[-1][2][2] == "text"


def test_columns_keep_persistence_block():
    assert columns_for(fold(DESIGN), ["marketing_input"]) == ["mp_t", "gp_to_assets", "销售费用率"]
    assert columns_for(fold(DESIGN), ["channel_supply"])[-2:] == ["客户集中度", "存货周转率_missing"]


def test_shapley_additive_game():
    w = {"a": 0.1, "b": 0.3, "c": -0.05}
    v = {frozenset(S): sum(w[p] for p in S) for S in
         [(), ("a",), ("b",), ("c",), ("a", "b"), ("a", "c"), ("b", "c"), ("a", "b", "c")]}
    assert shapley(v, list(w)) == pytest.approx(w)


def test_shapley_splits_interaction_and_sums_to_total():
    v = {frozenset(): 0.0, frozenset("a"): 0.0, frozenset("b"): 0.0, frozenset("ab"): 1.0}
    phi = shapley(v, ["a", "b"])
    assert phi == pytest.approx({"a": 0.5, "b": 0.5})
    assert sum(phi.values()) == pytest.approx(v[frozenset("ab")] - v[frozenset()])


def test_expanding_window_uses_only_earlier_report_years():
    X = pd.DataFrame({"报告年度": np.arange(2013, 2025)})
    m = window_masks(X, 2023)
    assert X.loc[m["train"], "报告年度"].max() == 2020
    assert X.loc[m["valid"], "报告年度"].tolist() == [2021, 2022]
    assert X.loc[m["refit"], "报告年度"].max() == 2022 and X.loc[m["test"], "报告年度"].tolist() == [2023]
