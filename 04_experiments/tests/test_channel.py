"""HHI construction from FN_FN010 rank shares."""
import numpy as np
import pandas as pd
import pytest

from build_channel_features import concentration
from mpdr.config import X_PATH
from mpdr.modules import module_of


def shares(**ranks):
    return pd.DataFrame([{r: ranks.get(f"r{r}", np.nan) for r in ["1", "2", "3", "4", "5", "6"]}])


def test_full_ranks_sum_of_squares():
    c = concentration(shares(r1=0.3, r2=0.2, r3=0.1, r4=0.05, r5=0.05, r6=0.7)).iloc[0]
    assert c["hhi"] == pytest.approx(0.09 + 0.04 + 0.01 + 0.0025 + 0.0025)
    assert c["top5"] == pytest.approx(0.7) and c["how"] == "完整名次"


def test_total_from_ranks_when_rank6_missing():
    c = concentration(shares(r1=0.2, r2=0.2, r3=0.2, r4=0.2, r5=0.2)).iloc[0]
    assert c["top5"] == pytest.approx(1.0) and c["hhi"] == pytest.approx(0.2)


def test_partial_ranks_spread_remainder_evenly():
    c = concentration(shares(r1=0.4, r6=0.6)).iloc[0]
    assert c["hhi"] == pytest.approx(0.16 + 4 * 0.05 ** 2) and c["how"] == "部分名次补齐"


def test_fewer_than_five_counterparties():
    c = concentration(shares(r1=0.5, r2=0.3, r6=0.8)).iloc[0]
    assert c["hhi"] == pytest.approx(0.25 + 0.09)


def test_total_only_has_no_hhi():
    c = concentration(shares(r6=0.5)).iloc[0]
    assert np.isnan(c["hhi"]) and c["how"] == "仅披露合计" and c["top5"] == 0.5


def test_ranks_without_valid_total_have_no_hhi():
    c = concentration(shares(r1=0.5, r2=0.3, r3=0.2, r4=0.1, r5=0.1)).iloc[0]  # ranks sum above 1
    assert np.isnan(c["hhi"]) and np.isnan(c["top5"]) and c["how"] == "无数据"


def test_ranks_contradicting_total_have_no_hhi():
    for c in (shares(r1=0.3, r2=0.2, r3=0.1, r4=0.05, r5=0.05, r6=0.5), shares(r1=0.9, r6=0.5)):
        c = c.pipe(concentration).iloc[0]
        assert np.isnan(c["hhi"]) and c["how"] == "名次与合计不符" and c["top5"] == 0.5


def test_x_hhi_bounded_by_top5_share():
    X = pd.read_csv(X_PATH, encoding="utf-8-sig", usecols=lambda c: "占比" in c or "集中度" in c)
    for p in ("客户", "供应商"):
        h, t, f = X[f"{p}集中度"], X[f"前五大{p}占比"], X[f"第一大{p}占比"]
        ok = h.notna()
        assert (h[ok] <= (t[ok] + 0.01) ** 2).all() and (h[ok] >= (t[ok] - 0.01).clip(lower=0) ** 2 / 5).all()
        ok &= f.notna()  # a partial disclosure can give ranks 2-5 without rank 1
        assert (h[ok] >= f[ok] ** 2 - 1e-9).all()


def test_channel_module_holds_table9_variables():
    for c in ("前五大客户占比", "客户集中度", "前五大供应商占比", "供应商集中度", "存货周转率", "存货周转率_missing"):
        assert module_of(c) == "channel_supply"
