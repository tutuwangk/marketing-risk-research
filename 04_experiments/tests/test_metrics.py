import numpy as np
import pytest

from mpdr import metrics as m

Y = np.array([1, -1, 2, -2, 3, -3, 4, -4, 5, -5], dtype=float)
YHAT = np.array([2, -1, 1, -3, 3, 1, 4, -2, 6, -4], dtype=float)
# e = yhat - y = [1, 0, -1, -1, 0, 4, 0, 2, 1, 1]


def test_point_metrics():
    assert m.mae(Y, YHAT) == pytest.approx(11 / 10)
    assert m.rmse(Y, YHAT) == pytest.approx(np.sqrt(25 / 10))
    # mean(y) = 0, SST = 110, SSE = 25
    assert m.r2_oos(Y, YHAT) == pytest.approx(1 - 25 / 110)
    # only index 5 has the wrong sign
    assert m.direction_accuracy(Y, YHAT) == pytest.approx(0.9)


def test_r2_uses_eval_mean():
    y = np.array([10.0, 12.0, 14.0])
    assert m.r2_oos(y, np.full(3, 12.0)) == pytest.approx(0.0)
    assert m.r2_oos(y, y) == pytest.approx(1.0)


def test_mask_selects_rows():
    mask = np.array([True] * 5 + [False] * 5)
    # e on first five rows = [1, 0, -1, -1, 0]
    assert m.mae(Y, YHAT, mask) == pytest.approx(3 / 5)
    assert m.direction_accuracy(Y, YHAT, mask) == pytest.approx(1.0)
    se = np.arange(1, 11, dtype=float)
    assert m.evaluate(Y, YHAT, se, mask) == m.evaluate(Y[mask], YHAT[mask], se[mask])


def test_tail_detection_with_ties():
    y = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float)
    yhat = np.array([-1, -1, -1, 0, 0, 1, 2, 3, 4, 5], dtype=float)
    # q=0.2: actual tail = y <= -3.2 -> {0, 1}; flagged = yhat <= -1 -> {0, 1, 2}
    out = m.tail_detection(y, yhat, q=0.2)
    assert out["tail_recall"] == pytest.approx(1.0)
    assert out["tail_precision"] == pytest.approx(2 / 3)
    assert out["tail_f1"] == pytest.approx(0.8)


def test_tail_detection_no_hits():
    y = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float)
    out = m.tail_detection(y, -y, q=0.2)
    assert out == {"tail_recall": 0.0, "tail_precision": 0.0, "tail_f1": 0.0}


def test_topk_hit_rate():
    y = np.array([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], dtype=float)
    yhat = np.array([-2, 4.5, -3, 0, 0, 1, 2, -1, 4, 5])
    # k = 2. Actual bottom: {0, 1}; most negative predictions: {2, 0} -> 1 hit
    assert m.topk_hit_rate(y, yhat, q=0.2) == pytest.approx(0.5)
    # Actual top: {8, 9}; most positive predictions: {9, 1} -> 1 hit
    assert m.topk_hit_rate(y, yhat, q=0.2, tail="positive") == pytest.approx(0.5)
    # default q = 0.1 on n = 10 -> k = 1: index 9 vs predicted index 2 / 9
    assert m.topk_hit_rate(y, yhat) == 0.0
    assert m.topk_hit_rate(y, yhat, tail="positive") == 1.0


def test_ewal_hand_computed():
    y = np.zeros(4)
    yhat = np.array([1.0, -1.0, 2.0, -2.0])
    se = np.array([1.0, 1.0, 2.0, 0.0])
    # loss = [2*1, 1*1, 2*2, 1*2]; W = se / 1 -> (2 + 1 + 8 + 0) / 4
    assert m.ewal(y, yhat, se) == pytest.approx(11 / 4)
    # missing SE weighs like SE = 0; W is scale-free
    assert m.ewal(y, yhat, [1.0, 1.0, 2.0, np.nan]) == pytest.approx(11 / 4)
    assert m.ewal(y, yhat, se * 1e8) == pytest.approx(11 / 4)


def test_ewal_asymmetry():
    # Over-predicting MPDR (risk underestimated) costs alpha = 2, under-predicting costs beta = 1
    assert m.ewal([0.0], [1.0], [5.0]) == pytest.approx(2.0)
    assert m.ewal([0.0], [-1.0], [5.0]) == pytest.approx(1.0)


def test_evaluate_returns_all_metrics():
    se = np.arange(1, 11, dtype=float)
    out = m.evaluate(Y, YHAT, se)
    assert set(out) == {"n", "mae", "rmse", "r2", "direction_acc", "tail_recall",
                        "tail_precision", "tail_f1", "topk_hit_neg", "topk_hit_pos", "ewal"}
    assert out["n"] == 10
    assert out["mae"] == pytest.approx(1.1)
    assert out["ewal"] == pytest.approx(m.ewal(Y, YHAT, se))
