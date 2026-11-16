"""Evaluation metrics for MPDR predictions. Every metric takes an optional boolean mask."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

TAIL_Q = 0.10
ALPHA, BETA = 2.0, 1.0


def _select(mask, *arrays):
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    if mask is None:
        return arrays
    mask = np.asarray(mask, dtype=bool)
    return [a[mask] for a in arrays]


def mae(y, yhat, mask=None):
    y, yhat = _select(mask, y, yhat)
    return float(np.mean(np.abs(yhat - y)))


def rmse(y, yhat, mask=None):
    y, yhat = _select(mask, y, yhat)
    return float(np.sqrt(np.mean((yhat - y) ** 2)))


def r2_oos(y, yhat, mask=None):
    """Out-of-sample R²: 1 - SSE / SS around the evaluation-set mean of y."""
    y, yhat = _select(mask, y, yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def direction_accuracy(y, yhat, mask=None):
    """Share of rows where the predicted sign matches the actual sign."""
    y, yhat = _select(mask, y, yhat)
    return float(np.mean(np.sign(y) == np.sign(yhat)))


def _bottom(v, q):
    return v <= np.quantile(v, q)


def tail_detection(y, yhat, q=TAIL_Q, mask=None):
    """Recall, precision and F1 for the actual bottom-q tail, flagging the bottom q of predictions."""
    y, yhat = _select(mask, y, yhat)
    actual, flagged = _bottom(y, q), _bottom(yhat, q)
    hits = np.sum(actual & flagged)
    recall, precision = hits / actual.sum(), hits / flagged.sum()
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return {"tail_recall": float(recall), "tail_precision": float(precision), "tail_f1": float(f1)}


def topk_hit_rate(y, yhat, q=TAIL_Q, tail="negative", mask=None):
    """Share of the ceil(q*n) most extreme predictions that are in the actual q tail on the same side."""
    y, yhat = _select(mask, y, yhat)
    if tail == "positive":
        y, yhat = -y, -yhat
    k = int(np.ceil(q * len(y)))
    top = np.argsort(yhat, kind="stable")[:k]
    return float(np.mean(_bottom(y, q)[top]))


def ewal(y, yhat, se, alpha=ALPHA, beta=BETA, mask=None):
    """Exposure-weighted asymmetric loss with e = yhat - y (e > 0: risk underestimated).

    W = SE / mean(SE) in the evaluation set; missing SE counts as 0, as in the label.
    """
    y, yhat, se = _select(mask, y, yhat, se)
    se = np.nan_to_num(se, nan=0.0)
    w = se / se.mean()
    e = yhat - y
    loss = alpha * np.maximum(e, 0) + beta * np.maximum(-e, 0)
    return float(np.sum(w * loss) / np.sum(w))


def evaluate(y, yhat, se, mask=None):
    """All metrics as one flat dict."""
    return {
        "n": int(len(_select(mask, y)[0])),
        "mae": mae(y, yhat, mask),
        "rmse": rmse(y, yhat, mask),
        "r2": r2_oos(y, yhat, mask),
        "direction_acc": direction_accuracy(y, yhat, mask),
        **tail_detection(y, yhat, mask=mask),
        "topk_hit_neg": topk_hit_rate(y, yhat, tail="negative", mask=mask),
        "topk_hit_pos": topk_hit_rate(y, yhat, tail="positive", mask=mask),
        "ewal": ewal(y, yhat, se, mask=mask),
    }


# Comparisons against a benchmark forecast, tail-event AUCs and their uncertainty.

def r2_vs(y, yhat, ybench, mask=None):
    """Campbell-Thompson R²_OS against a benchmark forecast: 1 - SSE(model) / SSE(benchmark)."""
    y, yhat, ybench = _select(mask, y, yhat, ybench)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - ybench) ** 2))


def _t_stat(d):
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def clark_west(y, ybench, yhat, mask=None):
    """Clark-West (2007) MSPE-adjusted t for a model nesting the benchmark; one-sided p (model better)."""
    y, ybench, yhat = _select(mask, y, ybench, yhat)
    t = _t_stat((y - ybench) ** 2 - ((y - yhat) ** 2 - (ybench - yhat) ** 2))
    return t, float(stats.norm.sf(t))


def diebold_mariano(y, yhat_a, yhat_b, mask=None):
    """Diebold-Mariano t on squared-error differences, positive when b is more accurate; two-sided p."""
    y, a, b = _select(mask, y, yhat_a, yhat_b)
    t = _t_stat((y - a) ** 2 - (y - b) ** 2)
    return t, float(2 * stats.norm.sf(abs(t)))


def yearly_tail(y, year, q=TAIL_Q, tail="negative"):
    """Boolean: y in the bottom (or top) q of its own year."""
    s = pd.Series(np.asarray(y, dtype=float))
    g = s.groupby(np.asarray(year))
    if tail == "negative":
        return (s <= g.transform(lambda v: v.quantile(q))).to_numpy()
    return (s >= g.transform(lambda v: v.quantile(1 - q))).to_numpy()


def tail_auc(y, yhat, year, tail="negative", q=TAIL_Q, mask=None):
    """AUC of the prediction for membership in the year's bottom (top) q; low predictions flag the bottom."""
    event = yearly_tail(y, year, q, tail)
    score = -np.asarray(yhat, dtype=float) if tail == "negative" else np.asarray(yhat, dtype=float)
    if mask is not None:
        event, score = event[np.asarray(mask, bool)], score[np.asarray(mask, bool)]
    return float(roc_auc_score(event, score))


def new_entry_mask(lag_in_tail):
    """Rows not in the tail last year (a missing lag counts as not in the tail)."""
    return ~pd.Series(lag_in_tail).fillna(False).astype(bool).to_numpy()


def cluster_resamples(firms, reps=500, seed=0):
    """Yields row-index arrays, each a resample of whole firms with replacement."""
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(firms))).groupby(np.asarray(firms)).indices
    keys = np.array(list(groups))
    for _ in range(reps):
        yield np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])


def cluster_bootstrap(stat, firms, reps=500, seed=0):
    """Percentile 95% CI of stat(idx) over firm-cluster resamples; stat takes an index array."""
    draws = []
    for idx in cluster_resamples(firms, reps, seed):
        try:
            draws.append(stat(idx))
        except ValueError:  # a resample with a single class
            continue
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


# Table 13 metrics that need the year (warnings and rankings are issued within each year).

def sign_detection(y, yhat, mask=None):
    """Recall, precision and F1 for actual MPDR < 0, flagging predicted MPDR < 0 (proposal Table 13)."""
    y, yhat = _select(mask, y, yhat)
    actual, flagged = y < 0, yhat < 0
    hits = np.sum(actual & flagged)
    recall = hits / actual.sum() if actual.any() else np.nan
    precision = hits / flagged.sum() if flagged.any() else np.nan
    f1 = 0.0 if hits == 0 else 2 * precision * recall / (precision + recall)
    return {"neg_recall": float(recall), "neg_precision": float(precision), "neg_f1": float(f1)}


def balanced_direction_accuracy(y, yhat, mask=None):
    """Mean of the hit rates on actual-negative and actual-positive rows."""
    y, yhat = _select(mask, y, yhat)
    return float(np.mean([np.mean(yhat[y < 0] < 0), np.mean(yhat[y >= 0] >= 0)]))


def yearly_topk_hit(y, yhat, year, q=TAIL_Q, tail="negative"):
    """Share of each year's q most extreme predictions (low for negative) that are in that year's actual q tail."""
    actual, flagged = yearly_tail(y, year, q, tail), yearly_tail(yhat, year, q, tail)
    return float(np.sum(actual & flagged) / flagged.sum())


def yearly_spearman(y, yhat, year):
    """Spearman rank correlation within each year, averaged with year weights n."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(yhat, float), "g": np.asarray(year)})
    rho = df.groupby("g").apply(lambda d: stats.spearmanr(d["y"], d["p"])[0], include_groups=False)
    return float(np.average(rho, weights=df.groupby("g").size()))


def ewal_ratio(y, yhat, ybench, se, alpha=ALPHA, beta=BETA):
    """EWAL relative to a benchmark forecast; below 1 means lower exposure-weighted loss."""
    return ewal(y, yhat, se, alpha, beta) / ewal(y, ybench, se, alpha, beta)
