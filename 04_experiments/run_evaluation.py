"""Proposal Table 13 evaluation of the two-stage test predictions, and accuracy by deviation direction and size.

Reads results_two_stage/test_predictions.csv from run_two_stage.py; nothing is refitted. Confidence intervals
are firm-cluster bootstrap percentiles. Writes to results_two_stage/:
  evaluation.csv        model x metric (seven dimensions), with CI and the difference from stage 1
  rq3_accuracy.csv      errors by actual deviation group (severe/moderate negative, moderate/severe positive)
  rq3_asymmetry.csv     downside minus upside contrasts per model, with CI

  python3 run_evaluation.py [--reps 1000]
"""
import argparse

import numpy as np
import pandas as pd

from mpdr.config import KEY, LABEL_PATH, SE, YEAR
from mpdr.metrics import (balanced_direction_accuracy, cluster_resamples, direction_accuracy, ewal, mae,
                          new_entry_mask, r2_oos, r2_vs, rmse, sign_detection, tail_auc, yearly_spearman,
                          yearly_tail, yearly_topk_hit)
from run_two_stage import OUT, load_data

MODELS = ["lag_mpdr", "stage1_only", "lasso", "tabnet", "bo_tabnet", "lgbm", "bo_lgbm"]
DIMENSION = {
    "mae": "regression_error", "rmse": "regression_error",
    "r2": "explanatory_power", "r2_vs_stage1": "explanatory_power",
    "direction_acc": "direction", "balanced_direction_acc": "direction",
    "neg_recall": "risk_warning", "neg_precision": "risk_warning", "neg_f1": "risk_warning",
    "auc_down": "risk_warning", "auc_down_new_entry": "risk_warning",
    "top5_hit_down": "ranking", "top10_hit_down": "ranking", "top20_hit_down": "ranking",
    "top10_hit_up": "ranking", "spearman": "ranking",
    "ewal": "economic_loss", "ewal_vs_stage1": "economic_loss", "ewal_logw": "economic_loss",
    "ewal_a1": "economic_loss", "ewal_a3": "economic_loss",
}
GROUPS = ["severe_negative", "moderate_negative", "moderate_positive", "severe_positive"]


def load_test():
    """Test predictions with SE, last year's MPDR and last year's tail flags."""
    P = pd.read_csv(OUT / "test_predictions.csv", dtype={"股票代码": str}, encoding="utf-8-sig")
    X = load_data()
    lab = pd.read_csv(LABEL_PATH, dtype={"股票代码": str}, encoding="utf-8-sig")
    lag = lab[KEY + ["mpdr"]].assign(**{YEAR: lab[YEAR] + 1}).rename(columns={"mpdr": "lag_mpdr"})
    T = (P.merge(X[KEY + [SE, "lag_down_tail", "lag_up_tail"]], on=KEY, how="left", validate="one_to_one")
         .merge(lag, on=KEY, how="left", validate="one_to_one"))
    # Naive persistence forecast: last year's deviation; firms without one get the expected level, 0.
    T["pred_lag_mpdr"] = T["lag_mpdr"].fillna(0.0)
    return T


def table13(y, P, year, se, lag_down):
    """{model: {metric: value}} for the proposal's seven dimensions."""
    new = new_entry_mask(lag_down)
    base = P["stage1_only"]
    out = {}
    for name, p in P.items():
        out[name] = {
            "mae": mae(y, p), "rmse": rmse(y, p), "r2": r2_oos(y, p), "r2_vs_stage1": r2_vs(y, p, base),
            "direction_acc": direction_accuracy(y, p), "balanced_direction_acc": balanced_direction_accuracy(y, p),
            **sign_detection(y, p),
            "auc_down": tail_auc(y, p, year, "negative"),
            "auc_down_new_entry": tail_auc(y, p, year, "negative", mask=new),
            "top5_hit_down": yearly_topk_hit(y, p, year, 0.05), "top10_hit_down": yearly_topk_hit(y, p, year, 0.10),
            "top20_hit_down": yearly_topk_hit(y, p, year, 0.20),
            "top10_hit_up": yearly_topk_hit(y, p, year, 0.10, "positive"), "spearman": yearly_spearman(y, p, year),
            "ewal": ewal(y, p, se), "ewal_vs_stage1": ewal(y, p, se) / ewal(y, base, se),
            # SE weights put about a third of the weight on the largest 1% of firm-years; log weights do not.
            "ewal_logw": ewal(y, p, np.log1p(se)),
            "ewal_a1": ewal(y, p, se, alpha=1.0), "ewal_a3": ewal(y, p, se, alpha=3.0),
        }
    return out


def deviation_group(y, year):
    """Actual deviation group: the year's bottom/top 10% are severe, the rest split at 0."""
    g = np.where(y < 0, "moderate_negative", "moderate_positive").astype(object)
    g[yearly_tail(y, year, tail="negative")] = "severe_negative"
    g[yearly_tail(y, year, tail="positive")] = "severe_positive"
    return g


def group_errors(y, P, group):
    rows = []
    base = P["stage1_only"]
    for name, p in P.items():
        for g in GROUPS:
            k = group == g
            e = p[k] - y[k]
            rows.append({"model": name, "group": g, "n": int(k.sum()), "mae": float(np.mean(np.abs(e))),
                         "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean()),
                         "mse_gain_vs_stage1": r2_vs(y[k], p[k], base[k])})
    return rows


def contrasts(y, P, year, group, lag_down, lag_up):
    """Downside minus upside, per model: errors on the severe groups, MSE gain over stage 1, tail AUCs."""
    new_d, new_u = new_entry_mask(lag_down), new_entry_mask(lag_up)
    down, up = group == "severe_negative", group == "severe_positive"
    base = P["stage1_only"]
    out = {}
    for name, p in P.items():
        c = {"mae_severe": mae(y, p, down) - mae(y, p, up),
             "abs_bias_severe": abs(np.mean(p[down] - y[down])) - abs(np.mean(p[up] - y[up])),
             "mse_gain_vs_stage1_severe": r2_vs(y[down], p[down], base[down]) - r2_vs(y[up], p[up], base[up]),
             "top10_hit": yearly_topk_hit(y, p, year, 0.10) - yearly_topk_hit(y, p, year, 0.10, "positive")}
        for q in (0.05, 0.10, 0.20):
            c[f"auc_q{int(q * 100)}"] = (tail_auc(y, p, year, "negative", q) - tail_auc(y, p, year, "positive", q))
        c["auc_q10_new_entry"] = (tail_auc(y, p, year, "negative", mask=new_d)
                                  - tail_auc(y, p, year, "positive", mask=new_u))
        out[name] = c
    return out


def with_ci(point, draws):
    """Long frame: point value and 95% percentile interval for every (model, metric)."""
    rows = []
    for name, metrics in point.items():
        for k, v in metrics.items():
            d = np.array([dr[name][k] for dr in draws])
            rows.append({"model": name, "metric": k, "value": v, "ci_low": np.nanpercentile(d, 2.5),
                         "ci_high": np.nanpercentile(d, 97.5)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()

    T = load_test()
    y, year, se = T["mpdr"].to_numpy(), T[YEAR].to_numpy(), T[SE].to_numpy()
    firms = T["股票代码"].to_numpy()
    P = {n: T[f"pred_{n}"].to_numpy() for n in MODELS}
    lag_d, lag_u = T["lag_down_tail"].to_numpy(), T["lag_up_tail"].to_numpy()
    group = deviation_group(y, year)

    def sub(idx):
        return y[idx], {n: p[idx] for n, p in P.items()}, year[idx]

    # One set of firm resamples serves every metric, so differences between models are paired.
    t13, t13_draws, rq3, rq3_draws = table13(y, P, year, se, lag_d), [], contrasts(y, P, year, group, lag_d, lag_u), []
    for idx in cluster_resamples(firms, a.reps):
        yy, pp, yr = sub(idx)
        t13_draws.append(table13(yy, pp, yr, se[idx], lag_d[idx]))
        rq3_draws.append(contrasts(yy, pp, yr, group[idx], lag_d[idx], lag_u[idx]))

    ev = with_ci(t13, t13_draws)
    ev.insert(1, "dimension", ev["metric"].map(DIMENSION))
    diff = [np.array([dr[n][k] - dr["stage1_only"][k] for dr in t13_draws]) for n, k in zip(ev["model"], ev["metric"])]
    ev["diff_vs_stage1"] = [t13[n][k] - t13["stage1_only"][k] for n, k in zip(ev["model"], ev["metric"])]
    ev["diff_ci_low"] = [np.nanpercentile(d, 2.5) for d in diff]
    ev["diff_ci_high"] = [np.nanpercentile(d, 97.5) for d in diff]
    order = {m: i for i, m in enumerate(DIMENSION)}
    ev = ev.sort_values(["model", "metric"], key=lambda s: s.map(order) if s.name == "metric" else s.map(MODELS.index))
    ev.to_csv(OUT / "evaluation.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(group_errors(y, P, group)).to_csv(OUT / "rq3_accuracy.csv", index=False, encoding="utf-8-sig")
    with_ci(rq3, rq3_draws).rename(columns={"metric": "contrast"}).to_csv(
        OUT / "rq3_asymmetry.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(ev.pivot(index="metric", columns="model", values="value").loc[list(DIMENSION), MODELS].round(3))


if __name__ == "__main__":
    main()
