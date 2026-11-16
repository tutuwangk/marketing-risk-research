"""Robustness checks for the two-stage models.

--part cutoffs   Tail cutoffs and EWAL sensitivity, from the saved test predictions of run_two_stage.py (nothing
                 is refitted). Tail AUC and top-k hit rate at q = 5%, 10%, 20% on both sides; EWAL for
                 alpha in {1, 1.5, 2, 3} (beta = 1) under four exposure weights (selling expense, sales, log selling
                 expense, equal), as a ratio to stage 1. Firm-cluster bootstrap CIs on the difference from stage 1.
                 Writes robustness_cutoffs.csv and robustness_ewal.csv next to the predictions.
--part expanding Expanding-window evaluation by test year: for test year Y, tune on train <= Y-3 and valid Y-2..Y-1,
                 refit on all report years <= Y-1 (their t+1 outcomes are known when year Y's reports are out) and
                 score Y. Every window re-runs the Bayesian searches. Writes results_expanding/.

The lagged (surprise) label needs no code: point the label and the output folder elsewhere and rerun the pipeline,
  MPDR_LABEL_PATH=../02_data/final/mpdr_label_lagged.csv MPDR_TWO_STAGE_DIR=results_two_stage_lagged \\
      sh -c 'python3 run_two_stage.py && python3 run_evaluation.py && python3 run_robustness.py --part cutoffs'

  python3 run_robustness.py --part cutoffs
  python3 run_robustness.py --part expanding [--quick]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mpdr import learners
from mpdr.config import KEY, PROJECT_DIR, SE, YEAR
from mpdr.metrics import clark_west, cluster_resamples, ewal, r2_vs, tail_auc, yearly_topk_hit
from run_evaluation import table13
from run_two_stage import OUT, build, load_data, stage2

EXP_OUT = Path(os.environ.get("MPDR_EXPANDING_DIR", PROJECT_DIR / "04_experiments" / "results_expanding"))
MODELS = ["stage1_only", "lasso", "bo_lgbm", "bo_tabnet"]
CUTOFFS = (0.05, 0.10, 0.20)
ALPHAS = (1.0, 1.5, 2.0, 3.0)
WEIGHTS = {"selling_expense": lambda T: T[SE].to_numpy(), "sales": lambda T: T["营业收入"].to_numpy(),
           "log_selling_expense": lambda T: np.log1p(T[SE].fillna(0).to_numpy()),
           "equal": lambda T: np.ones(len(T))}
EXP_YEARS = range(2019, 2025)


def cutoff_metrics(y, P, year, W):
    """{model: {metric: value}} for the tail-cutoff grid and the EWAL grid (EWAL as a ratio to stage 1)."""
    out = {}
    for name, p in P.items():
        r = {}
        for q in CUTOFFS:
            for tail, side in (("negative", "down"), ("positive", "up")):
                r[f"auc_{side}_q{int(q * 100)}"] = tail_auc(y, p, year, tail, q)
                r[f"top_hit_{side}_q{int(q * 100)}"] = yearly_topk_hit(y, p, year, q, tail)
        for w, wv in W.items():
            for a in ALPHAS:
                r[f"ewal_{w}_a{a:g}"] = ewal(y, p, wv, alpha=a) / ewal(y, P["stage1_only"], wv, alpha=a)
        out[name] = r
    return out


def cutoffs(reps):
    P0 = pd.read_csv(OUT / "test_predictions.csv", dtype={"股票代码": str}, encoding="utf-8-sig")
    X = load_data()
    T = P0.merge(X[KEY + [SE, "营业收入"]], on=KEY, how="left", validate="one_to_one")
    y, year, firms = T["mpdr"].to_numpy(), T[YEAR].to_numpy(), T["股票代码"].to_numpy()
    P = {n: T[f"pred_{n}"].to_numpy() for n in MODELS if f"pred_{n}" in T}
    W = {k: f(T) for k, f in WEIGHTS.items()}
    point = cutoff_metrics(y, P, year, W)
    draws = [cutoff_metrics(y[i], {n: p[i] for n, p in P.items()}, year[i], {k: v[i] for k, v in W.items()})
             for i in cluster_resamples(firms, reps)]
    rows = []
    for name in P:
        for k, v in point[name].items():
            d = np.array([dr[name][k] - dr["stage1_only"][k] for dr in draws])
            rows.append({"model": name, "metric": k, "value": v, "diff_vs_stage1": v - point["stage1_only"][k],
                         "diff_ci_low": np.nanpercentile(d, 2.5), "diff_ci_high": np.nanpercentile(d, 97.5)})
    R = pd.DataFrame(rows)
    is_ewal = R["metric"].str.startswith("ewal_")
    R[~is_ewal].to_csv(OUT / "robustness_cutoffs.csv", index=False, encoding="utf-8-sig")
    E = R[is_ewal].copy()
    E[["weight", "alpha"]] = E["metric"].str.extract(r"ewal_(.+)_a([\d.]+)$")
    E.rename(columns={"value": "ewal_vs_stage1"}).drop(columns="metric").to_csv(
        OUT / "robustness_ewal.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(R.pivot(index="metric", columns="model", values="value").round(3).to_string())


def window_masks(X, test_year):
    year = X[YEAR].to_numpy()
    m = {"train": year <= test_year - 3, "valid": np.isin(year, [test_year - 2, test_year - 1]),
         "test": year == test_year}
    m["refit"] = m["train"] | m["valid"]
    return m


def expanding(trials, seeds):
    EXP_OUT.mkdir(parents=True, exist_ok=True)
    X = load_data()
    has_text = "市场感知叙事强度" in X
    rows, tests, preds, params = [], [], [], []
    for ty in EXP_YEARS:
        m = window_masks(X, ty)
        T = X[m["test"]].reset_index(drop=True)
        folds, cat_dims, base, y = build(X, m, has_text)
        fl, ft = folds["lasso"], folds["tabnet"]
        best = {"bo_tabnet": learners.bayes_opt_tabnet(ft, cat_dims, trials)[0],
                "bo_lgbm": learners.bayes_opt_lgbm(fl, trials)[0]}
        params.append({"test_year": ty, **{k: str(v) for k, v in best.items()}})
        P = {n: base + stage2(n, fl, ft, cat_dims, best, seeds)[0] for n in MODELS}
        year, se = T[YEAR].to_numpy(), T[SE].to_numpy()
        for name, met in table13(y, P, year, se, T["lag_down_tail"].to_numpy()).items():
            rows.append({"test_year": ty, "model": name, "n": len(y), **met})
        for name in MODELS[1:]:
            t, p = clark_west(y, base, P[name])
            tests.append({"test_year": ty, "model": name, "r2_vs_stage1": r2_vs(y, P[name], base), "cw_t": t, "cw_p": p})
        preds.append(T[KEY + ["mpdr"]].assign(**{f"pred_{n}": p for n, p in P.items()}))
        print("window done", ty, flush=True)
    pd.DataFrame(rows).to_csv(EXP_OUT / "expanding_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).to_csv(EXP_OUT / "expanding_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(preds).to_csv(EXP_OUT / "expanding_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(params).to_csv(EXP_OUT / "expanding_bo_params.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(tests).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cutoffs", "expanding"], required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.trials, a.seeds = 50, 2, 1
        learners.MAX_EPOCHS, learners.PATIENCE = 20, 5
    if a.part == "cutoffs":
        cutoffs(a.reps)
    else:
        expanding(a.trials, tuple(range(a.seeds)))


if __name__ == "__main__":
    main()
