"""Cumulative module ablation (proposal Table 12) and a Shapley decomposition of each module's contribution.

Every specification keeps stage 1 (persistence OLS) and models its residual with the persistence block plus the
specification's modules:
  model 1  marketing input + operating base
  model 2  model 1 + annual-report text           (runs once 02_data/final/annual_report_text_scores.csv exists)
  model 3  model 2 + complaints and news          (no data yet, reported as pending)
  model 4  model 3 + macro and industry environment
  model 5  model 4 + channel and supply chain     (the full model)
Each specification gets its own Bayesian search for LightGBM and TabNet (Lasso tunes alpha on valid as usual), so
increments are not an artefact of hyperparameters fitted to the full model. Consecutive available models are
nested, so increments are tested with Clark-West and with paired firm-cluster bootstrap CIs.

The order of Table 12 decides how much each module appears to add. The Shapley decomposition removes that: stage 2
is fitted on every subset of the modules (persistence always in) and each module gets its average marginal
contribution over all orders, so the module shares add up to the full model's gain over the persistence-only stage 2.
Subsets re-tune Lasso and LightGBM (with --shapley-trials); TabNet reuses model 5's tuned parameters.

  python3 run_cumulative_ablation.py                # full run
  python3 run_cumulative_ablation.py --quick        # smoke run: 2 trials, 1 seed, short training

Writes to results_cumulative/: cumulative_metrics.csv, cumulative_increments.csv, shapley_subsets.csv,
shapley_modules.csv, predictions.csv, bo_params.json, bo_trials.csv.
"""
import argparse
import json
import os
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

from mpdr import learners
from mpdr.config import PROJECT_DIR, SE, YEAR
from mpdr.metrics import cluster_resamples, clark_west, r2_vs, sign_detection, tail_auc, yearly_topk_hit, ewal
from mpdr.modules import module_of
from mpdr.split import split_masks
from run_evaluation import table13
from run_two_stage import build, load_data

OUT = Path(os.environ.get("MPDR_CUMULATIVE_DIR", PROJECT_DIR / "04_experiments" / "results_cumulative"))
BASE_MODULE = "persistence"
# Table 12, in order: (model, module added). "complaints" has no columns until that data exists.
STEPS = [("model_1", ["marketing_input", "operating_base"]), ("model_2", ["text"]), ("model_3", ["complaints"]),
         ("model_4", ["macro_industry"]), ("model_5", ["channel_supply"])]
LEARNERS = ["lasso", "bo_lgbm", "bo_tabnet"]
# Metrics bootstrapped for increments and Shapley values; the point table has every Table 13 metric.
KEY_METRICS = ["r2_vs_stage1", "rmse", "neg_f1", "auc_down", "top10_hit_down", "ewal_logw"]


def key_metrics(y, p, base, year, se):
    return {"r2_vs_stage1": r2_vs(y, p, base), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "neg_f1": sign_detection(y, p)["neg_f1"], "auc_down": tail_auc(y, p, year, "negative"),
            "top10_hit_down": yearly_topk_hit(y, p, year, 0.10), "ewal_logw": ewal(y, p, np.log1p(se))}


def columns_for(fold, modules):
    """Design columns of the persistence block plus the given modules, in design order."""
    keep = {BASE_MODULE, *modules}
    return [c for c in fold.test.columns if module_of(c) in keep]


def fit(name, fl, ft, cat_dims, modules, params, seeds):
    """Test prediction of stage 2 (residual scale) for one learner on one module set."""
    fl, ft = fl.cols(columns_for(fl, modules)), ft.cols(columns_for(ft, modules))
    if name == "lasso":
        return learners.fit_lasso(fl)[0]
    if name == "bo_lgbm":
        return learners.fit_lgbm(fl, seeds, params)[0]
    return learners.fit_tabnet(ft, params, cat_dims, seeds)[0]


def tune(name, fl, ft, cat_dims, modules, trials):
    """Bayesian search on train/valid for one module set; (params, trials frame), or (None, None) for Lasso."""
    if name == "lasso":
        return None, None
    if name == "bo_lgbm":
        return learners.bayes_opt_lgbm(fl.cols(columns_for(fl, modules)), trials)
    return learners.bayes_opt_tabnet(ft.cols(columns_for(ft, modules)), cat_dims, trials)


def available_steps(fl):
    """Table 12 steps with their cumulative module lists; a step whose module has no columns is pending."""
    present = {module_of(c) for c in fl.test.columns}
    out, modules = [], []
    for step, added in STEPS:
        ok = all(m in present for m in added)
        if ok:
            modules = modules + added
        out.append((step, added, list(modules), ok))
    return out


def shapley(values, players):
    """{player: Shapley value} from {frozenset(subset): value} over all subsets of players."""
    n = len(players)
    w = {s: factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)}
    phi = {}
    for j in players:
        others = [p for p in players if p != j]
        phi[j] = sum(w[k] * (values[frozenset(S) | {j}] - values[frozenset(S)])
                     for k in range(n) for S in combinations(others, k))
    return phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--shapley-trials", type=int, default=20, help="LightGBM search budget per Shapley subset")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--learners", default=",".join(LEARNERS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.shapley_trials, a.seeds, a.reps = 2, 2, 1, 50
        learners.MAX_EPOCHS, learners.PATIENCE = 20, 5
    names = a.learners.split(",")
    seeds = tuple(range(a.seeds))
    OUT.mkdir(parents=True, exist_ok=True)

    X = load_data()
    has_text = "市场感知叙事强度" in X
    m = split_masks(X)
    T = X[m["test"]].reset_index(drop=True)
    folds, cat_dims, base, y = build(X, m, has_text)
    fl, ft = folds["lasso"], folds["tabnet"]
    year, se, firms = T[YEAR].to_numpy(), T[SE].to_numpy(), T["股票代码"].to_numpy()
    steps = available_steps(fl)
    print("steps:", [(s, mods, ok) for s, _, mods, ok in steps], flush=True)

    # Table 12: one Bayesian search per learner and specification.
    preds, params, trials = {}, {}, []
    for step, _, modules, ok in steps:
        if not ok:
            continue
        for name in names:
            p, tr = tune(name, fl, ft, cat_dims, modules, a.trials)
            if tr is not None:
                trials.append(tr.assign(spec=step, model=name))
                params[f"{step}:{name}"] = p
            preds[step, name] = base + fit(name, fl, ft, cat_dims, modules, p, seeds)
            print("done", step, name, flush=True)
    (OUT / "bo_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False))
    if trials:
        pd.concat(trials).to_csv(OUT / "bo_trials.csv", index=False, encoding="utf-8-sig")

    lag_down = T["lag_down_tail"].to_numpy()
    rows = []
    for step, added, modules, ok in steps:
        for name in names:
            if not ok:
                rows.append({"spec": step, "model": name, "added": "+".join(added), "status": "pending"})
                continue
            point = table13(y, {"stage1_only": base, name: preds[step, name]}, year, se, lag_down)[name]
            rows.append({"spec": step, "model": name, "added": "+".join(added), "status": "run",
                         "n_cols": len(columns_for(fl, modules)), **point})
    pd.DataFrame(rows).to_csv(OUT / "cumulative_metrics.csv", index=False, encoding="utf-8-sig")

    # Shapley: stage 2 on every subset of the non-persistence modules.
    players = [mod for _, _, mods, ok in steps if ok for mod in mods]
    players = list(dict.fromkeys(players))
    tabnet_params = params.get(f"{[s for s, _, _, ok in steps if ok][-1]}:bo_tabnet")
    sub_preds = {}
    for k in range(len(players) + 1):
        for S in combinations(players, k):
            for name in names:
                if name == "bo_tabnet":
                    p = tabnet_params
                else:
                    p = tune(name, fl, ft, cat_dims, list(S), a.shapley_trials)[0]
                sub_preds[frozenset(S), name] = base + fit(name, fl, ft, cat_dims, list(S), p, seeds)
            print("subset done", sorted(S), flush=True)

    # One set of firm resamples for increments and Shapley values, so all differences are paired.
    resamples = [None] + list(cluster_resamples(firms, a.reps))

    def metrics_on(idx, pred):
        k = slice(None) if idx is None else idx
        return key_metrics(y[k], pred[k], base[k], year[k], se[k])

    run_steps = [s for s, _, _, ok in steps if ok]
    sub_draws = {key: [metrics_on(idx, p) for idx in resamples] for key, p in sub_preds.items()}
    step_draws = {key: [metrics_on(idx, p) for idx in resamples] for key, p in preds.items()}

    inc = []
    pairs = list(zip(run_steps[:-1], run_steps[1:])) + [(run_steps[0], run_steps[-1])]
    for lo, hi in dict.fromkeys(pairs):
        if lo == hi:
            continue
        for name in names:
            t, pv = clark_west(y, preds[lo, name], preds[hi, name])
            for metric in KEY_METRICS:
                d = np.array([b[metric] - a_[metric] for a_, b in zip(step_draws[lo, name], step_draws[hi, name])])
                inc.append({"from": lo, "to": hi, "model": name, "metric": metric, "diff": d[0],
                            "ci_low": np.nanpercentile(d[1:], 2.5), "ci_high": np.nanpercentile(d[1:], 97.5),
                            "cw_t": t, "cw_p": pv})
    pd.DataFrame(inc).to_csv(OUT / "cumulative_increments.csv", index=False, encoding="utf-8-sig")

    sub_rows = [{"modules": "+".join(sorted(S)) or "(persistence only)", "n_modules": len(S), "model": name,
                 **sub_draws[S, name][0]} for S, name in sub_preds]
    pd.DataFrame(sub_rows).to_csv(OUT / "shapley_subsets.csv", index=False, encoding="utf-8-sig")

    sh = []
    full = frozenset(players)
    for name in names:
        for metric in KEY_METRICS:
            draws = [shapley({S: sub_draws[S, name][r][metric] for S, n in sub_preds if n == name}, players)
                     for r in range(len(resamples))]
            total = sub_draws[full, name][0][metric] - sub_draws[frozenset(), name][0][metric]
            for mod in players:
                d = np.array([dr[mod] for dr in draws])
                sh.append({"model": name, "metric": metric, "module": mod, "shapley": d[0],
                           "share_of_total": d[0] / total if total else np.nan, "total_gain": total,
                           "ci_low": np.nanpercentile(d[1:], 2.5), "ci_high": np.nanpercentile(d[1:], 97.5)})
    pd.DataFrame(sh).to_csv(OUT / "shapley_modules.csv", index=False, encoding="utf-8-sig")

    P = T[["股票代码", YEAR, "mpdr"]].assign(pred_stage1=base)
    for (step, name), p in preds.items():
        P[f"{step}:{name}"] = p
    for (S, name), p in sub_preds.items():
        P[f"subset[{'+'.join(sorted(S))}]:{name}"] = p
    P.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).set_index(["spec", "model"])[["status", "r2_vs_stage1", "neg_f1", "auc_down"]].round(4))
    print(pd.DataFrame(sh).query("metric == 'r2_vs_stage1'")[["model", "module", "shapley", "share_of_total"]].round(4))


if __name__ == "__main__":
    main()
