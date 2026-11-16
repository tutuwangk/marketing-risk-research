"""Two-stage MPDR models: persistence OLS (stage 1), then Lasso / LightGBM / TabNet / BO-TabNet on its residual.

Tuning uses train 2013-2020 and valid 2021-2022; final models refit on 2013-2022 and score test 2023-2024 once.
The text module joins the design and the ablation automatically once
02_data/final/annual_report_text_scores.csv exists.

  python3 run_two_stage.py              # full run (40 BO trials, 3 seeds)
  python3 run_two_stage.py --quick      # smoke run: 2 trials, 1 seed, short training
  python3 run_two_stage.py --params results_two_stage/bo_best_params.json   # reuse tuned parameters

Writes to results_two_stage/: metrics.csv, tests.csv, ablation.csv, importance_features.csv,
importance_modules.csv, test_predictions.csv, bo_tabnet_trials.csv, bo_lgbm_trials.csv, bo_best_params.json.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mpdr import learners
from mpdr.config import CATEGORICAL, KEY, LABEL_PATH, PROJECT_DIR, SE, ST, YEAR, YEAR_LEVEL
from mpdr.features import FeaturePreprocessor
from mpdr.learners import Fold
from mpdr.metrics import (clark_west, cluster_bootstrap, diebold_mariano, evaluate, new_entry_mask, r2_vs,
                          tail_auc, yearly_tail)
from mpdr.modules import TextBlock, attach_text, design, load_text, module_columns, module_of
from mpdr.split import split_masks
from mpdr.stages import PersistenceStage
from run_lasso import load

OUT = Path(os.environ.get("MPDR_TWO_STAGE_DIR", PROJECT_DIR / "04_experiments" / "results_two_stage"))
MODELS = ["stage1_only", "lasso", "lgbm", "bo_lgbm", "tabnet", "bo_tabnet"]


def load_data():
    """X with MPDR, last year's MPDR tail flags, restatement flag and (when scored) text columns."""
    X = load()
    lab = pd.read_csv(LABEL_PATH, dtype={"股票代码": str}, encoding="utf-8-sig")
    lab["down_tail"] = yearly_tail(lab["mpdr"], lab[YEAR], tail="negative")
    lab["up_tail"] = yearly_tail(lab["mpdr"], lab[YEAR], tail="positive")
    lag = lab[KEY + ["down_tail", "up_tail"]].assign(**{YEAR: lab[YEAR] + 1})
    X = X.merge(lag.rename(columns={"down_tail": "lag_down_tail", "up_tail": "lag_up_tail"}), on=KEY,
                how="left", validate="one_to_one")
    return attach_text(X, load_text())


def build(X, m, has_text):
    """Design matrices fitted on train (tuning) and on refit (final), plus stage-1 residuals.

    The lasso design is standardized with one-hot categoricals. TabNet gets the same standardized numeric
    columns plus integer-coded categoricals for its embeddings: on unscaled inputs it stopped learning
    after a few epochs on valid.
    """
    parts, pres = {}, {}
    for fit_on in ("train", "refit"):
        pre = pres["lasso", fit_on] = FeaturePreprocessor("lasso").fit(X[m[fit_on]])
        codes = pres["tabnet", fit_on] = FeaturePreprocessor("tabnet").fit(X[m[fit_on]])
        txt = TextBlock("lasso").fit(X[m[fit_on]]) if has_text else None
        for r in (("train", "valid") if fit_on == "train" else ("refit", "test")):
            D = design(pre, txt, X[m[r]]).reset_index(drop=True)
            numeric = [c for c in D.columns if not c.startswith(tuple(f"{c}=" for c in CATEGORICAL))]
            parts["lasso", r] = D
            parts["tabnet", r] = pd.concat(
                [D[numeric], codes.transform(X[m[r]])[CATEGORICAL].reset_index(drop=True)], axis=1)
    y = {k: X.loc[m[k], "mpdr"].to_numpy() for k in ("train", "valid", "refit", "test")}
    s1_train = PersistenceStage().fit(parts["lasso", "train"], y["train"])
    s1_refit = PersistenceStage().fit(parts["lasso", "refit"], y["refit"])
    res = {"train": s1_train.residual(parts["lasso", "train"], y["train"]),
           "valid": s1_train.residual(parts["lasso", "valid"], y["valid"]),
           "refit": s1_refit.residual(parts["lasso", "refit"], y["refit"])}
    folds = {mode: Fold(parts[mode, "train"], parts[mode, "valid"], parts[mode, "refit"], parts[mode, "test"],
                        res["train"], res["valid"], res["refit"]) for mode in ("lasso", "tabnet")}
    # Code ranges can differ between the train-fitted and refit-fitted encoders; size embeddings for both.
    cat_dims = {c: max(pres["tabnet", "train"].cat_dims[i], pres["tabnet", "refit"].cat_dims[i])
                for i, c in enumerate(CATEGORICAL)}
    return folds, cat_dims, s1_refit.predict(parts["lasso", "test"]), y["test"]


def subsets(T):
    st = T[ST].astype(bool).to_numpy()
    out = {"all": None, "non_st": ~st, "no_restated": ~T["restated"].to_numpy()}
    out.update({f"year_{y}": (T[YEAR] == y).to_numpy() for y in sorted(T[YEAR].unique())})
    return out


def score(y, pred, base, T, mask):
    """Point, tail and new-entry metrics for one prediction on one subset."""
    year, se = T[YEAR].to_numpy(), T[SE].to_numpy()
    keep = np.ones(len(y), bool) if mask is None else mask
    new_down = keep & new_entry_mask(T["lag_down_tail"])
    new_up = keep & new_entry_mask(T["lag_up_tail"])
    return {**evaluate(y, pred, se, mask), "r2_vs_stage1": r2_vs(y, pred, base, mask),
            "auc_down": tail_auc(y, pred, year, "negative", mask=keep),
            "auc_up": tail_auc(y, pred, year, "positive", mask=keep),
            "auc_down_new_entry": tail_auc(y, pred, year, "negative", mask=new_down),
            "auc_up_new_entry": tail_auc(y, pred, year, "positive", mask=new_up),
            "n_new_down": int(new_down.sum())}


def auc_gain_ci(y, a, b, T, tail, new_entry, reps):
    """Firm-cluster bootstrap CI of AUC(b) - AUC(a) for the year's tail, optionally among new entrants."""
    year = T[YEAR].to_numpy()
    event = yearly_tail(y, year, tail=tail)
    keep = new_entry_mask(T[f"lag_{'down' if tail == 'negative' else 'up'}_tail"]) if new_entry else np.ones(len(y), bool)
    sign = -1 if tail == "negative" else 1
    idx_all = np.where(keep)[0]

    def stat(idx):
        return roc_auc_score(event[idx], sign * b[idx]) - roc_auc_score(event[idx], sign * a[idx])

    return cluster_bootstrap(lambda i: stat(idx_all[i]), T["股票代码"].to_numpy()[idx_all], reps)


def stage2(name, fold_l, fold_t, cat_dims, params, seeds):
    if name == "stage1_only":
        return np.zeros(len(fold_l.test)), {}
    if name == "lasso":
        return learners.fit_lasso(fold_l)
    if name in ("lgbm", "bo_lgbm"):
        return learners.fit_lgbm(fold_l, seeds, params.get(name))
    return learners.fit_tabnet(fold_t, params[name], cat_dims, seeds)


def importance_rows(info, model):
    rows = []
    for kind in ("gain", "mask", "coef"):
        if kind in info:
            v = info[kind].abs()
            v = v / v.sum()
            for col, val in v.items():
                rows.append({"model": model, "measure": kind, "feature": col, "share": val})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--params", type=Path, help="bo_best_params.json from an earlier run; skips those searches")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.seeds, a.reps = 2, 1, 50
        learners.MAX_EPOCHS, learners.PATIENCE = 20, 5
    seeds = tuple(range(a.seeds))
    OUT.mkdir(parents=True, exist_ok=True)

    X = load_data()
    has_text = "市场感知叙事强度" in X
    m = split_masks(X)
    T = X[m["test"]].reset_index(drop=True)
    folds, cat_dims, base, y = build(X, m, has_text)
    fl, ft = folds["lasso"], folds["tabnet"]
    print(f"text module: {'on' if has_text else 'off (scores not in 02_data/final yet)'}", flush=True)

    # Both BO searches use the same budget and sampler; LightGBM's fixed settings and TabNet's defaults
    # are each search's first trial.
    best = json.loads(a.params.read_text()) if a.params else {}
    searches = {"bo_tabnet": lambda: learners.bayes_opt_tabnet(ft, cat_dims, a.trials),
                "bo_lgbm": lambda: learners.bayes_opt_lgbm(fl, a.trials)}
    for name, search in searches.items():
        if name not in best:
            best[name], trials = search()
            trials.to_csv(OUT / f"{name}_trials.csv", index=False, encoding="utf-8-sig")
    (OUT / "bo_best_params.json").write_text(json.dumps(best, indent=2))
    params = {"tabnet": dict(learners.TABNET_DEFAULT), **best}
    print("BO best:", best, flush=True)

    preds, imp = {}, []
    for name in MODELS:
        s2, info = stage2(name, fl, ft, cat_dims, params, seeds)
        preds[name] = base + s2
        imp += importance_rows(info, name)
        print("done", name, flush=True)

    sub = subsets(T)
    metrics = pd.DataFrame([{"model": n, "subset": s, **score(y, p, base, T, k)}
                            for n, p in preds.items() for s, k in sub.items()])
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")

    tests = []
    for n in MODELS[1:]:
        for s in ("all", "year_2023", "year_2024"):
            t, p = clark_west(y, base, preds[n], sub[s])
            tests.append({"test": "clark_west_vs_stage1", "model": n, "against": "stage1_only", "subset": s, "t": t, "p": p})
        for tail, new in (("negative", False), ("negative", True), ("positive", False)):
            lo, hi = auc_gain_ci(y, base, preds[n], T, tail, new, a.reps)
            tests.append({"test": f"auc_gain_{tail}{'_new_entry' if new else ''}", "model": n, "against": "stage1_only",
                          "subset": "all", "ci_low": lo, "ci_high": hi})
    for other in ("lasso", "lgbm", "bo_lgbm", "tabnet"):
        t, p = diebold_mariano(y, preds[other], preds["bo_tabnet"])
        tests.append({"test": "diebold_mariano", "model": "bo_tabnet", "against": other, "subset": "all", "t": t, "p": p})
    pd.DataFrame(tests).to_csv(OUT / "tests.csv", index=False, encoding="utf-8-sig")

    # Leave-one-module-out on stage 2; the full model nests each reduced one, so Clark-West tests the module.
    # macro_industry is also split into its year-level variables and its industry categoricals.
    mods = module_columns(list(fl.test.columns))
    drops = {mod: (lambda n, mod=mod: module_of(n) == mod) for mod in mods}
    drops["macro_industry:macro"] = lambda n: n in YEAR_LEVEL
    drops["macro_industry:industry"] = lambda n: module_of(n) == "macro_industry" and n not in YEAR_LEVEL
    abl = []
    for mod, dropped in drops.items():
        cols_l = [c for c in fl.test.columns if not dropped(c)]
        cols_t = [c for c in ft.test.columns if not dropped(c)]
        for name in ("lasso", "lgbm", "bo_lgbm", "bo_tabnet"):
            s2, _ = stage2(name, fl.cols(cols_l), ft.cols(cols_t), cat_dims, params, seeds)
            p_red = base + s2
            t, pv = clark_west(y, p_red, preds[name])
            abl.append({"model": name, "dropped": mod, "n_cols_dropped": len(fl.test.columns) - len(cols_l),
                        **{k: v for k, v in score(y, p_red, base, T, None).items()
                           if k in ("r2", "r2_vs_stage1", "auc_down", "auc_up", "auc_down_new_entry", "auc_up_new_entry")},
                        "full_minus_reduced_r2_vs_stage1": r2_vs(y, preds[name], base) - r2_vs(y, p_red, base),
                        "cw_t_full_vs_reduced": t, "cw_p": pv})
        print("ablation done", mod, flush=True)
    pd.DataFrame(abl).to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    imp = pd.DataFrame(imp)
    imp["module"] = imp["feature"].map(module_of)
    imp.to_csv(OUT / "importance_features.csv", index=False, encoding="utf-8-sig")
    imp.groupby(["model", "measure", "module"], as_index=False)["share"].sum().to_csv(
        OUT / "importance_modules.csv", index=False, encoding="utf-8-sig")
    T[KEY + [ST, "mpdr"]].assign(pred_stage1=base, **{f"pred_{n}": p for n, p in preds.items()}).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(metrics[metrics.subset == "all"].round(4).to_string())


if __name__ == "__main__":
    main()
