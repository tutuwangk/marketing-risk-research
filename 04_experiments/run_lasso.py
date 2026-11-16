"""Lasso baseline: tune alpha on valid, refit on train+valid, evaluate once on test.

Writes results/lasso_metrics.csv, results/lasso_coefficients.csv and results/lasso_test_predictions.csv.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression

from mpdr.config import KEY, LABEL_PATH, RESTATEMENT_PATH, RESULTS_DIR, SE, ST
from mpdr.features import FeaturePreprocessor, load_x
from mpdr.metrics import evaluate
from mpdr.split import split_masks

ALPHAS = np.logspace(-4, -0.5, 36)
# Persistence baseline: last year's performance plus the benchmark's own regressors.
PERSISTENCE = ["mp_t", "gp_nonpos", "ln1p_se", "se_missing_or_zero", "企业规模", "成立年限", "资产负债率",
               "经营现金流资产比", "资本支出强度", "capex_missing"]


def load():
    """X merged with the MPDR label, the lagged label, and the restatement look-ahead flag."""
    X = load_x()
    lab = pd.read_csv(LABEL_PATH, dtype={"股票代码": str}, encoding="utf-8-sig")
    X = X.merge(lab[KEY + ["mpdr"]], on=KEY, how="left", validate="one_to_one")
    lag = lab[KEY + ["mpdr"]].assign(报告年度=lab["报告年度"] + 1).rename(columns={"mpdr": "mpdr_lag"})
    X = X.merge(lag, on=KEY, how="left", validate="one_to_one")
    rs = pd.read_csv(RESTATEMENT_PATH, dtype={"股票代码": str}, encoding="utf-8-sig")
    X = X.merge(rs[KEY + ["疑似使用更正后数值"]], on=KEY, how="left", validate="one_to_one")
    X["restated"] = X["疑似使用更正后数值"].fillna(False).astype(bool)
    assert X["mpdr"].notna().all()
    return X


def fit_lasso(X, fit_mask, alpha):
    pre = FeaturePreprocessor("lasso").fit(X[fit_mask])
    model = Lasso(alpha=alpha, max_iter=50_000).fit(pre.transform(X[fit_mask]), X.loc[fit_mask, "mpdr"])
    return pre, model


def tune(X, m):
    """Validation RMSE for each alpha, with the model fitted on train only."""
    pre = FeaturePreprocessor("lasso").fit(X[m["train"]])
    Xt, Xv = pre.transform(X[m["train"]]), pre.transform(X[m["valid"]])
    yt, yv = X.loc[m["train"], "mpdr"], X.loc[m["valid"], "mpdr"].to_numpy()
    rows = []
    for a in ALPHAS:
        pred = Lasso(alpha=a, max_iter=50_000).fit(Xt, yt).predict(Xv)
        rows.append({"alpha": a, "valid_rmse": float(np.sqrt(np.mean((yv - pred) ** 2)))})
    return pd.DataFrame(rows)


def run(X, sample):
    """Tune, refit and score one sample definition; returns (metrics rows, coefficients, predictions)."""
    m = split_masks(X)
    grid = tune(X, m)
    alpha = grid.loc[grid["valid_rmse"].idxmin(), "alpha"]
    pre, model = fit_lasso(X, m["refit"], alpha)
    T = X[m["test"]].reset_index(drop=True)
    y, se = T["mpdr"].to_numpy(), T[SE].to_numpy()
    R = X[m["refit"]]
    persist = LinearRegression().fit(pre.transform(R)[PERSISTENCE], R["mpdr"])
    preds = {"lasso": model.predict(pre.transform(T)),
             "persistence": persist.predict(pre.transform(T)[PERSISTENCE]),
             "zero": np.zeros(len(T)),
             "lagged_mpdr": T["mpdr_lag"].fillna(0).to_numpy()}
    subsets = {"all": None, "non_st": ~T[ST].astype(bool).to_numpy(),
               "no_restated": ~T["restated"].to_numpy()}
    rows = [{"sample": sample, "model": name, "subset": sub, "alpha": alpha if name == "lasso" else np.nan,
             **evaluate(y, p, se, mask)}
            for name, p in preds.items() for sub, mask in subsets.items()]
    coef = pd.DataFrame({"sample": sample, "feature": pre.feature_names, "coef": model.coef_})
    out = T[KEY + [ST, "mpdr"]].assign(sample=sample, pred_lasso=preds["lasso"])
    return rows, coef, out


def main():
    X = load()
    results = [run(X, "main"), run(X[~X[ST].astype(bool)].reset_index(drop=True), "exclude_st")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([r for res in results for r in res[0]])
    metrics.to_csv(RESULTS_DIR / "lasso_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([r[1] for r in results]).to_csv(RESULTS_DIR / "lasso_coefficients.csv", index=False,
                                              encoding="utf-8-sig")
    pd.concat([r[2] for r in results]).to_csv(RESULTS_DIR / "lasso_test_predictions.csv", index=False,
                                              encoding="utf-8-sig")
    print(metrics.round(4).to_string())


if __name__ == "__main__":
    main()
