"""Research question 3 (do downside and upside deviations have different drivers?) and firm-level diagnosis.

Uses the two-stage design and the tuned parameters in results_two_stage/bo_best_params.json; BO-TabNet and
BO-LightGBM are both reported. Attribution groups are the modules, with macro_industry split into year-level
variables (macro) and industry columns (industry). Writes to results_two_stage/:
  rq3_ablation.csv          leave-one-group-out cost on downside vs upside firms, with firm-cluster CIs
  rq3_quantile.csv          quantile regressions of MPDR at tau 0.1-0.9, with firm-cluster bootstrap tests
  rq3_signal_structure.csv  group Shapley attributions of downside vs upside firms, with CIs
  firm_diagnosis.csv        per test firm-year: predictions, warning band, stage-1 and stage-2 group attributions,
                            stage-2 feature drivers
  firm_signal_types.csv     k-means signal types among firms flagged downside or upside
  ablation_predictions.csv  test predictions of every reduced model

  python3 run_rq3_diagnosis.py            # full run
  python3 run_rq3_diagnosis.py --quick    # smoke run
"""
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tools.sm_exceptions import IterationLimitWarning

from mpdr import learners
from mpdr.attribution import group_shapley, top_drivers
from mpdr.config import INDUSTRY_RAW, KEY, ST, YEAR
from mpdr.features import FeaturePreprocessor
from mpdr.metrics import cluster_resamples, new_entry_mask, tail_auc, yearly_tail
from mpdr.modules import TextBlock, attribution_groups, design, module_of
from mpdr.split import split_masks
from mpdr.stages import PersistenceStage
from run_evaluation import deviation_group
from run_two_stage import OUT, build, load_data, stage2

MODELS = ["bo_tabnet", "bo_lgbm"]
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
BOOT_TAUS = (0.1, 0.5, 0.9)
BANDS = {"downside_warning": "negative", "upside_opportunity": "positive"}


def mse(y, p, k):
    return float(np.mean((y[k] - p[k]) ** 2))


def direction_cost(y, full, red, year, group, lag_d, lag_u):
    """Cost of dropping a group, on each side: relative MSE increase in the severe groups and AUC loss."""
    out = {}
    for side, g, tail, lag in (("down", "severe_negative", "negative", lag_d), ("up", "severe_positive", "positive", lag_u)):
        k = group == g
        out[f"rel_dmse_{side}"] = mse(y, red, k) / mse(y, full, k) - 1
        out[f"dauc_{side}"] = tail_auc(y, full, year, tail) - tail_auc(y, red, year, tail)
        new = new_entry_mask(lag)
        out[f"dauc_{side}_new_entry"] = tail_auc(y, full, year, tail, mask=new) - tail_auc(y, red, year, tail, mask=new)
    out["rel_dmse_down_minus_up"] = out["rel_dmse_down"] - out["rel_dmse_up"]
    out["dauc_down_minus_up"] = out["dauc_down"] - out["dauc_up"]
    return out


def ci_rows(point, draws, keys):
    rows = []
    for k, v in point.items():
        d = np.array([dr[k] for dr in draws])
        rows.append({**keys, "metric": k, "value": v, "ci_low": np.nanpercentile(d, 2.5),
                     "ci_high": np.nanpercentile(d, 97.5)})
    return rows


def band_of(pred, year):
    b = np.full(len(pred), "normal", dtype=object)
    for band, tail in BANDS.items():
        b[yearly_tail(pred, year, tail=tail)] = band
    return b


def signal_structure(phi, sel_down, sel_up):
    """Mean attribution per group (stage 1 included) and each stage-2 group's share of stage-2 |attribution|,
    for two sets of firms and their difference."""
    stage2 = phi.drop(columns="stage1").abs()
    share = stage2.div(stage2.sum(axis=1), axis=0)
    out = {}
    for g in phi.columns:
        out[f"{g}|mean_phi_down"] = phi.loc[sel_down, g].mean()
        out[f"{g}|mean_phi_up"] = phi.loc[sel_up, g].mean()
        if g == "stage1":
            continue
        out[f"{g}|abs_share_down"] = share.loc[sel_down, g].mean()
        out[f"{g}|abs_share_up"] = share.loc[sel_up, g].mean()
        out[f"{g}|abs_share_down_minus_up"] = out[f"{g}|abs_share_down"] - out[f"{g}|abs_share_up"]
    return out


def signal_types(phi, sel, seed=0):
    """k-means on signed shares of stage-2 |attribution| (k = 2..5 by silhouette); returns labels, k, silhouette."""
    Z = phi[sel].div(phi[sel].abs().sum(axis=1), axis=0).to_numpy()
    fits = {k: KMeans(k, n_init=20, random_state=seed).fit(Z) for k in range(2, 6)}
    k = max(fits, key=lambda k: silhouette_score(Z, fits[k].labels_))
    return fits[k].labels_, k, silhouette_score(Z, fits[k].labels_)


def independent_columns(D):
    """Columns kept in order while each one raises the rank (with a constant); drops exact collinearity."""
    keep, Z = [], np.ones((len(D), 1))
    for c in D.columns:
        cand = np.column_stack([Z, D[c].to_numpy(float)])
        if np.linalg.matrix_rank(cand) > Z.shape[1]:
            keep.append(c)
            Z = cand
    return keep


def quantile_regressions(X, m, has_text, reps, seed=0):
    """MPDR on the standardized design over 2013-2024 at each tau; industry dummies are controls.

    Preprocessing is fitted on 2013-2022. Bootstrap SEs resample whole firms; the 0.1 vs 0.9 difference
    tests whether a variable moves the lower and upper tails of MPDR by different amounts.
    """
    keep = np.asarray(m["refit"] | m["test"])
    pre = FeaturePreprocessor("lasso").fit(X[m["refit"]])
    txt = TextBlock("lasso").fit(X[m["refit"]]) if has_text else None
    D = design(pre, txt, X[keep]).reset_index(drop=True)
    D = D[[c for c in D.columns if not c.startswith("消费类型=")]]
    D = D[independent_columns(D.loc[:, D.std() > 0])]
    Z = sm.add_constant(D)
    y = X.loc[keep, "mpdr"].to_numpy()
    firms = X.loc[keep, "股票代码"].to_numpy()

    def fit(idx, tau):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IterationLimitWarning)
            return sm.QuantReg(y[idx], Z.iloc[idx]).fit(q=tau, max_iter=5000).params

    all_idx = np.arange(len(y))
    point = {t: fit(all_idx, t) for t in TAUS}
    boots = {t: [] for t in BOOT_TAUS}
    for i, idx in enumerate(cluster_resamples(firms, reps, seed)):
        for t in BOOT_TAUS:
            boots[t].append(fit(idx, t))
        if (i + 1) % 25 == 0:
            print(f"  quantile bootstrap {i + 1}/{reps}", flush=True)
    B = {t: pd.DataFrame(boots[t]) for t in BOOT_TAUS}
    rows = []
    for c in Z.columns:
        if c == "const" or c.startswith("industry="):
            continue
        r = {"feature": c, "group": attribution_groups([c]).popitem()[0], **{f"b_{t}": point[t][c] for t in TAUS}}
        for t in BOOT_TAUS:
            se = B[t][c].std()
            r[f"se_{t}"] = se
            r[f"p_{t}"] = float(2 * min((B[t][c] <= 0).mean(), (B[t][c] >= 0).mean()))
        diff = B[0.1][c] - B[0.9][c]
        r["diff_0.1_0.9"] = point[0.1][c] - point[0.9][c]
        r["diff_ci_low"], r["diff_ci_high"] = np.percentile(diff, [2.5, 97.5])
        r["diff_p"] = float(2 * min((diff <= 0).mean(), (diff >= 0).mean()))
        rows.append(r)
    return pd.DataFrame(rows), len(y), Z.shape[1] - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--qr-reps", type=int, default=200)
    ap.add_argument("--background", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.reps, a.qr_reps, a.background = 1, 20, 4, 8
        learners.MAX_EPOCHS, learners.PATIENCE = 20, 5
    seeds = tuple(range(a.seeds))

    X = load_data()
    has_text = "市场感知叙事强度" in X
    m = split_masks(X)
    T = X[m["test"]].reset_index(drop=True)
    folds, cat_dims, base, y = build(X, m, has_text)
    fl, ft = folds["lasso"], folds["tabnet"]
    params = json.loads((OUT / "bo_best_params.json").read_text())
    year, firms = T[YEAR].to_numpy(), T["股票代码"].to_numpy()
    lag_d, lag_u = T["lag_down_tail"].to_numpy(), T["lag_up_tail"].to_numpy()
    group = deviation_group(y, year)

    s1 = PersistenceStage().fit(fl.refit, X.loc[m["refit"], "mpdr"].to_numpy())
    assert np.allclose(s1.predict(fl.test), base)

    # Full models, kept for attributions; they reproduce run_two_stage's predictions.
    fits = {"bo_lgbm": learners.fit_lgbm(fl, seeds, params["bo_lgbm"]),
            "bo_tabnet": learners.fit_tabnet(ft, params["bo_tabnet"], cat_dims, seeds)}
    full = {n: base + fits[n][0] for n in MODELS}
    saved = pd.read_csv(OUT / "test_predictions.csv", dtype={"股票代码": str}, encoding="utf-8-sig")
    for n in MODELS:
        print(f"{n}: max |pred - saved pred| = {np.abs(full[n] - saved[f'pred_{n}'].to_numpy()).max():.2e}", flush=True)

    # 1. Leave-one-group-out, scored separately on the downside and the upside.
    groups = {"lasso": attribution_groups(list(fl.test.columns)), "tabnet": attribution_groups(list(ft.test.columns))}
    reduced, abl = {}, []
    resamples = list(cluster_resamples(firms, a.reps))
    for g in groups["lasso"]:
        cols_l = [c for c in fl.test.columns if c not in groups["lasso"][g]]
        cols_t = [c for c in ft.test.columns if c not in groups["tabnet"][g]]
        for n in MODELS:
            red = reduced[n, g] = base + stage2(n, fl.cols(cols_l), ft.cols(cols_t), cat_dims, params, seeds)[0]
            point = direction_cost(y, full[n], red, year, group, lag_d, lag_u)
            draws = [direction_cost(y[i], full[n][i], red[i], year[i], group[i], lag_d[i], lag_u[i]) for i in resamples]
            abl += ci_rows(point, draws, {"model": n, "dropped": g})
        print("ablation done", g, flush=True)
    pd.DataFrame(abl).to_csv(OUT / "rq3_ablation.csv", index=False, encoding="utf-8-sig")
    T[KEY].assign(**{f"pred_{n}_without_{g}": p for (n, g), p in reduced.items()}).to_csv(
        OUT / "ablation_predictions.csv", index=False, encoding="utf-8-sig")

    # 2. Attributions. Stage 1 depends only on persistence columns, so its part is its own deviation from the
    # background mean; stage 2 gets exact group Shapley values (same background rows for both models).
    bg = np.random.default_rng(0).choice(len(fl.refit), a.background, replace=False)
    lgbm_models, tab_models = fits["bo_lgbm"][1]["models"], fits["bo_tabnet"][1]["models"]
    s1_bg = float(s1.predict(fl.refit.iloc[bg]).mean())
    phi_s1 = s1.predict(fl.test) - s1_bg

    def predict_lgbm(D):
        return np.mean([r.predict(D) for r in lgbm_models], axis=0)

    def predict_tabnet(D):
        return np.mean([t.predict(D.to_numpy(np.float32)).ravel() for t in tab_models], axis=0)

    phi, phi_base = {}, {}
    phi["bo_lgbm"], phi_base["bo_lgbm"] = group_shapley(predict_lgbm, fl.test, fl.refit.iloc[bg], groups["lasso"])
    print("shapley done bo_lgbm", flush=True)
    phi["bo_tabnet"], phi_base["bo_tabnet"] = group_shapley(predict_tabnet, ft.test, ft.refit.iloc[bg], groups["tabnet"])
    print("shapley done bo_tabnet", flush=True)
    for n in MODELS:
        phi[n].insert(0, "stage1", phi_s1)
        gap = np.abs(phi[n].sum(axis=1) + s1_bg + phi_base[n] - full[n]).max()
        print(f"{n}: max |sum(phi) + base - pred| = {gap:.2e}", flush=True)

    # 3. Signal structure: firms flagged downside vs upside (by prediction), and actual severe groups.
    struct = []
    for n in MODELS:
        band = band_of(full[n], year)
        for basis, (down, up) in {"predicted_band": (band == "downside_warning", band == "upside_opportunity"),
                                  "actual_group": (group == "severe_negative", group == "severe_positive")}.items():
            point = signal_structure(phi[n], down, up)
            draws = [signal_structure(phi[n].iloc[i].reset_index(drop=True), down[i], up[i]) for i in resamples]
            for r in ci_rows(point, draws, {"model": n, "basis": basis}):
                r["group"], r["metric"] = r["metric"].split("|")
                struct.append(r)
    pd.DataFrame(struct).to_csv(OUT / "rq3_signal_structure.csv", index=False, encoding="utf-8-sig")

    # 4. Firm diagnosis: group attributions for both models, stage-2 feature drivers for BO-LightGBM
    # (TreeSHAP) and BO-TabNet's per-firm attention masks.
    raw_pre = FeaturePreprocessor("tabnet").fit(X[m["refit"]])
    raw = raw_pre.transform(T).reset_index(drop=True)
    values = fl.test.copy()
    shared = [c for c in values.columns if c in raw.columns]
    values[shared] = raw[shared].to_numpy()
    tree = np.mean([r.predict(fl.test, pred_contrib=True) for r in lgbm_models], axis=0)
    contrib = pd.DataFrame(tree[:, :-1], columns=fl.test.columns)
    mask = np.mean([t.explain(ft.test.to_numpy(np.float32))[0] for t in tab_models], axis=0)
    mask = pd.DataFrame(mask / mask.sum(axis=1, keepdims=True), columns=ft.test.columns)

    diag = T[KEY + [INDUSTRY_RAW, ST]].copy()
    diag.insert(2, "forecast_year", diag[YEAR] + 1)
    diag["mpdr_actual"], diag["actual_group"] = y, group
    diag["pred_stage1"], diag["phi_stage1"] = base, phi_s1
    types = []
    for n in MODELS:
        diag[f"pred_{n}"] = full[n]
        diag[f"pct_{n}"] = pd.Series(full[n]).groupby(year).rank(pct=True).to_numpy()
        diag[f"band_{n}"] = band = band_of(full[n], year)
        diag[f"phi_base_{n}"] = s1_bg + phi_base[n]
        P2 = phi[n].drop(columns="stage1")
        for g in P2.columns:
            diag[f"phi_{g}_{n}"] = P2[g].to_numpy()
        diag[f"signal_type_{n}"] = ""
        for b, tail in BANDS.items():
            sel = band == b
            labels, k, sil = signal_types(P2, sel)
            diag.loc[sel, f"signal_type_{n}"] = [f"{b}_{j + 1}" for j in labels]
            hit = yearly_tail(y, year, tail=tail)
            for j in range(k):
                rows = np.flatnonzero(sel)[labels == j]
                P = P2.iloc[rows]
                share = P.div(P.abs().sum(axis=1), axis=0).mean()
                lead = P.mean().idxmin() if tail == "negative" else P.mean().idxmax()
                types.append({"model": n, "band": b, "signal_type": f"{b}_{j + 1}", "k": k, "silhouette": sil,
                              "n": len(rows), "leading_stage2_group": lead, "mean_pred": full[n][rows].mean(),
                              "mean_actual": y[rows].mean(), "hit_rate_actual_tail": hit[rows].mean(),
                              "mean_phi_stage1": phi_s1[rows].mean(),
                              **{f"mean_phi_{g}": P[g].mean() for g in P.columns},
                              **{f"signed_share_{g}": share[g] for g in P.columns}})
    diag["lgbm_stage2_down_drivers"] = top_drivers(contrib, values, sign=-1)
    diag["lgbm_stage2_up_drivers"] = top_drivers(contrib, values, sign=1)
    top_mask = np.argsort(-mask.to_numpy(), axis=1)[:, :3]
    diag["tabnet_top_attention"] = ["; ".join(f"{mask.columns[j]}({mask.iat[i, j]:.2f})" for j in row)
                                    for i, row in enumerate(top_mask)]
    diag.to_csv(OUT / "firm_diagnosis.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(types).to_csv(OUT / "firm_signal_types.csv", index=False, encoding="utf-8-sig")
    print("diagnosis done", flush=True)

    # 5. Quantile regressions (in-sample structure over 2013-2024).
    qr, n_obs, n_cols = quantile_regressions(X, m, has_text, a.qr_reps)
    qr.insert(0, "n_obs", n_obs)
    qr.to_csv(OUT / "rq3_quantile.csv", index=False, encoding="utf-8-sig")
    print(f"quantile regressions done: {n_obs} obs, {n_cols} regressors", flush=True)


if __name__ == "__main__":
    main()
