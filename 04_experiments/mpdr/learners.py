"""Stage-2 learners. Each tunes on (train, valid), refits on train + valid and predicts test.

Every learner takes a Fold of design matrices and residual targets and returns test predictions.
"""
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.linear_model import Lasso

from .config import CATEGORICAL

ALPHAS = np.logspace(-4, -0.5, 36)
SEEDS = (0, 1, 2)
LGBM = dict(n_estimators=3000, learning_rate=0.02, num_leaves=15, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
# pytorch-tabnet defaults, written out so the untuned model is explicit.
TABNET_DEFAULT = dict(n_d=8, n_a=8, n_steps=3, gamma=1.3, lambda_sparse=1e-3, lr=2e-2, mask_type="sparsemax",
                      cat_emb_dim=1, batch_size=1024)
MAX_EPOCHS, PATIENCE, VIRTUAL_BATCH = 300, 30, 128


@dataclass
class Fold:
    """Design matrices (one column set) and stage-1 residuals for train, valid, refit and test."""
    train: pd.DataFrame
    valid: pd.DataFrame
    refit: pd.DataFrame
    test: pd.DataFrame
    r_train: np.ndarray
    r_valid: np.ndarray
    r_refit: np.ndarray

    def cols(self, columns):
        return Fold(self.train[columns], self.valid[columns], self.refit[columns], self.test[columns],
                    self.r_train, self.r_valid, self.r_refit)


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def fit_lasso(f):
    err = [mse(f.r_valid, Lasso(alpha=a, max_iter=50_000).fit(f.train, f.r_train).predict(f.valid)) for a in ALPHAS]
    alpha = float(ALPHAS[int(np.argmin(err))])
    model = Lasso(alpha=alpha, max_iter=50_000).fit(f.refit, f.r_refit)
    return model.predict(f.test), {"alpha": alpha, "coef": pd.Series(model.coef_, index=f.refit.columns)}


def fit_lgbm(f, seeds=SEEDS, params=None):
    params = LGBM if params is None else params
    preds, gains, models = [], [], []
    for s in seeds:
        m = lgb.LGBMRegressor(**params, random_state=s).fit(
            f.train, f.r_train, eval_set=[(f.valid, f.r_valid)], callbacks=[lgb.early_stopping(100, verbose=False)])
        n = max(m.best_iteration_, 20)
        r = lgb.LGBMRegressor(**dict(params, n_estimators=n), random_state=s).fit(f.refit, f.r_refit)
        preds.append(r.predict(f.test))
        gains.append(pd.Series(r.booster_.feature_importance("gain"), index=f.refit.columns))
        models.append(r)
    return np.mean(preds, axis=0), {"gain": pd.concat(gains, axis=1).mean(axis=1), "models": models}


def _cat_args(columns, cat_dims):
    idx = [i for i, c in enumerate(columns) if c in CATEGORICAL]
    return {"cat_idxs": idx, "cat_dims": [cat_dims[columns[i]] for i in idx]}


def _tabnet(p, columns, cat_dims, seed):
    torch.manual_seed(seed)
    return TabNetRegressor(
        n_d=p["n_d"], n_a=p["n_a"], n_steps=p["n_steps"], gamma=p["gamma"], lambda_sparse=p["lambda_sparse"],
        mask_type=p["mask_type"], cat_emb_dim=p["cat_emb_dim"], optimizer_params={"lr": p["lr"]},
        seed=seed, verbose=0, device_name="cpu", **_cat_args(list(columns), cat_dims))


def tabnet_valid(f, p, cat_dims, seed=0):
    """Fit on train with early stopping on valid; returns (valid MSE, best epoch count)."""
    m = _tabnet(p, f.train.columns, cat_dims, seed)
    m.fit(f.train.to_numpy(np.float32), f.r_train.reshape(-1, 1),
          eval_set=[(f.valid.to_numpy(np.float32), f.r_valid.reshape(-1, 1))], eval_metric=["mse"],
          max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=p["batch_size"], virtual_batch_size=VIRTUAL_BATCH)
    pred = m.predict(f.valid.to_numpy(np.float32)).ravel()
    return mse(f.r_valid, pred), int(m.best_epoch) + 1


def fit_tabnet(f, p, cat_dims, seeds=SEEDS):
    """Per seed: epoch count from the train/valid run, then refit on train + valid for that many epochs.

    Returns the seed-averaged test prediction, the mean test attention mask per column and the fitted models.
    """
    preds, masks, epochs, models = [], [], [], []
    for s in seeds:
        _, n_epochs = tabnet_valid(f, p, cat_dims, s)
        m = _tabnet(p, f.refit.columns, cat_dims, s)
        m.fit(f.refit.to_numpy(np.float32), f.r_refit.reshape(-1, 1), max_epochs=n_epochs, patience=0,
              batch_size=p["batch_size"], virtual_batch_size=VIRTUAL_BATCH)
        Xs = f.test.to_numpy(np.float32)
        preds.append(m.predict(Xs).ravel())
        explain, _ = m.explain(Xs)
        masks.append(pd.Series(explain.mean(axis=0), index=f.test.columns))
        epochs.append(n_epochs)
        models.append(m)
    return np.mean(preds, axis=0), {"mask": pd.concat(masks, axis=1).mean(axis=1), "epochs": epochs, "models": models}


def search_space(trial):
    n = trial.suggest_int("n_d", 8, 64, log=True)
    return dict(n_d=n, n_a=n, n_steps=trial.suggest_int("n_steps", 3, 8), gamma=trial.suggest_float("gamma", 1.0, 2.0),
                lambda_sparse=trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
                lr=trial.suggest_float("lr", 5e-3, 5e-2, log=True),
                mask_type=trial.suggest_categorical("mask_type", ["sparsemax", "entmax"]),
                cat_emb_dim=trial.suggest_int("cat_emb_dim", 1, 4),
                batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]))


def bayes_opt_tabnet(f, cat_dims, n_trials, seed=0):
    """Gaussian-process Bayesian optimization of valid MSE; the default config is the first trial."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GPSampler(seed=seed))
    study.enqueue_trial({k: v for k, v in TABNET_DEFAULT.items() if k != "n_a"})

    def objective(trial):
        loss, n_epochs = tabnet_valid(f, search_space(trial), cat_dims, seed)
        trial.set_user_attr("epochs", n_epochs)
        return loss

    study.optimize(objective, n_trials=n_trials)
    best = search_space(optuna.trial.FixedTrial(study.best_params))
    return best, study.trials_dataframe()


def lgbm_space(trial):
    return dict(num_leaves=trial.suggest_int("num_leaves", 7, 63, log=True),
                learning_rate=trial.suggest_float("learning_rate", 5e-3, 1e-1, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 200, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True))


def bayes_opt_lgbm(f, n_trials, seed=0):
    """Same search protocol as BO-TabNet; the fixed LGBM settings are the first trial."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GPSampler(seed=seed))
    study.enqueue_trial({k: LGBM[k] for k in ("num_leaves", "learning_rate", "min_child_samples", "subsample",
                                              "colsample_bytree", "reg_lambda")})

    def objective(trial):
        p = dict(LGBM, **lgbm_space(trial))
        m = lgb.LGBMRegressor(**p, random_state=seed).fit(
            f.train, f.r_train, eval_set=[(f.valid, f.r_valid)], callbacks=[lgb.early_stopping(100, verbose=False)])
        return mse(f.r_valid, m.predict(f.valid))

    study.optimize(objective, n_trials=n_trials)
    return dict(LGBM, **lgbm_space(optuna.trial.FixedTrial(study.best_params))), study.trials_dataframe()
