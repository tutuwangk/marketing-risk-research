"""Year-t feature construction with every statistic fitted on the training fold only."""
import numpy as np
import pandas as pd

from .config import (
    ASSETS, CAPEX, CATEGORICAL, CONSUMPTION, CONTINUOUS, FLAGS, GP, GROUP_SUFFIX, INDUSTRY_RAW,
    INV_TURNOVER, MIN_INDUSTRY_OBS, MISSING_FLAG_MERGE_AGREEMENT, MISSING_FLAG_MIN_RATE, OTHER,
    PRE2021_INDUSTRIES, SE, ST, WINSOR_Q, X_PATH, YEAR_LEVEL,
)

MODES = ("lasso", "tabnet")
WINSORIZED = [c for c in CONTINUOUS if c not in YEAR_LEVEL]


def load_x(path=X_PATH):
    """Read model_features_X.csv with 股票代码 kept as 6-digit text."""
    return pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig", low_memory=False)


def _industry_key_map():
    """Industry label -> 2013-2020 label; an exact match wins over membership in a combined label."""
    mapping = {}
    for combined in PRE2021_INDUSTRIES:
        for part in combined.split("|"):
            mapping.setdefault(part, combined)
    mapping.update({lbl: lbl for lbl in PRE2021_INDUSTRIES})
    return mapping


INDUSTRY_KEY = _industry_key_map()


def consumption_group(consumption):
    """Group label for industries pooled into their consumption type."""
    return consumption + GROUP_SUFFIX


def harmonize_industry(df):
    """对应目标行业 mapped onto one key across SW versions; unmatched labels go to their 消费类型 group."""
    return df[INDUSTRY_RAW].map(INDUSTRY_KEY).fillna(consumption_group(df[CONSUMPTION]))


class FeaturePreprocessor:
    """fit(X_train) learns fill values, winsor cutoffs, missing flags, category levels and
    scalers; transform(X_any) applies them row by row.

    mode="lasso": standardized numeric block plus one-hot categoricals (train levels only, so
    an unseen "other" category is an all-zero row). mode="tabnet": unscaled numeric block plus
    integer-coded categoricals, with code len(levels) reserved for "other"; see cat_idxs/cat_dims.
    """

    def __init__(self, mode="lasso"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def fit(self, X):
        gp = X[GP]
        self.gp_log_min_ = float(np.log(gp[gp > 0]).min())

        C = self._continuous(X)
        self.missing_rate_ = C.isna().mean()
        self.missing_groups_ = _group_missing_flags(
            C[[c for c in CONTINUOUS if self.missing_rate_[c] > MISSING_FLAG_MIN_RATE]].isna()
        )
        self.lower_ = C[WINSORIZED].quantile(WINSOR_Q[0])
        self.upper_ = C[WINSORIZED].quantile(WINSOR_Q[1])
        self.medians_ = self._clip(C).median()

        ind = harmonize_industry(X)
        counts = ind.value_counts()
        self.kept_industries_ = sorted(counts.index[counts >= MIN_INDUSTRY_OBS])
        cons_levels = sorted(X[CONSUMPTION].unique())
        # every consumption-type group is a level, so unmatched or rare industries never fall to "other"
        self.levels_ = {
            CONSUMPTION: cons_levels,
            "industry": sorted(set(self.kept_industries_) | {consumption_group(c) for c in cons_levels}),
        }

        num = self._numeric(X)
        self.mean_ = num.mean()
        self.scale_ = num.std(ddof=0).replace(0, 1.0)  # constant columns (e.g. st in a no-ST sample)
        return self

    def transform(self, X):
        num = self._numeric(X)
        codes = self._codes(X)
        if self.mode == "tabnet":
            return pd.concat([num, codes], axis=1)
        num = (num - self.mean_) / self.scale_
        onehot = [
            pd.DataFrame(
                (codes[c].to_numpy()[:, None] == np.arange(len(self.levels_[c]))).astype(float),
                index=X.index,
                columns=[f"{c}={lvl}" for lvl in self.levels_[c]],
            )
            for c in CATEGORICAL
        ]
        return pd.concat([num, *onehot], axis=1)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def numeric_names(self):
        return CONTINUOUS + FLAGS + [f"{c}_missing" for c in self.missing_groups_]

    @property
    def feature_names(self):
        if self.mode == "tabnet":
            return self.numeric_names + CATEGORICAL
        return self.numeric_names + [f"{c}={lvl}" for c in CATEGORICAL for lvl in self.levels_[c]]

    @property
    def cat_idxs(self):
        """Positions of the categorical columns in tabnet output."""
        n = len(self.numeric_names)
        return list(range(n, n + len(CATEGORICAL)))

    @property
    def cat_dims(self):
        """Cardinality of each categorical, including the "other" code."""
        return [len(self.levels_[c]) + 1 for c in CATEGORICAL]

    def category_label(self, column, code):
        """Level name for a code, with OTHER for the reserved code."""
        levels = self.levels_[column]
        return levels[code] if code < len(levels) else OTHER

    def _continuous(self, X):
        """Continuous features before winsorizing; mp_t is filled once gp_log_min_ is known."""
        gp = X[GP]
        C = pd.DataFrame(index=X.index)
        C["mp_t"] = np.log(gp.where(gp > 0)).fillna(self.gp_log_min_)
        C["gp_to_assets"] = gp / X[ASSETS]
        C["ln1p_se"] = np.log1p(X[SE].fillna(0))
        for c in CONTINUOUS[3:]:
            C[c] = X[c]
        C[INV_TURNOVER] = np.log1p(X[INV_TURNOVER].clip(lower=0))
        return C

    def _clip(self, C):
        C = C.copy()
        C[WINSORIZED] = C[WINSORIZED].clip(self.lower_, self.upper_, axis=1)
        return C

    def _flags(self, X):
        se = X[SE]
        return pd.DataFrame({
            "gp_nonpos": X[GP] <= 0,
            "se_missing_or_zero": se.isna() | (se <= 0),
            "st": X[ST].astype(bool),
            "capex_missing": X[CAPEX].isna(),
        }, index=X.index).astype(float)

    def _numeric(self, X):
        C = self._clip(self._continuous(X))
        miss = pd.DataFrame(
            {f"{name}_missing": C[cols].isna().any(axis=1) for name, cols in self.missing_groups_.items()},
            index=X.index,
        ).astype(float)
        return pd.concat([C.fillna(self.medians_), self._flags(X), miss], axis=1)

    def _industry_labels(self, X):
        ind = harmonize_industry(X)
        return ind.where(ind.isin(self.kept_industries_), consumption_group(X[CONSUMPTION]))

    def _codes(self, X):
        """Integer codes per categorical; anything outside the train levels gets the "other" code."""
        raw = {CONSUMPTION: X[CONSUMPTION], "industry": self._industry_labels(X)}
        out = {}
        for c in CATEGORICAL:
            codes = pd.Index(self.levels_[c]).get_indexer(raw[c])
            out[c] = np.where(codes < 0, len(self.levels_[c]), codes)
        return pd.DataFrame(out, index=X.index)


def _group_missing_flags(M):
    """Group columns whose train missing patterns agree above the threshold with a group's first
    column; returns {first column: members}. A group's flag is 1 when any member is missing."""
    groups = {}
    for c in M.columns:
        lead = next((g for g in groups if (M[g] == M[c]).mean() > MISSING_FLAG_MERGE_AGREEMENT), None)
        groups.setdefault(lead or c, []).append(c)
    return groups
