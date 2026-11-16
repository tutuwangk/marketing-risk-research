"""Predictor modules for the ablation, and the annual-report text block."""
import pandas as pd

from .config import KEY, MODULES, TEXT_COLUMNS, TEXT_LATE, TEXT_PATH, YEAR_LEVEL


def load_text(path=TEXT_PATH):
    """Seven text scores (and the late-disclosure flag when the file has it) keyed by 股票代码 + 报告年度,
    or None while the file does not exist yet."""
    if not path.exists():
        return None
    T = pd.read_csv(path, dtype={"股票代码": str}, encoding="utf-8-sig")
    return T[KEY + TEXT_COLUMNS + ([TEXT_LATE] if TEXT_LATE in T else [])]


def attach_text(X, text):
    """X with the text columns merged in (NaN where a report has no score)."""
    if text is None:
        return X
    return X.merge(text, on=KEY, how="left", validate="one_to_one")


class TextBlock:
    """Text scores with train-median fill, a text_missing flag and the late-disclosure flag text_late; standardized
    in lasso mode."""

    def __init__(self, mode="lasso"):
        self.mode = mode

    def fit(self, X):
        T = X[TEXT_COLUMNS].astype(float)
        self.medians_ = T.median()
        filled = self._block(X)
        self.mean_ = filled.mean()
        self.scale_ = filled.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X):
        B = self._block(X)
        return B if self.mode == "tabnet" else (B - self.mean_) / self.scale_

    def _block(self, X):
        T = X[TEXT_COLUMNS].astype(float)
        out = T.fillna(self.medians_)
        out["text_missing"] = T.isna().any(axis=1).astype(float)
        out["text_late"] = X[TEXT_LATE].fillna(0).astype(float) if TEXT_LATE in X else 0.0
        return out


def module_of(name):
    """Module of one design column; missing flags follow the column they flag, dummies go to macro_industry."""
    if name.startswith(("industry=", "消费类型=")) or name in ("industry", "消费类型"):
        return "macro_industry"
    base = name[: -len("_missing")] if name.endswith("_missing") else name
    for candidate in (name, base):
        for module, cols in MODULES.items():
            if candidate in cols:
                return module
    raise KeyError(f"no module for column {name}")


def module_columns(names):
    """{module: [design columns]} over the given column names, in MODULES order."""
    out = {m: [] for m in MODULES}
    for n in names:
        out[module_of(n)].append(n)
    return {m: c for m, c in out.items() if c}


def design(pre, text_block, X):
    """Preprocessed features plus the text block when one is fitted."""
    D = pre.transform(X)
    if text_block is not None:
        D = pd.concat([D, text_block.transform(X)], axis=1)
    return D


def attribution_groups(names):
    """{group: [columns]} for attributions and the ablation: the modules, with macro_industry split into
    year-level variables ("macro") and industry / consumption-type columns ("industry")."""
    out = {}
    for n in names:
        g = module_of(n)
        if g == "macro_industry":
            g = "macro" if n in YEAR_LEVEL else "industry"
        out.setdefault(g, []).append(n)
    order = [m for m in MODULES if m != "macro_industry"] + ["macro", "industry"]
    return {g: out[g] for g in sorted(out, key=order.index)}
