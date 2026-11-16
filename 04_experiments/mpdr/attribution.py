"""Per-row attributions: exact Shapley values over feature groups, and signed top drivers."""
from math import factorial

import numpy as np
import pandas as pd


def group_shapley(predict, X, background, groups, chunk=500):
    """Exact Shapley values of feature groups for each row of X, interventional over the background rows.

    v(S) for a row is the mean prediction when the columns in S come from that row and all other columns
    from a background row. predict maps a DataFrame to a 1-d array. Returns (phi, base): phi is rows x
    groups and each row sums to predict(row) - base, where base is the mean prediction on the background.
    """
    names, cols = list(groups), list(X.columns)
    idx = {g: [cols.index(c) for c in groups[g]] for g in names}
    G, B = len(names), len(background)
    bg = background[cols].to_numpy(float)
    weight = {s: factorial(s) * factorial(G - s - 1) / factorial(G) for s in range(G)}
    out = []
    for start in range(0, len(X), chunk):
        x = X.iloc[start:start + chunk][cols].to_numpy(float)
        n = len(x)
        rows, tiled = np.repeat(x, B, axis=0), np.tile(bg, (n, 1))
        v = {}
        for S in range(2 ** G):
            M = tiled.copy()
            for j, g in enumerate(names):
                if S >> j & 1:
                    M[:, idx[g]] = rows[:, idx[g]]
            v[S] = np.asarray(predict(pd.DataFrame(M, columns=cols))).reshape(n, B).mean(axis=1)
        phi = np.zeros((n, G))
        for j in range(G):
            for S in range(2 ** G):
                if not S >> j & 1:
                    phi[:, j] += weight[bin(S).count("1")] * (v[S | 1 << j] - v[S])
        out.append(phi)
    return pd.DataFrame(np.vstack(out), columns=names, index=X.index), float(v[0].mean())


def top_drivers(contrib, values, k=3, sign=-1):
    """Per row, the k columns with the most negative (sign=-1) or positive (sign=1) contribution,
    formatted "column=value(contribution)"; contributions on the wrong side are left out."""
    C = contrib.to_numpy() * sign
    order = np.argsort(-C, axis=1)[:, :k]
    cols = np.asarray(contrib.columns)
    out = []
    for i, row in enumerate(order):
        parts = [f"{cols[j]}={values.iat[i, values.columns.get_loc(cols[j])]:.3g}({contrib.iat[i, j]:+.3f})"
                 for j in row if C[i, j] > 0]
        out.append("; ".join(parts))
    return out
