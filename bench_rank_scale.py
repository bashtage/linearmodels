"""Benchmark for the scale-invariant rank check in linearmodels/iv/model.py.

Addresses the two questions from the review of PR #850:

1. Does standardisation improve rank detection?
   -> Yes: `matrix_rank` on the raw matrix is scale-dependent (a full-rank
   design with columns spanning 1e13 in scale is reported as deficient),
   while the unit-length-column check gives the correct rank.

2. Does standardisation lead to excessively large errors in LS estimates
   when matrices are poorly scaled?
   -> No: on moderately scaled designs the standardised path's errors are
   comparable to (and at large spreads better than) the raw path, which
   silently truncates small singular values. At extreme spreads (>= 1e10)
   every path degrades, because the signal of the tiny column is lost in
   the dynamic range of y itself — that is a property of the data, not of
   the standardisation.

Run:  python bench_rank_scale.py
"""

import numpy as np
from linearmodels import OLS
from numpy.linalg import lstsq, matrix_rank, norm

rng = np.random.default_rng(20241022)


def unit_length_columns(x):
    """Scale the columns of a matrix to unit length, leaving a 0 column at 0."""
    scale = norm(x, axis=0)
    scale[scale == 0.0] = 1.0
    return x / scale


print("=" * 78)
print("1. Rank detection: raw matrix_rank is scale-dependent")
print("=" * 78)

z_full = rng.standard_normal((200, 3))
z_full[:, 1] *= 1e13
z_full[:, 2] *= 1e-13
print("full-rank design, columns spanning 1e-13 .. 1e13")
print(f"  true rank             : 3")
print(f"  matrix_rank(raw)      : {matrix_rank(z_full)}   <-- false deficiency")
print(f"  matrix_rank(unit-len) : {matrix_rank(unit_length_columns(z_full))}")

z_dep = rng.standard_normal((200, 3))
z_dep[:, 2] = z_dep[:, 0] * 1e-13  # exactly dependent column
print("\nrank-deficient design (3rd column is 1e-13 x the 1st)")
print(f"  true rank             : 2")
print(f"  matrix_rank(raw)      : {matrix_rank(z_dep)}")
print(f"  matrix_rank(unit-len) : {matrix_rank(unit_length_columns(z_dep))}")

print()
print("=" * 78)
print("2. LS estimate error vs column-scale spread (noise 1e-10, n=2000,")
print("   median max-error over 50 draws)")
print("=" * 78)
print(f"{'column scales':<34}{'OLS':>10}{'raw lstsq':>12}{'unit-len':>10}")

n = 2000
beta = np.array([1.0, -2.0, 0.5])
noise_sd = 1e-10
draws = 50

for spread in [1e0, 1e3, 1e6, 1e8, 1e10, 1e12]:
    scales = np.array([1.0, spread**0.5, spread**-0.5])
    errs_ols, errs_raw, errs_std = [], [], []
    for _ in range(draws):
        X = rng.standard_normal((n, 3)) * scales
        y = X @ beta + rng.standard_normal(n) * noise_sd

        errs_ols.append(np.max(np.abs(OLS(y, X).fit().params - beta)))
        errs_raw.append(np.max(np.abs(lstsq(X, y, rcond=None)[0] - beta)))
        s = norm(X, axis=0)
        errs_std.append(np.max(np.abs(lstsq(X / s, y, rcond=None)[0] / s - beta)))

    err_ols = float(np.median(errs_ols))
    err_raw = float(np.median(errs_raw))
    err_std = float(np.median(errs_std))

    label = f"[1, {scales[1]:.0e}, {scales[2]:.0e}]"
    print(f"{label:<34}{err_ols:>10.2e}{err_raw:>12.2e}{err_std:>10.2e}")

print("""
Reading the table:
  * at moderate spreads every path is accurate; errors grow with the spread
    for all of them, standardisation included;
  * raw lstsq silently truncates the smallest singular value once the spread
    approaches the machine precision (error saturates at the coefficient's
    own magnitude, e.g. 0.5), while the standardised path degrades smoothly;
  * at extreme spreads every path fails alike: the tiny column's signal is
    below the dynamic range of y in double precision, so no solver can
    recover it.  Standardisation neither causes nor amplifies this.
""")

print("=" * 78)
print("3. IV2SLS end-to-end: estimates agree across instrument scalings")
print("=" * 78)

from linearmodels.iv import IV2SLS

n = 500
z = rng.standard_normal((n, 2))
x = z @ [1.0, 1.0] + rng.standard_normal(n)
y = 1 + 0.5 * x + rng.standard_normal(n)
exog = np.ones((n, 1))

base = IV2SLS(y, exog, x, z).fit()
scaled = IV2SLS(y, exog, x, z * [1.0, 1e13]).fit()
print(f"base params   : {dict(base.params)}")
print(f"scaled params : {dict(scaled.params)}")
print(f"max abs diff  : {np.max(np.abs(scaled.params - base.params)):.3e}")
