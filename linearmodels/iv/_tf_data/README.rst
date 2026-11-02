Generated tF calibration tables
===============================

The NPZ and JSON files are copied without modification from
https://github.com/finite-sample/tf-inference at commit
``ebaa0f8ff0286203b0a1488abdf3330ace84c4fe`` (version 0.1.0).
The accompanying LICENSE is that project's MIT license.

Each NPZ stores the F grid and the regularized squared critical value
``Z(F) = (F - boundary) * critical(F)**2``. The matching JSON records the
boundary, calibrated plateau, construction settings, numerical size-search
diagnostics, midpoint interpolation error and SHA-256 checksum. Runtime lookup
verifies the checksum once, constructs a PCHIP interpolator for Z, and caches it.
F at or below the boundary returns infinity. Above it, cap F at ``plateau_f``
and evaluate ``sqrt(Z(F) / (F - boundary))``.

These are generated numerical approximations to Appendix B.3 of Lee, McCrary,
Moreira and Porter (2022), https://doi.org/10.1257/aer.20211063. They do not use
the rounded printed tables. Calibration uses a finite nuisance-parameter grid
and local optimization, not a certified supremum over the unbounded domain.
Interpolation is not claimed to preserve the printed table's conservative bound.

The source package's ``validation/results.json`` records 4,032 held-out
rejection-probability checks and comparisons against refined curves. The largest
observed relative critical-value difference across the 18 levels was 1.76e-7.
These are numerical checks, not uniform error or coverage guarantees.

To regenerate, check out that source commit, install it, run ``make tables``
and ``make validate``, then copy its ``src/tf_inference/data/*.npz`` and
``src/tf_inference/data/*.json`` into this directory. Retain its license and
update this provenance record. The source package also exports CSV files for
other software; linearmodels needs only NPZ and JSON.
