"""
Linear factor models for applications in asset pricing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from formulaic import model_matrix
from formulaic.materializers.types import NAAction
from formulaic.utils.context import capture_context
import numpy as np
from numpy.linalg import lstsq
from pandas import DataFrame
from scipy.optimize import minimize

from linearmodels.asset_pricing.covariance import (
    HeteroskedasticCovariance,
    HeteroskedasticWeight,
    KernelCovariance,
    KernelWeight,
)
from linearmodels.asset_pricing.results import (
    GMMFactorModelResults,
    LinearFactorModelResults,
)
from linearmodels.iv.covariance import KERNEL_LOOKUP
from linearmodels.iv.data import IVData, IVDataLike
from linearmodels.shared import hypotheses
from linearmodels.shared.alignment import check_alignment, row_labels
from linearmodels.shared.exceptions import (
    check_cov_config,
    lower_option,
    missing_warning,
    unknown_option_error,
)
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.rsquared import r_squared
from linearmodels.shared.typed_getters import get_float, get_string
from linearmodels.shared.utility import AttrDict
import linearmodels.typing.data

# The names of the covariance estimators of the models
COV_TYPES = ("heteroskedastic", "kernel", "robust")
_COV_KEYS = {"heteroskedastic": (), "robust": (), "kernel": ("kernel", "bandwidth")}


def callback_factory(
    obj: (
        Callable[
            [
                linearmodels.typing.data.Float64Array,
                bool,
                linearmodels.typing.data.Float64Array,
            ],
            float,
        ]
        | Callable[
            [
                linearmodels.typing.data.Float64Array,
                bool,
                HeteroskedasticWeight | KernelWeight,
            ],
            float,
        ]
    ),
    args: tuple[bool, Any],
    disp: bool | int = 1,
) -> Callable[[linearmodels.typing.data.Float64Array], None]:
    """
    Create a callback that prints the value of the objective during optimization.

    Parameters
    ----------
    obj : callable
        The objective function, which is called as ``obj(params, *args)``.
    args : tuple
        The arguments of ``obj`` after the parameters.
    disp : bool or int, optional
        Number of iterations between printed updates. The default is 1. A value
        that is 0 or negative suppresses the output.

    Returns
    -------
    callable
        Function of the parameters that prints the iteration and the value of the
        objective every ``disp`` iterations.
    """
    d = {"iter": 0}
    disp = int(disp)

    def callback(params: linearmodels.typing.data.Float64Array) -> None:
        fval = obj(params, *args)
        if disp > 0 and (d["iter"] % disp == 0):
            print("Iteration: {}, Objective: {}".format(d["iter"], fval))
        d["iter"] += 1

    return callback


def _j_statistic(
    alphas: linearmodels.typing.data.Float64Array,
    alpha_vcv: linearmodels.typing.data.Float64Array,
    restrictions: linearmodels.typing.data.Float64Array,
) -> WaldTestStatistic:
    r"""
    J statistic of the pricing errors of a two-pass estimator.

    Parameters
    ----------
    alphas : ndarray
        The pricing errors (nportfolio by 1), which satisfy
        ``restrictions.T @ alphas = 0``.
    alpha_vcv : ndarray
        The estimated covariance of the pricing errors (nportfolio by nportfolio).
    restrictions : ndarray
        The nportfolio by nrestriction matrix of the restrictions that the pricing
        errors satisfy, which has full column rank.

    Returns
    -------
    WaldTestStatistic
        The statistic :math:`\hat{\alpha}^{\prime}\hat{V}^{+}\hat{\alpha}`, where
        :math:`\hat{V}^{+}=Q(Q^{\prime}\hat{V}Q)^{-1}Q^{\prime}` is the generalized
        inverse of :math:`\hat{V}` with rank nportfolio less nrestriction, and the
        columns of :math:`Q` are an orthonormal basis of the space that is
        orthogonal to the columns of ``restrictions``. An ``InvalidTestStatistic``
        if :math:`Q^{\prime}\hat{V}Q` is singular.

    Notes
    -----
    The statistic depends on the space that is orthogonal to the columns of
    ``restrictions`` and not on the basis :math:`Q` of the space, so that it does
    not depend on the order of the portfolios. No tolerance decides the rank of
    :math:`\hat{V}`, which is the dimension of that space.
    """
    name = "J-statistic"
    null = "All alphas are 0"
    nportfolio, nrestriction = restrictions.shape
    df = nportfolio - nrestriction
    if df == 0:
        return WaldTestStatistic(0.0, null, df, name=name)
    # The pricing errors are in the span of the last df columns of u
    q = np.linalg.svd(restrictions)[0][:, nrestriction:]
    vcv = q.T @ alpha_vcv @ q
    vals, vecs = np.linalg.eigh((vcv + vcv.T) / 2)
    if not vals[0] > df * np.finfo(float).eps * vals[-1]:
        reason = (
            "The estimated covariance of the pricing errors has a rank smaller "
            f"than the {df} degrees of freedom of the statistic, so that the "
            "J-statistic cannot be computed. This occurs when the number of "
            "observations is small relative to the number of portfolios."
        )
        return hypotheses.InvalidTestStatistic(reason, name=name)
    coefs = vecs.T @ (q.T @ alphas)
    stat = float(np.sum(coefs**2 / vals[:, None]))
    return WaldTestStatistic(stat, null, df, name=name)


class _FactorModelBase:
    r"""
    Base class for all factor models.

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio)
    factors : array_like
        Priced factor returns (nobs by nfactor)
    """

    def __init__(self, portfolios: IVDataLike, factors: IVDataLike):
        self.portfolios = IVData(portfolios, var_name="portfolio")
        self.factors = IVData(factors, var_name="factor")
        check_alignment(
            row_labels(self.portfolios),
            {"factors": self.factors},
            reference="portfolios",
        )
        self._name = self.__class__.__name__
        self._formula: str | None = None
        self._validate_data()

    def __str__(self) -> str:
        out = self.__class__.__name__
        f, p = self.factors.shape[1], self.portfolios.shape[1]
        out += f" with {f} factors, {p} test portfolios"
        return out

    def __repr__(self) -> str:
        return self.__str__() + f"\nid: {hex(id(self))}"

    def _drop_missing(self) -> linearmodels.typing.data.BoolArray:
        data = (self.portfolios, self.factors)
        missing = cast(
            "linearmodels.typing.data.BoolArray",
            np.any(np.c_[[dh.isnull for dh in data]], 0),
        )
        if any(missing):
            if all(missing):
                raise ValueError(
                    "All observations contain missing data. "
                    "Model cannot be estimated."
                )
            self.portfolios.drop(missing)
            self.factors.drop(missing)
        missing_warning(missing)
        return missing

    def _validate_data(self) -> None:
        p = self.portfolios.ndarray
        f = self.factors.ndarray
        if p.shape[0] != f.shape[0]:
            raise ValueError(
                "The number of observations in portfolios and "
                "factors is not the same."
            )
        self._drop_missing()

        p = cast("linearmodels.typing.data.Float64Array", self.portfolios.ndarray)
        f = cast("linearmodels.typing.data.Float64Array", self.factors.ndarray)
        if has_constant(p)[0] or np.linalg.matrix_rank(p) < min(p.shape):
            raise ValueError(
                "portfolios must not contain a constant or "
                "equivalent and must not have rank\n"
                "less than the dimension of the smaller shape."
            )
        rank = np.linalg.matrix_rank(f)
        if has_constant(f)[0]:
            raise ValueError("factors must not contain a constant or equivalent.")
        if rank < f.shape[1]:
            raise ValueError(
                "Model cannot be estimated. factors do not have full column rank."
            )
        if p.shape[0] < (f.shape[1] + 1):
            raise ValueError(
                "Model cannot be estimated. portfolios must have factors + 1 or "
                "more returns to\nestimate the model parameters."
            )

    def _check_debiased(self, debiased: bool) -> None:
        """Debiasing needs more observations than regressors in the regressions"""
        nobs, nfactor = self.factors.shape
        if debiased and nobs <= nfactor + 1:
            raise ValueError(
                "The covariance cannot be debiased when the number of observations "
                f"({nobs}) is not larger than the number of factors plus one "
                f"({nfactor + 1}), the number of regressors of each time-series "
                "regression, since there are no residual degrees of freedom. Use "
                "debiased=False or more observations."
            )

    @property
    def formula(self) -> str | None:
        """Formula used to construct the model, if it was constructed using one."""
        return self._formula

    @formula.setter
    def formula(self, value: str | None) -> None:  # numpydoc ignore=GL08
        self._formula = value

    @staticmethod
    def _prepare_data_from_formula(
        formula: str,
        data: DataFrame,
        portfolios: DataFrame | None,
        context: Mapping[str, Any] | None,
    ) -> tuple[DataFrame, DataFrame, str]:
        orig_formula = formula
        # Missing values are kept so that the model drops the observations with a
        # missing value in the portfolios or the factors, as it does for arrays
        na_action = NAAction("ignore")
        # capture_context returns None if the frames of the caller are unavailable,
        # when formulaic looks in the namespace of this function
        eval_context: int | Mapping[str, Any] = 0 if context is None else context
        if portfolios is not None:
            factors_mm = model_matrix(
                formula + " + 0",
                data,
                context=eval_context,
                ensure_full_rank=True,
                na_action=na_action,
            )
            factors = DataFrame(factors_mm)
        else:
            formula_components = formula.split("~")
            portfolios_mm = model_matrix(
                formula_components[0].strip() + " + 0",
                data,
                context=eval_context,
                ensure_full_rank=False,
                na_action=na_action,
            )
            portfolios = DataFrame(portfolios_mm)
            factors_mm = model_matrix(
                formula_components[1].strip() + " + 0",
                data,
                context=eval_context,
                ensure_full_rank=False,
                na_action=na_action,
            )
            factors = DataFrame(factors_mm)

        return factors, portfolios, orig_formula


class TradedFactorModel(_FactorModelBase):
    r"""
    Linear factor models estimator applicable to traded factors.

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio).
    factors : array_like
        Priced factor returns (nobs by nfactor).

    Notes
    -----
    Implements both time-series estimators of risk premia, factor loadings
    and zero-alpha tests.

    The model estimated is

    .. math::

        r_{it}^e = \alpha_i + f_t \beta_i + \epsilon_{it}

    where :math:`r_{it}^e` is the excess return on test portfolio i and
    :math:`f_t` are the traded factor returns.  The model is directly
    tested using the estimated values :math:`\hat{\alpha}_i`. Risk premia,
    :math:`\lambda_i` are estimated using the sample averages of the factors,
    which must be excess returns on traded portfolios.

    Rows are paired by position, so that the first row of ``factors`` is the
    factor return in the period of the first row of ``portfolios``. If both
    ``portfolios`` and ``factors`` carry labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates), ``factors`` must use the same index, in the
    same order, as ``portfolios``, so that a return cannot be attached to the
    wrong period. A ``ValueError`` is raised if the index differs. Either
    reindex ``factors`` to the index of ``portfolios``, or call
    ``to_numpy()`` to pair its values by position. NumPy arrays and pandas
    objects with a default index do not carry labels and are always paired by
    position.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(self, portfolios: IVDataLike, factors: IVDataLike):
        super().__init__(portfolios, factors)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: DataFrame,
        *,
        portfolios: DataFrame | None = None,
    ) -> TradedFactorModel:
        """
        Specify a model using the formula interface.

        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        portfolios : array_like, optional
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.

        Returns
        -------
        TradedFactorModel
            Model instance.

        Notes
        -----
        The formula can be used in one of two ways.  The first specifies only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specifies the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        The formula can use the variables and functions that are available
        where ``from_formula`` is called, in addition to the columns of ``data``
        and the transforms of formulaic.

        .. versionchanged:: 7.1

           Variables and functions of the caller could not be used in the
           formula, and raised a ``FactorEvaluationError``.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import TradedFactorModel
        >>> data = french.load()
        >>> formula = "S1M1 + S1M5 + S3M3 + S5M1 + S5M5 ~ MktRF + SMB + HML"
        >>> mod = TradedFactorModel.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[["S1M1", "S1M5", "S3M1", "S3M5", "S5M1", "S5M5"]]
        >>> formula = "MktRF + SMB + HML"
        >>> mod = TradedFactorModel.from_formula(formula, data, portfolios=portfolios)
        """
        context = capture_context(1)
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios, context
        )
        mod = cls(portfolios, factors)
        mod.formula = formula
        return mod

    def fit(
        self,
        cov_type: str = "robust",
        debiased: bool = True,
        **cov_config: str | float,
    ) -> LinearFactorModelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : {"robust", "heteroskedastic", "kernel"}, optional
            Name of covariance estimator. The default is "robust". The names are
            not case sensitive. "heteroskedastic" is an alias of "robust". See Notes.
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True. See Notes.
        **cov_config : dict
            Additional covariance-specific options for the kernel covariance
            estimator: ``kernel`` (str) and ``bandwidth`` (float). The other
            covariance estimators do not use any. See Notes.

        Returns
        -------
        LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.
        ValueError
            If ``debiased`` is True and the number of observations is not larger
            than the number of factors plus one, or if the bandwidth is selected
            from the data and there are too few observations. See Notes.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 ignored the key.

        Notes
        -----
        Supported covariance estimators are:

        * "robust", "heteroskedastic" - Heteroskedasticity-robust covariance
          estimator
        * "kernel" - Heteroskedasticity and Autocorrelation consistent (HAC)
          covariance estimator

        The kernel covariance estimator takes the optional arguments
        ``kernel``, which is one of "bartlett", "newey-west", "parzen", "gallant",
        "qs", "quadratic-spectral" or "andrews", and ``bandwidth`` (a non-negative
        number). The names of the kernels are not case sensitive. The default
        kernel is "bartlett". If ``bandwidth`` is not given it
        is selected from the data, which needs at least 4 observations with the
        Bartlett kernel, 5 with the Parzen kernel and 6 with the quadratic
        spectral kernel. ``fit`` raises a ``ValueError`` for fewer observations,
        and if the selected bandwidth is not finite. Pass ``bandwidth`` to use
        the kernel with fewer observations. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.

        If ``debiased`` is True, the covariance of the parameters of the
        time-series regressions (the alphas and the loadings) is divided by
        nobs - nfactor - 1, the number of observations less the number of
        regressors, instead of by nobs. The risk premia are the sample means of
        the factors, so that their covariance, which is the block of ``cov`` for
        the risk premia and the source of ``risk_premia_se``, is divided by
        nobs - 1. The covariance between the two groups of parameters is divided
        by the geometric mean of the two scales, so that the covariance matrix
        remains positive semi-definite. If ``debiased`` is False, the covariance
        is divided by nobs.

        .. versionchanged:: 7.1

           The block of ``cov`` for the risk premia was divided by
           nobs - nfactor - 1, while ``risk_premia_se`` used nobs - 1. Both use
           nobs - 1 now, so that the block of ``cov`` for the risk premia is
           smaller by the factor (nobs - nfactor - 1) / (nobs - 1), for example by
           5.1% with 60 observations and 3 factors, and equals
           ``risk_premia_se`` squared. The covariance between the risk premia
           and the other parameters, which used nobs - nfactor - 1, is now smaller
           by the square root of the same factor. The covariance of the
           alphas and the loadings, ``risk_premia_se`` and the covariance with
           ``debiased=False`` do not change.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _COV_KEYS, KERNEL_LOOKUP)
        self._check_debiased(debiased)
        p = self.portfolios.ndarray
        f = self.factors.ndarray
        nportfolio = p.shape[1]
        nobs, nfactor = f.shape
        fc = np.c_[np.ones((nobs, 1)), f]
        rp = f.mean(0)[:, None]
        fe = f - f.mean(0)
        b = np.linalg.pinv(fc) @ p
        eps = p - fc @ b
        alphas = b[:1].T

        nloading = (nfactor + 1) * nportfolio
        xpxi = np.eye(nloading + nfactor)
        xpxi[:nloading, :nloading] = np.kron(
            np.eye(nportfolio), np.linalg.pinv(fc.T @ fc / nobs)
        )
        f_rep = np.tile(fc, (1, nportfolio))
        eps_rep = np.tile(eps, (nfactor + 1, 1))  # 1 2 3 ... 25 1 2 3 ...
        eps_rep = eps_rep.ravel(order="F")
        eps_rep = np.reshape(eps_rep, (nobs, (nfactor + 1) * nportfolio), order="F")
        xe = f_rep * eps_rep
        xe = np.c_[xe, fe]
        if cov_type in ("robust", "heteroskedastic"):
            cov_est = HeteroskedasticCovariance(
                xe, inv_jacobian=xpxi, center=False, debiased=debiased, df=fc.shape[1]
            )
            rp_cov_est = HeteroskedasticCovariance(
                fe, jacobian=np.eye(f.shape[1]), center=False, debiased=debiased, df=1
            )
        elif cov_type == "kernel":
            kernel = get_string(cov_config, "kernel")
            bandwidth = get_float(cov_config, "bandwidth")
            cov_est = KernelCovariance(
                xe,
                inv_jacobian=xpxi,
                center=False,
                debiased=debiased,
                df=fc.shape[1],
                bandwidth=bandwidth,
                kernel=kernel,
            )
            bw = cov_est.bandwidth
            _cov_config = dict(cov_config.items())
            _cov_config["bandwidth"] = bw
            rp_cov_est = KernelCovariance(
                fe,
                jacobian=np.eye(f.shape[1]),
                center=False,
                debiased=debiased,
                df=1,
                bandwidth=bw,
                kernel=kernel,
            )
        else:
            raise unknown_option_error("cov_type", cov_type, COV_TYPES)
        full_vcv = cov_est.cov
        rp_cov = rp_cov_est.cov
        if debiased:
            # The risk premia are sample means, so their covariance uses nobs - 1 and
            # not the nobs - (nfactor + 1) of the time-series regressions. The
            # covariance between the two uses the geometric mean of the two scales
            adj = np.ones(nloading + nfactor)
            adj[nloading:] = np.sqrt((nobs - fc.shape[1]) / (nobs - 1))
            full_vcv = full_vcv * np.outer(adj, adj)
        vcv = full_vcv[:nloading, :nloading]

        # Rearrange VCV
        order: np.ndarray = np.reshape(
            np.arange((nfactor + 1) * nportfolio), (nportfolio, nfactor + 1)
        )
        order = order.T.ravel()
        vcv = vcv[order][:, order]

        # Return values
        alpha_vcv = vcv[:nportfolio, :nportfolio]
        stat = float(np.squeeze(alphas.T @ np.linalg.pinv(alpha_vcv) @ alphas))
        jstat = WaldTestStatistic(
            stat, "All alphas are 0", nportfolio, name="J-statistic"
        )
        params = b.T
        betas = b[1:].T
        residual_ss = (eps**2).sum()
        e = p - p.mean(0)[None, :]
        total_ss = (e**2).sum()
        r2 = r_squared(residual_ss, total_ss, (p**2).sum(), p.shape[0])
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append(f"alpha-{portfolio}")
            param_names.extend(
                [f"beta-{portfolio}-{factor}" for factor in self.factors.cols]
            )
        param_names.extend([f"lambda-{factor}" for factor in self.factors.cols])

        res = AttrDict(
            params=params,
            cov=full_vcv,
            betas=betas,
            rp=rp,
            rp_cov=rp_cov,
            alphas=alphas,
            alpha_vcv=alpha_vcv,
            jstat=jstat,
            rsquared=r2,
            total_ss=total_ss,
            residual_ss=residual_ss,
            param_names=param_names,
            portfolio_names=self.portfolios.cols,
            factor_names=self.factors.cols,
            name=self._name,
            cov_type=cov_type,
            model=self,
            nobs=nobs,
            rp_names=self.factors.cols,
            cov_est=cov_est,
        )

        return LinearFactorModelResults(res)


class _LinearFactorModelBase(_FactorModelBase):
    r"""
    Linear factor model base class
    """

    def __init__(
        self,
        portfolios: IVDataLike,
        factors: IVDataLike,
        *,
        risk_free: bool = False,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        self._risk_free = bool(risk_free)
        super().__init__(portfolios, factors)
        self._validate_additional_data()
        if sigma is None:
            self._sigma_m12 = self._sigma_inv = self._sigma = np.eye(
                self.portfolios.shape[1]
            )
        else:
            self._sigma = self._validate_sigma(sigma)
            vals, vecs = np.linalg.eigh(self._sigma)
            self._sigma_m12 = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            self._sigma_inv = np.linalg.inv(self._sigma).astype(float, copy=False)

    def _validate_sigma(
        self, sigma: linearmodels.typing.data.ArrayLike
    ) -> linearmodels.typing.data.FloatArray2D:
        """Check that sigma is a symmetric positive definite matrix of the right size"""
        nportfolio = self.portfolios.shape[1]
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.shape != (nportfolio, nportfolio):
            raise ValueError(
                f"sigma must be a square array with shape ({nportfolio}, "
                f"{nportfolio}), where {nportfolio} is the number of test portfolios. "
                f"The shape of sigma is {sigma_arr.shape}."
            )
        if not np.all(np.isfinite(sigma_arr)):
            raise ValueError("sigma must not contain missing or infinite values.")
        scale = np.max(np.abs(sigma_arr))
        if np.max(np.abs(sigma_arr - sigma_arr.T)) > 1e-8 * scale:
            raise ValueError("sigma must be symmetric.")
        vals = np.linalg.eigvalsh((sigma_arr + sigma_arr.T) / 2)
        if not vals[0] > nportfolio * np.finfo(float).eps * vals[-1]:
            raise ValueError("sigma must be positive definite.")
        return sigma_arr

    def __str__(self) -> str:
        out = super().__str__()
        if np.any(self._sigma != np.eye(self.portfolios.shape[1])):
            out += " using GLS"
        out += f"\nEstimated risk-free rate: {self._risk_free}"

        return out

    def _validate_additional_data(self) -> None:
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nrp = f.shape[1] + int(self._risk_free)
        if p.shape[1] < nrp:
            raise ValueError(
                "The number of test portfolio must be at least as "
                "large as the number of risk premia, including the "
                "risk free rate if estimated."
            )

    def _boundaries(self) -> tuple[int, int, int, int, int, int, int]:
        nobs, nf = self.factors.ndarray.shape
        nport = self.portfolios.ndarray.shape[1]
        nrf = int(bool(self._risk_free))

        s1 = (nf + 1) * nport
        s2 = s1 + (nf + nrf)
        s3 = s2 + nport

        return nobs, nf, nport, nrf, s1, s2, s3


class LinearFactorModel(_LinearFactorModelBase):
    r"""
    Linear factor model estimator.

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio).
    factors : array_like
        Priced factor returns (nobs by nfactor).
    risk_free : bool, optional
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate. The default
        is False.
    sigma : array_like, optional
        Positive definite residual covariance (nportfolio by nportfolio) that
        weights the test portfolios in the second stage. A ``ValueError`` is
        raised if it does not have this shape, contains missing or infinite
        values, or is not symmetric and positive definite. If not provided, the
        identity matrix is used.

    Notes
    -----
    Suitable for traded or non-traded factors.

    Implements a 2-step estimator of risk premia, factor loadings and model
    tests.

    The first stage model estimated is

    .. math::

        r_{it} = c_i + f_t \beta_i + \epsilon_{it}

    where :math:`r_{it}` is the return on test portfolio i and
    :math:`f_t` are the traded factor returns.  The parameters :math:`c_i`
    are required to allow non-traded to be tested, but are not economically
    interesting.  These are not reported.

    The second stage model uses the estimated factor loadings from the first
    and is

    .. math::

        \bar{r}_i = \lambda_0 + \hat{\beta}_i^\prime \lambda + \eta_i

    where :math:`\bar{r}_i` is the average excess return to portfolio i and
    :math:`\lambda_0` is only included if estimating the risk-free rate. GLS
    is used in the second stage if ``sigma`` is provided.

    The model is tested using the estimated values
    :math:`\hat{\alpha}_i=\hat{\eta}_i`, the pricing errors. With :math:`N`
    test portfolios, :math:`K` factors and :math:`n_{rf}` equal to 1 if the
    risk-free rate is estimated and 0 if it is not, the second stage estimates
    make the pricing errors orthogonal to the :math:`K + n_{rf}` columns of
    :math:`\Sigma^{-1}\hat{\beta}_c`. Here :math:`\hat{\beta}_c` contains the
    estimated factor loadings, preceded by a column of ones if the risk-free
    rate is estimated, and :math:`\Sigma` is ``sigma``, or the identity if it
    is not provided. The pricing errors therefore lie in a space with
    :math:`N - K - n_{rf}` dimensions, and the covariance of the pricing errors
    has rank :math:`N - K - n_{rf}`.

    The J-statistic is the Wald statistic of the pricing errors in the space
    where they lie,

    .. math::

        J = (Q^\prime\hat{\alpha})^\prime (Q^\prime \hat{V} Q)^{-1}
            (Q^\prime\hat{\alpha})

    where :math:`\hat{V}` is the estimated covariance of the pricing errors and
    :math:`Q` is an :math:`N` by :math:`N - K - n_{rf}` matrix whose orthonormal
    columns span the space that is orthogonal to the columns of
    :math:`\Sigma^{-1}\hat{\beta}_c`. The :math:`N - K - n_{rf}` elements of
    :math:`Q^\prime\hat{\alpha}` are the coordinates of the pricing errors in
    that space, and :math:`Q^\prime \hat{V} Q` is their estimated covariance.
    The statistic tests the null that all of the pricing errors are zero, and
    has an asymptotic :math:`\chi^2_{N-K-n_{rf}}` distribution. It can also be
    written :math:`\hat{\alpha}^\prime \hat{V}^{+} \hat{\alpha}` for
    :math:`\hat{V}^{+}=Q(Q^\prime \hat{V} Q)^{-1}Q^\prime`, which has rank
    :math:`N - K - n_{rf}`. If :math:`\hat{V}` were exactly singular in the
    directions of the columns of :math:`\Sigma^{-1}\hat{\beta}_c`, as the
    covariance of the pricing errors is in theory, :math:`\hat{V}^{+}` would be
    its Moore-Penrose pseudo-inverse.

    In a sample, :math:`\hat{V}` is not exactly singular. The Moore-Penrose
    pseudo-inverse of :math:`\hat{V}` would either invert the directions that
    are null in theory, which makes the statistic too large, or need a
    tolerance that decides which eigenvalues are treated as zero. The
    J-statistic instead projects the pricing errors on the space where they lie
    and uses the rank that the model implies, :math:`N - K - n_{rf}`, so that no
    tolerance decides which eigenvalues are treated as zero.

    The statistic depends on the space and not on the basis of the space: any
    other orthonormal basis, :math:`QR` for an orthogonal matrix :math:`R`,
    gives the same value. It therefore does not depend on the order of the test
    portfolios. Reordering the columns of ``portfolios``, and the rows and
    columns of ``sigma`` in the same way, reorders :math:`\hat{\alpha}`,
    :math:`\hat{V}` and the rows of :math:`\Sigma^{-1}\hat{\beta}_c` in the
    same way and does not change the statistic. It also does not depend on the
    order of the observations if ``cov_type`` is ``"robust"``, but the kernel
    covariance estimators use the order of the observations.

    If :math:`Q^\prime \hat{V} Q` is numerically singular, which can occur when
    there are few observations, the J-statistic is an ``InvalidTestStatistic``.
    If the number of test portfolios is the number of risk premia,
    :math:`N = K + n_{rf}`, the pricing errors are zero and the statistic is 0
    with 0 degrees of freedom.

    .. versionchanged:: 7.1

       The J-statistic used the Moore-Penrose pseudo-inverse of
       :math:`\hat{V}` with its default tolerance, which also inverts the
       directions in which the covariance is null in theory. Its eigenvalues
       are small but not zero in a sample. The statistic was therefore too
       large: by 4% in the median and by up to 30% in 200 simulated samples of
       a correct model with 250 observations, 25 portfolios and 3 factors,
       where the 5% test rejected 12.5% of the samples (10.0% now; with 1,000
       observations 7.5% and 5.5%). The covariance can be far from singular in
       those directions when there are few portfolios: the J-statistic of the
       five portfolios and three factors of the Fama-French example in the
       documentation is 41.6 and was 100.6. If the risk-free rate was
       estimated, an eigenvalue that is numerically zero was inverted as well,
       and the statistic changed in the third digit when the observations were
       only reordered.

    Rows are paired by position, so that the first row of ``factors`` is the
    factor return in the period of the first row of ``portfolios``. If both
    ``portfolios`` and ``factors`` carry labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates), ``factors`` must use the same index, in the
    same order, as ``portfolios``, so that a return cannot be attached to the
    wrong period. A ``ValueError`` is raised if the index differs. Either
    reindex ``factors`` to the index of ``portfolios``, or call
    ``to_numpy()`` to pair its values by position. NumPy arrays and pandas
    objects with a default index do not carry labels and are always paired by
    position.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self,
        portfolios: IVDataLike,
        factors: IVDataLike,
        *,
        risk_free: bool = False,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> None:
        super().__init__(portfolios, factors, risk_free=risk_free, sigma=sigma)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: DataFrame,
        *,
        portfolios: DataFrame | None = None,
        risk_free: bool = False,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> LinearFactorModel:
        """
        Specify a model using the formula interface.

        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        portfolios : array_like, optional
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool, optional
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate. The
            default is False.
        sigma : array_like, optional
            Positive definite residual covariance (nportfolio by nportfolio) that
            weights the test portfolios in the second stage. A ``ValueError`` is
            raised if it does not have this shape, contains missing or infinite
            values, or is not symmetric and positive definite. If not provided,
            the identity matrix is used.

        Returns
        -------
        LinearFactorModel
            Model instance.

        Notes
        -----
        The formula can be used in one of two ways.  The first specifies only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specifies the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        The formula can use the variables and functions that are available
        where ``from_formula`` is called, in addition to the columns of ``data``
        and the transforms of formulaic.

        .. versionchanged:: 7.1

           Variables and functions of the caller could not be used in the
           formula, and raised a ``FactorEvaluationError``.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import LinearFactorModel
        >>> data = french.load()
        >>> formula = "S1M1 + S1M5 + S3M3 + S5M1 + S5M5 ~ MktRF + SMB + HML"
        >>> mod = LinearFactorModel.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[["S1M1", "S1M5", "S3M1", "S3M5", "S5M1", "S5M5"]]
        >>> formula = "MktRF + SMB + HML"
        >>> mod = LinearFactorModel.from_formula(formula, data, portfolios=portfolios)
        """
        context = capture_context(1)
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios, context
        )
        mod = cls(portfolios, factors, risk_free=risk_free, sigma=sigma)
        mod.formula = formula
        return mod

    def fit(
        self,
        cov_type: str = "robust",
        debiased: bool = True,
        **cov_config: bool | int | str,
    ) -> LinearFactorModelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        cov_type : {"robust", "heteroskedastic", "kernel"}, optional
            Name of covariance estimator. The default is "robust". The names are
            not case sensitive. "heteroskedastic" is an alias of "robust".
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True.
        **cov_config : dict
            Additional covariance-specific options for the kernel covariance
            estimator: ``kernel`` (str) and ``bandwidth`` (float). The other
            covariance estimators do not use any. See Notes.

        Returns
        -------
        LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.
        ValueError
            If ``debiased`` is True and the number of observations is not larger
            than the number of factors plus one, or if the bandwidth is selected
            from the data and there are too few observations. See Notes.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 ignored the key.

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, which is one of "bartlett", "newey-west", "parzen", "gallant",
        "qs", "quadratic-spectral" or "andrews", and ``bandwidth`` (a non-negative
        number). The names of the kernels are not case sensitive. The default
        kernel is "bartlett". If ``bandwidth`` is not given it
        is selected from the data, which needs at least 4 observations with the
        Bartlett kernel, 5 with the Parzen kernel and 6 with the quadratic
        spectral kernel. ``fit`` raises a ``ValueError`` for fewer observations,
        and if the selected bandwidth is not finite. Pass ``bandwidth`` to use
        the kernel with fewer observations. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _COV_KEYS, KERNEL_LOOKUP)
        nobs, nf, nport, _, s1, s2, s3 = self._boundaries()
        excess_returns = not self._risk_free
        self._check_debiased(debiased)
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nport = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = lstsq(fc, p, rcond=None)[0].astype(float)  # nf+1 by np
        eps = p - fc @ b
        if excess_returns:
            betas = b[1:].T
        else:
            betas = b.T.copy()
            betas[:, 0] = 1.0

        sigma_m12 = self._sigma_m12
        lam = lstsq(sigma_m12 @ betas, sigma_m12 @ p.mean(0)[:, None], rcond=None)[0]
        expected = betas @ lam
        pricing_errors = p - expected.T
        # Moments
        alphas = pricing_errors.mean(0)[:, None]
        moments = self._moments(eps, betas, alphas, pricing_errors)
        # Jacobian
        jacobian = self._jacobian(betas, lam, alphas)

        if cov_type not in COV_TYPES:
            raise unknown_option_error("cov_type", cov_type, COV_TYPES)
        if cov_type in ("robust", "heteroskedastic"):
            cov_est_inst = HeteroskedasticCovariance(
                moments,
                jacobian=jacobian,
                center=False,
                debiased=debiased,
                df=fc.shape[1],
            )
        else:  # "kernel":
            bandwidth = get_float(cov_config, "bandwidth")
            kernel = get_string(cov_config, "kernel")
            cov_est_inst = KernelCovariance(
                moments,
                jacobian=jacobian,
                center=False,
                debiased=debiased,
                df=fc.shape[1],
                kernel=kernel,
                bandwidth=bandwidth,
            )

        # VCV
        full_vcv = cov_est_inst.cov
        alpha_vcv = full_vcv[s2:, s2:]
        # The second stage estimates make the pricing errors orthogonal to the
        # columns of sigma^-1 [1 betas]
        jstat = _j_statistic(alphas, alpha_vcv, self._sigma_inv @ betas)

        total_ss = ((p - p.mean(0)[None, :]) ** 2).sum()
        residual_ss = (eps**2).sum()
        r2 = r_squared(residual_ss, total_ss, (p**2).sum(), p.shape[0])
        rp = lam
        rp_cov = full_vcv[s1:s2, s1:s2]
        betas = betas if excess_returns else betas[:, 1:]
        params = np.c_[alphas, betas]
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append(f"alpha-{portfolio}")
            param_names.extend(
                [f"beta-{portfolio}-{factor}" for factor in self.factors.cols]
            )
        if not excess_returns:
            param_names.append("lambda-risk_free")
        param_names.extend([f"lambda-{factor}" for factor in self.factors.cols])

        # Pivot vcv to remove unnecessary and have correct order
        order: np.ndarray = np.reshape(np.arange(s1), (nport, nf + 1))
        order[:, 0] = np.arange(s2, s3)
        order = order.ravel()
        order = np.r_[order, s1:s2]
        full_vcv = full_vcv[order][:, order]
        factor_names = list(self.factors.cols)
        rp_names = factor_names[:]
        if not excess_returns:
            rp_names.insert(0, "risk_free")
        res = AttrDict(
            params=params,
            cov=full_vcv,
            betas=betas,
            rp=rp,
            rp_cov=rp_cov,
            alphas=alphas,
            alpha_vcv=alpha_vcv,
            jstat=jstat,
            rsquared=r2,
            total_ss=total_ss,
            residual_ss=residual_ss,
            param_names=param_names,
            portfolio_names=self.portfolios.cols,
            factor_names=factor_names,
            name=self._name,
            cov_type=cov_type,
            model=self,
            nobs=nobs,
            rp_names=rp_names,
            cov_est=cov_est_inst,
        )

        return LinearFactorModelResults(res)

    def _jacobian(
        self,
        betas: linearmodels.typing.data.Float64Array,
        lam: linearmodels.typing.data.Float64Array,
        alphas: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        nobs, nf, nport, nrf, s1, s2, s3 = self._boundaries()
        f = self.factors.ndarray
        fc = np.c_[np.ones((nobs, 1)), f]
        excess_returns = not self._risk_free
        bc = betas
        sigma_inv = self._sigma_inv

        jac = np.eye((nport * (nf + 1)) + (nf + nrf) + nport)
        fpf = fc.T @ fc / nobs
        jac[:s1, :s1] = np.kron(np.eye(nport), fpf)

        b_tilde = sigma_inv @ bc
        alpha_tilde = sigma_inv @ alphas
        _lam = lam if excess_returns else lam[1:]
        for i in range(nport):
            block = np.zeros((nf + nrf, nf + 1))
            block[:, 1:] = b_tilde[[i]].T @ _lam.T
            block[nrf:, 1:] -= alpha_tilde[i] * np.eye(nf)
            jac[s1:s2, (i * (nf + 1)) : ((i + 1) * (nf + 1))] = block
        jac[s1:s2, s1:s2] = bc.T @ sigma_inv @ bc
        zero_lam = np.r_[[[0]], _lam]
        jac[s2:s3, :s1] = np.kron(np.eye(nport), zero_lam.T)
        jac[s2:s3, s1:s2] = bc

        return jac

    def _moments(
        self,
        eps: linearmodels.typing.data.Float64Array,
        betas: linearmodels.typing.data.Float64Array,
        alphas: linearmodels.typing.data.Float64Array,
        pricing_errors: linearmodels.typing.data.Float64Array,
    ) -> linearmodels.typing.data.Float64Array:
        sigma_inv = self._sigma_inv

        f = self.factors.ndarray
        nobs, nf, nport, _, _, _, _ = self._boundaries()
        fc = np.c_[np.ones((nobs, 1)), f]
        f_rep = np.tile(fc, (1, nport))
        eps_rep = np.tile(eps, (nf + 1, 1))
        eps_rep = np.reshape(eps_rep.T, (nport * (nf + 1), nobs)).T

        # Moments
        g1 = f_rep * eps_rep
        g2 = pricing_errors @ sigma_inv @ betas
        g3 = pricing_errors - alphas.T

        return np.c_[g1, g2, g3]


class LinearFactorModelGMM(_LinearFactorModelBase):
    r"""
    GMM estimator of Linear factor models.

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio).
    factors : array_like
        Priced factors values (nobs by nfactor).
    risk_free : bool, optional
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate. The default
        is False.

    Notes
    -----
    Suitable for traded or non-traded factors.

    Implements a GMM estimator of risk premia, factor loadings and model
    tests.

    The moments are

    .. math::

        \left[\begin{array}{c}
           \epsilon_{t}\otimes f_{c,t}\\
           f_{t}-\mu
        \end{array}\right]

    and

    .. math::

      \epsilon_{t}=r_{t}-\left[1_{N}\;\beta\right]\lambda-\beta\left(f_{t}-\mu\right)

    where :math:`r_{it}` is the return on test portfolio i and
    :math:`f_t` are the factor returns.

    There are :math:`N (K + 1) + K` moment conditions, where :math:`N` is the
    number of test portfolios and :math:`K` is the number of factors. The number
    of observations must exceed the number of moment conditions, since the
    covariance of the moment conditions, which weights them, is singular
    otherwise. A ``ValueError`` is raised if it does not.

    .. versionchanged:: 7.1

       A ``ValueError`` is raised if the number of observations does not exceed
       the number of moment conditions. The estimator raised a ``LinAlgError``
       or returned a J-statistic that was NaN or arbitrary, such as -4.6e35.

    The model is tested using the optimized objective function using the
    usual GMM J statistic.

    The alphas are the averages of the moment conditions :math:`\epsilon_t`
    that correspond to the intercept, and their standard errors account for the
    estimation of the parameters. If :math:`S` is the covariance of the moment
    conditions and :math:`G` is the Jacobian of their average, the covariance of
    the average moment conditions at the efficient GMM estimates is

    .. math::

        \left(S - G (G^\prime S^{-1} G)^{-1} G^\prime\right) / (T - df)

    where :math:`df` is the number of factors if the covariance is debiased and
    0 if it is not. This is the same :math:`S`, :math:`G` and scale as the
    covariance of the parameters :math:`(G^\prime S^{-1} G)^{-1} / (T - df)`,
    which depends on ``cov_type`` (and ``kernel`` and ``bandwidth`` for the
    kernel estimator), ``center`` and ``debiased`` of ``fit``. The covariance of
    the alphas has rank equal to the number of degrees of freedom of the
    J-statistic. If the model is just identified, the alphas are zero and so is
    their variance.

    .. versionchanged:: 7.1

       The standard errors of the alphas were the square roots of the diagonal
       of the covariance of the average moment conditions, :math:`S / T`, which
       ignores that the parameters were estimated, and which did not depend on
       ``cov_type``, ``center`` or ``debiased``. This overstated the standard
       errors: in a simulation of 60 samples with 400 observations, 6 portfolios
       and 2 factors the mean standard error was 0.00143 to 0.00145 for all six
       alphas, where their sampling standard deviations were 0.00047 to 0.00136.
       The mean standard errors are now 0.00049 to 0.00133 and track them. The
       t-statistics and p-values of the alphas, which are in ``tstats`` and
       ``pvalues``, change with them. The covariance of the parameters, the risk
       premia and the J-statistic do not change.

    Rows are paired by position, so that the first row of ``factors`` is the
    factor return in the period of the first row of ``portfolios``. If both
    ``portfolios`` and ``factors`` carry labels (a pandas Series or DataFrame
    with an index other than the default 0, 1, ..., n-1, or an xarray
    DataArray with coordinates), ``factors`` must use the same index, in the
    same order, as ``portfolios``, so that a return cannot be attached to the
    wrong period. A ``ValueError`` is raised if the index differs. Either
    reindex ``factors`` to the index of ``portfolios``, or call
    ``to_numpy()`` to pair its values by position. NumPy arrays and pandas
    objects with a default index do not carry labels and are always paired by
    position.
    Dates and time deltas are compared by value, so that the same dates at
    different resolutions, e.g., ``datetime64[us]`` and ``datetime64[ns]``, are
    the same labels.
    """

    def __init__(
        self, portfolios: IVDataLike, factors: IVDataLike, *, risk_free: bool = False
    ) -> None:
        super().__init__(portfolios, factors, risk_free=risk_free)

    def _validate_additional_data(self) -> None:
        super()._validate_additional_data()
        nobs, nfactor = self.factors.shape
        nportfolio = self.portfolios.shape[1]
        nmoment = nportfolio * (nfactor + 1) + nfactor
        if nobs <= nmoment:
            raise ValueError(
                f"The number of observations ({nobs}) must exceed the number of "
                f"moment conditions ({nmoment}) of the GMM estimator, since the "
                "covariance of the moment conditions is singular otherwise. The "
                "number of moment conditions is nportfolio * (nfactor + 1) + "
                f"nfactor, where nportfolio is {nportfolio} and nfactor is "
                f"{nfactor}. Use more observations, fewer portfolios or fewer "
                "factors."
            )

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: DataFrame,
        *,
        portfolios: DataFrame | None = None,
        risk_free: bool = False,
    ) -> LinearFactorModelGMM:
        """
        Specify a model using the formula interface.

        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes.
        data : DataFrame
            DataFrame containing the variables used in the formula.
        portfolios : array_like, optional
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool, optional
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate. The
            default is False.

        Returns
        -------
        LinearFactorModelGMM
            Model instance.

        Notes
        -----
        The formula can be used in one of two ways.  The first specifies only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specifies the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

        The formula can use the variables and functions that are available
        where ``from_formula`` is called, in addition to the columns of ``data``
        and the transforms of formulaic.

        .. versionchanged:: 7.1

           Variables and functions of the caller could not be used in the
           formula, and raised a ``FactorEvaluationError``.

        Examples
        --------
        >>> from linearmodels.datasets import french
        >>> from linearmodels.asset_pricing import LinearFactorModelGMM
        >>> data = french.load()
        >>> formula = "S1M1 + S1M5 + S3M3 + S5M1 + S5M5 ~ MktRF + SMB + HML"
        >>> mod = LinearFactorModelGMM.from_formula(formula, data)

        Using only factors

        >>> portfolios = data[["S1M1", "S1M5", "S3M1", "S3M5", "S5M1", "S5M5"]]
        >>> formula = "MktRF + SMB + HML"
        >>> mod = LinearFactorModelGMM.from_formula(
        ...     formula, data, portfolios=portfolios
        ... )
        """
        context = capture_context(1)
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios, context
        )
        mod = cls(portfolios, factors, risk_free=risk_free)
        mod.formula = formula
        return mod

    def fit(
        self,
        *,
        center: bool = True,
        use_cue: bool = False,
        steps: int = 2,
        disp: int = 10,
        max_iter: int = 1000,
        cov_type: str = "robust",
        debiased: bool = True,
        starting: linearmodels.typing.data.ArrayLike | None = None,
        opt_options: dict[str, Any] | None = None,
        **cov_config: bool | int | str,
    ) -> GMMFactorModelResults:
        """
        Estimate model parameters.

        Parameters
        ----------
        center : bool, optional
            Flag indicating to center the moment conditions before computing
            the weighting matrix. The default is True.
        use_cue : bool, optional
            Flag indicating to use continuously updating estimator. The default is
            False.
        steps : int, optional
            Number of steps to use when estimating parameters.  2 corresponds
            to the standard efficient GMM estimator. Higher values will
            iterate until convergence or up to the number of steps given. The
            default is 2.
        disp : int, optional
            Number of iterations between printed update. 0 or negative values
            suppresses output. The default is 10.
        max_iter : int, optional
            Maximum number of iterations when minimizing objective. Must be
            positive. The default is 1000.
        cov_type : {"robust", "heteroskedastic", "kernel"}, optional
            Name of covariance estimator. The default is "robust". The names are
            not case sensitive. "heteroskedastic" is an alias of "robust".
        debiased : bool, optional
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment. The default is True.
        starting : array_like, optional
            Starting values to use in optimization.  If not provided, 2SLS
            estimates are used.
        opt_options : dict, optional
            Additional options to pass to scipy.optimize.minimize when
            optimizing the objective function. If not provided, defers to
            scipy to choose an appropriate optimizer. All minimize inputs
            except ``fun``, ``x0``, and ``args`` can be overridden.
        **cov_config : dict
            Additional covariance-specific options for the kernel covariance
            estimator: ``kernel`` (str) and ``bandwidth`` (float). The other
            covariance estimators do not use any. See Notes.

        Returns
        -------
        GMMFactorModelResults
            Results class with parameter estimates, covariance and test statistics.

        Raises
        ------
        UnknownOptionError
            If ``cov_type`` or ``kernel`` is not supported. It is both a
            ``KeyError`` and a ``ValueError``.
        ValueError
            If ``starting`` does not have one element for each parameter, or if
            the bandwidth is selected from the data and there are too few
            observations. See Notes.
            A key of ``cov_config`` that the covariance estimator that ``cov_type``
            selects does not accept also raises it. The message names the key and lists
            the accepted keys. This is checked before the model is estimated. Releases
            before 7.1 ignored the key.

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, which is one of "bartlett", "newey-west", "parzen", "gallant",
        "qs", "quadratic-spectral" or "andrews", and ``bandwidth`` (a non-negative
        number). The names of the kernels are not case sensitive. The default
        kernel is "bartlett". If ``bandwidth`` is not given it
        is selected from the data, for the weight matrix and for the covariance,
        which needs at least 4 observations with the Bartlett kernel, 5 with the
        Parzen kernel and 6 with the quadratic spectral kernel. ``fit`` raises a
        ``ValueError`` for fewer observations, and if the selected bandwidth is
        not finite. Pass ``bandwidth`` to use the kernel with fewer
        observations. See
        :func:`~linearmodels.iv.covariance.kernel_optimal_bandwidth`.
        """
        cov_type = lower_option(cov_type)
        check_cov_config(cov_config, cov_type, _COV_KEYS, KERNEL_LOOKUP)

        nobs, n = self.portfolios.shape
        k = self.factors.shape[1]
        excess_returns = not self._risk_free
        nrf = int(not bool(excess_returns))
        # 1. Starting Values - use 2 pass
        mod = LinearFactorModel(
            self.portfolios, self.factors, risk_free=self._risk_free
        )
        # The starting values do not use the covariance, which cannot be debiased
        # if there are no residual degrees of freedom in the first stage
        res = mod.fit(debiased=False)
        betas_1d = np.asarray(res.betas).ravel()
        lam = np.asarray(res.risk_premia)
        mu = self.factors.ndarray.mean(0)
        sv = np.r_[betas_1d, lam, mu][:, None]
        if starting is not None:
            starting = np.asarray(starting)
            if starting.ndim == 1:
                starting = starting[:, None]
            if starting.shape != sv.shape:
                raise ValueError(f"Starting values must have {sv.shape} elements.")
            sv = starting

        g = self._moments(sv, excess_returns)
        g -= g.mean(0)[None, :] if center else 0
        kernel: str | None = None
        bandwidth: float | None = None
        if cov_type not in COV_TYPES:
            raise unknown_option_error("cov_type", cov_type, COV_TYPES)
        if cov_type in ("robust", "heteroskedastic"):
            weight_est_instance = HeteroskedasticWeight(g, center=center)
            cov_est = HeteroskedasticCovariance
        else:  # "kernel":
            kernel = get_string(cov_config, "kernel")
            bandwidth = get_float(cov_config, "bandwidth")
            weight_est_instance = KernelWeight(
                g, center=center, kernel=kernel, bandwidth=bandwidth
            )
            cov_est = KernelCovariance

        w = weight_est_instance.w(g)

        args = (excess_returns, w)

        # 2. Step 1 using w = inv(s) from SV
        callback = callback_factory(self._j, args, disp=disp)
        options = {"maxiter": max_iter}
        opt_options = {} if opt_options is None else opt_options
        options.update(opt_options.get("options", {}))

        def minimize_kwargs(
            callback: Callable[[linearmodels.typing.data.Float64Array], None],
        ) -> dict[str, Any]:
            # The options of the user apply to every minimization
            return {"callback": callback, **opt_options, "options": options}

        opt_res = minimize(
            fun=self._j,
            x0=np.squeeze(sv),
            args=args,
            **minimize_kwargs(callback),
        )
        params = opt_res.x
        last_obj = opt_res.fun
        iters = 1
        # 3. Step 2 using step 1 estimates
        if not use_cue:
            while iters < steps:
                iters += 1
                g = self._moments(params, excess_returns)
                w = weight_est_instance.w(g)
                args = (excess_returns, w)

                # 2. Step 1 using w = inv(s) from SV
                callback = callback_factory(self._j, args, disp=disp)
                opt_res = minimize(
                    self._j,
                    params,
                    args=args,
                    **minimize_kwargs(callback),
                )
                params = opt_res.x
                obj = opt_res.fun
                if np.abs(obj - last_obj) < 1e-6:
                    break
                last_obj = obj

        else:
            cue_args = (excess_returns, weight_est_instance)
            callback = callback_factory(self._j_cue, cue_args, disp=disp)
            opt_res = minimize(
                self._j_cue,
                params,
                args=cue_args,
                **minimize_kwargs(callback),
            )
            params = opt_res.x

        # 4. Compute final S and G for inference
        g = self._moments(params, excess_returns)
        jac = self._jacobian(params, excess_returns)
        if cov_est is HeteroskedasticCovariance:
            cov_est_inst = HeteroskedasticCovariance(
                g,
                jacobian=jac,
                center=center,
                debiased=debiased,
                df=self.factors.shape[1],
            )
        else:
            cov_est_inst = KernelCovariance(
                g,
                jacobian=jac,
                center=center,
                debiased=debiased,
                df=self.factors.shape[1],
                kernel=kernel,
                bandwidth=bandwidth,
            )

        full_vcv = cov_est_inst.cov
        sel = slice((n * k), (n * k + k + nrf))
        rp = params[sel]
        rp_cov = full_vcv[sel, sel]
        sel = slice(0, (n * (k + 1)), (k + 1))
        alphas = g.mean(0)[sel, None]
        # The alphas are averages of moments, whose covariance at the estimates
        # accounts for the estimation of the parameters
        alpha_vcv = cov_est_inst._moment_cov[sel, sel]
        if use_cue:
            # The objective that was minimized updates the weight matrix
            stat = self._j_cue(params, excess_returns, weight_est_instance)
        else:
            stat = self._j(params, excess_returns, w)
        jstat = WaldTestStatistic(
            stat, "All alphas are 0", n - k - nrf, name="J-statistic"
        )

        # R2 calculation
        betas = np.reshape(params[: (n * k)], (n, k))
        resids = self.portfolios.ndarray - self.factors.ndarray @ betas.T
        resids -= resids.mean(0)[None, :]
        residual_ss = (resids**2).sum()
        total = self.portfolios.ndarray
        y_ss = (self.portfolios.ndarray**2).sum()
        total = total - total.mean(0)[None, :]
        total_ss = (total**2).sum()
        r2 = r_squared(residual_ss, total_ss, y_ss, self.portfolios.shape[0])
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.extend(
                [f"beta-{portfolio}-{factor}" for factor in self.factors.cols]
            )
        if not excess_returns:
            param_names.append("lambda-risk_free")
        param_names.extend([f"lambda-{f}" for f in self.factors.cols])
        param_names.extend([f"mu-{f}" for f in self.factors.cols])
        rp_names = list(self.factors.cols)[:]
        if not excess_returns:
            rp_names.insert(0, "risk_free")
        params = np.c_[alphas, betas]
        # 5. Return values
        res_dict = AttrDict(
            params=params,
            cov=full_vcv,
            betas=betas,
            rp=rp,
            rp_cov=rp_cov,
            alphas=alphas,
            alpha_vcv=alpha_vcv,
            jstat=jstat,
            rsquared=r2,
            total_ss=total_ss,
            residual_ss=residual_ss,
            param_names=param_names,
            portfolio_names=self.portfolios.cols,
            factor_names=self.factors.cols,
            name=self._name,
            cov_type=cov_type,
            model=self,
            nobs=nobs,
            rp_names=rp_names,
            iter=iters,
            cov_est=cov_est_inst,
        )

        return GMMFactorModelResults(res_dict)

    def _moments(
        self, parameters: linearmodels.typing.data.Float64Array, excess_returns: bool
    ) -> linearmodels.typing.data.Float64Array:
        """Calculate nobs by nmoments moment conditions"""
        nrf = int(not excess_returns)
        p = np.asarray(self.portfolios.ndarray, dtype=float)
        nobs, n = p.shape
        f = np.asarray(self.factors.ndarray, dtype=float)
        k = f.shape[1]
        s1, s2 = n * k, n * k + k + nrf
        betas = parameters[:s1]
        lam = parameters[s1:s2]
        mu = parameters[s2:]
        betas = np.reshape(betas, (n, k))
        expected = np.c_[np.ones((n, nrf)), betas] @ lam
        fe = f - mu.T
        eps = p - expected.T - fe @ betas.T
        f = np.column_stack((np.ones((nobs, 1)), f))
        f = np.tile(f, (1, n))
        eps = np.reshape(np.tile(eps, (k + 1, 1)).T, (n * (k + 1), nobs)).T
        g = np.c_[eps * f, fe]
        return g

    def _j(
        self,
        parameters: linearmodels.typing.data.Float64Array,
        excess_returns: bool,
        w: linearmodels.typing.data.Float64Array,
    ) -> float:
        """Objective function"""
        g = self._moments(parameters, excess_returns)
        nobs = self.portfolios.shape[0]
        gbar = g.mean(0)[:, None]
        return nobs * float(np.squeeze(gbar.T @ w @ gbar))

    def _j_cue(
        self,
        parameters: linearmodels.typing.data.Float64Array,
        excess_returns: bool,
        weight_est: HeteroskedasticWeight | KernelWeight,
    ) -> float:
        """CUE Objective function"""
        g = self._moments(parameters, excess_returns)
        gbar = g.mean(0)[:, None]
        nobs = self.portfolios.shape[0]
        w = weight_est.w(g)
        return nobs * float(np.squeeze(gbar.T @ w @ gbar))

    def _jacobian(
        self, params: linearmodels.typing.data.Float64Array, excess_returns: bool
    ) -> linearmodels.typing.data.Float64Array:
        """Jacobian matrix for inference"""
        nobs, k = self.factors.shape
        n = self.portfolios.shape[1]
        nrf = int(bool(not excess_returns))
        jac = np.zeros((n * k + n + k, params.shape[0]))
        s1, s2 = (n * k), (n * k) + k + nrf
        betas = params[:s1]
        betas = np.reshape(betas, (n, k))
        lam = params[s1:s2]
        mu = params[-k:]
        lam_tilde = lam if excess_returns else lam[1:]
        f = self.factors.ndarray
        fe = f - mu.T + lam_tilde.T
        f_aug = np.c_[np.ones((nobs, 1)), f]
        fef = f_aug.T @ fe / nobs
        r1 = n * (k + 1)
        jac[:r1, :s1] = np.kron(np.eye(n), fef)

        jac12 = np.zeros((r1, (k + nrf)))
        jac13 = np.zeros((r1, k))
        iota = np.ones((nobs, 1))
        for i in range(n):
            if excess_returns:
                b = betas[[i]]
            else:
                b = np.c_[[1], betas[[i]]]
            jac12[(i * (k + 1)) : (i + 1) * (k + 1)] = f_aug.T @ (iota @ b) / nobs

            b = betas[[i]]
            jac13[(i * (k + 1)) : (i + 1) * (k + 1)] = -f_aug.T @ (iota @ b) / nobs
        jac[:r1, s1:s2] = jac12
        jac[:r1, s2:] = jac13
        jac[-k:, -k:] = np.eye(k)

        return jac
