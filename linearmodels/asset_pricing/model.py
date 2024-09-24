"""
Linear factor models for applications in asset pricing
"""

from __future__ import annotations

from typing import Any, Callable, cast

from formulaic import model_matrix
from formulaic.materializers.types import NAAction
import numpy as np
from numpy.linalg import lstsq
import pandas
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
from linearmodels.iv.data import IVData, IVDataLike
from linearmodels.shared.exceptions import missing_warning
from linearmodels.shared.hypotheses import WaldTestStatistic
from linearmodels.shared.linalg import has_constant
from linearmodels.shared.typed_getters import get_float, get_string
from linearmodels.shared.utility import AttrDict
import linearmodels.typing.data


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
    d = {"iter": 0}
    disp = int(disp)

    def callback(params: linearmodels.typing.data.Float64Array) -> None:
        fval = obj(params, *args)
        if disp > 0 and (d["iter"] % disp == 0):
            print("Iteration: {}, Objective: {}".format(d["iter"], fval))
        d["iter"] += 1

    return callback


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
            linearmodels.typing.data.BoolArray,
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

        p = cast(linearmodels.typing.data.Float64Array, self.portfolios.ndarray)
        f = cast(linearmodels.typing.data.Float64Array, self.factors.ndarray)
        if has_constant(p)[0]:
            raise ValueError(
                "portfolios must not contains a constant or "
                "equivalent and must not have rank\n"
                "less than the dimension of the smaller shape."
            )
        if has_constant(f)[0]:
            raise ValueError("factors must not contain a constant or equivalent.")
        if np.linalg.matrix_rank(f) < f.shape[1]:
            raise ValueError(
                "Model cannot be estimated. factors do not have full column rank."
            )
        if p.shape[0] < (f.shape[1] + 1):
            raise ValueError(
                "Model cannot be estimated. portfolios must have factors + 1 or "
                "more returns to\nestimate the model parameters."
            )

    @property
    def formula(self) -> str | None:
        return self._formula

    @formula.setter
    def formula(self, value: str | None) -> None:
        self._formula = value

    @staticmethod
    def _prepare_data_from_formula(
        formula: str, data: pandas.DataFrame, portfolios: pandas.DataFrame | None
    ) -> tuple[DataFrame, DataFrame, str]:
        orig_formula = formula
        na_action = NAAction("raise")
        if portfolios is not None:
            factors_mm = model_matrix(
                formula + " + 0",
                data,
                context=0,  # TODO: self._eval_env,
                ensure_full_rank=True,
                na_action=na_action,
            )
            factors = DataFrame(factors_mm)
        else:
            formula_components = formula.split("~")
            portfolios_mm = model_matrix(
                formula_components[0].strip() + " + 0",
                data,
                context=0,  # TODO: self._eval_env,
                ensure_full_rank=False,
                na_action=na_action,
            )
            portfolios = DataFrame(portfolios_mm)
            factors_mm = model_matrix(
                formula_components[1].strip() + " + 0",
                data,
                context=0,  # TODO: self._eval_env,
                ensure_full_rank=False,
                na_action=na_action,
            )
            factors = DataFrame(factors_mm)

        return factors, portfolios, orig_formula


class TradedFactorModel(_FactorModelBase):
    r"""
    Linear factor models estimator applicable to traded factors

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio)
    factors : array_like
        Priced factor returns (nobs by nfactor)

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
    """

    def __init__(self, portfolios: IVDataLike, factors: IVDataLike):
        super().__init__(portfolios, factors)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pandas.DataFrame,
        *,
        portfolios: pandas.DataFrame | None = None,
    ) -> TradedFactorModel:
        """
        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array_like
            Portfolios to be used in the model

        Returns
        -------
        TradedFactorModel
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

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
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios
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
        Estimate model parameters

        Parameters
        ----------
        cov_type : str
            Name of covariance estimator
        debiased : bool
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        cov_config : dict
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        Supported covariance estimators are:

        * "robust" - Heteroskedasticity-robust covariance estimator
        * "kernel" - Heteroskedasticity and Autocorrelation consistent (HAC)
          covariance estimator

        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of "bartlett", "parzen" or "qs" (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """
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
            _cov_config = {k: v for k, v in cov_config.items()}
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
            raise ValueError(f"Unknown cov_type: {cov_type}")
        full_vcv = cov_est.cov
        rp_cov = rp_cov_est.cov
        vcv = full_vcv[:nloading, :nloading]

        # Rearrange VCV
        order = np.reshape(
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
        r2 = 1 - residual_ss / total_ss
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append(f"alpha-{portfolio}")
            for factor in self.factors.cols:
                param_names.append(f"beta-{portfolio}-{factor}")
        for factor in self.factors.cols:
            param_names.append(f"lambda-{factor}")

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
            self._sigma = np.asarray(sigma)
            vals, vecs = np.linalg.eigh(sigma)
            self._sigma_m12 = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
            self._sigma_inv = np.linalg.inv(self._sigma)

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
    Linear factor model estimator

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio)
    factors : array_like
        Priced factor returns (nobs by nfactor)
    risk_free : bool
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate.
    sigma : array_like
        Positive definite residual covariance (nportfolio by nportfolio)

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
    :math:`\hat{\alpha}_i=\hat{\eta}_i`.
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
        data: pandas.DataFrame,
        *,
        portfolios: pandas.DataFrame | None = None,
        risk_free: bool = False,
        sigma: linearmodels.typing.data.ArrayLike | None = None,
    ) -> LinearFactorModel:
        """
        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array_like
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate.
        sigma : array_like
            Positive definite residual covariance (nportfolio by nportfolio)

        Returns
        -------
        LinearFactorModel
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

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
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios
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
        Estimate model parameters

        Parameters
        ----------
        cov_type : str
            Name of covariance estimator
        debiased : bool
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        LinearFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of "bartlett", "parzen" or "qs" (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """
        nobs, nf, nport, nrf, s1, s2, s3 = self._boundaries()
        excess_returns = not self._risk_free
        f = self.factors.ndarray
        p = self.portfolios.ndarray
        nport = p.shape[1]

        # Step 1, n regressions to get B
        fc = np.c_[np.ones((nobs, 1)), f]
        b = lstsq(fc, p, rcond=None)[0]  # nf+1 by np
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

        if cov_type not in ("robust", "heteroskedastic", "kernel"):
            raise ValueError(f"Unknown weight: {cov_type}")
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
        stat = float(np.squeeze(alphas.T @ np.linalg.pinv(alpha_vcv) @ alphas))
        jstat = WaldTestStatistic(
            stat, "All alphas are 0", nport - nf - nrf, name="J-statistic"
        )

        total_ss = ((p - p.mean(0)[None, :]) ** 2).sum()
        residual_ss = (eps**2).sum()
        r2 = 1 - residual_ss / total_ss
        rp = lam
        rp_cov = full_vcv[s1:s2, s1:s2]
        betas = betas if excess_returns else betas[:, 1:]
        params = np.c_[alphas, betas]
        param_names = []
        for portfolio in self.portfolios.cols:
            param_names.append(f"alpha-{portfolio}")
            for factor in self.factors.cols:
                param_names.append(f"beta-{portfolio}-{factor}")
        if not excess_returns:
            param_names.append("lambda-risk_free")
        for factor in self.factors.cols:
            param_names.append(f"lambda-{factor}")

        # Pivot vcv to remove unnecessary and have correct order
        order = np.reshape(np.arange(s1), (nport, nf + 1))
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
        nobs, nf, nport, _, s1, s2, s3 = self._boundaries()
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
    GMM estimator of Linear factor models

    Parameters
    ----------
    portfolios : array_like
        Test portfolio returns (nobs by nportfolio)
    factors : array_like
        Priced factors values (nobs by nfactor)
    risk_free : bool
        Flag indicating whether the risk-free rate should be estimated
        from returns along other risk premia.  If False, the returns are
        assumed to be excess returns using the correct risk-free rate.

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

    The model is tested using the optimized objective function using the
    usual GMM J statistic.
    """

    def __init__(
        self, portfolios: IVDataLike, factors: IVDataLike, *, risk_free: bool = False
    ) -> None:
        super().__init__(portfolios, factors, risk_free=risk_free)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pandas.DataFrame,
        *,
        portfolios: pandas.DataFrame | None = None,
        risk_free: bool = False,
    ) -> LinearFactorModelGMM:
        """
        Parameters
        ----------
        formula : str
            Formula modified for the syntax described in the notes
        data : DataFrame
            DataFrame containing the variables used in the formula
        portfolios : array_like
            Portfolios to be used in the model. If provided, must use formula
            syntax containing only factors.
        risk_free : bool
            Flag indicating whether the risk-free rate should be estimated
            from returns along other risk premia.  If False, the returns are
            assumed to be excess returns using the correct risk-free rate.

        Returns
        -------
        LinearFactorModelGMM
            Model instance

        Notes
        -----
        The formula can be used in one of two ways.  The first specified only the
        factors and uses the data provided in ``portfolios`` as the test portfolios.
        The second specified the portfolio using ``+`` to separate the test portfolios
        and ``~`` to separate the test portfolios from the factors.

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
        factors, portfolios, formula = cls._prepare_data_from_formula(
            formula, data, portfolios
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
        Estimate model parameters

        Parameters
        ----------
        center : bool
            Flag indicating to center the moment conditions before computing
            the weighting matrix.
        use_cue : bool
            Flag indicating to use continuously updating estimator
        steps : int
            Number of steps to use when estimating parameters.  2 corresponds
            to the standard efficient GMM estimator. Higher values will
            iterate until convergence or up to the number of steps given
        disp : int
            Number of iterations between printed update. 0 or negative values
            suppresses output
        max_iter : int
            Maximum number of iterations when minimizing objective. Must be positive.
        cov_type : str
            Name of covariance estimator
        debiased : bool
            Flag indicating whether to debias the covariance estimator using
            a degree of freedom adjustment
        starting : array_like
            Starting values to use in optimization.  If not provided, 2SLS
            estimates are used.
        opt_options : dict
            Additional options to pass to scipy.optimize.minimize when
            optimizing the objective function. If not provided, defers to
            scipy to choose an appropriate optimizer. All minimize inputs
            except ``fun``, ``x0``, and ``args`` can be overridden.
        cov_config
            Additional covariance-specific options.  See Notes.

        Returns
        -------
        GMMFactorModelResults
            Results class with parameter estimates, covariance and test statistics

        Notes
        -----
        The kernel covariance estimator takes the optional arguments
        ``kernel``, one of "bartlett", "parzen" or "qs" (quadratic spectral)
        and ``bandwidth`` (a positive integer).
        """

        nobs, n = self.portfolios.shape
        k = self.factors.shape[1]
        excess_returns = not self._risk_free
        nrf = int(not bool(excess_returns))
        # 1. Starting Values - use 2 pass
        mod = LinearFactorModel(
            self.portfolios, self.factors, risk_free=self._risk_free
        )
        res = mod.fit()
        betas = np.asarray(res.betas).ravel()
        lam = np.asarray(res.risk_premia)
        mu = self.factors.ndarray.mean(0)
        sv = np.r_[betas, lam, mu][:, None]
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
        if cov_type not in ("robust", "heteroskedastic", "kernel"):
            raise ValueError(f"Unknown weight: {cov_type}")
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
        _default_options: dict[str, Any] = {"callback": callback}
        options = {"disp": bool(disp), "maxiter": max_iter}
        opt_options = {} if opt_options is None else opt_options
        options.update(opt_options.get("options", {}))
        _default_options.update(opt_options)
        _default_options["options"] = options
        opt_res = minimize(
            fun=self._j,
            x0=np.squeeze(sv),
            args=args,
            **_default_options,
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
                    callback=callback,
                    options={"disp": bool(disp), "maxiter": max_iter},
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
                callback=callback,
                options={"disp": bool(disp), "maxiter": max_iter},
            )
            params = opt_res.x

        # 4. Compute final S and G for inference
        g = self._moments(params, excess_returns)
        s = g.T @ g / nobs
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
        alpha_vcv = s[sel, sel] / nobs
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
        total = total - total.mean(0)[None, :]
        total_ss = (total**2).sum()
        r2 = 1.0 - residual_ss / total_ss
        param_names = []
        for portfolio in self.portfolios.cols:
            for factor in self.factors.cols:
                param_names.append(f"beta-{portfolio}-{factor}")
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
