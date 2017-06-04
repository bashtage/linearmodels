"""
Estimators for systems of equations

References
----------
Greene, William H. "Econometric analysis 4th edition." International edition,
    New Jersey: Prentice Hall (2000).
StataCorp. 2013. Stata 13 Base Reference Manual. College Station, TX: Stata
    Press.
Henningsen, A., & Hamann, J. (2007). systemfit: A Package for Estimating
    Systems of Simultaneous Equations in R. Journal of Statistical Software,
    23(4), 1 - 40. doi:http://dx.doi.org/10.18637/jss.v023.i04
"""
from collections import Mapping, OrderedDict

import numpy as np
from numpy import (asarray, cumsum, diag, eye, hstack, inf, ones_like, prod,
                   reshape, sqrt, vstack, zeros)
from numpy.linalg import inv, lstsq, solve
from pandas import Series
from patsy.highlevel import dmatrices
from patsy.missing import NAAction

from linearmodels.iv.data import IVData
from linearmodels.system._utility import (blocked_column_product,
                                          blocked_diag_product, inv_matrix_sqrt)
from linearmodels.system.covariance import (HeteroskedasticCovariance,
                                            HomoskedasticCovariance)
from linearmodels.system.results import SURResults
from linearmodels.utility import (AttrDict, WaldTestStatistic, has_constant,
                                  matrix_rank)

__all__ = ['SUR']

UNKNOWN_EQ_TYPE = """
Contents of each equation must be either a dictionary with keys 'dependent'
and 'exog' or a 2-element tuple of he form (dependent, exog).
equations[{key}] was {type}
"""


class SUR(object):
    r"""
    Seemingly unrelated regression estimation (SUR/SURE)

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent and exogenous variable
        values.  Each element must be either a tuple of the form (dependent,
        exog, [weights]) or a dictionary with keys 'dependent' and 'exog' and
        the optional key 'weights'
    sigma : array-like
        Pre-specified residual covariance to use in GLS estimation. If not
        provided, FGLS is implemented based on an estimate of sigma.

    Notes
    -----
    Estimates a set of regressions which are seemingly unrelated in the sense
    that separate estimation would lead to consistent parameter estiamtes.
    Each equation is of the form

    .. math::

        y_{i,k} = x_{i,k}\beta_i + \epsilon_{i,k}

    where k denotes the equation and i denoted the observation index. By
    stacking vertically arrays of dependent and placing the exogenous
    variables into a block diagonal array, the entire system can be compactly
    expressed as

    .. math::

        Y = X\beta + \epsilon

    where

    .. math::

        Y = \left[\begin{array}{x}Y_1 \\ Y_2 \\ \vdots \\ Y_K\end{array}\right]

    and

    .. math::

        X = \left[\begin{array}{cccc}
                 X_1 & 0 & \ldots & 0 \\
                 0 & X_2 & \dots & 0 \\
                 \vdots & \vdots & \ddots & \vdots \\
                 0 & 0 & \dots & X_K
            \end{array}\right]

    The system OLS estimator is

    .. math::

        \hat{\beta}_{OLS} = (X'X)^{-1}X'Y

    When certain conditions are satisfied, a GLS estimator of the form

    .. math::

        \hat{\beta}_{GLS} = (X'\Omega^{-1}X)^{-1}X'\Omega^{-1}Y

    can improve accuracy of coefficient estimates where

    .. math::

        \Omega = \Sigma \otimes I_n

    where :math:`\Sigma` is the covariance matrix of the residuals.
    """

    def __init__(self, equations, *, sigma=None):
        if not isinstance(equations, Mapping):
            raise TypeError('dependent must be a dictionary-like')

        # Ensure nearly deterministic equation ordering
        if not isinstance(equations, OrderedDict):
            pairs = [(str(key), key) for key in equations]
            pairs = sorted(pairs, key=lambda p: p[0])
            ordered_eqn = OrderedDict()
            for key, value in pairs:
                ordered_eqn[value] = equations[value]
            equations = ordered_eqn

        self._equations = equations
        self._sigma = asarray(sigma) if sigma is not None else None
        self._param_names = []
        self._eq_labels = []
        self._dependent = []
        self._exog = []
        self._weights = []
        self._constraints = []
        self._constant_loc = None
        self._has_constant = None
        self._common_exog = False
        self._validate_data()

    def _validate_data(self):
        ids = []
        for i, key in enumerate(self._equations):
            self._eq_labels.append(key)
            eq_data = self._equations[key]
            dep_name = 'dependent_' + str(i)
            exog_name = 'exog_' + str(i)
            if isinstance(eq_data, (tuple, list)):
                self._dependent.append(IVData(eq_data[0], var_name=dep_name))
                ids.append(id(eq_data[1]))
                self._exog.append(IVData(eq_data[1], var_name=exog_name))
                if len(eq_data) == 3:
                    self._weights.append(IVData(eq_data[2]))
                else:
                    dep = self._dependent[-1].ndarray
                    self._weights.append(ones_like(dep))

            elif isinstance(eq_data, dict):
                self._dependent.append(IVData(eq_data['dependent'], var_name=dep_name))
                ids.append(id(eq_data['exog']))
                self._exog.append(IVData(eq_data['exog'], var_name=exog_name))
                if 'weights' in eq_data:
                    self._weights.append(IVData(eq_data['weights']))
                else:
                    dep = self._dependent[-1].ndarray
                    self._weights.append(ones_like(dep))

            else:
                msg = UNKNOWN_EQ_TYPE.format(key=key, type=type(vars))
                raise TypeError(msg)

        self._common_exog = len(set(ids)) == 1
        constant = []
        constant_loc = []
        for lhs, rhs in zip(self._dependent, self._exog):
            self._param_names.extend([lhs.cols[0] + '_' + col for col in rhs.cols])
            rhs_a = rhs.ndarray
            lhs_a = lhs.ndarray
            if lhs_a.shape[0] != rhs_a.shape[0]:
                raise ValueError('Dependent and exogenous do not have the same'
                                 ' number of observations')
            if lhs_a.shape[0] <= rhs_a.shape[1]:
                raise ValueError('Fewer observations than variables')
            if matrix_rank(rhs_a) < rhs_a.shape[1]:
                raise ValueError('Exogenous variable arrays are not all full '
                                 'rank')
            const, const_loc = has_constant(rhs_a)
            constant.append(const)
            constant_loc.append(const_loc)
        self._has_constant = Series(constant,
                                    index=[d.cols[0] for d in self._dependent])
        self._constant_loc = constant_loc

    @property
    def has_constant(self):
        """Vector indicating which equations contain constants"""
        return self._has_constant

    @classmethod
    def multivariate_ls(cls, dependent, exog):
        """
        Interface for specification of multivariate regression models

        Parameters
        ----------
        dependent : array-like
            nobs by ndep array of dependent variables
        exog : array-like
            nobs by nvar array of exogenous regressors common to all models

        Returns
        -------
        model : SUR
            Model instance

        Notes
        -----
        Utility function to simplify the construction of multivariate
        regression models which all use the same regressors. Constructs
        the dictionary of equations from the variables using the common
        exogenous variable.

        Examples
        --------
        A simple CAP-M can be estimated as a multivariate regression

        >>> from linearmodels.datasets import french
        >>> data = french.load()
        >>> portfolios = data[['S1V1','S1V5','S5V1','S5V5']]
        >>> factors = data['MktRf'].copy()
        >>> factors['alpha'] = 1
        >>> mod = SUR.multivariate_ls(portfolios, factors)
        """
        equations = OrderedDict()
        dependent = IVData(dependent, var_name='dependent')
        exog = IVData(exog, var_name='exog')
        for col in dependent.pandas:
            equations[col] = (dependent.pandas[[col]], exog.pandas)
        return cls(equations)

    @classmethod
    def from_formula(cls, formula, data, *, sigma=None):
        """
        Parameters
        ----------
        formula : {str, dict-like}
            Either a string or a dictionary of strings where each value in
            the dictionary represents a single equation. See Notes for a
            description of the accepted syntax
        data : DataFrame
            Frame containing named variables
        sigma : array-like
            Pre-specified residual covariance to use in GLS estimation. If not
            provided, FGLS is implemented based on an estimate of sigma.

        Returns
        -------
        model : SUR
            Model instance

        Notes
        -----
        Models can be specified in one of two ways. The first uses curly
        braces to encapsulate equations.  The second uses a dictionary
        where each key is an equation name.

        Examples
        --------
        The simplest format uses standard Patsy formulas for each equation
        in a dictionary.  Best practice is to use an Ordered Dictionary

        >>> formula = {'eq1': 'y1 ~ 1 + x1_1', 'eq2': 'y2 ~ 1 + x2_1'}
        >>> mod = SUR.from_formula(formula, data)

        The second format uses curly braces {} to surround distinct equations

        >>> formula = '{y1 ~ 1 + x1_1} {y2 ~ 1 + x2_1}'
        >>> mod = SUR.from_formula(formula, data)
        """
        na_action = NAAction(on_NA='raise', NA_types=[])
        if not isinstance(formula, (Mapping, str)):
            raise TypeError('formula must be a string or dictionary-like')

        eqns = OrderedDict()
        if isinstance(formula, Mapping):
            for key in formula:
                f = formula[key]
                f = '~ 0 +'.join(f.split('~'))
                dep, exog = dmatrices(f, data, return_type='dataframe',
                                      NA_action=na_action)
                eqns[key] = {'dependent': dep, 'exog': exog}
            return SUR(eqns, sigma=sigma)

        formula = formula.replace('\n', ' ').strip()
        parts = formula.split('}')
        for i, part in enumerate(parts):
            part = part.strip()
            if part == '':
                continue
            part = part.replace('{', '')
            f = '~ 0 +'.join(part.split('~'))
            dep, exog = dmatrices(f, data, return_type='dataframe',
                                  NA_action=na_action)
            base_key = key = f.split('~')[0].strip()
            count = 0
            while key in eqns:
                key = base_key + '.{0}'.format(count)
                count += 1
            eqns[key] = {'dependent': dep, 'exog': exog}
        return SUR(eqns, sigma=sigma)

    def _multivariate_ls_fit(self):
        dep, exog = self._dependent, self._exog
        k = len(dep)
        eps = []
        full_y = []
        total_cols = 0
        ci = [0]
        beta0 = []
        for i in range(k):
            x = exog[i].ndarray
            y = dep[i].ndarray
            full_y.append(y)
            total_cols += x.shape[1]
            ci.append(total_cols)
            b = lstsq(x, y)[0]
            eps.append(y - x @ b)
            beta0.append(b)
        beta = vstack(beta0)
        eps = hstack(eps)
        return beta, eps

    @staticmethod
    def _f_stat(stats, debiased):
        cov = stats.cov
        nvar = k = cov.shape[0]
        sel = list(range(k))
        if stats.has_constant:
            sel.pop(stats.constant_loc)
        cov = cov[sel][:, sel]
        params = stats.params[sel]
        stat = float(params.T @ inv(cov) @ params)
        df = params.shape[0]
        nobs = stats.nobs
        null = 'All parameters ex. constant are zero'
        name = 'Equation F-statistic'

        if debiased:
            wald = WaldTestStatistic(stat / df, null, df, nobs - nvar,
                                     name=name)
        else:
            return WaldTestStatistic(stat, null=null, df=df, name=name)

        return wald

    def _multivariate_ls_finalize(self, beta, eps, cov_type, **cov_config):
        nobs = eps.shape[0]
        sigma = eps.T @ eps / nobs
        k = len(self._dependent)

        # Covariance estimation
        x = [ex.ndarray for ex in self._exog]
        if cov_type == 'unadjusted':
            cov_est = HomoskedasticCovariance
        else:
            cov_est = HeteroskedasticCovariance
        cov = cov_est(x, eps, sigma, gls=False, **cov_config).cov

        individual = AttrDict()
        loc = 0
        debiased = cov_config.get('debiased', False)
        for i in range(k):
            stats = AttrDict()
            stats['eq_label'] = self._eq_labels[i]
            stats['dependent'] = self._dependent[i].cols[0]
            stats['method'] = 'OLS'
            stats['cov_type'] = cov_type
            stats['index'] = self._dependent[i].rows
            stats['iter'] = 0

            y = self._dependent[i].ndarray
            x = self._exog[i].ndarray
            nobs, df = x.shape
            b = beta[loc:loc + df]
            stats['params'] = b

            e = eps[:, [i]]

            stats['resid'] = e
            stats['wresid'] = e
            stats['nobs'] = e.shape[0]
            stats['df_model'] = df
            stats['resid_ss'] = float(e.T @ e)
            stats['debiased'] = debiased

            ye = y
            cons = int(self.has_constant.iloc[i])
            if cons:
                ye = y - y.mean()
            stats['total_ss'] = float(ye.T @ ye)
            stats['r2'] = 1.0 - stats.resid_ss / stats.total_ss
            stats['r2a'] = 1.0 - (stats.resid_ss / (nobs - df)) / (stats.total_ss / (nobs - cons))
            stats['has_constant'] = bool(cons)
            stats['constant_loc'] = self._constant_loc[i]
            stats['cov'] = cov[loc:loc + df, loc:loc + df]
            stats['f_stat'] = self._f_stat(stats, debiased)
            names = self._param_names[loc:loc + df]
            offset = len(stats.dependent) + 1
            stats['param_names'] = [n[offset:] for n in names]
            key = self._dependent[i].cols[0]
            individual[key] = stats

            loc += df
        results = AttrDict()
        results['method'] = 'OLS'
        results['iter'] = 0
        results['cov_type'] = cov_type
        results['index'] = self._dependent[0].rows
        results['nobs'] = prod(eps.shape)
        results['individual'] = individual
        results['params'] = beta
        results['df_model'] = beta.shape[0]
        results['param_names'] = self._param_names
        results['cov'] = cov
        results['sigma'] = sigma
        results['debiased'] = debiased

        total_ss = resid_ss = 0.0
        resid = []
        for key in individual:
            total_ss += individual[key].total_ss
            resid_ss += individual[key].resid_ss
            resid.append(individual[key].resid)

        resid = hstack(resid)
        results['r2'] = 1.0 - resid_ss / total_ss
        results['resid_ss'] = resid_ss
        results['total_ss'] = total_ss
        results['resid'] = resid
        results['wresid'] = resid
        return SURResults(results)

    def _gls_finalize(self, beta, sigma, gls_eps, eps, cov_type, iter_count, **cov_config):
        k = len(self._dependent)

        # Covariance estimation
        x = [ex.ndarray for ex in self._exog]
        if cov_type == 'unadjusted':
            cov_est = HomoskedasticCovariance
        else:
            cov_est = HeteroskedasticCovariance
        gls_eps = reshape(gls_eps, (k, gls_eps.shape[0] // k)).T
        eps = reshape(eps, (k, eps.shape[0] // k)).T
        cov = cov_est(x, gls_eps, sigma, gls=True, **cov_config).cov

        # TODO: Pass in, DRY
        sigma_m12 = inv_matrix_sqrt(sigma)
        ys = [dep.ndarray for dep in self._dependent]
        cs = [ones_like(dep.ndarray) for dep in self._dependent]
        block_y = blocked_column_product(ys, sigma_m12)
        block_c = blocked_diag_product(cs, sigma_m12)
        const_gls_eps = block_y - block_c @ lstsq(block_c, block_y)[0]

        individual = AttrDict()
        loc = 0
        debiased = cov_config.get('debiased', False)
        for i in range(k):
            stats = AttrDict()
            stats['eq_label'] = self._eq_labels[i]
            stats['dependent'] = self._dependent[i].cols[0]
            stats['method'] = 'GLS'
            stats['cov_type'] = cov_type
            stats['index'] = self._dependent[i].rows
            stats['iter'] = iter_count

            x = self._exog[i].ndarray
            nobs, df = x.shape
            b = beta[loc:loc + df]

            stats['params'] = b
            sel = slice(i * nobs, (i + 1) * nobs)
            e = gls_eps[:, [i]]
            stats['wresid'] = e
            stats['nobs'] = e.shape[0]
            stats['df_model'] = df
            stats['resid'] = eps[:, [i]]
            stats['resid_ss'] = float(e.T @ e)
            stats['debiased'] = debiased

            cons = int(self.has_constant.iloc[i])
            if cons:
                ye = const_gls_eps[sel]
            else:
                ye = block_y[sel]

            stats['total_ss'] = float(ye.T @ ye)
            stats['r2'] = 1.0 - stats.resid_ss / stats.total_ss
            stats['r2a'] = 1.0 - (stats.resid_ss / (nobs - df)) / (stats.total_ss / (nobs - cons))
            stats['has_constant'] = bool(cons)
            stats['constant_loc'] = self._constant_loc[i]
            stats['cov'] = cov[loc:loc + df, loc:loc + df]
            stats['f_stat'] = self._f_stat(stats, debiased)
            names = self._param_names[loc:loc + df]
            offset = len(stats.dependent) + 1
            stats['param_names'] = [n[offset:] for n in names]
            key = self._dependent[i].cols[0]
            individual[key] = stats

            loc += df

        results = AttrDict()
        results['method'] = 'GLS'
        results['cov_type'] = cov_type
        results['iter'] = iter_count
        results['index'] = self._dependent[0].rows
        results['nobs'] = block_y.shape[0]
        results['sigma'] = sigma
        results['individual'] = individual
        results['params'] = beta
        results['df_model'] = beta.shape[0]
        results['param_names'] = self._param_names
        results['cov'] = cov
        results['sigma'] = sigma
        results['debiased'] = debiased

        total_ss = resid_ss = 0.0
        resid = []
        wresid = []
        for key in individual:
            total_ss += individual[key].total_ss
            resid_ss += individual[key].resid_ss
            resid.append(individual[key].resid)
            wresid.append(individual[key].wresid)

        resid = hstack(resid)
        wresid = hstack(wresid)
        results['r2'] = 1.0 - resid_ss / total_ss
        results['resid_ss'] = resid_ss
        results['total_ss'] = total_ss
        results['resid'] = resid
        results['wresid'] = wresid

        return SURResults(results)

    def _gls_estimate(self, eps, nobs, total_cols, k, ci, full_cov):
        sigma = self._sigma
        if sigma is None:
            sigma = eps.T @ eps / nobs
        if not full_cov:
            sigma = diag(diag(sigma))
        sigma_inv = inv(sigma)
        xpx = zeros((total_cols, total_cols))
        xpy = zeros((total_cols, 1))
        for i in range(k):
            xi = self._exog[i].ndarray
            for j in range(i, k):
                xj = self._exog[j].ndarray
                yj = self._dependent[j].ndarray
                s_ij = sigma_inv[i, j]
                prod = s_ij * (xi.T @ xj)
                xpx[ci[i]:ci[i + 1], ci[j]:ci[j + 1]] = prod
                xpx[ci[j]:ci[j + 1], ci[i]:ci[i + 1]] = prod.T
                xpy[ci[i]:ci[i + 1]] += s_ij * (xi.T @ yj)

        for i in range(k):
            sy = zeros((nobs, 1))
            for j in range(k):
                sy += sigma_inv[i, j] * self._dependent[j].ndarray
            xpy[ci[i]:ci[i + 1]] = self._exog[i].ndarray.T @ sy

        beta = solve(xpx, xpy)
        loc = 0
        epss = []
        for j in range(k):
            x = self._exog[j].ndarray
            y = self._dependent[j].ndarray
            kx = x.shape[1]
            epss.append(y - x @ beta[loc:loc + kx])
            loc += kx

        eps = hstack(epss)

        return beta, eps, sigma

    def fit(self, *, use_gls=True, full_cov=True, iterate=False, iter_limit=100, tol=1e-6,
            cov_type='robust', **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        use_gls : bool
            Flag indicating whether to use gls
        full_cov : bool
            Flag indicating whether to utilize information in correlations
            when estimating the model with GLS
        iterate : bool
            Flag indicating to iterate GLS until convergence of iter limit
            iterations have been completed
        iter_limit : int
            Maximum number of iterations for iterative GLS
        tol : float
            Tolerance to use when checking for convergence in iterative GLS
        cov_type : str
            Name of covariance estimator. Valid options are

            * 'unadjusted', 'homoskedastic' - Classic covariance estimator
            * 'robust', 'heteroskedastic' - Heteroskedasticit robust
              covariance estimator

        **cov_config
            Additional parameters to pass to covariance estimator. All
            estimators support debiased which employs a small-sample adjustment

        Returns
        -------
        results : SURResults
            Estimation results
        """
        cov_type = cov_type.lower()
        if cov_type not in ('unadjusted', 'robust', 'homoskedastic', 'heteroskedastic'):
            raise ValueError('Unknown cov_type: {0}'.format(cov_type))
        cov_type = 'unadjusted' if cov_type in ('unadjusted', 'homoskedastic') else 'robust'
        k = len(self._dependent)
        col_sizes = [0] + list(map(lambda v: v.ndarray.shape[1], self._exog))
        ci = cumsum(col_sizes)
        total_cols = ci[-1]
        beta0, eps = self._multivariate_ls_fit()
        if not use_gls or self._common_exog:
            return self._multivariate_ls_finalize(beta0, eps, cov_type, **cov_config)

        beta_hist = [beta0]
        nobs = eps.shape[0]
        iter_count = 0
        delta = inf
        while ((iter_count < iter_limit and iterate) or iter_count == 0) and delta > tol:
            beta, eps, sigma = self._gls_estimate(eps, nobs, total_cols, k, ci, full_cov)
            beta_hist.append(beta)
            delta = beta_hist[-1] - beta_hist[-2]
            delta = sqrt(np.mean(delta ** 2))
            iter_count += 1

        sigma_m12 = inv_matrix_sqrt(sigma)
        ys = [dep.ndarray for dep in self._dependent]
        xs = [exog.ndarray for exog in self._exog]
        y = blocked_column_product(ys, sigma_m12)
        x = blocked_diag_product(xs, sigma_m12)
        gls_eps = y - x @ beta

        y = blocked_column_product(ys, eye(k))
        x = blocked_diag_product(xs, eye(k))
        eps = y - x @ beta

        return self._gls_finalize(beta, sigma, gls_eps, eps, cov_type, iter_count, **cov_config)

    @property
    def constraints(self):
        """Model constraints"""
        return self._constraints

    def add_constraints(self, constraints):
        """
        Parameters
        ----------
        constraints :
            Constraints to add to the model
        """
        raise NotImplementedError
        self._constraints.append(constraints)

    def reset_constraints(self):
        """Remove all model constraints"""
        self._constraints = []

    @property
    def param_names(self):
        """Model parameter names"""
        return self._param_names
