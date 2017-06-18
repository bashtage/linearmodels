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
from numpy import (asarray, cumsum, diag, eye, hstack, inf, nanmean,
                   ones_like, reshape, sqrt, zeros)
from numpy.linalg import inv, solve
from pandas import Series
from patsy.highlevel import dmatrices
from patsy.missing import NAAction

from linearmodels.iv.data import IVData
from linearmodels.system._utility import LinearConstraint, blocked_column_product, \
    blocked_diag_product, blocked_inner_prod, inv_matrix_sqrt
from linearmodels.system.covariance import (HeteroskedasticCovariance,
                                            HomoskedasticCovariance)
from linearmodels.system.results import SURResults
from linearmodels.utility import (AttrDict, WaldTestStatistic, has_constant,
                                  matrix_rank, missing_warning, InvalidTestStatistic)

__all__ = ['SUR']

UNKNOWN_EQ_TYPE = """
Contents of each equation must be either a dictionary with keys 'dependent'
and 'exog' or a 2-element tuple of he form (dependent, exog).
equations[{key}] was {type}
"""


def _missing_weights(keys):
    if not keys:
        return
    import warnings
    msg = 'Weights not for for equation labels:\n{0}'.format(', '.join(keys))
    warnings.warn(msg, UserWarning)


class SUR(object):
    r"""
    Seemingly unrelated regression estimation (SUR/SURE)

    Parameters
    ----------
    equations : dict
        Dictionary-like structure containing dependent and exogenous variable
        values.  Each key is an equations label and must be a string. Each
        value must be either a tuple of the form (dependent,
        exog, [weights]) or a dictionary with keys 'dependent' and 'exog' and
        the optional key 'weights'.
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
        for key in equations:
            if not isinstance(key, str):
                raise ValueError('Equation lebels (keys) must be strings')

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
        self._y = []
        self._x = []
        self._wy = []
        self._wx = []
        self._w = []

        self._weights = []
        self._constraints = None
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
                    self._weights.append(IVData(ones_like(dep)))

            elif isinstance(eq_data, dict):
                self._dependent.append(IVData(eq_data['dependent'], var_name=dep_name))
                ids.append(id(eq_data['exog']))
                self._exog.append(IVData(eq_data['exog'], var_name=exog_name))
                if 'weights' in eq_data:
                    self._weights.append(IVData(eq_data['weights']))
                else:
                    dep = self._dependent[-1].ndarray
                    self._weights.append(IVData(ones_like(dep)))

            else:
                msg = UNKNOWN_EQ_TYPE.format(key=key, type=type(vars))
                raise TypeError(msg)
        for lhs, rhs in zip(self._dependent, self._exog):
            rhs_a = rhs.ndarray
            lhs_a = lhs.ndarray
            if lhs_a.shape[0] != rhs_a.shape[0]:
                raise ValueError('Dependent and exogenous do not have the same'
                                 ' number of observations')

        self._drop_missing()
        self._common_exog = len(set(ids)) == 1
        constant = []
        constant_loc = []
        for lhs, rhs, label in zip(self._dependent, self._exog, self._eq_labels):
            self._param_names.extend([label + '_' + col for col in rhs.cols])
            rhs_a = rhs.ndarray
            lhs_a = lhs.ndarray
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

        for dep, exog, w in zip(self._dependent, self._exog, self._weights):
            y = dep.ndarray
            x = exog.ndarray
            w = w.ndarray
            w = w / nanmean(w)
            w_sqrt = np.sqrt(w)
            self._w.append(w)
            self._y.append(y)
            self._x.append(x)
            self._wy.append(y * w_sqrt)
            self._wx.append(x * w_sqrt)

    def _drop_missing(self):
        k = len(self._dependent)
        nobs = self._dependent[0].shape[0]
        missing = np.zeros(nobs, dtype=np.bool)

        for i in range(k):
            missing |= self._dependent[i].isnull
            missing |= self._exog[i].isnull
            missing |= self._weights[i].isnull

        missing_warning(missing)
        if np.any(missing):
            for i in range(k):
                self._dependent[i].drop(missing)
                self._exog[i].drop(missing)
                self._weights[i].drop(missing)

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
    def from_formula(cls, formula, data, *, sigma=None, weights=None):
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
            Pre-specified residual covariance to use in GLS estimation. If
            not provided, FGLS is implemented based on an estimate of sigma.
        weights : dict-like
            Dictionary like object (e.g. a DataFrame) containing variable
            weights.  Each entry must have the same number of observations as
            data.  If an equation label is not a key weights, the weights will
            be set to unity

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

        It is also possible to include equation labels when using curly braces

        >>> formula = '{eq1: y1 ~ 1 + x1_1} {eq2: y2 ~ 1 + x2_1}'
        >>> mod = SUR.from_formula(formula, data)
        """
        na_action = NAAction(on_NA='raise', NA_types=[])
        if not isinstance(formula, (Mapping, str)):
            raise TypeError('formula must be a string or dictionary-like')

        missing_weight_keys = []
        eqns = OrderedDict()
        if isinstance(formula, Mapping):
            for key in formula:
                f = formula[key]
                f = '~ 0 +'.join(f.split('~'))
                dep, exog = dmatrices(f, data, return_type='dataframe',
                                      NA_action=na_action)
                eqns[key] = {'dependent': dep, 'exog': exog}
                if weights is not None:
                    if key in weights:
                        eqns[key]['weights'] = weights[key]
                    else:
                        missing_weight_keys.append(key)
            _missing_weights(missing_weight_keys)
            return SUR(eqns, sigma=sigma)

        formula = formula.replace('\n', ' ').strip()
        parts = formula.split('}')
        for i, part in enumerate(parts):
            base_key = None
            part = part.strip()
            if part == '':
                continue
            part = part.replace('{', '')
            if ':' in part.split('~')[0]:
                base_key, part = part.split(':')
                key = base_key = base_key.strip()
                part = part.strip()
            f = '~ 0 +'.join(part.split('~'))
            dep, exog = dmatrices(f, data, return_type='dataframe',
                                  NA_action=na_action)
            if base_key is None:
                base_key = key = f.split('~')[0].strip()
            count = 0
            while key in eqns:
                key = base_key + '.{0}'.format(count)
                count += 1
            eqns[key] = {'dependent': dep, 'exog': exog}
            if weights is not None:
                if key in weights:
                    eqns[key]['weights'] = weights[key]
                else:
                    missing_weight_keys.append(key)

        _missing_weights(missing_weight_keys)

        return SUR(eqns, sigma=sigma)

    def _multivariate_ls_fit(self):
        wy, wx = self._wy, self._wx
        k = len(wx)
        if self.constraints is not None:
            cons = self.constraints
            x = blocked_diag_product(wx, eye(len(wx)))
            y = np.vstack(wy)
            xt = x @ cons.t
            # TODO: Make more memory efficient
            # Replace with t.T @ xpx @ t
            xpx = xt.T @ xt
            # Replace with t.T @ xpy - t.T @ xpx @ a.T
            xpy = xt.T @ (y - x @ cons.a.T)
            paramsc = np.linalg.solve(xpx, xpy)
            params = cons.t @ paramsc + cons.a.T
        else:
            xpx = blocked_inner_prod(wx, eye(len(wx)))
            xpy = []
            for i in range(k):
                xpy.append(wx[i].T @ wy[i])
            xpy = np.vstack(xpy)
            xpx, xpy = xpx, xpy
            params = solve(xpx, xpy)

        beta = params
        loc = 0
        eps = []
        for i in range(k):
            nb = wx[i].shape[1]
            b = beta[loc:loc + nb]
            eps.append(wy[i] - wx[i] @ b)
            loc += nb
        eps = hstack(eps)

        return beta, eps

    def _f_stat(self, stats, debiased):
        cov = stats.cov
        k = cov.shape[0]
        sel = list(range(k))
        if stats.has_constant:
            sel.pop(stats.constant_loc)
        cov = cov[sel][:, sel]
        params = stats.params[sel]
        df = params.shape[0]
        nobs = stats.nobs
        null = 'All parameters ex. constant are zero'
        name = 'Equation F-statistic'
        try:
            stat = float(params.T @ inv(cov) @ params)

        except np.linalg.LinAlgError:
            return InvalidTestStatistic('Covariance is singular, possibly due '
                                        'to constraints.', name=name)

        if debiased:
            total_reg = np.sum(list(map(lambda s: s.shape[1], self._wx)))
            df_denom = len(self._wx) * nobs - total_reg
            wald = WaldTestStatistic(stat / df, null, df, df_denom=df_denom,
                                     name=name)
        else:
            return WaldTestStatistic(stat, null=null, df=df, name=name)

        return wald

    def _multivariate_ls_finalize(self, beta, eps, sigma, cov_type, **cov_config):
        k = len(self._wx)

        # Covariance estimation
        if cov_type == 'unadjusted':
            cov_est = HomoskedasticCovariance
        else:
            cov_est = HeteroskedasticCovariance
        cov = cov_est(self._wx, eps, sigma, sigma, gls=False,
                      constraints=self._constraints, **cov_config).cov

        individual = AttrDict()
        debiased = cov_config.get('debiased', False)
        for i in range(k):
            wy = wye = self._wy[i]
            w = self._w[i]
            cons = int(self.has_constant.iloc[i])
            if cons:
                wc = np.ones_like(wy) * np.sqrt(w)
                wye = wy - wc @ np.linalg.lstsq(wc, wy)[0]
            total_ss = float(wye.T @ wye)

            stats = self._common_indiv_results(i, beta, cov, eps, eps, 'OLS',
                                               cov_type, 0, debiased, cons, total_ss)
            key = self._eq_labels[i]
            individual[key] = stats

        nobs = eps.size
        results = self._common_results(beta, cov, 'OLS', 0, nobs, cov_type,
                                       sigma, individual, debiased)
        results['wresid'] = results.resid

        return SURResults(results)

    def _common_indiv_results(self, index, beta, cov, wresid, resid, method,
                              cov_type, iter_count, debiased, constant, total_ss):
        loc = 0
        for i in range(index):
            loc += self._wx[i].shape[1]
        i = index
        stats = AttrDict()
        # Static properties
        stats['eq_label'] = self._eq_labels[i]
        stats['dependent'] = self._dependent[i].cols[0]
        stats['method'] = method
        stats['cov_type'] = cov_type
        stats['index'] = self._dependent[i].rows
        stats['iter'] = iter_count
        stats['debiased'] = debiased
        stats['has_constant'] = bool(constant)
        stats['constant_loc'] = self._constant_loc[i]

        # Parameters, errors and measures of fit
        wxi = self._wx[i]
        nobs, df = wxi.shape
        b = beta[loc:loc + df]
        e = wresid[:, [i]]
        nobs = e.shape[0]
        df_c = (nobs - constant)
        df_r = (nobs - df)

        stats['params'] = b
        stats['cov'] = cov[loc:loc + df, loc:loc + df]
        stats['wresid'] = e
        stats['nobs'] = nobs
        stats['df_model'] = df
        stats['resid'] = resid[:, [i]]
        stats['resid_ss'] = float(resid[:, [i]].T @ resid[:, [i]])
        stats['total_ss'] = total_ss
        stats['r2'] = 1.0 - stats.resid_ss / stats.total_ss
        stats['r2a'] = 1.0 - (stats.resid_ss / df_r) / (stats.total_ss / df_c)

        names = self._param_names[loc:loc + df]
        offset = len(stats.eq_label) + 1
        stats['param_names'] = [n[offset:] for n in names]

        # F-statistic
        stats['f_stat'] = self._f_stat(stats, debiased)

        return stats

    def _common_results(self, beta, cov, method, iter_count, nobs, cov_type,
                        sigma, individual, debiased):
        results = AttrDict()
        results['method'] = method
        results['iter'] = iter_count
        results['nobs'] = nobs
        results['cov_type'] = cov_type
        results['index'] = self._dependent[0].rows
        results['sigma'] = sigma
        results['individual'] = individual
        results['params'] = beta
        results['df_model'] = beta.shape[0]
        results['param_names'] = self._param_names
        results['cov'] = cov
        results['debiased'] = debiased

        total_ss = resid_ss = 0.0
        resid = []
        for key in individual:
            total_ss += individual[key].total_ss
            resid_ss += individual[key].resid_ss
            resid.append(individual[key].resid)
        resid = hstack(resid)

        results['resid_ss'] = resid_ss
        results['total_ss'] = total_ss
        results['r2'] = 1.0 - results.resid_ss / results.total_ss
        results['resid'] = resid
        results['constraints'] = self._constraints
        results['model'] = self

        return results

    def _gls_finalize(self, beta, sigma, full_sigma, gls_eps, eps,
                      cov_type, iter_count, **cov_config):
        """Collect results to return after GLS estimation"""
        wx = self._wx
        k = len(self._wy)

        # Covariance estimation
        if cov_type == 'unadjusted':
            cov_est = HomoskedasticCovariance
        else:
            cov_est = HeteroskedasticCovariance
        gls_eps = reshape(gls_eps, (k, gls_eps.shape[0] // k)).T
        eps = reshape(eps, (k, eps.shape[0] // k)).T
        cov = cov_est(wx, gls_eps, sigma, full_sigma, gls=True,
                      constraints=self._constraints, **cov_config).cov

        # Repackage results for individual equations
        individual = AttrDict()
        debiased = cov_config.get('debiased', False)
        method = 'Iterative GLS' if iter_count > 1 else 'GLS'
        for i in range(k):
            cons = int(self.has_constant.iloc[i])

            if cons:
                c = np.sqrt(self._w[i])
                ye = self._wy[i] - c @ np.linalg.lstsq(c, self._wy[i])[0]
            else:
                ye = self._wy[i]
            total_ss = float(ye.T @ ye)
            stats = self._common_indiv_results(i, beta, cov, gls_eps, eps,
                                               method, cov_type, iter_count,
                                               debiased, cons, total_ss)

            key = self._eq_labels[i]
            individual[key] = stats

        # Populate results dictionary
        nobs = eps.size
        results = self._common_results(beta, cov, method, iter_count, nobs,
                                       cov_type, sigma, individual, debiased)

        # wresid is different between GLS and OLS
        wresid = []
        for key in individual:
            wresid.append(individual[key].wresid)
        wresid = hstack(wresid)
        results['wresid'] = wresid

        return SURResults(results)

    def _sigma_scale(self, debiased):
        if not debiased:
            return 1.0
        nobs = float(self._wx[0].shape[0])
        scales = np.array([nobs - x.shape[1] for x in self._wx], dtype=np.float64)
        scales = np.sqrt(nobs / scales)
        return scales[:, None] @ scales[None, :]

    def _gls_estimate(self, eps, nobs, total_cols, ci, full_cov, debiased):
        """Core estimation routine for iterative GLS"""
        wx, wy = self._wx, self._wy
        sigma = self._sigma
        if sigma is None:
            sigma = eps.T @ eps / nobs
            sigma *= self._sigma_scale(debiased)

        if not full_cov:
            sigma = diag(diag(sigma))
        sigma_inv = inv(sigma)

        k = len(wy)
        if self.constraints is not None:
            cons = self.constraints
            sigma_m12 = inv_matrix_sqrt(sigma)
            x = blocked_diag_product(wx, sigma_m12)
            y = blocked_column_product(wy, sigma_m12)
            xt = x @ cons.t
            xpx = xt.T @ xt
            xpy = xt.T @ (y - x @ cons.a.T)
            paramsc = solve(xpx, xpy)
            params = cons.t @ paramsc + cons.a.T
        else:
            xpx = blocked_inner_prod(wx, sigma_inv)
            xpy = zeros((total_cols, 1))
            for i in range(k):
                sy = zeros((nobs, 1))
                for j in range(k):
                    sy += sigma_inv[i, j] * wy[j]
                xpy[ci[i]:ci[i + 1]] = wx[i].T @ sy

            params = solve(xpx, xpy)

        beta = params

        loc = 0
        for j in range(k):
            _wx = wx[j]
            _wy = wy[j]
            kx = _wx.shape[1]
            eps[:, [j]] = _wy - _wx @ beta[loc:loc + kx]
            loc += kx

        return beta, eps, sigma

    def fit(self, *, method=None, full_cov=True, iterate=False, iter_limit=100, tol=1e-6,
            cov_type='robust', **cov_config):
        """
        Estimate model parameters

        Parameters
        ----------
        method : {None, 'gls', 'ols'}
            Estimation method.  Default auto selects based on regressors,
            using OLS only if all regressors are identical. The other two
            arguments force the use of GLS or OLS.
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
        col_idx = cumsum(col_sizes)
        total_cols = col_idx[-1]
        beta, eps = self._multivariate_ls_fit()
        nobs = eps.shape[0]
        debiased = cov_config.get('debiased', False)
        full_sigma = sigma = (eps.T @ eps / nobs) * self._sigma_scale(debiased)
        if (self._common_exog and method is None and self._constraints is None) or method == 'ols':
            return self._multivariate_ls_finalize(beta, eps, sigma, cov_type, **cov_config)

        beta_hist = [beta]
        nobs = eps.shape[0]
        iter_count = 0
        delta = inf
        while ((iter_count < iter_limit and iterate) or iter_count == 0) and delta >= tol:
            beta, eps, sigma = self._gls_estimate(eps, nobs, total_cols, col_idx,
                                                  full_cov, debiased)
            beta_hist.append(beta)
            delta = beta_hist[-1] - beta_hist[-2]
            delta = sqrt(np.mean(delta ** 2))
            iter_count += 1

        sigma_m12 = inv_matrix_sqrt(sigma)
        wy = blocked_column_product(self._wy, sigma_m12)
        wx = blocked_diag_product(self._wx, sigma_m12)
        gls_eps = wy - wx @ beta

        y = blocked_column_product(self._y, eye(k))
        x = blocked_diag_product(self._x, eye(k))
        eps = y - x @ beta

        return self._gls_finalize(beta, sigma, full_sigma, gls_eps,
                                  eps, cov_type, iter_count, **cov_config)

    @property
    def constraints(self):
        """
        Model constraints

        Returns
        -------
        cons : LinearConstraint
            Constraint object
        """
        return self._constraints

    def add_constraints(self, r, q=None):
        r"""
        Parameters
        ----------
        r : DataFrame
            Constraint matrix. nconstraints by nparameters
        q : Series, optional
            Constraint values (nconstraints).  If not set, set to 0

        Notes
        -----
        Constraints are of the form

        .. math ::

            r \beta = q

        The property `param_names` can be used to determine the order of
        parameters.
        """
        self._constraints = LinearConstraint(r, q=q, num_params=len(self._param_names),
                                             require_pandas=True)

    def reset_constraints(self):
        """Remove all model constraints"""
        self._constraints = None

    @property
    def param_names(self):
        """
        Model parameter names

        Returns
        -------
        names : list[str]
            Normalized, unique model parameter names
        """
        return self._param_names
