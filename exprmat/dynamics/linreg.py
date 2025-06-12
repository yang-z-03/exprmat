
import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix, issparse
import warnings


def prod_sum(a1, a2, axis) -> ndarray:

    if issparse(a1): return a1.multiply(a2).sum(axis=axis).A1
    elif axis == 0: return np.einsum("ij, ij -> j", a1, a2) if a1.ndim > 1 else (a1 * a2).sum()
    elif axis == 1: return np.einsum("ij, ij -> i", a1, a2) if a1.ndim > 1 else (a1 * a2).sum()


def sum(a, axis = None) -> ndarray:
    
    if a.ndim == 1:
        axis = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a.sum(axis=axis).A1 if issparse(a) else a.sum(axis=axis)


def multiply(a, b):

    if issparse(a): return a.multiply(b)
    elif issparse(b): return b.multiply(a)
    else: return a * b


class linreg:
    '''
    Extreme quantile and constraint least square linear regression.

    Parameters
    ----------
    percentile
        Percentile of data on which linear regression line is fit. If `None`, all data
        is used, if a single value is given, it is interpreted as the upper quantile.
        Defaults to `None`.

    fit_intercept
        Whether to calculate the intercept for model. Defaults to `False`.

    positive_intercept
        Whether the intercept it constraint to positive values. Only plays a role when
        `fit_intercept=True`. Defaults to `True`.

    constrain_ratio
        Ratio to which coefficients are clipped. If `None`, the coefficients are not
        constraint. Defaults to `None`.

    Attributes
    ----------
    coefficient
        Estimated coefficients of the linear regression line.

    intercept
        Fitted intercept of linear model. Set to `0.0` if `fit_intercept=False`.
    '''

    def __init__(
        self,
        percentile = None,
        fit_intercept: bool = False,
        positive_intercept: bool = True,
        constrain_ratio = None,
    ):
        
        if not fit_intercept and isinstance(percentile, (list, tuple)):
            self.percentile = percentile[1]
        else: self.percentile = percentile
        self.fit_intercept = fit_intercept
        self.positive_intercept = positive_intercept

        if constrain_ratio is None:
            self.constrain_ratio = [-np.inf, np.inf]
        elif len(constrain_ratio) == 1: self.constrain_ratio = [-np.inf, constrain_ratio]
        else: self.constrain_ratio = constrain_ratio


    def trim_data_extreme(self, data):

        if not isinstance(data, list):
            data = [data]

        data = np.array([
            data_mat.toarray() if issparse(data_mat) else data_mat
            for data_mat in data
        ])

        normalized_data = np.sum(
            data / data.max(axis = 1, keepdims = True)
            .clip(1e-3, None), axis = 0
        )

        bound = np.percentile(normalized_data, self.percentile, axis=0)

        if bound.ndim == 1: trimmer = csr_matrix(normalized_data >= bound).astype(bool)
        else: trimmer = csr_matrix(
            (normalized_data <= bound[0]) | (normalized_data >= bound[1])
        ).astype(bool)

        return [trimmer.getnnz(axis=0)] + [
            trimmer.multiply(data_mat).tocsr() for data_mat in data
        ]


    def fit(self, x: ndarray, y: ndarray):
        
        n_obs = x.shape[0]
        if self.percentile is not None:
            n_obs, x, y = self.trim_data_extreme(data=[x, y])

        _xx = prod_sum(x, x, axis = 0)
        _xy = prod_sum(x, y, axis = 0)

        if self.fit_intercept:
            _x = sum(x, axis = 0) / n_obs
            _y = sum(y, axis = 0) / n_obs
            self.coefficient = (_xy / n_obs - _x * _y) / (_xx / n_obs - _x**2)
            self.intercept = _y - self.coefficient * _x

            if self.positive_intercept:
                idx = self.intercept < 0
                if self.coefficient.ndim > 0:
                    self.coefficient[idx] = _xy[idx] / _xx[idx]
                else:
                    self.coefficient = _xy / _xx
                self.intercept = np.clip(self.intercept, 0, None)
        else:
            self.coefficient = _xy / _xx
            self.intercept = np.zeros(x.shape[1]) if x.ndim > 1 else 0

        if not np.isscalar(self.coefficient):
            self.coefficient[np.isnan(self.coefficient)] = 0
            self.intercept[np.isnan(self.intercept)] = 0
        else:
            if np.isnan(self.coefficient): self.coefficient = 0
            if np.isnan(self.intercept): self.intercept = 0

        self.coefficient = np.clip(self.coefficient, *self.constrain_ratio)
        return self
