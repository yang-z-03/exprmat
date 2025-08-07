from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel  # type: ignore
from joblib import delayed
from joblib import parallel_backend
from scipy.optimize import minimize  # type: ignore

from exprmat.deseq import utils

from abc import ABC
from abc import abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class inference(ABC):
    
    @abstractmethod
    def lin_reg_mu(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        """
        Estimate mean of negative binomial model using a linear regression.
        Used to initialize genewise dispersion models.

        Parameters
        ----------
        counts : ndarray
            Raw counts.

        size_factors : ndarray
            Sample-wise scaling factors (obtained from median-of-ratios).

        design_matrix : ndarray
            Design matrix.

        min_mu : float
            Lower threshold for fitted means, for numerical stability.
            (default: ``0.5``).

        Returns
        -------
        ndarray
            Estimated mean.
        """


    @abstractmethod
    def irls(
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        min_mu: float,
        beta_tol: float,
        min_beta: float = -30,
        max_beta: float = 30,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
        maxiter: int = 250,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Fit a NB GLM wit log-link to predict counts from the design matrix.
        See equations (1-2) in the DESeq2 paper.

        Parameters
        ----------
        counts : ndarray
            Raw counts.

        size_factors : ndarray
            Sample-wise scaling factors (obtained from median-of-ratios).

        design_matrix : ndarray
            Design matrix.

        disp : ndarray
            Gene-wise dispersion prior.

        min_mu : ndarray
            Lower bound on estimated means, to ensure numerical stability.
            (default: ``0.5``).

        beta_tol : float
            Stopping criterion for IRWLS:
            :math:`\vert dev - dev_{old}\vert / \vert dev + 0.1 \vert < \beta_{tol}`.
            (default: ``1e-8``).

        min_beta : float
            Lower-bound on LFC. (default: ``-30``).

        max_beta : float
            Upper-bound on LFC. (default: ``-30``).

        optimizer : str
            Optimizing method to use in case IRLS starts diverging.
            Accepted values: 'BFGS' or 'L-BFGS-B'.
            NB: only 'L-BFGS-B' ensures that LFCS will
            lay in the [min_beta, max_beta] range. (default: ``'L-BFGS-B'``).

        maxiter : int
            Maximum number of IRLS iterations to perform before switching to L-BFGS-B.
            (default: ``250``).

        Returns
        -------
        beta: ndarray
            Fitted (basemean, lfc) coefficients of negative binomial GLM.

        mu: ndarray
            Means estimated from size factors and beta:
            :math:`\mu = s_{ij} \exp(\beta^t X)`.

        H: ndarray
            Diagonal of the :math:`W^{1/2} X (X^t W X)^-1 X^t W^{1/2}`
            covariance matrix.

        converged: ndarray
            Whether IRLS or the optimizer converged. If not and if dimension allows it,
            perform grid search.
        """


    @abstractmethod
    def alpha_mle(
        self,
        counts: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        prior_disp_var: float | None = None,
        cr_reg: bool = True,
        prior_reg: bool = False,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the dispersion parameter of a negative binomial GLM.

        Parameters
        ----------
        counts : ndarray
            Raw counts.

        design_matrix : ndarray
            Design matrix.

        mu : ndarray
            Mean estimation for the NB model.

        alpha_hat : ndarray
            Initial dispersion estimate.

        min_disp : float
            Lower threshold for dispersion parameters.

        max_disp : float
            Upper threshold for dispersion parameters.

        prior_disp_var : float, optional
            Prior dispersion variance.

        cr_reg : bool
            Whether to use Cox-Reid regularization. (default: ``True``).

        prior_reg : bool
            Whether to use prior log-residual regularization. (default: ``False``).

        optimizer : str
            Optimizing method to use. Accepted values: 'BFGS' or 'L-BFGS-B'.
            (default: ``'L-BFGS-B'``).

        Returns
        -------
        ndarray
            Dispersion estimate.

        ndarray
            Whether L-BFGS-B converged. If not, dispersion is estimated
            using grid search.
        """


    @abstractmethod
    def wald_test(
        self,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        lfc: np.ndarray,
        mu: np.ndarray,
        ridge_factor: np.ndarray,
        contrast: np.ndarray,
        lfc_null: np.ndarray,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run Wald test for differential expression.
        Computes Wald statistics, standard error and p-values from dispersion 
        and LFC estimates.

        Parameters
        ----------
        design_matrix : ndarray
            Design matrix.

        disp : float
            Dispersion estimate.

        lfc : ndarray
            Log-fold change estimate (in natural log scale).

        mu : float
            Mean estimation for the NB model.

        ridge_factor : ndarray
            Regularization factors.

        contrast : ndarray
            Vector encoding the contrast that is being tested.

        lfc_null : float
            The (log2) log fold change under the null hypothesis.

        alt_hypothesis : str or None
            The alternative hypothesis for computing wald p-values.

        Returns
        -------
        wald_p_value : ndarray
            Estimated p-value.

        wald_statistic : ndarray
            Wald statistic.

        wald_se : ndarray
            Standard error of the Wald statistic.
        """


    @abstractmethod
    def fit_rough_dispersions(
        self, normed_counts: np.ndarray, design_matrix: np.ndarray
    ) -> np.ndarray:
        """
        'Rough dispersion' estimates from linear model, as per the R code.
        Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
        <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

        Parameters
        ----------
        normed_counts : ndarray
            Array of deseq2-normalized read counts. Rows: samples, columns: genes.

        design_matrix : pandas.DataFrame
            A DataFrame with experiment design information (to split cohorts).
            Indexed by sample barcodes. Unexpanded, *with* intercept.

        Returns
        -------
        ndarray
            Estimated dispersion parameter for each gene.
        """


    @abstractmethod
    def fit_moments_dispersions(
        self, normed_counts: np.ndarray, size_factors: np.ndarray
    ) -> np.ndarray:
        """
        Dispersion estimates based on moments, as per the R code.
        Used as initial estimates in :meth:`DeseqDataSet.fit_genewise_dispersions()
        <pydeseq2.dds.DeseqDataSet.fit_genewise_dispersions>`.

        Parameters
        ----------
        normed_counts : ndarray
            Array of deseq2-normalized read counts. Rows: samples, columns: genes.

        size_factors : ndarray
            DESeq2 normalization factors.

        Returns
        -------
        ndarray
            Estimated dispersion parameter for each gene.
        """


    @abstractmethod
    def dispersion_trend_gamma_glm(
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Fit a gamma glm on gene dispersions. The intercept should be concatenated 
        in this method and the first returned coefficient should be the intercept.

        Parameters
        ----------
        covariates : pd.Series
            Covariates for the regression (num_genes,).

        targets : pd.Series
            Targets for the regression (num_genes,).

        Returns
        -------
        coeffs : ndarray
            Coefficients of the regression.

        predictions : ndarray
            Predictions of the regression.

        converged : bool
            Whether the optimization converged.
        """


    @abstractmethod
    def lfc_shrink_nbinom_glm(
        self,
        design_matrix: np.ndarray,
        counts: np.ndarray,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit a negative binomial MAP LFC using an apeGLM prior.
        Only the LFC is shrinked, and not the intercept.

        Parameters
        ----------
        design_matrix : ndarray
            Design matrix.

        counts : ndarray
            Raw counts.

        size : ndarray
            Size parameter of NB family (inverse of dispersion).

        offset : ndarray
            Natural logarithm of size factor.

        prior_no_shrink_scale : float
            Prior variance for the intercept.

        prior_scale : float
            Prior variance for the LFC parameter.

        optimizer : str
            Optimizing method to use in case IRLS starts diverging.
            Accepted values: 'L-BFGS-B', 'BFGS' or 'Newton-CG'.

        shrink_index : int
            Index of the LFC coordinate to shrink. (default: ``1``).

        Returns
        -------
        beta: ndarray
            2-element array, containing the intercept (first) and the LFC (second).

        inv_hessian: ndarray
            Inverse of the Hessian of the objective at the estimated MAP LFC.

        converged: ndarray
            Whether L-BFGS-B converged for each optimization problem.
        """


class default_deseq_inference(inference):
    """
    Default DESeq2-related inference methods, using scipy/sklearn/numpy.

    This object contains the interface to the default inference routines and uses
    joblib internally for parallelization. Inherit this class or its parent to write
    custom inference routines.

    Parameters
    ----------
    joblib_verbosity : int
        The verbosity level for joblib tasks. The higher the value, the more updates
        are reported. (default: ``0``).
    batch_size : int
        Number of tasks to allocate to each joblib parallel worker. (default: ``128``).
    n_cpus : int
        Number of cpus to use. If None, all available cpus will be used.
        (default: ``None``).
    backend : str
        Joblib backend.
    """

    fit_rough_dispersions = staticmethod(utils.fit_rough_dispersions)  # type: ignore
    fit_moments_dispersions = staticmethod(utils.fit_moments_dispersions)  # type: ignore


    def __init__(
        self,
        joblib_verbosity: int = 0,
        batch_size: int = 128,
        n_cpus: int | None = None,
        backend: str = "loky",
    ):
        self._joblib_verbosity = joblib_verbosity
        self._batch_size = batch_size
        self._n_cpus = utils.get_num_processes(n_cpus)
        self._backend = backend


    @property
    def n_cpus(self) -> int:  # noqa: D102
        return self._n_cpus


    @n_cpus.setter
    def n_cpus(self, n_cpus: int) -> None:
        self._n_cpus = utils.get_num_processes(n_cpus)


    def lin_reg_mu(  # noqa: D102
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        min_mu: float,
    ) -> np.ndarray:
        
        with parallel_backend(self._backend, inner_max_num_threads=1):
            mu_hat_ = np.array(
                Parallel(
                    n_jobs=self.n_cpus,
                    verbose=self._joblib_verbosity,
                    batch_size=self._batch_size,
                )(
                    delayed(utils.fit_lin_mu)(
                        counts=counts[:, i],
                        size_factors=size_factors,
                        design_matrix=design_matrix,
                        min_mu=min_mu,
                    )
                    for i in range(counts.shape[1])
                )
            )
        return mu_hat_.T


    def irls(  # noqa: D102
        self,
        counts: np.ndarray,
        size_factors: np.ndarray,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        min_mu: float,
        beta_tol: float,
        min_beta: float = -30,
        max_beta: float = 30,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
        maxiter: int = 250,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(utils.irls_solver)(
                    counts=counts[:, i],
                    size_factors=size_factors,
                    design_matrix=design_matrix,
                    disp=disp[i],
                    min_mu=min_mu,
                    beta_tol=beta_tol,
                    min_beta=min_beta,
                    max_beta=max_beta,
                    optimizer=optimizer,
                    maxiter=maxiter,
                )
                for i in range(counts.shape[1])
            )
        res = zip(*res)
        MLE_lfcs_, mu_hat_, hat_diagonals_, converged_ = (np.array(m) for m in res)

        return (
            MLE_lfcs_,
            mu_hat_.T,
            hat_diagonals_.T,
            converged_,
        )


    def alpha_mle(  # noqa: D102
        self,
        counts: np.ndarray,
        design_matrix: np.ndarray,
        mu: np.ndarray,
        alpha_hat: np.ndarray,
        min_disp: float,
        max_disp: float,
        prior_disp_var: float | None = None,
        cr_reg: bool = True,
        prior_reg: bool = False,
        optimizer: Literal["BFGS", "L-BFGS-B"] = "L-BFGS-B",
    ) -> tuple[np.ndarray, np.ndarray]:
        
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(utils.fit_alpha_mle)(
                    counts=counts[:, i],
                    design_matrix=design_matrix,
                    mu=mu[:, i],
                    alpha_hat=alpha_hat[i],
                    min_disp=min_disp,
                    max_disp=max_disp,
                    prior_disp_var=prior_disp_var,
                    cr_reg=cr_reg,
                    prior_reg=prior_reg,
                    optimizer=optimizer,
                )
                for i in range(counts.shape[1])
            )
        res = zip(*res)
        dispersions_, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return dispersions_, l_bfgs_b_converged_


    def wald_test(  # noqa: D102
        self,
        design_matrix: np.ndarray,
        disp: np.ndarray,
        lfc: np.ndarray,
        mu: np.ndarray,
        ridge_factor: np.ndarray,
        contrast: np.ndarray,
        lfc_null: np.ndarray,
        alt_hypothesis: (
            Literal["greaterAbs", "lessAbs", "greater", "less"] | None
        ) = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        num_genes = mu.shape[1]
        with parallel_backend(self._backend, inner_max_num_threads=1):
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(utils.wald_test)(
                    design_matrix=design_matrix,
                    disp=disp[i],
                    lfc=lfc[i],
                    mu=mu[:, i],
                    ridge_factor=ridge_factor,
                    contrast=contrast,
                    lfc_null=lfc_null,  # Convert log2 to natural log
                    alt_hypothesis=alt_hypothesis,
                )
                for i in range(num_genes)
            )
        res = zip(*res)
        pvals, stats, se = (np.array(m) for m in res)

        return pvals, stats, se


    def dispersion_trend_gamma_glm(  # noqa: D102
        self, covariates: pd.Series, targets: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        covariates_w_intercept = covariates.to_frame()
        covariates_w_intercept.insert(0, "intercept", 1)
        covariates_fit = covariates_w_intercept.values
        targets_fit = targets.values

        def loss(coeffs):
            mu = covariates_fit @ coeffs
            return np.nanmean(targets_fit / mu + np.log(mu), axis=0)

        def grad(coeffs):
            mu = covariates_fit @ coeffs
            return -np.nanmean(
                ((targets_fit / mu - 1)[:, None] * covariates_fit) / mu[:, None], axis=0
            )

        try:
            res = minimize(
                loss,
                x0=np.array([1.0, 1.0]),
                jac=grad,
                method="L-BFGS-B",
                bounds=[(1e-12, np.inf)],
            )
        except RuntimeWarning:  # Could happen if the coefficients fall to zero
            return np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), False

        coeffs = res.x
        return coeffs, covariates_fit @ coeffs, res.success


    def lfc_shrink_nbinom_glm(  # noqa: D102
        self,
        design_matrix: np.ndarray,
        counts: np.ndarray,
        size: np.ndarray,
        offset: np.ndarray,
        prior_no_shrink_scale: float,
        prior_scale: float,
        optimizer: str,
        shrink_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        with parallel_backend(self._backend, inner_max_num_threads=1):
            num_genes = counts.shape[1]
            res = Parallel(
                n_jobs=self.n_cpus,
                verbose=self._joblib_verbosity,
                batch_size=self._batch_size,
            )(
                delayed(utils.nbinom_glm)(
                    design_matrix=design_matrix,
                    counts=counts[:, i],
                    size=size[i],
                    offset=offset,
                    prior_no_shrink_scale=prior_no_shrink_scale,
                    prior_scale=prior_scale,
                    optimizer=optimizer,
                    shrink_index=shrink_index,
                )
                for i in range(num_genes)
            )
        res = zip(*res)
        lfcs, inv_hessians, l_bfgs_b_converged_ = (np.array(m) for m in res)
        return lfcs, inv_hessians, l_bfgs_b_converged_