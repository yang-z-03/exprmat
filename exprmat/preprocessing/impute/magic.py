
from scipy import sparse
from scipy import spatial
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError

import graphtools
import matplotlib.pyplot as plt
import numbers
import numpy as np
import pandas as pd
import warnings

from exprmat.ansi import error, warning, info
import exprmat.utils as utils


class magic(BaseEstimator):
    """
    MAGIC operator which performs dimensionality reduction.

    Markov Affinity-based Graph Imputation of Cells (MAGIC) is an algorithm for denoising 
    and transcript recover of single cells applied to single-cell RNA sequencing data.

    Parameters
    ----------
    knn : int, optional, default: 5
        Number of nearest neighbors from which to compute kernel bandwidth

    knn_max : int, optional, default: None
        Maximum number of nearest neighbors with nonzero connection.
        If `None`, will be set to 3 * `knn`

    decay : int, optional, default: 1
        Sets decay rate of kernel tails.
        If None, alpha decaying kernel is not used

    t : int, optional, default: 3
        Power to which the diffusion operator is powered. This sets the level of diffusion. 
        If 'auto', t is selected according to the Procrustes disparity of the diffused data.

    n_pca : int, optional, default: 100
        Number of principal components to use for calculating neighborhoods. For extremely 
        large datasets, using n_pca < 20 allows neighborhoods to be calculated in
        roughly log(n_samples) time.

    solver : str, optional, default: 'exact'
        Which solver to use. "exact" uses the implementation described in van Dijk et al. (2018). 
        "approximate" uses a faster implementation that performs imputation in the PCA space and 
        then projects back to the gene space. Note, the "approximate" solver may return negative values.

    knn_dist : string, optional, default: 'euclidean'
        Distance metric for building kNN graph. Recommended values:
        'euclidean', 'cosine'. Any metric from `scipy.spatial.distance` can be
        used. Custom distance functions of form `f(x, y) = d` are also accepted

    random_state : integer or numpy.RandomState, optional, default: None
        The generator used to initialize random PCA. If an integer is given, it fixes the seed
        Defaults to the global `numpy` random number generator

    """

    def __init__(
        self,
        knn = 5,
        knn_max = None,
        decay = 1,
        t = 3,
        n_pca = 100,
        solver = "exact",
        knn_dist = "euclidean",
        n_jobs = 1,
        random_state = None,
        verbose = 1
    ):
        self.knn = knn
        self.knn_max = knn_max
        self.decay = decay
        self.t = t
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.solver = solver
        self.graph = None
        self.input = None
        self.imputed = None
        self.check_params()
        self.verbose = verbose

    @property
    def knn_max(self):
        if self._knn_max is not None:
            return self._knn_max
        else:
            return self.knn * 3

    @knn_max.setter
    def knn_max(self, value):
        self._knn_max = value

    @property
    def diff_op(self):
        """ The diffusion operator calculated from the data """
        if self.graph is not None:
            return self.graph.diff_op
        else:
            warning("This magic instance is not fitted yet.")
            error("Call 'fit' with appropriate arguments before using this property.")


    def check_params(self):

        utils.check_positive(knn = self.knn)
        utils.check_int(knn = self.knn, n_jobs = self.n_jobs)
        utils.check_if_not(
            None,
            utils.check_positive,
            utils.check_int,
            n_pca = self.n_pca,
            knn_max = self.knn_max,
        )

        utils.check_if_not(None, utils.check_positive, decay = self.decay)
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t = self.t)
        utils.check_in(["exact", "approximate"], solver = self.solver)
        if not callable(self.knn_dist):
            utils.check_in([
                 "euclidean",
                "cosine",
                "correlation",
                "cityblock",
                "l1",
                "l2",
                "manhattan",
                "braycurtis",
                "canberra",
                "chebyshev",
                "dice",
                "hamming",
                "jaccard",
                "kulsinski",
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",
                "sokalsneath",
                "sqeuclidean",
                "yule",
            ], knn_dist = self.knn_dist)


    def set_graph_params(self, **params):
        try: self.graph.set_params(**params)
        except AttributeError: pass


    def set_params(self, **params):

        reset_kernel = False
        reset_imputation = False
        # diff potential parameters
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            reset_imputation = True
            del params["t"]

        # kernel parameters
        if "knn" in params and params["knn"] != self.knn:
            self.knn = params["knn"]
            reset_kernel = True
            del params["knn"]
        if "knn_max" in params and params["knn_max"] != self.knn_max:
            self.knn_max = params["knn_max"]
            reset_kernel = True
            del params["knn_max"]
        if "decay" in params and params["decay"] != self.decay:
            self.decay = params["decay"]
            reset_kernel = True
            del params["decay"]
        if "n_pca" in params and params["n_pca"] != self.n_pca:
            self.n_pca = params["n_pca"]
            reset_kernel = True
            del params["n_pca"]
        if "knn_dist" in params and params["knn_dist"] != self.knn_dist:
            self.knn_dist = params["knn_dist"]
            reset_kernel = True
            del params["knn_dist"]

        # parameters that don't change the embedding
        if "solver" in params and params["solver"] != self.solver:
            self.solver = params["solver"]
            reset_imputation = True
            del params["solver"]
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
            self.set_graph_params(n_jobs=params["n_jobs"])
            del params["n_jobs"]
        if "random_state" in params:
            self.random_state = params["random_state"]
            self.set_graph_params(random_state=params["random_state"])
            del params["random_state"]
        if "verbose" in params:
            self.verbose = params["verbose"]
            self.set_graph_params(verbose=params["verbose"])
            del params["verbose"]

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self.graph = None
            reset_imputation = True
        if reset_imputation:
            self.imputed = None

        self.check_params()
        return self


    def fit(self, X, graph = None):
        """
        Computes the diffusion operator

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_features` dimensions. 
            Accepted data types: `numpy.ndarray`, `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`.

        graph : `graphtools.Graph`, optional (default: None)
            If given, provides a precomputed kernel matrix with which to perform diffusion.

        Returns
        -------
        magic_operator
            The estimator object
        """

        if self.n_pca is None or X.shape[1] <= self.n_pca: n_pca = None
        else: n_pca = self.n_pca

        info("running MAGIC on {} cells and {} genes.".format(X.shape[0], X.shape[1]))

        if graph is None:
            graph = self.graph
            if self.input is not None and not utils.matrix_is_equivalent(X, self.input):
                
                # if the same data is used, we can reuse existing kernel and
                # diffusion matrices. Otherwise we have to recompute.
                info("reset the graph due to difference in input data")
                graph = None

            elif graph is not None:
                try:
                    graph.set_params(
                        decay = self.decay,
                        knn = self.knn,
                        knn_max = self.knn_max,
                        distance = self.knn_dist,
                        n_jobs = self.n_jobs,
                        verbose = self.verbose,
                        n_pca = n_pca,
                        thresh = 1e-4,
                        random_state = self.random_state,
                    )

                except ValueError as e:
                    # something changed that should have invalidated the graph
                    warning("reset the graph due to {}".format(str(e)))
                    graph = None
        
        else:
            self.knn = graph.knn
            self.alpha = graph.decay
            self.n_pca = graph.n_pca
            self.knn_dist = graph.distance
            try: self.knn_max = graph.knn_max
            except AttributeError:
                # not all graphs have knn_max
                self.knn_max = None

        self.input = X

        if utils.has_empty_columns(X):
            warning('you should remove empty genes before imputing with MAGIC.')

        if graph is not None:
            info("Using precomputed graph and diffusion operator ...")
            self.graph = graph
        
        else:
            self.imputed = None
            self.graph = graphtools.Graph(
                X,
                n_pca = n_pca,
                knn = self.knn,
                knn_max = self.knn_max,
                decay = self.decay,
                thresh = 1e-4,
                n_jobs = self.n_jobs,
                verbose = self.verbose,
                random_state = self.random_state,
            )

        return self


    def find_gene_index(self, X, genes):

        is_sparse_x = sparse.issparse(X) or utils.is_sparse_dataframe(X)
        is_large_x = np.prod(X.shape) > 5000 * 20000
        if genes is None and is_sparse_x and is_large_x:
            warning(f"returning imputed values for all genes on a ({X.shape[0]} x {X.shape[1]}) matrix")
            warning(f"will require approximately {np.prod(X.shape) * 8 / (1024 ** 3):.2f}GiB of memory.")
            warning(f"suppress this warning by explicitly set `genes='all_genes'`.")

        if isinstance(genes, str) and genes == "all_genes":
            genes = None
        elif isinstance(genes, str) and genes == "pca_only":
            if not hasattr(self.graph, "data_pca"):
                error("cannot return pca as it is not performed.")
        
        return utils.find_gene_index(X, genes)


    def transform(self, X = None, genes = None, t_max = 20, plot_optimal_t = False, ax = None):
        """
        Computes the values of genes after diffusion

        Parameters
        ----------
        X : array, optional, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_features` dimensions. Not required, since 
            MAGIC does not embed cells not given in the input matrix to `MAGIC.fit()`.
            Accepted data types: `numpy.ndarray`, `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`.

        genes : list or {"all_genes", "pca_only"}, optional (default: None)
            List of genes, either as integer indices or column names if input data is a pandas 
            DataFrame. If "all_genes", the entire
            smoothed matrix is returned. If "pca_only", PCA on the smoothed
            data is returned. If None, the entire matrix is also
            returned, but a warning may be raised if the resultant matrix
            is very large.

        t_max : int, optional, default: 20
            maximum t to test if `t` is set to 'auto'

        plot_optimal_t : boolean, optional, default: False
            If true and `t` is set to 'auto', plot the disparity used to select t

        ax : matplotlib.axes.Axes, optional
            If given and `plot_optimal_t` is true, plot will be drawn on the given axis.

        Returns
        -------
        X_magic : array, shape = [n_samples, n_genes]
            The gene expression values after diffusion
        """

        if self.graph is None:
            if self.input is not None:
                self.fit(self.input)
            else: error('the MAGIC estimator is not trained. use fit() to train the model.')

        if X is not None and not utils.matrix_is_equivalent(X, self.graph.data):
            extrapolation = True
            store_result = False
            warning('running MAGIC on unfitted result may not provide sensible result.')

        else:
            extrapolation = False
            X = self.input
            data = self.graph
            store_result = True

        genes = self.find_gene_index(X, genes)
        if genes is None: genes_is_short = False
        else: genes_is_short = len(genes) < self.graph.data_nu.shape[1]

        if isinstance(genes, str) and genes == "pca_only":
            # have to use PCA to return it
            solver = "approximate"
        elif self.imputed is None and genes_is_short:
            # faster to skip PCA
            solver = "exact"
            store_result = False
        else: solver = self.solver

        if store_result and self.imputed is not None:
            x_imp = self.imputed
        
        else:
            if extrapolation:
                n_pca = self.n_pca if solver == "approximate" else None
                data = graphtools.base.Data(X, n_pca = n_pca)
            
            if solver == "approximate":
                # select PCs
                X_input = data.data_nu

            else:
                X_input = utils.to_array_or_spmatrix(data.data)
                if genes is not None and not (isinstance(genes, str) and genes != "pca_only"):
                    X_input = utils.select_columns(X_input, idx = genes)
            
            if solver == "exact" and X_input.shape[1] > 6000:
                warning(f"running MAGIC with `solver='exact'` on {X_input.shape[1]}-dimensional data may take a long time.")
                warning("consider denoising specific genes with `genes` or using `solver='approximate'`.")
            
            x_imp, errors, t_o = self.imputation(X_input, t_max = t_max, plot = plot_optimal_t, ax = ax)
            self.t_opt = t_o
            self.errors = errors
            if store_result: self.imputed = x_imp

        # return selected genes
        if isinstance(genes, str) and genes == "pca_only":
            x_imp = PCA().fit_transform(x_imp)
            genes = ["PC{}".format(i + 1) for i in range(x_imp.shape[1])]
        elif solver == "approximate":
            x_imp = data.inverse_transform(x_imp, columns=genes)
        elif genes is not None and len(genes) != x_imp.shape[1]:
            # select genes
            x_imp = utils.select_columns(x_imp, idx = genes)

        # convert back to pandas dataframe, if necessary
        x_imp = utils.convert_to_same_format(
            x_imp, X, columns = genes, prevent_sparse = True)
        return x_imp


    def fit_transform(self, X, graph = None, **kwargs):

        self.fit(X, graph = graph)
        imp = self.transform(**kwargs)
        return imp


    def calculate_errors(
        self, data, data_prev = None, weights = None, subsample_genes = None
    ):
        """
        Calculates difference before and after diffusion

        Parameters
        ----------
        data : array-like
            Current data matrix

        data_prev : array-like, optional (default: None)
            Previous data matrix. If None, `data` is simply prepared for comparison and no error is returned.

        weights : list-like, optional (default: None)
            Weightings for dimensions of data. If None, dimensions are equally weighted.

        subsample_genes : like-like, optional (default: None)
            Genes to select in subsampling. If None, no subsampling is performed.

        Returns
        -------
        error : float
            Procrustes disparity value.

        data_curr : array-like
            Transformed data to use for the next comparison.
        """

        if subsample_genes is not None:
            data = data[:, subsample_genes]
        if weights is None:
            weights = np.ones(data.shape[1]) / data.shape[1]
        if data_prev is not None:
            _, _, error = spatial.procrustes(data_prev, data)
        else: error = None
        return error, data


    def imputation(
        self, data, t_max = 20, plot = False, ax = None,
        max_genes_compute_t = 500,
        threshold = 0.001,
    ):
        """
        Peform MAGIC imputation

        Parameters
        ----------
        data : array-like
            Input data

        t_max : int, optional (default: 20)
            Maximum value of t to consider for optimal t selection

        plot : bool, optional (default: False)
            Plot the optimal t selection graph

        ax : matplotlib.Axes, optional (default: None)
            Axis on which to plot. If None, a new axis is created

        max_genes_compute_t : int, optional (default: 500)
            Above this number, genes will be subsampled for optimal t selection

        threshold : float, optional (default: 0.001)
            Threshold after which Procrustes disparity is considered to have converged 
            for optimal t selection

        Returns
        -------
        imputed : array-like, shape = [n_samples, n_pca]
            Imputed data.
        
        errors:
            Rounds of errors at different tested level t.
        
        t_opt:
            Optimal t value, if not converged, return None.
        """

        data_imputed = utils.toarray(data)

        if data_imputed.shape[1] > max_genes_compute_t:
            subsample_genes = np.random.choice(
                data_imputed.shape[1], max_genes_compute_t, replace=False)
        else: subsample_genes = None

        if hasattr(data, "data_pca"):
            weights = None  # data.data_pca.explained_variance_ratio_
        else: weights = None

        if self.t == "auto":
            _, data_prev = self.calculate_errors(
                data_imputed,
                data_prev = None,
                weights = weights,
                subsample_genes = subsample_genes,
            )
            error_vec = []
            t_opt = None

        else: t_opt = self.t


        # classical magic
        # the diffusion matrix is powered when t has been specified by
        # the user, and the dimensions of the diffusion matrix are lesser
        # than those of the data matrix. (M^t) * D
        if (t_opt is not None) and (self.diff_op.shape[1] < data_imputed.shape[1]):
            diff_op_t = np.linalg.matrix_power(
                utils.toarray(self.diff_op), t_opt)
            data_imputed = diff_op_t.dot(data_imputed)

        # fast magic
        # a while loop is used when the dimensions of the diffusion matrix
        # are greater than those of the data matrix, or when t is not specified
        # (so as to allow for the calculation of the optimal t value)
        else:
            i = 0
            while (t_opt is None and i < t_max) or \
                (t_opt is not None and i < t_opt):

                i += 1
                data_imputed = self.diff_op.dot(data_imputed)
                if self.t == "auto":
                    error, data_prev = self.calculate_errors(
                        data_imputed,
                        data_prev,
                        weights=weights,
                        subsample_genes=subsample_genes,
                    )

                    error_vec.append(error)
                    if error < threshold and t_opt is None:
                        t_opt = i + 1
                        info("automatically selected t = {}".format(t_opt))

        if t_opt is None:
            # never converged
            warning("optimal t > t_max ({}) never converged.".format(t_max))
            errors = None

        else:
            data_overimputed = data_imputed
            while i < t_max:
                i += 1
                data_overimputed = self.diff_op.dot(data_overimputed)
                error, data_prev = self.calculate_errors(
                    data_overimputed,
                    data_prev,
                    weights=weights,
                    subsample_genes=subsample_genes,
                )
                error_vec.append(error)

            x = np.arange(len(error_vec)) + 1
            errors = pd.DataFrame({'x': x, 'errors': error_vec})
                
        return data_imputed, errors, t_opt