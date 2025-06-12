
import warnings
import os
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse
warnings.simplefilter(action = "ignore", category = FutureWarning)

from exprmat.ansi import error, warning, info
from exprmat.dynamics.linreg import linreg, prod_sum
from exprmat.dynamics.moments import moments, second_order_moments, get_moments
from exprmat.dynamics.optimization import leastsq_generalized, maximum_likelihood
from exprmat.dynamics.parallel import parallelize, get_n_jobs
from exprmat.dynamics.transition import transition_matrix
from exprmat.reduction.pca import pca
from exprmat.dynamics.utils import (
    groups_to_bool, make_dense, r_square, strings_to_categoricals, 
    cosine_correlation, get_indices, get_iterative_indices,
    get_n_neighs, get_neighs, l2_norm
)


class velocity_inference:

    def __init__(
        self,
        adata = None,
        Ms = None,
        Mu = None,
        groups_for_fit = None,
        groupby = None,
        residual = None,
        constrain_ratio = None,
        min_r2 = 0.01,
        min_ratio = 0.01,
        use_highly_variable = 'hvg.vst',
        r2_adjusted = True,
        use_raw = False,
    ):
        
        self.adata = adata
        self.ms, self.mu = Ms, Mu
        if Ms is None:
            self.ms = adata.layers["spliced"] if use_raw else adata.layers["ms"]
        if Mu is None:
            self.mu = adata.layers["unspliced"] if use_raw else adata.layers["mu"]
        self.ms, self.mu = make_dense(self.ms), make_dense(self.mu)

        n_obs, n_vars = self.ms.shape
        self.residual, self.residual2 = residual, None
        self.offset = np.zeros(n_vars, dtype=np.float32)
        self.offset2 = np.zeros(n_vars, dtype=np.float32)
        self.gamma = np.zeros(n_vars, dtype=np.float32)
        self.qreg_ratio = np.zeros(n_vars, dtype=np.float32)
        self.r2 = np.zeros(n_vars, dtype=np.float32)
        self.beta = np.ones(n_vars, dtype=np.float32)
        self.velocity_genes = np.ones(n_vars, dtype=bool)
        self.groups_for_fit = groups_to_bool(adata, groups_for_fit, groupby)
        self.constrain_ratio = constrain_ratio
        self.r2_adj = r2_adjusted
        self.min_r2 = min_r2
        self.min_ratio = min_ratio
        self.hvg = None
        if use_highly_variable is not None and adata is not None:
            if use_highly_variable in adata.var.keys():
                self.hvg = adata.var[use_highly_variable]


    def compute_deterministic(self, fit_offset = False, perc = None):

        subset = self.groups_for_fit
        Ms = self.ms if subset is None else self.ms[subset]
        Mu = self.mu if subset is None else self.mu[subset]

        lr = linreg(fit_intercept = fit_offset, percentile = perc)
        lr.fit(Ms, Mu)
        self.offset = lr.intercept
        self.gamma = lr.coefficient

        if self.constrain_ratio is not None:
            if np.size(self.constrain_ratio) < 2:
                self.constrain_ratio = [None, self.constrain_ratio]
            cr = self.constrain_ratio
            self.gamma = np.clip(self.gamma, cr[0], cr[1])

        self.residual = self.mu - self.gamma * self.ms
        if fit_offset:
            self.residual -= self.offset
        res = self.residual

        # velocity genes
        if self.r2_adj:
            lr = linreg(fit_intercept=fit_offset)
            lr.fit(Ms, Mu)
            off = lr.intercept
            gamma = lr.coefficient

            res = self.mu - gamma * self.ms
            if fit_offset:
                res -= off

        self.qreg_ratio = np.array(self.gamma)  # quantile regression ratio
        self.r2 = r_square(res, total=self.mu - self.mu.mean(0))
        self.velocity_genes = (
            (self.r2 > self.min_r2)
            & (self.gamma > self.min_ratio)
            & (np.max(self.ms > 0, 0) > 0)
            & (np.max(self.mu > 0, 0) > 0)
        )

        if self.hvg is not None:
            self.velocity_genes &= self.hvg

        if np.sum(self.velocity_genes) < 2:
            min_r2 = np.percentile(self.r2, 80)
            self.velocity_genes = self.r2 > min_r2
            min_r2 = np.round(min_r2, 4)
            warning(
                f"you seem to have very low signal in splicing dynamics. "
                f"the correlation threshold has been reduced to {min_r2}. "
                f"be cautious when interpreting results."
            )


    def compute_stochastic(
        self, fit_offset = False, fit_offset2 = False, mode = None, perc = None,
        neighbor_key = 'neighbors',
    ):
        
        if self.residual is None:
            self.compute_deterministic(fit_offset = fit_offset, perc = perc)

        idx = np.ones(self.velocity_genes.shape, dtype=bool)
        if np.any(self.velocity_genes):
            idx = self.velocity_genes
        is_subset = len(set(idx)) > 1

        adata = self.adata[:, idx] if is_subset else self.adata
        ms = self.ms[:, idx] if is_subset else self.ms
        mu = self.mu[:, idx] if is_subset else self.mu
        res = self.residual[:, idx] if is_subset else self.residual

        _Mss, _Mus = second_order_moments(adata, neighbor_key = neighbor_key)

        var_ss = 2 * _Mss - ms
        cov_us = 2 * _Mus + mu

        lr = linreg(fit_intercept=fit_offset2)
        lr.fit(var_ss, cov_us)
        off2 = lr.intercept
        gamma2 = lr.coefficient

        # initialize covariance matrix
        res_std = res.std(0)
        res2_std = (cov_us - gamma2 * var_ss - off2).std(0)

        # solve multiple regression
        self.offset[idx], self.offset2[idx], self.gamma[idx] = (
            maximum_likelihood(ms, mu, _Mus, _Mss, fit_offset, fit_offset2)
            if mode == "bayes"
            else leastsq_generalized(
                ms, mu, var_ss, cov_us, res_std, res2_std,
                fit_offset, fit_offset2, perc,
            )
        )

        self.residual = self.mu - self.gamma * self.ms
        if fit_offset: self.residual -= self.offset

        res2 = (cov_us - 2 * ms * mu) - self.gamma[idx] * (
            var_ss - 2 * ms ** 2
        )

        if fit_offset: res2 += 2 * self.offset[idx] * ms
        if fit_offset2: res2 -= self.offset2[idx]
        if is_subset:
            self.residual2 = np.zeros(self.ms.shape, dtype=np.float32)
            self.residual2[:, idx] = res2
        else: self.residual2 = res2


    def get_pars(self):
        
        return (
            self.offset,
            self.offset2,
            self.beta,
            self.gamma,
            self.qreg_ratio,
            self.r2,
            self.velocity_genes,
        )


    def get_pars_names(self):

        return [
            "offset",
            "offset2",
            "beta",
            "gamma",
            "qreg.ratio",
            "r2",
            "genes",
        ]


def write_residuals(adata, vkey, residual = None, cell_subset = None):
    
    if residual is not None:
        if cell_subset is None: adata.layers[vkey] = residual
        else:
            if vkey not in adata.layers.keys():
                adata.layers[vkey] = np.zeros(adata.shape, dtype = np.float32)
            adata.layers[vkey][cell_subset] = residual


def write_pars(adata, vkey, pars, pars_names, add_key = None):

    for i, key in enumerate(pars_names):
        key = f"{vkey}.{key}.{add_key}" if add_key is not None else f"{vkey}.{key}"
        if len(set(pars[i])) > 1: adata.var[key] = pars[i]
        elif key in adata.var.keys(): del adata.var[key]


def velocity(
    adata, vkey = "velocity", mode = "stochastic",
    fit_offset = False, fit_offset2 = False,
    filter_genes = False,
    groups = None, groupby = None, groups_for_fit = None,
    constrain_ratio = None,
    use_raw = False,
    use_latent_time = None,
    perc = None,
    min_r2 = 1e-2,
    min_likelihood = 1e-3,
    r2_adjusted = None,
    use_highly_variable = 'vst.hvg',
    neighbor_key = 'neighbors',
    diff_kinetics = None,
    **kwargs
):
    """
    Estimates velocities in a gene-specific manner.

    The steady-state model :cite:p:`LaManno18` determines velocities by quantifying how
    observations deviate from a presumed steady-state equilibrium ratio of unspliced to
    spliced mRNA levels. This steady-state ratio is obtained by performing a linear
    regression restricting the input data to the extreme quantiles. By including
    second-order moments, the stochastic model :cite:p:`Bergen20` exploits not only the balance
    of unspliced to spliced mRNA levels but also their covariation. By contrast, the
    likelihood-based dynamical model :cite:p:`Bergen20` solves the full splicing kinetics and
    generalizes RNA velocity estimation to transient systems. It is also capable of
    capturing non-observed steady states.

    Parameters
    ----------

    data: anndata.AnnData
        Annotated data matrix.

    vkey: str (default: `'velocity'`)
        Name under which to refer to the computed velocities for `velocity_graph` and 
        `velocity_embedding`.

    mode: `'deterministic'`, `'stochastic'` or `'dynamical'` (default: `'stochastic'`)
        Whether to run the estimation using the steady-state/deterministic, stochastic or
        dynamical model of transcriptional dynamics. The dynamical model requires to run 
        `recover_dynamics` first.

    fit_offset: bool (default: `False`)
        Whether to fit with offset for first order moment dynamics.

    fit_offset2: bool, (default: `False`)
        Whether to fit with offset for second order moment dynamics.

    filter_genes: bool (default: `True`)
        Whether to remove genes that are not used for further velocity analysis.

    groups: str, list (default: `None`)
        Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’], to which velocity analysis shall 
        be restricted. You are recommended to infer velocity on lineage-related clusters.

    groupby: str, list or np.ndarray (default: `None`)
        Key of observations grouping to consider.

    groups_for_fit: str, list or np.ndarray (default: `None`)
        Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’], to which steady-state fitting 
        shall be restricted. You are recommended to infer velocity on lineage-related clusters.

    constrain_ratio: float or tuple of type float or None: (default: `None`)
        Bounds for the steady-state ratio.

    use_raw: bool (default: `False`)
        Whether to use raw data for estimation.

    use_latent_time: bool or None (default: `None`)
        Whether to use latent time as a regularization for velocity estimation.

    perc: float (default: `[5, 95]`)
        Percentile, e.g. 98, for extreme quantile fit.

    min_r2: float (default: 0.01)
        Minimum threshold for coefficient of determination

    min_likelihood: float (default: `None`)
        Minimal likelihood for velocity genes to fit the model on.

    r2_adjusted: bool (default: `None`)
        Whether to compute coefficient of determination on full data fit (adjusted) 
        or extreme quantile fit (None)

    use_highly_variable: str (default: 'hvg.vst')
        Whether to use highly variable genes only, stored in .var

    Returns
    -------
    velocity: `.layers`
        Velocity vectors for each individual cell

    velocity.genes, velocity.beta, velocity.gamma, velocity.r2: `.var`
        Parameters on per-gene velocity inference
    """

    if perc is None: perc = [5, 95]
    if not use_raw and "ms" not in adata.layers.keys():
        error('you should run moments() first to generate ms slot.')

    info("computing velocities ...")
    strings_to_categoricals(adata)

    if mode is None or (mode == "dynamical" and "fit.alpha" not in adata.var.keys()):
        mode = "stochastic"
        warning('for dynamical model, you should run recover_dynamics() first.')
        warning('here, we fallback to stochastic model.')

    if mode in {"dynamical", "dynamical_residuals"}:

        pass

        # from ._em_model_utils import get_divergence, get_reads, get_vars

        # gene_subset = ~np.isnan(adata.var["fit_alpha"].values)
        # vdata = adata[:, gene_subset]
        # alpha, beta, gamma, scaling, t_ = get_vars(vdata)

        # connect = not adata.uns["recover_dynamics"]["use_raw"]
        # kwargs_ = {
        #     "kernel_width": None,
        #     "normalized": True,
        #     "var_scale": True,
        #     "reg_par": None,
        #     "min_confidence": 1e-2,
        #     "constraint_time_increments": False,
        #     "fit_steady_states": True,
        #     "fit_basal_transcription": None,
        #     "use_connectivities": connect,
        #     "time_connectivities": connect,
        #     "use_latent_time": use_latent_time,
        # }
        # kwargs_.update(adata.uns["recover_dynamics"])
        # kwargs_.update(**kwargs)

        # if "residuals" in mode:
        #     u, s = get_reads(vdata, use_raw=adata.uns["recover_dynamics"]["use_raw"])
        #     if kwargs_["fit_basal_transcription"]:
        #         u, s = u - adata.var["fit_u0"], s - adata.var["fit_s0"]
        #     o = vdata.layers["fit_t"] < t_
        #     vt = u * beta - s * gamma  # ds/dt
        #     wt = (alpha * o - beta * u) * scaling  # du/dt
        # else:
        #     vt, wt = get_divergence(vdata, mode="velocity", **kwargs_)

        # vgenes = adata.var.fit_likelihood > min_likelihood
        # if min_r2 is not None:
        #     if "fit_r2" not in adata.var.keys():
        #         velo = velocity_inference(
        #             adata,
        #             groups_for_fit=groups_for_fit,
        #             groupby=groupby,
        #             constrain_ratio=constrain_ratio,
        #             min_r2=min_r2,
        #             use_highly_variable=use_highly_variable,
        #             use_raw=use_raw,
        #         )
        #         velo.compute_deterministic(fit_offset=fit_offset, perc=perc)
        #         adata.var["fit_r2"] = velo.r2
        #     vgenes &= adata.var.fit_r2 > min_r2

        # lb, ub = np.nanpercentile(adata.var.fit_scaling, [10, 90])
        # vgenes = (
        #     vgenes
        #     & (adata.var.fit_scaling > np.min([lb, 0.03]))
        #     & (adata.var.fit_scaling < np.max([ub, 3]))
        # )

        # adata.var[f"{vkey}_genes"] = vgenes

        # adata.layers[vkey] = np.ones(adata.shape) * np.nan
        # adata.layers[vkey][:, gene_subset] = vt

        # adata.layers[f"{vkey}_u"] = np.ones(adata.shape) * np.nan
        # adata.layers[f"{vkey}_u"][:, gene_subset] = wt

        # if filter_genes and len(set(vgenes)) > 1:
        #     adata._inplace_subset_var(vgenes)

    elif mode in {"steady_state", "deterministic", "stochastic"}:
        
        categories = (
            adata.obs[groupby].cat.categories
            if groupby is not None and groups is None and groups_for_fit is None
            else [None]
        )

        for cat in categories:
            groups = cat if cat is not None else groups

            cell_subset = groups_to_bool(adata, groups, groupby)
            _adata = adata if groups is None else adata[cell_subset]
            velo = velocity_inference(
                _adata,
                groups_for_fit = groups_for_fit,
                groupby = groupby,
                constrain_ratio = constrain_ratio,
                min_r2 = min_r2,
                r2_adjusted = r2_adjusted,
                use_highly_variable = use_highly_variable,
                use_raw = use_raw,
            )

            velo.compute_deterministic(fit_offset = fit_offset, perc = perc)

            if mode == "stochastic":

                if filter_genes and len(set(velo.velocity_genes)) > 1:
                    adata._inplace_subset_var(velo.velocity_genes)
                    residual = velo.residual[:, velo.velocity_genes]
                    _adata = adata if groups is None else adata[cell_subset]
                    velo = velocity_inference(
                        _adata,
                        residual = residual,
                        groups_for_fit = groups_for_fit,
                        groupby = groupby,
                        constrain_ratio = constrain_ratio,
                        use_highly_variable = use_highly_variable,
                    )
                    
                velo.compute_stochastic(
                    fit_offset, fit_offset2, mode, 
                    perc = perc, neighbor_key = neighbor_key
                )

            write_residuals(adata, vkey, velo.residual, cell_subset)
            write_residuals(adata, f"variance.{vkey}", velo.residual2, cell_subset)
            write_pars(adata, vkey, velo.get_pars(), velo.get_pars_names(), add_key=cat)

            if filter_genes and len(set(velo.velocity_genes)) > 1:
                adata._inplace_subset_var(velo.velocity_genes)

    else: error("mode can only be one of these: deterministic, stochastic or dynamical.")

    if f"{vkey}.genes" in adata.var.keys() and np.sum(adata.var[f"{vkey}.genes"]) < 10:
        warning("too few genes are selected as velocity genes. (< 10)")
        warning("consider setting a lower threshold for min_r2 or min_likelihood.")

    if diff_kinetics:
        
        if not isinstance(diff_kinetics, str):
            diff_kinetics = "fit.diff.kinetics"
        
        if diff_kinetics in adata.var.keys():
            if 'fit_diff_kinetics' in adata.uns["recover.dynamics"]:
                groupby = adata.uns["recover.dynamics"]["fit_diff_kinetics"]
            else: groupby = "clusters"
            
            clusters = adata.obs[groupby]
            for i, v in enumerate(np.array(adata.var[diff_kinetics].values, dtype=str)):
                if len(v) > 0 and v != "nan":
                    idx = 1 - clusters.isin([a.strip() for a in v.split(",")])
                    adata.layers[vkey][:, i] *= idx
                    if mode == "dynamical":
                        adata.layers[f"{vkey}.u"][:, i] *= idx

    adata.uns[f"{vkey}.params"] = {
        "mode": mode, 
        "fit_offset": fit_offset, 
        "perc": perc
    }

    return None


def velocity_genes(
    adata,
    vkey = "velocity",
    min_r2 = 0.01,
    min_ratio = 0.01,
    use_highly_variable = 'hvg.vst'
):
    """
    Estimates velocities in a gene-specific manner.

    Parameters
    ----------

    data: anndata.AnnData
        Annotated data matrix.

    vkey: str (default: `'velocity'`)
        Name under which to refer to the computed velocities.

    min_r2: float (default: 0.01)
        Minimum threshold for coefficient of determination

    min_ratio: float (default: 0.01)
        Minimum threshold for quantile regression un/spliced ratio.

    use_highly_variable: str (default: `vst.hvg`)
        Whether to use highly variable genes only, stored in .var[].

    Returns
    -------
    velocity.genes: `.var`
        genes to be used for further velocity analysis (velocity graph and embedding)
    """
    
    if f"{vkey}.genes" not in adata.var.keys(): velocity(adata, vkey)
    vgenes = np.ones(adata.n_vars, dtype = bool)

    if "ms" in adata.layers.keys() and "mu" in adata.layers.keys():
        vgenes &= np.max(adata.layers["ms"] > 0, 0) > 0
        vgenes &= np.max(adata.layers["mu"] > 0, 0) > 0

    if min_r2 is not None and f"{vkey}.r2" in adata.var.keys():
        vgenes &= adata.var[f"{vkey}.r2"] > min_r2

    if min_ratio is not None and f"{vkey}.qreg.ratio" in adata.var.keys():
        vgenes &= adata.var[f"{vkey}.qreg.ratio"] > min_ratio

    if use_highly_variable and (use_highly_variable in adata.var.keys()):
        vgenes &= adata.var[use_highly_variable].values

    if np.sum(vgenes) < 2:
        warning("you seem to have very low signal in splicing dynamics.")
        warning("consider reducing the thresholds and be cautious with interpretations.")

    adata.var[f"{vkey}.genes"] = vgenes
    info("number of obtained velocity.genes:", np.sum(adata.var[f"{vkey}.genes"]))
    return None


def vals_to_csr(vals, rows, cols, shape, split_negative = False):

    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()
        return graph.tocsr(), graph_neg.tocsr()

    else: return graph.tocsr()


class velocity_graph_inference:
    
    def __init__(
        self,
        adata,
        vkey = "velocity",
        xkey = "ms",
        tkey = None,
        n_neighbors = None,
        sqrt_transform = None,
        n_recurse_neighbors = None,
        random_neighbors_at_max = None,
        gene_subset = None,
        approx = None,
        report = False,
        compute_uncertainties = None,
        neighbor_key = 'neighbors',
        mode_neighbors = "distances",
    ):
        
        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}.genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}.genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].toarray()[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )

        V = np.array(
            adata.layers[vkey].toarray()[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            xpca, pc, _, _, _, _, _ = pca(X, n_comp = 30, svd_solver = "arpack")
            self.X = np.array(xpca, dtype = np.float32)
            self.V = (V - V.mean(0)).dot(pc.T)
            self.V[V.sum(1) == 0] = 0
        
        else:
            self.X = np.array(X, dtype = np.float32)
            self.V = np.array(V, dtype = np.float32)
        
        self.v_raw = np.array(self.V)
        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}.params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis = 1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else: self.n_recurse_neighbors = 2

        if neighbor_key not in adata.uns.keys():
            error('you should run neighbors before inferring velocity graph.')

        if np.min((get_neighs(adata, neighbor_key, "distances") > 0).sum(1).A1) == 0:
            error("your neighbor graph seems to be corrupted.")

        if n_neighbors is None or n_neighbors <= get_n_neighs(adata, neighbor_key):
            self.indices = get_indices(
                dist = get_neighs(adata, neighbor_key, "distances"),
                n_neighbors = n_neighbors,
                mode_neighbors = mode_neighbors,
            )[0]

        else: error('specified n_neighbors is greater than the existing run.')
        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}.graph", f"{vkey}.graph.neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        if tkey in adata.obs.keys():
            self.t0 = adata.obs[tkey].astype("category").copy()
            init = min(self.t0) if isinstance(min(self.t0), int) else 0
            self.t0 = self.t0.cat.set_categories(
                np.arange(init, len(self.t0.cat.categories)), rename = True)
            self.t1 = self.t0.copy()
            self.t1 = self.t1.cat.set_categories(
                self.t0.cat.categories + 1, rename=True)
        
        else: self.t0 = None

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata
        self.neighbor_key = neighbor_key
        self.mode_neighbor = mode_neighbors


    def compute_cosines(
        self, n_jobs = None, backend = "loky", show_progress_bar: bool = True
    ):
        
        n_jobs = get_n_jobs(n_jobs = n_jobs)
        n_obs = self.X.shape[0]

        res = parallelize(
            self.compute_cosine_task,
            range(self.X.shape[0]),
            n_jobs = n_jobs,
            unit = "cells",
            backend = backend,
            as_array = False,
            show_progress_bar = show_progress_bar,
        )()

        uncertainties, vals, rows, cols = map(flatten, zip(*res))
        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0
        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape = (n_obs, n_obs), split_negative = True)

        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape = (n_obs, n_obs), split_negative = False)
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).toarray().flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)


    def compute_cosine_task(self, obs_idx, queue):

        vals, rows, cols, uncertainties = [], [], [], []
        if self.compute_uncertainties:
            moments = get_moments(
                self.adata, np.sign(self.v_raw), second_order = True,
                neighbor_key = self.neighbor_key, mode = self.mode_neighbor
            )

        for obs_id in obs_idx:
            if self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs
                )

                if self.t0 is not None:
                    t0, t1 = self.t0[obs_id], self.t1[obs_id]
                    if t0 >= 0 and t1 > 0:
                        
                        t1_idx = np.where(self.t0 == t1)[0]
                        if len(t1_idx) > len(neighs_idx):
                            t1_idx = np.random.choice(
                                t1_idx, len(neighs_idx), replace=False
                            )

                        if len(t1_idx) > 0:
                            neighs_idx = np.unique(np.concatenate([neighs_idx, t1_idx]))

                dX = self.X[neighs_idx] - self.X[obs_id, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)
                val = cosine_correlation(dX, self.V[obs_id])  # 40% of runtime

                if self.compute_uncertainties:
                    dX /= l2_norm(dX)[:, None]
                    uncertainties.extend(
                        np.nansum(dX**2 * moments[obs_id][None, :], 1)
                    )

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * obs_id)
                cols.extend(neighs_idx)

            if queue is not None:
                queue.put(1)

        if queue is not None:
            queue.put(None)

        return uncertainties, vals, rows, cols


def flatten(iterable):
    return [i for it in iterable for i in it]


def velocity_graph(
    adata,
    vkey = "velocity",
    xkey = "ms",
    tkey = None,
    n_neighbors = None,
    n_recurse_neighbors = None,
    random_neighbors_at_max = None,
    sqrt_transform = None,
    variance_stabilization = None,
    gene_subset = None,
    compute_uncertainties = None,
    approx = None,
    neighbor_key = 'neighbors',
    mode_neighbors = "distances",
    n_jobs = None,
    backend = "loky",
    show_progress_bar: bool = True,
):
    """
    Computes velocity graph based on cosine similarities.

    The cosine similarities are computed between velocities and potential cell state
    transitions, i.e. it measures how well a corresponding change in gene expression
    :math:`\delta_{ij} = x_j - x_i` matches the predicted change according to the
    velocity vector :math:`\nu_i`,

    Parameters
    ----------

    data: :class:`~anndata.AnnData`
        Annotated data matrix.

    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.

    xkey: `str` (default: `'Ms'`)
        Layer key to extract count data from.

    tkey: `str` (default: `None`)
        Observation key to extract time data from.

    n_neighbors: `int` or `None` (default: None)
        Use fixed number of neighbors or do recursive neighbor search (if `None`).

    n_recurse_neighbors: `int` (default: `None`)
        Number of recursions for neighbors search. Defaults to 2 if mode_neighbors is 
        'distances', and 1 if mode_neighbors is 'connectivities'.

    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual cell is higher than this
        threshold, a random selection of such are chosen as reference neighbors.

    sqrt_transform: `bool` (default: `False`)
        Whether to variance-transform the cell states changes and velocities before 
        computing cosine similarities.

    gene_subset: `list` of `str`, subset of adata.var_names or `None`(default: `None`)
        Subset of genes to compute velocity graph on exclusively.

    compute_uncertainties: `bool` (default: `None`)
        Whether to compute uncertainties along with cosine correlation.

    approx: `bool` or `None` (default: `None`)
        If True, first 30 pc's are used instead of the full count matrix
        
    mode_neighbors: 'str' (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or 'connectivities'. 
        The latter yields a symmetric graph.

    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.

    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid options.
    
    show_progress_bar
        Whether to show a progress bar.
    """

    if vkey not in adata.layers.keys():
        error('should run velocity() first.')
    
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = velocity_graph_inference(
        adata,
        vkey = vkey,
        xkey = xkey,
        tkey = tkey,
        n_neighbors = n_neighbors,
        approx = approx,
        n_recurse_neighbors = n_recurse_neighbors,
        random_neighbors_at_max = random_neighbors_at_max,
        sqrt_transform = sqrt_transform,
        gene_subset = gene_subset,
        compute_uncertainties = compute_uncertainties,
        report = True,
        neighbor_key = neighbor_key,
        mode_neighbors = mode_neighbors,
    )

    n_jobs = get_n_jobs(n_jobs = n_jobs)
    info(f"computing velocity graph (using {n_jobs}/{os.cpu_count()} cores)")
    vgraph.compute_cosines(
        n_jobs = n_jobs, 
        backend = backend, 
        show_progress_bar = show_progress_bar
    )

    adata.uns[f"{vkey}.graph"] = vgraph.graph
    adata.uns[f"{vkey}.graph.neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}.graph.uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}.self.transition"] = vgraph.self_prob

    if f"{vkey}.params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}.params"]:
            del adata.uns[f"{vkey}.params"]["embeddings"]
    else: adata.uns[f"{vkey}.params"] = {}
    adata.uns[f"{vkey}.params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}.params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors
    return


def quiver_autoscale(x_emb, v_emb):

    import matplotlib.pyplot as pl
    
    # just so that it handles very large values
    scale_factor = np.abs(x_emb).max()  
    fig, ax = pl.subplots()
    Q = ax.quiver(
        x_emb[:, 0] / scale_factor,
        x_emb[:, 1] / scale_factor,
        v_emb[:, 0],
        v_emb[:, 1],
        angles = "xy",
        scale_units = "xy",
        scale = None,
    )

    Q._init()
    fig.clf()
    pl.close(fig)
    return Q.scale / scale_factor


def velocity_embedding(
    adata,
    basis = 'umap',
    vkey = "velocity",
    neighbor_key = 'neighbors',
    scale = 10,
    self_transitions = True,
    use_negative_cosines = True,
    direct_pca_projection = None,
    retain_scale = False,
    autoscale = True,
    all_comps = True,
    T = None
):
    """
    Projects the single cell velocities into any embedding.
    Given normalized difference of the embedding positions

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix.

    basis: str (default: `'umap'`)
        Which embedding to use.

    vkey: str (default: `'velocity'`)
        Name of velocity estimates to be used.

    scale: int (default: 10)
        Scale parameter of gaussian kernel for transition matrix.

    self_transitions: bool (default: `True`)
        Whether to allow self transitions, based on the confidences of transitioning to
        neighboring cells.

    use_negative_cosines: bool (default: `True`)
        Whether to project cell-to-cell transitions with negative cosines into
        negative/opposite direction.

    direct_pca_projection: bool (default: `None`)
        Whether to directly project the velocities into PCA space, thus skipping 
        the velocity graph.

    retain_scale: bool (default: `False`)
        Whether to retain scale from high dimensional space in embedding.

    autoscale: bool (default: `True`)
        Whether to scale the embedded velocities by a scalar multiplier,
        which simply ensures that the arrows in the embedding are properly scaled.

    all_comps: bool (default: `True`)
        Whether to compute the velocities on all embedding components.

    T: csr_matrix (default: `None`)
        Allows the user to directly pass a transition matrix.
        
    Returns
    -------
    velocity.{embedding}: `.obsm`
        coordinates of velocity projection on embedding (e.g., basis = 'umap')
    """

    basis = "pca" if direct_pca_projection else basis
    if basis not in adata.obsm_keys():
        error(f"you need to compute the embedding `{basis}` first.")

    if direct_pca_projection and "pca" in basis:
        warning("directly projecting velocities into pca space is merely for exploratory analysis on principal components.")
        warning("it does not reflect the actual velocity field from high dimensional gene expression space.")
        warning("to visualize velocities, consider applying `direct_pca_projection = False`.")

    info("computing velocity embedding ...")

    V = np.array(adata.layers[vkey])
    vgenes = np.ones(adata.n_vars, dtype = bool)
    if f"{vkey}.genes" in adata.var.keys():
        vgenes &= np.array(adata.var[f"{vkey}.genes"], dtype = bool)
    vgenes &= ~ np.isnan(V.sum(0))
    V = V[:, vgenes]

    if direct_pca_projection and "pca" in basis:
        pcomp = adata.varm["pca"] if all_comps else adata.varm["pca"][:, :2]
        pcomp = pcomp[vgenes]
        x_emb = adata.obsm[basis]
        v_emb = (V - V.mean(0)).dot(pcomp)

    else:
        x_emb = adata.obsm[basis] if all_comps else \
            adata.obsm[basis][:, :2]
        v_emb = np.zeros(x_emb.shape)

        T = transition_matrix(
            adata,
            vkey = vkey,
            scale = scale,
            self_transitions = self_transitions,
            use_negative_cosines = use_negative_cosines,
            neighbor_key = neighbor_key
        ) if T is None else T
        
        T.setdiag(0)
        T.eliminate_zeros()

        densify = adata.n_obs < 1e4
        TA = T.toarray() if densify else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(adata.n_obs):
                indices = T[i].indices
                dX = x_emb[indices] - x_emb[i, None]  # shape (n_neighbors, 2)
                if not retain_scale:
                    dX /= l2_norm(dX)[:, None]
                dX[np.isnan(dX)] = 0  # zero diff in a steady-state
                probs = TA[i, indices] if densify else T[i].data
                v_emb[i] = probs.dot(dX) - probs.mean() * dX.sum(0)

        if retain_scale:
            X = adata.layers["ms"] \
                if "ms" in adata.layers.keys() \
                else adata.layers["spliced"]
            delta = T.dot(X[:, vgenes]) - X[:, vgenes]
            if issparse(delta): delta = delta.toarray()
            cos_proj = (V * delta).sum(1) / l2_norm(delta)
            v_emb *= np.clip(cos_proj[:, None] * 10, 0, 1)

    if autoscale: v_emb /= 3 * quiver_autoscale(x_emb, v_emb)

    if f"{vkey}.params" in adata.uns.keys():
        adata.uns[f"{vkey}.params"]["embeddings"] = \
            [] if "embeddings" not in adata.uns[f"{vkey}.params"] \
            else list(adata.uns[f"{vkey}.params"]["embeddings"])
        adata.uns[f"{vkey}.params"]["embeddings"].extend([basis])

    vkey += f".{basis}"
    adata.obsm[vkey] = v_emb


def velocity_confidence(adata, vkey = "velocity", neighbor_key = 'neighbors'):
    """
    Computes confidences of velocities.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix.

    vkey: str (default: `'velocity'`)
        Name of velocity estimates to be used.

    copy: bool (default: `False`)
        Return a copy instead of writing to adata.

    Returns
    -------

    velocity.length: `.obs`
        Length of the velocity vectors for each individual cell

    velocity.confidence: `.obs`
        Confidence for each cell
    """

    if vkey not in adata.layers.keys():
        error("You need to run `velocity()` first.")

    V = np.array(adata.layers[vkey])

    tmp_filter = np.invert(np.isnan(np.sum(V, axis = 0)))
    if f"{vkey}.genes" in adata.var.keys():
        tmp_filter &= np.array(adata.var[f"{vkey}.genes"], dtype = bool)
    if "spearmans.score" in adata.var.keys():
        tmp_filter &= adata.var["spearmans.score"].values > 0.1

    V = V[:, tmp_filter]
    V -= V.mean(1)[:, None]
    vnorm = l2_norm(V, axis = 1)
    R = np.zeros(adata.n_obs)

    indices = get_indices(dist = get_neighs(adata, neighbor_key, "distances"))[0]
    for i in range(adata.n_obs):
        vi_neighs = V[indices[i]]
        vi_neighs -= vi_neighs.mean(1)[:, None]
        R[i] = np.mean(
            np.einsum("ij, j", vi_neighs, V[i]) / 
            (l2_norm(vi_neighs, axis=1) * vnorm[i])[None, :]
        )

    adata.obs[f"{vkey}.length"] = vnorm.round(2)
    adata.obs[f"{vkey}.confidence"] = np.clip(R, 0, None)

    if f"{vkey}.confidence.transition" not in adata.obs.keys():
        velocity_confidence_transition(adata, vkey)


def velocity_confidence_transition(
    adata, vkey = "velocity", neighbor_key = 'neighbors', 
    scale = 10
):
    if vkey not in adata.layers.keys():
        error("You need to run `velocity()` first.")

    X = np.array(adata.layers["ms"])
    V = np.array(adata.layers[vkey])

    tmp_filter = np.invert(np.isnan(np.sum(V, axis=0)))
    if f"{vkey}.genes" in adata.var.keys():
        tmp_filter &= np.array(adata.var[f"{vkey}.genes"], dtype = bool)
    if "spearmans.score" in adata.var.keys():
        tmp_filter &= adata.var["spearmans.score"].values > 0.1

    V = V[:, tmp_filter]
    X = X[:, tmp_filter]

    T = transition_matrix(adata, vkey = vkey, scale = scale, neighbor_key = neighbor_key)
    dX = T.dot(X) - X
    dX -= dX.mean(1)[:, None]
    V -= V.mean(1)[:, None]

    norms = l2_norm(dX, axis = 1) * l2_norm(V, axis = 1)
    norms += norms == 0
    adata.obs[f"{vkey}.confidence.transition"] = prod_sum(dX, V, axis = 1) / norms

