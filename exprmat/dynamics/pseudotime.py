
import numpy as np
from scipy.sparse import issparse, linalg, spdiags, csr_matrix
from scanpy.tools._dpt import DPT

from exprmat.ansi import error, warning, info
from exprmat.dynamics.velocity import velocity_graph_inference
from exprmat.dynamics.velocity import transition_matrix
from exprmat.dynamics.utils import (
    get_connectivities, groups_to_bool, 
    scale, strings_to_categoricals, get_plasticity_score
)


def velocity_map(adata = None, T = None, n_dcs = 10, return_model = False):

    vpt = velocity_pseudotime_model(adata, n_dcs = n_dcs)
    if T is None:
        T = adata.uns["velocity.graph"] - adata.uns["velocity.graph.neg"]
        vpt._connectivities = T + T.T
    vpt.compute_transitions()
    vpt.compute_eigen(n_dcs)
    adata.obsm["vmap"] = vpt.eigen_basis
    return vpt if return_model else None


class velocity_pseudotime_model(DPT):
    
    def set_iroot(self, neighbor_key = 'neighbors', root = None):

        if (
            isinstance(root, str)
            and root in self._adata.obs.keys()
            and self._adata.obs[root].max() != 0
        ):
            self.iroot = get_connectivities(self._adata, neighbor_key).dot(self._adata.obs[root])
            self.iroot = scale(self.iroot).argmax()
        elif isinstance(root, str) and root in self._adata.obs_names:
            self.iroot = self._adata.obs_names.get_loc(root)
        elif isinstance(root, (int, np.integer)) and root < self._adata.n_obs:
            self.iroot = root
        else: self.iroot = None


    def compute_transitions(self, density_normalize = True):

        T = self._connectivities
        if density_normalize:
            q = np.asarray(T.sum(axis=0))
            q += q == 0
            Q = spdiags(1.0 / q, 0, T.shape[0], T.shape[0]) \
                if issparse(T) else np.diag(1.0 / q)
            K = Q.dot(T).dot(Q)

        else: K = T
        z = np.sqrt(np.asarray(K.sum(axis=0)))

        Z = spdiags(1.0 / z, 0, K.shape[0], K.shape[0]) \
            if issparse(K) else np.diag(1.0 / z)
        self._transitions_sym = Z.dot(K).dot(Z)


    def compute_eigen(self, n_comps = 10, sym = None, sort = "decrease"):

        if self._transitions_sym is None:
            error("run `compute_transitions()` first.")
        n_comps = min(self._transitions_sym.shape[0] - 1, n_comps)
        evals, evecs = linalg.eigsh(self._transitions_sym, k=n_comps, which="LM")
        self._eigen_values = evals[::-1]
        self._eigen_basis = evecs[:, ::-1]


    def compute_pseudotime(self, inverse = False):

        if self.iroot is not None:
            self._set_pseudotime()
            self.pseudotime = 1 - self.pseudotime if inverse else self.pseudotime
            self.pseudotime[~ np.isfinite(self.pseudotime)] = np.nan
        else:
            self.pseudotime = np.empty(self._adata.n_obs)
            self.pseudotime[:] = np.nan


def velocity_pseudotime(
    adata,
    vkey = "velocity",
    neighbor_key = 'neighbors',
    groupby = None,
    groups = None,
    root_key = 'root.cells',
    end_key = 'endpoints',
    n_dcs = 10,
    use_velocity_graph = True,
    save_diffmap = None,
    return_model = None,
    **kwargs,
):
    """
    Computes a pseudotime based on the velocity graph.

    Velocity pseudotime is a random-walk based distance measures on the velocity graph.
    After computing a distribution over root cells obtained from the velocity-inferred
    transition matrix, it measures the average number of steps it takes to reach a cell
    after start walking from one of the root cells. Contrarily to diffusion pseudotime,
    it implicitly infers the root cells and is based on the directed velocity graph
    instead of the similarity-based diffusion kernel.

    Parameters
    ----------
    adata: anndata.AnnData
        Annotated data matrix

    vkey: str (default: `'velocity'`)
        Name of velocity estimates to be used.

    groupby: str, list or np.ndarray (default: `None`)
        Key of observations grouping to consider.

    groups: str, list or np.ndarray (default: `None`)
        Groups selected to find terminal states on. Must be an element of adata.obs[groupby]. 
        Only to be set, if each group is assumed to have a distinct lineage with an independent 
        root and end point.

    root_key: `int` (default: `None`)
        Index of root cell to be used. Computed from velocity-inferred transition matrix if not specified.

    end_key: int (default: `None`)
        Index of end point to be used. Computed from velocity-inferred transition matrix if not specified.

    n_dcs: int (default: 10)
        The number of diffusion components to use.

    use_velocity_graph: bool (default: `True`)
        Whether to use the velocity graph. If False, it uses the similarity-based diffusion kernel.
    
    save_diffmap: bool (default: `None`)
        Whether to store diffmap coordinates.

    return_model: bool (default: `None`)
        Whether to return the vpt object for further inspection.
    """

    strings_to_categoricals(adata)
    if root_key is None and "root.cells" in adata.obs.keys():
        root0 = adata.obs["root.cells"][0]
        if not np.isnan(root0) and not isinstance(root0, str): root_key = "root.cells"
    if end_key is None and "endpoints" in adata.obs.keys():
        end0 = adata.obs["endpoints"][0]
        if not np.isnan(end0) and not isinstance(end0, str): end_key = "endpoints"

    groupby = "fate" if groupby is None and "fate" in adata.obs.keys() else groupby
    if groupby is not None:
        warning("only set groupby when you have evident distinct clusters/lineages.")
        warning("each with own root and endpoint.")

    categories = (
        adata.obs[groupby].cat.categories
        if groupby is not None and groups is None
        else [None]
    )

    for cat in categories:
        groups = cat if cat is not None else groups
        if (
            root_key is None
            or root_key in adata.obs.keys()
            and np.max(adata.obs[root_key]) == np.min(adata.obs[root_key])
        ):
            terminal_states(adata, vkey = vkey, groupby = groupby, groups = groups)
            root_key, end_key = "root.cells", "endpoints"

        cell_subset = groups_to_bool(adata, groups = groups, groupby = groupby)
        data = adata.copy() if cell_subset is None else adata[cell_subset].copy()
        if "allow_kendall_tau_shift" not in kwargs:
            kwargs["allow_kendall_tau_shift"] = True

        vpt = velocity_pseudotime_model(
            data, n_dcs = n_dcs, neighbors_key = neighbor_key, **kwargs)

        if use_velocity_graph:
            T = data.uns[f"{vkey}.graph"] - data.uns[f"{vkey}.graph.neg"]
            vpt._connectivities = T + T.T

        vpt.compute_transitions()
        vpt.compute_eigen(n_comps = n_dcs)

        vpt.set_iroot(neighbor_key = neighbor_key, root = root_key)
        vpt.compute_pseudotime()
        dpt_root = vpt.pseudotime

        if end_key is not None:
            vpt.set_iroot(end_key)
            vpt.compute_pseudotime(inverse = True)
            dpt_end = vpt.pseudotime

            # merge dpt_root and inverse dpt_end together
            vpt.pseudotime = np.nan_to_num(dpt_root) + np.nan_to_num(dpt_end)
            vpt.pseudotime[np.isfinite(dpt_root) & np.isfinite(dpt_end)] /= 2
            vpt.pseudotime = scale(vpt.pseudotime)
            vpt.pseudotime[np.isnan(dpt_root) & np.isnan(dpt_end)] = np.nan

        if "n_branchings" in kwargs and kwargs["n_branchings"] > 0:
            vpt.branchings_segments()
        else: vpt.indices = vpt.pseudotime.argsort()

        if f"{vkey}.pseudotime" not in adata.obs.keys():
            pseudotime = np.empty(adata.n_obs)
            pseudotime[:] = np.nan
        else: pseudotime = adata.obs[f"{vkey}.pseudotime"].values
        pseudotime[cell_subset] = vpt.pseudotime
        adata.obs[f"{vkey}.pseudotime"] = np.array(pseudotime, dtype=np.float64)

        if save_diffmap:
            diffmap = np.empty(shape = (adata.n_obs, n_dcs))
            diffmap[:] = np.nan
            diffmap[cell_subset] = vpt.eigen_basis
            adata.obsm[f"vdiff.{groups}" if groups is not None else 'vdiff'] = diffmap

    return vpt if return_model else None


def cell_fate(
    adata,
    groupby = "leiden",
    neighbor_key: str = 'neighbors',
    disconnected_groups = None,
    self_transitions = False,
    n_neighbors = None
):
    '''
    Computes individual cell endpoints.

    Parameters
    ----------
    data: anndata.AnnData
        Annotated data matrix.

    groupby: str (default: `'clusters'`)
        Key to which to assign the fates.

    disconnected_groups: list of str (default: `None`)
        Which groups to treat as disconnected for fate assignment.

    self_transitions: bool (default: `False`)
        Whether to include self-transitions.

    n_neighbors: int (default: `None`)
        Number of neighbors to restrict transitions to.

    Returns
    -------
    cell_fate: `.obs`
        most likely cell fate for each individual cell
    cell_fate_confidence: `.obs`
        confidence of transitioning to the assigned fate
    '''

    info("computing cell fates")
    n_neighbors = 10 if n_neighbors is None else n_neighbors
    _adata = adata.copy()
    vgraph = velocity_graph_inference(
        _adata, n_neighbors = n_neighbors, approx = True, 
        neighbor_key = neighbor_key, n_recurse_neighbors = 1
    )

    vgraph.compute_cosines()
    _adata.uns["velocity.graph"] = vgraph.graph
    _adata.uns["velocity.graph.neg"] = vgraph.graph_neg

    T = transition_matrix(
        _adata, self_transitions = self_transitions,
        neighbor_key = neighbor_key
    )

    fate = np.linalg.inv(np.eye(_adata.n_obs) - T)
    if issparse(T): fate = fate.toarray()
    cell_fates = np.array(_adata.obs[groupby][fate.argmax(1)])
    if disconnected_groups is not None:
        idx = _adata.obs[groupby].isin(disconnected_groups)
        cell_fates[idx] = _adata.obs[groupby][idx]

    adata.obs["cell.fate"] = cell_fates
    adata.obs["cell.fate.confidence"] = fate.max(1) / fate.sum(1)
    strings_to_categoricals(adata)


def cell_origin(
    adata,
    groupby = "clusters",
    neighbor_key: str = 'neighbors',
    disconnected_groups = None,
    self_transitions = False,
    n_neighbors = None
):
    
    info("computing cell origins")

    n_neighbors = 10 if n_neighbors is None else n_neighbors
    _adata = adata.copy()
    vgraph = velocity_graph_inference(
        _adata, n_neighbors = n_neighbors, approx = True, 
        neighbor_key = neighbor_key, n_recurse_neighbors = 1
    )

    vgraph.compute_cosines()
    _adata.uns["velocity.graph"] = vgraph.graph
    _adata.uns["velocity.graph.neg"] = vgraph.graph_neg

    T = transition_matrix(
        _adata, 
        self_transitions = self_transitions, 
        neighbor_key = neighbor_key, 
        backward = True
    )

    fate = np.linalg.inv(np.eye(_adata.n_obs) - T)
    if issparse(T): fate = fate.toarray()
    cell_fates = np.array(_adata.obs[groupby][fate.argmax(1)])
    if disconnected_groups is not None:
        idx = _adata.obs[groupby].isin(disconnected_groups)
        cell_fates[idx] = _adata.obs[groupby][idx]

    adata.obs["cell.origin"] = cell_fates
    adata.obs["cell.origin.confidence"] = fate.max(1) / fate.sum(1)
    strings_to_categoricals(adata)


def eigs(T, k = 10, eps = 1e-3, perc = None, random_state = None, v0 = None):

    if random_state is not None:
        np.random.seed(random_state)
        v0 = np.random.rand(min(T.shape))
    
    try:
        # find k eigs with largest real part, and sort in descending order of eigenvals
        eigvals, eigvecs = linalg.eigs(T.T, k = k, which = "LR", v0 = v0)
        p = np.argsort(eigvals)[::-1]
        eigvals = eigvals.real[p]
        eigvecs = eigvecs.real[:, p]

        # select eigenvectors with eigenvalue of 1 - eps.
        idx = eigvals >= 1 - eps
        eigvals = eigvals[idx]
        eigvecs = np.absolute(eigvecs[:, idx])

        if perc is not None:
            lbs, ubs = np.percentile(eigvecs, perc, axis=0)
            eigvecs[eigvecs < lbs] = 0
            eigvecs = np.clip(eigvecs, 0, ubs)
            eigvecs /= eigvecs.max(0)

    except ValueError as e:
        warning(f"failed to fine k = {k} egenvalues with real part: {e}")
        eigvals, eigvecs = np.empty(0), np.zeros(shape=(T.shape[0], 0))

    return eigvals, eigvecs


def verify_roots(adata, roots, modality = "ms"):

    if "gene.count.corr" in adata.var.keys():
        p = get_plasticity_score(adata, modality)
        p_ub, root_ub = p > 0.5, roots > 0.9
        n_right_assignments = np.sum(root_ub * p_ub) / np.sum(p_ub)
        n_false_assignments = np.sum(root_ub * np.invert(p_ub)) / np.sum(
            np.invert(p_ub)
        )

        n_randn_assignments = np.mean(root_ub)
        if n_right_assignments > 3 * n_randn_assignments:  # mu + 2*mu (std=mu)
            roots *= p_ub
        elif (
            n_false_assignments > n_randn_assignments
            or n_right_assignments < n_randn_assignments
        ): warning("uncertain or fuzzy root cell identification.")

    return roots


def write_to_obs(adata, key, vals, cell_subset=None):

    if cell_subset is None:
        adata.obs[key] = vals
    
    else:
        vals_all = (
            adata.obs[key].copy() 
            if key in adata.obs.keys() 
            else np.zeros(adata.n_obs)
        )

        vals_all[cell_subset] = vals
        adata.obs[key] = vals_all


def terminal_states(
    adata,
    vkey = "velocity",
    neighbor_key: str = 'neighbors',
    modality = "ms",
    groupby = None,
    groups = None,
    self_transitions = False,
    eps = 1e-3,
    random_state = 42,
    **kwargs,
):
    """
    Computes terminal states (root and endpoints).

    The end points and root cells are obtained as stationary states of the
    velocity-inferred transition matrix and its transposed, respectively,
    which is given by left eigenvectors corresponding to an eigenvalue of 1
    """
    
    info("computing terminal states ...")
    strings_to_categoricals(adata)

    if groupby is not None:
        warning("only set groupby when you have evident distinct clusters/lineages.")
        warning("each with own root and endpoint.")

    kwargs.update({"self_transitions": self_transitions})
    categories = [None]
    if groupby is not None and groups is None:
        categories = adata.obs[groupby].cat.categories
    
    for cat in categories:

        groups = cat if cat is not None else groups
        cell_subset = groups_to_bool(adata, groups=groups, groupby=groupby)
        _adata = adata if groups is None else adata[cell_subset]
        connectivities = get_connectivities(_adata, neighbor_key = neighbor_key, mode = "distances")

        T = transition_matrix(
            _adata, vkey = vkey, backward = True, 
            neighbor_key = neighbor_key, **kwargs
        )

        eigvecs_roots = eigs(T, eps=eps, perc=[2, 98], random_state = random_state)[1]
        roots = csr_matrix.dot(connectivities, eigvecs_roots).sum(1)
        roots = scale(np.clip(roots, 0, np.percentile(roots, 98)))
        roots = verify_roots(_adata, roots, modality)
        write_to_obs(adata, "root.cells", roots, cell_subset)

        T = transition_matrix(_adata, vkey=vkey, backward=False, **kwargs)
        eigvecs_ends = eigs(T, eps=eps, perc=[2, 98], random_state = random_state)[1]
        ends = csr_matrix.dot(connectivities, eigvecs_ends).sum(1)
        ends = scale(np.clip(ends, 0, np.percentile(ends, 98)))
        write_to_obs(adata, "endpoints", ends, cell_subset)

        n_roots, n_ends = eigvecs_roots.shape[1], eigvecs_ends.shape[1]
        groups_str = f" ({groups})" if isinstance(groups, str) else ""
        roots_str = f"{n_roots} {'regions' if n_roots > 1 else 'region'}"
        ends_str = f"{n_ends} {'regions' if n_ends > 1 else 'region'}"

        info(f"identified {roots_str} of root cells and {ends_str} of end points {groups_str}.")

    return