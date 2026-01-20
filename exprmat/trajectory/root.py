
from typing import Optional, Union
from anndata import AnnData
import numpy as np
import pandas as pd
import igraph

import itertools
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from exprmat.utils import error, warning, info


def define_root(
    adata: AnnData,
    root: Union[int, str],
    tips_only: bool = False,
    min_val: bool = False,
    trajectory_key = 'ppt',
):
    """
    Define the root of the trajectory.

    Parameters
    ----------
    adata
        Annotated data matrix.

    root
        Either an Id (int) of the tip of the fork to be considered as a root. 
        Or a key (str) from obs/X (such as CytoTRACE) for automatic selection.
        
    layer
        If key is in X, choose which layer to use for the averaging.
    """

    if f"{trajectory_key}.graph" not in adata.uns:
        error("you need to run `principle_tree` first to compute a princal graph before choosing a root.")

    graph = adata.uns[f"{trajectory_key}.graph"]
    circle = len(graph["tips"]) == 0

    if type(root) == str:
        if root in adata.obs:
            root_val = adata.obs[root]
        else: error(f"{root} not present in adata.obs")
        info("automatic root selection using " + root + " values", time = False)
        nodes = np.arange(adata.obsm[trajectory_key].shape[1])
        avgs = pd.Series(np.nan, index = nodes)
        
        # handle empty nodes
        unassigned = np.array([adata.obsm[trajectory_key][:, n].sum() for n in nodes]) > 0
        nodes = nodes[unassigned]
        avgs_temp = list(
            map(lambda n: np.average(
                root_val, weights = adata.obsm[trajectory_key][:, n]), nodes,
        ))

        avgs.loc[nodes] = avgs_temp
        if tips_only:
            mask = np.ones(avgs.shape, bool)
            mask[adata.uns[f"{trajectory_key}.graph"]["tips"]] = False
            avgs[mask] = 0

        if min_val:
            if tips_only: avgs[mask] = avgs.max()
            root = np.argmin(avgs)

        else: root = np.argmax(avgs)

    if circle:

        d = 1e-6 + pairwise_distances(
            graph["points"].T, graph["points"].T, metric = graph["metrics"])
        to_g = graph["adjacencies"] * d
        csr = csr_matrix(to_g)

        g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode = "undirected")
        g.es["weight"] = to_g[to_g.nonzero()]
        root_dist_matrix = shortest_path(csr, directed = False, indices = root)
        
        pp_info = pd.DataFrame({
            "pp": g.vs.indices,
            "time": root_dist_matrix,
            "seg": np.zeros(csr.shape[0]),
        })

        furthest = pp_info.time.idxmax()
        pp_info.loc[g.get_shortest_paths(
            v = root, to = furthest, weights = "weight")[0][:-1], "seg"
        ] = 1

        s = pp_info.seg.copy()
        s[furthest] = 0
        g.vs["group"] = s
        g.vs["label"] = s.index

        sub_g = g.vs.select(group = 0).subgraph()
        a, b = np.argwhere(np.array(sub_g.degree()) == 1).ravel()
        dst_0 = sub_g.distances(a, b, weights = "weight")[0][0]
        a, b = sub_g.vs["label"][a], sub_g.vs["label"][b]
        dst_1 = g.distances(root, furthest, weights = "weight")[0][0]
        pp_seg = pd.concat([
            pd.Series([0, a, b, dst_0], index=["n", "from", "to", "d"]),
            pd.Series([1, root, furthest, dst_1], index=["n", "from", "to", "d"]),
        ], axis = 1).T

        pp_seg.n = pp_seg.n.astype(int).astype(str)
        pp_seg["from"] = pp_seg["from"].astype(int)
        pp_seg.to = pp_seg.to.astype(int)
        pp_info.seg = pp_info.seg.astype(int).astype(str)

        adata.uns[f"{trajectory_key}.graph"]["root"] = root
        adata.uns[f"{trajectory_key}.graph"]["root2"] = pp_seg["from"][0]
        adata.uns[f"{trajectory_key}.graph"]["pp_info"] = pp_info
        adata.uns[f"{trajectory_key}.graph"]["pp_seg"] = pp_seg
        adata.uns[f"{trajectory_key}.graph"]["forks"] = np.array([furthest])
        adata.uns[f"{trajectory_key}.graph"]["tips"] = np.array([furthest, root, pp_seg["from"][0]])

    else:

        d = 1e-6 + pairwise_distances(
            graph["points"].T, graph["points"].T, metric = graph["metrics"]
        )

        to_g = graph["adjacencies"] * d
        csr = csr_matrix(to_g)
        g = igraph.Graph.Adjacency((to_g > 0).tolist(), mode = "undirected")
        g.es["weight"] = to_g[to_g.nonzero()]

        root_dist_matrix = shortest_path(csr, directed = False, indices = root)
        pp_info = pd.DataFrame({
            "pp": g.vs.indices,
            "time": root_dist_matrix,
            "seg": np.zeros(csr.shape[0]),
        })

        nodes = np.argwhere(
            np.apply_along_axis(arr = (csr > 0).toarray(), axis = 0, func1d = np.sum) != 2
        ).flatten()
        nodes = np.unique(np.append(nodes, root))

        pp_seg = list()
        for node1, node2 in itertools.combinations(nodes, 2):
            paths12 = g.get_shortest_paths(node1, node2)
            paths12 = np.array([val for sublist in paths12 for val in sublist])

            if np.sum(np.isin(nodes, paths12)) == 2:
                fromto = np.array([node1, node2])
                path_root = root_dist_matrix[[node1, node2]]
                fro = fromto[np.argmin(path_root)]
                to = fromto[np.argmax(path_root)]
                pp_info.loc[paths12, "seg"] = len(pp_seg) + 1
                pp_seg.append(
                    pd.DataFrame({
                        "n": len(pp_seg) + 1,
                        "from": fro,
                        "to": to,
                        "d": shortest_path(csr, directed=False, indices=fro)[to],
                    }, index=[len(pp_seg) + 1])
                )

        pp_seg = pd.concat(pp_seg, axis=0)
        pp_seg["n"] = pp_seg["n"].astype(int).astype(str)

        pp_seg["from"] = pp_seg["from"].astype(int)
        pp_seg["to"] = pp_seg["to"].astype(int)
        pp_info["seg"] = pp_info["seg"].astype(int).astype(str)
        graph["pp_info"] = pp_info
        graph["pp_seg"] = pp_seg
        graph["root"] = root
        adata.uns[f"{trajectory_key}.graph"] = graph

    info("node " + str(root) + " selected as a root")