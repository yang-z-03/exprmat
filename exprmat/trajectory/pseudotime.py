
from anndata import AnnData
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from typing import Optional, Union

from exprmat.utils import error, warning, info
from exprmat import pprog


class progress_parallel(Parallel):

    def __init__(
        self, use_tqdm = True, total = None, 
        file = None, desc = None, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self._desc = desc
        self._file = file
        super().__init__(*args, **kwargs)


    def __call__(self, *args, **kwargs):
        with pprog(
            disable = not self._use_tqdm,
            total = self._total,
            desc = self._desc,
            file = self._file,
        ) as self.pbar:
            return Parallel.__call__(self, *args, **kwargs)


    def print_progress(self):
        if self._total is None:
            self.pbar.total = self.n_dispatched_tasks
        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()


def pseudotime(
    adata: AnnData,
    n_jobs: int = 1,
    n_map: int = 1,
    seed: Optional[int] = None,
    trajectory_key = 'ppt',
    cmap = 'category20b'
):
    """
    Projects cells onto the tree, and uses distance from the root 
    as a pseudotime value.

    Parameters
    ----------
    adata
        Annotated data matrix.
        
    n_map
        number of probabilistic mapping of cells onto the tree to use. If n_map = 1 then 
        likelihood cell mapping is used.

    seed
        A numpy random seed for reproducibility for muliple mappings
    """

    if f"{trajectory_key}.graph" not in adata.uns:
        error("you need to run `principle_tree` first to compute a princal graph before choosing a root.")

    graph = adata.uns[f"{trajectory_key}.graph"]
    pp_seg = adata.uns[f"{trajectory_key}.graph"]["pp_seg"]
    pp_info = adata.uns[f"{trajectory_key}.graph"]["pp_info"]

    reassign, recolor = False, False
    if f"{trajectory_key}.milestones" in adata.obs:
        if adata.obs[f"{trajectory_key}.milestones"].dtype.name == "category":
            tmp_mil = adata.obs[f"{trajectory_key}.milestones"].cat.categories.copy()
            reassign = True
        if f"{trajectory_key}.milestones.colors" in adata.uns:
            tmp_mil_col = adata.uns[f"{trajectory_key}.milestones.colors"].copy()
            recolor = True

    info("projecting cells onto the principal graph")
    from sklearn.metrics import pairwise_distances
    P = pairwise_distances(graph["points"].T, metric = graph["metrics"])

    if n_map == 1:
        if graph["method"] == "ppt": 
            df_l = [map_cells(graph, R = adata.obsm[trajectory_key], P = P, multi = False, verbose = True)]
        else: df_l = [map_cells_epg(graph, adata)]
    
    else:
        if seed is not None:
            np.random.seed(seed)
            map_seeds = np.random.randint(999999999, size = n_map)
        else: map_seeds = [None for i in range(n_map)]
        
        df_l = progress_parallel(
            n_jobs = n_jobs, total = n_map, desc = "mappings"
        )(
            delayed(map_cells)(
                graph = graph, R = adata.obsm[trajectory_key], 
                P = P, multi = True, map_seed = map_seeds[m]
            ) for m in range(n_map)
        )

    # formatting cell projection data
    for i in range(len(df_l)): df_l[i].index = adata.obs_names
    df_summary = df_l[0]
    df_summary["seg"] = df_summary["seg"].astype("category")
    df_summary["edge"] = df_summary["edge"].astype("category")

    # remove pre-existing palette to avoid errors with plotting
    if f"{trajectory_key}.seg.colors" in adata.uns:
        del adata.uns[f"{trajectory_key}.seg.colors"]

    for col in df_summary.columns.tolist():
        adata.obs[f"{trajectory_key}." + (col if col != 't' else 'pseudotime')] = df_summary[col]

    names = np.arange(len(df_l)).astype(str).tolist()
    dictionary = dict(zip(names, df_l))
    adata.uns[f"{trajectory_key}.pseudotime"] = dictionary

    if n_map > 1:
        
        adata.obs[f"{trajectory_key}.pseudotime.sd"] = (pd.concat(
            list(map(
                lambda x: pd.Series(x["t"]),
                list(adata.uns[f"{trajectory_key}.pseudotime"].values()),
            )), axis = 1).apply(np.std, axis = 1).values
        )

        # reassign cells to their closest segment
        root = adata.uns[f"{trajectory_key}.graph"]["root"]
        tips = adata.uns[f"{trajectory_key}.graph"]["tips"]
        endpoints = tips[tips != root]

        allsegs = pd.concat([df.seg for df in adata.uns[f"{trajectory_key}.pseudotime"].values()], axis = 1)
        allsegs = allsegs.apply(lambda x: x.value_counts(), axis = 1)
        adata.obs[f"{trajectory_key}.seg"] = allsegs.idxmax(axis=1)
        adata.obs[f"{trajectory_key}.pseudotime"] = pd.concat(
            [df.t for df in adata.uns[f"{trajectory_key}.pseudotime"].values()], axis=1
        ).mean(axis=1)

        for s in pp_seg.n:
            df_seg = adata.obs.loc[adata.obs[f"{trajectory_key}.seg"] == s, f"{trajectory_key}.pseudotime"]

            # reassign cells below minimum pseudotime of their assigned seg
            if any(int(s) == pp_seg.index[pp_seg["from"] != root]):
                start_t = pp_info.loc[pp_seg["from"], "time"].iloc[int(s) - 1]
                cells_back = allsegs.loc[df_seg[df_seg < start_t].index]
                ncells = cells_back.shape[0]
                
                if ncells != 0:

                    filter_from = pd.concat(
                        [pp_info.loc[pp_seg["from"], "time"] for i in range(ncells)], axis = 1,
                    ).T.values
                    filter_to = pd.concat(
                        [pp_info.loc[pp_seg["to"], "time"] for i in range(ncells)], axis = 1,
                    ).T.values
                    t_cells = adata.obs.loc[cells_back.index, f"{trajectory_key}.pseudotime"]

                    boo = (filter_from < t_cells.values.reshape((-1, 1))) & (
                        filter_to > t_cells.values.reshape((-1, 1)))

                    cells_back = (cells_back.fillna(0) * boo).apply(
                        lambda x: x.index[np.argsort(x)][::-1], axis = 1)
                    cells_back = cells_back.apply(lambda x: x[x != s][0])
                    adata.obs.loc[cells_back.index, f"{trajectory_key}.seg"] = cells_back.values

            # reassign cells over maximum pseudotime of their assigned seg
            if any(int(s) == pp_seg.index[~ pp_seg.to.isin(endpoints)]):
                end_t = pp_info.loc[pp_seg["to"], "time"].iloc[int(s) - 1]
                cells_front = allsegs.loc[df_seg[df_seg > end_t].index]
                ncells = cells_front.shape[0]
                if ncells != 0:
                    filter_from = pd.concat(
                        [pp_info.loc[pp_seg["from"], "time"] for i in range(ncells)], axis = 1,
                    ).T.values
                    filter_to = pd.concat(
                        [pp_info.loc[pp_seg["to"], "time"] for i in range(ncells)], axis = 1,
                    ).T.values
                    t_cells = adata.obs.loc[cells_front.index, f"{trajectory_key}.pseudotime"]

                    boo = (filter_to > t_cells.values.reshape((-1, 1))) & (
                        filter_from < t_cells.values.reshape((-1, 1)))
                    cells_front = (cells_front.fillna(0) * boo).apply(
                        lambda x: x.index[np.argsort(x)][::-1], axis = 1)
                    cells_front = cells_front.apply(lambda x: x[x != s][0])
                    adata.obs.loc[cells_front.index, f"{trajectory_key}.seg"] = cells_front.values

    milestones = pd.Series(index = adata.obs_names, dtype = str)
    for seg in pp_seg.n:
        cell_seg = adata.obs.loc[adata.obs[f"{trajectory_key}.seg"] == seg, f"{trajectory_key}.pseudotime"]
        if len(cell_seg) > 0:
            milestones[cell_seg.index[
                (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) < 0)
            ]] = pp_seg.loc[int(seg), "from"]

            milestones[cell_seg.index[
                (cell_seg - min(cell_seg) - (max(cell_seg - min(cell_seg)) / 2) > 0)
            ]] = pp_seg.loc[int(seg), "to"]

    adata.obs[f"{trajectory_key}.milestones"] = milestones
    adata.obs[f"{trajectory_key}.milestones"] = adata.obs[f"{trajectory_key}.milestones"] \
        .fillna(-1).astype(int).astype("str").replace('-1', pd.NA).astype("category")

    adata.uns[f"{trajectory_key}.graph"]["milestones"] = {
        x: int(x) for x in adata.obs[f"{trajectory_key}.milestones"].cat.categories.tolist()
    }

    # setting consistent color palettes
    from exprmat.plotting.palettes import get_palette
    adata.uns[f"{trajectory_key}.milestones.colors"] = get_palette(
        cmap, len(adata.obs[f"{trajectory_key}.milestones"].cat.categories))
    
    while reassign:
        if "tmp_mil_col" not in locals(): break
        if len(tmp_mil_col) != len(adata.obs[f"{trajectory_key}.milestones"].cat.categories): break
        rename_milestones(adata, trajectory_key, tmp_mil)
        if recolor: adata.uns[f"{trajectory_key}.milestones.colors"] = tmp_mil_col
        reassign = False

    adata.uns[f"{trajectory_key}.seg.colors"] = [
        str(np.array(adata.uns[f"{trajectory_key}.milestones.colors"])[
            pd.Series(adata.uns[f"{trajectory_key}.graph"]["milestones"]) == int(t)
        ][0]) if str(t) in adata.uns[f"{trajectory_key}.graph"]["milestones"].keys()
        else '#e0e0e0' for t in adata.uns[f"{trajectory_key}.graph"]["pp_seg"]['to'].tolist()
    ]


def map_cells(graph, R, P, multi = False, map_seed = None, verbose = False):
    
    import igraph
    g = igraph.Graph.Adjacency((graph["adjacencies"] > 0).tolist(), mode = "undirected")
    # add edge weights and node labels.
    g.es["weight"] = np.array(P[graph["adjacencies"].nonzero()].ravel()).tolist()[0]

    if multi:
        np.random.seed(map_seed)
        rrm = (np.apply_along_axis(
            lambda x: np.random.choice(np.arange(len(x)), size = 1, p = x),
            axis = 1, arr = R,
        )).T.flatten()
    else: rrm = np.apply_along_axis(np.argmax, axis = 1, arr = R)

    def map_on_edges(v):

        vcells = np.argwhere(rrm == v)
        if vcells.shape[0] > 0:
            nv = np.array(g.neighborhood(v, order=1))[1:]
            nvd = g.distances(v, nv, weights=g.es["weight"])
            ndf = pd.DataFrame({
                "cell": vcells.flatten(),
                "v0": v,
                "v1": nv[np.argmin(nvd)],
                "d": np.min(nvd),
            })

            p0 = R[vcells, v].flatten()
            p1 = np.array(
                list(map(lambda x: R[vcells[x], ndf.v1[x]], range(len(vcells))))
            ).flatten()

            ndf["t"] = [
                np.average(
                    graph["pp_info"].time[ndf.iloc[i, [1, 2]].astype(int)],
                    weights=[p0[i], p1[i]],
                ) for i in range(ndf.shape[0])
            ]

            ndf["seg"] = '0'
            isinfork = (graph["pp_info"].loc[ndf.v0, "pp"].isin(graph["forks"])).values
            ndf.loc[isinfork, "seg"] = (
                graph["pp_info"].loc[ndf.loc[isinfork, "v1"], "seg"].values)
            ndf.loc[~isinfork, "seg"] = (
                graph["pp_info"].loc[ndf.loc[~isinfork, "v0"], "seg"].values)

            return ndf
        else: return None

    from exprmat import pprog
    df = list(map(map_on_edges, pprog(range(graph["adjacencies"].shape[1]), desc = 'mapping', disable = not verbose)))
    df = pd.concat(df)
    df.sort_values("cell", inplace = True)
    # df.index = graph["cells_fitted"]

    df["edge"] = df.apply(lambda x: str(int(x.iloc[1])) + "|" + str(int(x.iloc[2])), axis=1)
    df.drop(["cell", "v0", "v1", "d"], axis = 1, inplace = True)
    return df


def map_cells_epg(graph, adata):

    import elpigraph
    from exprmat.trajectory.elpi import get_data

    EPG = adata.uns["epg"]
    ndims_rep = None if "ndims_rep" not in graph else graph["ndims_rep"]
    X, use_rep = get_data(adata, graph["use_rep"], ndims_rep)
    elpigraph.utils.getPseudotime(X.values, EPG, graph["root"])
    edges = EPG["Edges"][0]
    eid = EPG["projection"]["edge_id"]

    df = pd.DataFrame({
        "t": EPG["pseudotime"],
        "seg": graph["pp_info"].loc[EPG["projection"]["node_id"], "seg"].values,
        "edge": ["|".join(edges[eid, :][i, :].astype(str)) for i in range(len(eid))],
    }, index = adata.obs_names)
    return df


def rename_milestones(adata, trajectory_key, new: Union[list, dict], copy: bool = False):

    adata = adata.copy() if copy else adata
    if isinstance(new, dict) is False:
        new = dict(zip(adata.obs[f"{trajectory_key}.milestones"].cat.categories, new))

    milestones = pd.Series(
        adata.uns[f"{trajectory_key}.graph"]["milestones"].keys(),
        index=adata.uns[f"{trajectory_key}.graph"]["milestones"].values(),
    )

    replace = pd.Series(new)
    replace.index = [(milestones == n).idxmax() for n in replace.index]
    milestones.loc[replace.index] = replace.values

    adata.uns[f"{trajectory_key}.graph"]["milestones"] = dict(zip(milestones.values, milestones.index))
    adata.obs[f"{trajectory_key}.milestones"] = adata.obs[f"{trajectory_key}.milestones"].cat.rename_categories(new)

    return adata if copy else None
