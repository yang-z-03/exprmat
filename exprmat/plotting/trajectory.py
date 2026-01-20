
import numpy as np
import pandas as pd
import igraph
from anndata import AnnData
import matplotlib.collections
from typing import Union, Optional, Sequence, Tuple, List

from pandas.api.types import is_categorical_dtype
from scanpy.plotting._utils import savefig_or_show

import matplotlib.patheffects as path_effects
import matplotlib.text as mtext
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, rgb2hex
from numba import njit
import math

from exprmat.plotting.milestone import milestones as milestones_plot
from exprmat import error, warning, info
from exprmat.plotting.palettes import mpl


def graph(
    adata: AnnData,
    background_color = None,
    basis: Union[None, str] = None,
    size_nodes: float = None,
    alpha_nodes: float = 1,
    linewidth: float = 2,
    alpha_seg: float = 1,
    color_cells: Union[str, None] = None,
    tips: bool = True,
    forks: bool = True,
    nodes: Optional[List] = [],
    rasterized: bool = True,
    ax = None,
    trajectory_key = 'ppt',
    embedding_dims = [0, 1],
    contour_plot = False,
    figsize = (3, 3),
    ** kwargs,
):

    if f"{trajectory_key}.graph" not in adata.uns:
        error(f'do not find trajectory key `{trajectory_key}`')

    graph = adata.uns[f"{trajectory_key}.graph"]
    emb = adata.obsm[basis]
    emb = emb[:, embedding_dims]
    R = adata.obsm[trajectory_key]
    proj = (np.dot(emb.T, R) / R.sum(axis = 0)).T
    B = graph["adjacencies"]

    from exprmat.reduction.plot import embedding
    
    if ax: embedding(
        adata, basis = basis, ax = ax, embedding_dims = embedding_dims, 
        contour_plot = contour_plot, color = background_color, **kwargs
    )
        
    else: ax = embedding(
        adata, basis = basis, embedding_dims = embedding_dims, figsize = figsize,
        contour_plot = contour_plot, color = background_color, **kwargs
    ).axes[0]

    al = np.array(igraph.Graph.Adjacency((B > 0).tolist(), mode = "undirected").get_edgelist())
    segs = al.tolist()
    vertices = proj.tolist()
    lines = [[tuple(vertices[j]) for j in i] for i in segs]
    lc = matplotlib.collections.LineCollection(
        lines, colors = "k", linewidths = linewidth, 
        alpha = alpha_seg, rasterized = rasterized
    )

    ax.add_collection(lc)
    ax.scatter(
        proj[:, 0], proj[:, 1], 
        s = size_nodes, c = "k", alpha = alpha_nodes, 
        rasterized = rasterized
    )

    bbox = dict(facecolor = "white", alpha = 0.6, edgecolor = "white", pad = 0.1)

    if tips:
        for tip in graph["tips"]:
            ax.annotate(
                tip,
                (proj[tip, 0], proj[tip, 1]),
                ha = "center",
                va = "center",
                xytext = (0, 0),
                textcoords = "offset points",
                bbox = bbox,
            )

    if forks:
        for fork in graph["forks"]:
            ax.annotate(
                fork,
                (proj[fork, 0], proj[fork, 1]),
                ha = "center",
                va = "center",
                xytext = (0, 0),
                textcoords = "offset points",
                bbox = bbox,
            )

    if nodes:
        for node in nodes:
            ax.annotate(
                node,
                (proj[node, 0], proj[node, 1]),
                ha = "center",
                va = "center",
                xytext = (0, 0),
                textcoords = "offset points",
                bbox = bbox,
            )
    
    return ax.figure
            

def trajectory(
    adata: AnnData,
    basis: Union[None, str] = None,
    color_seg = "segments",
    background_color = None,
    cmap_seg: str = "viridis",
    layer_seg: Union[str, None] = "fitted",
    perc_seg: Union[List, None] = None,
    color_cells: Union[str, None] = None,
    scale_path: float = 1,
    rasterized: bool = True,
    ax = None,
    trajectory_key = 'ppt',
    embedding_dims = [0, 1],
    contour_plot = False,
    figsize = (3, 3),
    annotate = False,
    annotate_fontsize = 9,
    **kwargs,
):
    
    from exprmat.reduction.plot import embedding
    if ax: embedding(
        adata, basis = basis, ax = ax, embedding_dims = embedding_dims, annotate = False,
        contour_plot = contour_plot, color = background_color, **kwargs
    )
        
    else: ax = embedding(
        adata, basis = basis, embedding_dims = embedding_dims, figsize = figsize, annotate = False,
        contour_plot = contour_plot, color = background_color, **kwargs
    ).axes[0]

    if f"{trajectory_key}.graph" not in adata.uns:
        raise ValueError("you need to run `principle_tree_pseudotime` first before plotting.")

    graph = adata.uns[f"{trajectory_key}.graph"]
    emb = adata.obsm[f"{basis}"]
    emb = emb[:, embedding_dims]
    R = adata.obsm[trajectory_key]

    nodes = graph["pp_info"].index
    proj = pd.DataFrame((np.dot(emb.T, R) / R.sum(axis=0)).T, index = nodes)

    B = graph["adjacencies"]
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode = "undirected")
    g.vs[:]["name"] = [v.index for v in g.vs]

    miles_ids = np.concatenate([graph["tips"], graph["forks"]])

    al = np.array(g.get_edgelist())
    edges = [g.vs[e.tolist()]["name"] for e in al]
    lines = [[tuple(proj.loc[j]) for j in i] for i in edges]
    miles_ids = miles_ids[np.isin(miles_ids, proj.index)]

    if color_seg == "milestones":
        from matplotlib.colors import LinearSegmentedColormap

        rev_dict = dict(zip(graph["milestones"].values(), graph["milestones"].keys()))
        miles_cat = adata.obs[f"{trajectory_key}.milestones"].cat.categories
        mil_col = np.array(adata.uns[f"{trajectory_key}.milestones.colors"])

        def get_milestones_gradients(i):
            start = graph["pp_seg"].iloc[i, 1]
            end = graph["pp_seg"].iloc[i, 2]
            mil_path = graph["pp_info"].time[g.get_all_shortest_paths(start, end)[0]]
            mil_path = (mil_path - mil_path.min()) / (mil_path.max() - mil_path.min())
            start_col = mil_col[miles_cat == rev_dict[start]][0] if start in rev_dict.keys() else '#e0e0e0'
            end_col = mil_col[miles_cat == rev_dict[end]][0] if end in rev_dict.keys() else '#e0e0e0'

            edges_mil = pd.Series([
                mil_path[[first, second]].mean()
                for first, second in zip(mil_path.index, mil_path.index[1:])],
                index = [(first, second) for first, second in zip(mil_path.index, mil_path.index[1:])]
            )

            cmap = LinearSegmentedColormap.from_list("mil", [start_col, end_col])
            return pd.Series(
                [rgb2hex(c) for c in cmap(edges_mil)], index = edges_mil.index
            )

        edge_colors = pd.concat(
            [get_milestones_gradients(i) for i in range(graph["pp_seg"].shape[0])]
        )

        edges_tuples = [tuple(e) for e in edges]
        edge_colors.index = [
            e if any((np.array(edges_tuples) == e).sum(axis=1) == 2) else e[::-1]
            for e in edge_colors.index
        ]
        edge_colors = edge_colors[edges_tuples]

        color_segs = [hex2color(c) for c in edge_colors]
        color_mils = []
        for m in miles_ids:
            if m in rev_dict.keys(): color_mils.append(mil_col[miles_cat.tolist().index(rev_dict[m])])
            else: color_mils.append('#e0e0e0')


    elif color_seg == "segments":
        
        seg_edges = (
            graph["pp_info"]
            .loc[np.array(edges).ravel(), "seg"]
            .values.reshape(-1, 2)[:, 0]
        )

        seg_col = pd.Series(
            adata.uns[f"{trajectory_key}.seg.colors"], 
            index = graph["pp_seg"].n
        )

        color_segs = [hex2color(seg_col.loc[s]) for s in seg_edges]
        
        color_mils = []
        for m in miles_ids:
            if m in graph["milestones"].values(): 
                color_mils.append(hex2color(seg_col.loc[graph["pp_info"].loc[m].seg]))
            else: color_mils.append('#e0e0e0')
    

    lc = matplotlib.collections.LineCollection(
        lines,
        colors = "k",
        linewidths = 7.5 * scale_path,
        zorder = 100,
        path_effects = [path_effects.Stroke(capstyle = "round")],
        rasterized = rasterized,
    )

    ax.add_collection(lc)
    g = igraph.Graph.Adjacency((B > 0).tolist(), mode = "undirected")
    seg = graph["pp_seg"].loc[:, ["from", "to"]].values.tolist()
    bbox = dict(facecolor = "white", alpha = 0.6, edgecolor = "white", pad = 0.1)

    lc = matplotlib.collections.LineCollection(
        lines,
        colors = color_segs,
        linewidths = 5 * scale_path,
        zorder = 104,
        path_effects = [path_effects.Stroke(capstyle = "round")],
        rasterized = rasterized,
    )

    # milestones

    ax.scatter(
        proj.loc[miles_ids, 0],
        proj.loc[miles_ids, 1],
        zorder = 103,
        c = "k",
        s = 200 * scale_path,
        rasterized = False,
    )


    ax.add_collection(lc)
    ax.scatter(
        proj.loc[miles_ids, 0],
        proj.loc[miles_ids, 1],
        zorder = 105,
        c = color_mils,
        s = 140 * scale_path,
        rasterized = rasterized,
    )

    if annotate:

        if color_seg == 'milestones':

            for x in miles_ids.tolist():
                
                if "milestone_names" in graph.keys():
                    names = graph['milestone_names']
                else: names = {}
                
                text = mtext.Text(
                    x = proj.loc[x, 0], y = proj.loc[x, 1], # fontproperties = 'bold',
                    text = str(x) if str(x) not in names.keys() else names[str(x)], 
                    color = 'black', zorder = 106,
                    ha = 'center', va = 'center', size = annotate_fontsize
                )

                text.set_path_effects([
                    matplotlib.patheffects.Stroke(linewidth = 3, foreground = 'white'),
                    matplotlib.patheffects.Normal()
                ])

                ax.add_artist(text)
                pass
            
        elif color_seg == 'segments':

            unique_edges = list(set(seg_edges))
            for uniseg in unique_edges:
                line_segs = []
                for l, seg in zip(lines, seg_edges):
                    if seg == uniseg: line_segs.append(l)

                points = len(line_segs)
                (fromx, fromy), (tox, toy) = line_segs[points // 2]
                text = mtext.Text(
                    x = (fromx + tox) / 2, y = (fromy + toy) / 2, # fontproperties = 'bold',
                    text = str(uniseg), 
                    color = 'black', zorder = 106,
                    ha = 'center', va = 'center', size = annotate_fontsize
                )

                text.set_path_effects([
                    matplotlib.patheffects.Stroke(linewidth = 3, foreground = 'white'),
                    matplotlib.patheffects.Normal()
                ])

                ax.add_artist(text)
                pass
    
    return ax.figure


@njit()
def cdist_numba(coords, out):
    for i in range(0, coords.shape[0] - 1):
        out[i] = math.sqrt(
            (coords[i, 0] - coords[i + 1, 0]) ** 2
            + (coords[i, 1] - coords[i + 1, 1]) ** 2
        )


def trace(
    adata,
    trace_key = 'trace',
    trajectory_key = 'ppt',
    values = 'fitted',
    show_hvg_only = None,
    show_leiden_only = None,
    show_genes = None,
    p_cutoff = 0.0001,
    cmap = 'turbo',
    figsize = (4, 8),
    show_gene_names = None
):
    trace = adata.uns[trace_key]
    if show_genes:
        trace = trace[[x in show_genes for x in trace.obs['gene']], :].copy()
    else:
        if show_hvg_only: trace = trace[adata.var[show_hvg_only], :].copy()
        if show_leiden_only: trace = trace[[int(x) in show_leiden_only for x in trace.obs['leiden']], :].copy()
        trace = trace[(trace.obs['p'] < p_cutoff), :].copy()

    fig, ax = plt.subplots(1, 1, figsize = figsize)

    if values == 'expression': 
        expr = trace.layers['expression'].copy()
        expr[expr < 0] = 0
        expr = expr / expr.max(1, keepdims = True)
    elif values == 'fitted':
        expr = trace.layers['scaled'].copy()

    ax.imshow(expr, aspect = 'auto', cmap = mpl(cmap), interpolation = 'nearest')

    # show gene names
    if show_gene_names is None:
        show_gene_names = expr.shape[0] < 50
    
    if show_gene_names:
        ax.set_yticks(
            [x for x in range(expr.shape[0])],
            trace.obs['gene']
        )
        ax.set_xlabel('Pseudotime')

    else:
        prev_leiden = None
        leidens = []
        for x in trace.obs['leiden']:
            if x != prev_leiden:
                leidens.append(str(x))
                prev_leiden = x
            else: leidens.append('')

        ax.set_yticks(
            [x for x in range(expr.shape[0])],
            leidens
        )
        ax.set_ylabel('Gene behavior patterns')
        ax.set_xlabel('Pseudotime')

    return fig
