
import numpy as np
import pandas as pd
from typing import Literal, Union
import scanpy as sc
from matplotlib import pyplot as plt
import networkx as nx
import seaborn as sns

from exprmat.lr.utils import default_anndata_keys as K
from exprmat.lr.utils import default_primary_columns as D
from exprmat.lr.utils import default_params as V
from exprmat.utils import error, warning, info


def check_variable(liana_res, var_name, var):
    if var is None:
        raise error(f'`{var_name}` must be provided.')
    if var not in liana_res.columns:
        raise error(f'`{var}` ({var_name}) must be one of {liana_res.columns}')


def get_ligand_receptor(adata, liana_res, uns_key = K.uns_key):
    if adata is not None:
        assert uns_key in adata.uns.keys()
        return adata.uns[uns_key].copy()
    if liana_res is not None:
        return liana_res.copy()
    if (liana_res is None) & (uns_key is None):
        error('`lr` or AnnData with `uns_key` must be provided!')


def prepare_lr(
    adata = None, liana_res = None,
    source_labels = None, target_labels = None,
    ligand_complex = None, receptor_complex = None,
    uns_key = K.uns_key
):

    liana_res = get_ligand_receptor(adata, liana_res, uns_key)

    # subset to only cell labels of interest
    liana_res = filter_labels(liana_res, labels = source_labels, label_type = 'source')
    liana_res = filter_labels(liana_res, labels = target_labels, label_type = 'target')

    liana_res['interaction'] = liana_res[D.ligand_complex] + ' > ' + liana_res[D.receptor_complex]

    if ligand_complex is not None:
        liana_res = liana_res[np.isin(liana_res[D.ligand_complex], ligand_complex)]
    if receptor_complex is not None:
        liana_res = liana_res[np.isin(liana_res[D.receptor_complex], receptor_complex)]

    return liana_res


def filter_labels(liana_res, labels, label_type):
    if labels is not None:
        if labels is str:
            labels = [labels]
        covered = np.isin(labels, liana_res[label_type])
        if not covered.all():
            not_covered = np.array(labels)[~covered]
            raise ValueError(f"{not_covered} not found in `lr['{label_type}']`!")
        msk = np.isin(liana_res[label_type], labels)
        liana_res = liana_res[msk]

    return liana_res


def aggregate_scores(res, what, how, absolute, entities):
    res['score'] = np.absolute(res[what]) if absolute else res[what]
    res = res.groupby(entities, observed = False).agg(score = ('score', how)).reset_index()
    return res


def invert_scores(score):
    return -np.log10(score + np.finfo(float).eps)


def filter_by(liana_res, filter_fun):
    if filter_fun is not None:
        msk = filter_fun(liana_res).astype(bool)
        relevant_interactions = np.unique(liana_res[msk].interaction)
        # liana_res = liana_res[np.isin(liana_res['interaction'], relevant_interactions)]
        liana_res = liana_res[msk]

    return liana_res


def topn(liana_res, top_n, orderby, orderby_ascending, orderby_absolute):

    if top_n is not None:
        # get the top_n for each interaction
        if orderby is None:
            error("specify the column to order the interactions.")
        if orderby_ascending is None:
            error("specify if `orderby` is ascending or not.")
        if orderby_ascending: how = 'min'
        else: how = 'max'

        top_lrs = aggregate_scores(
            liana_res, what = orderby, how = how,
            absolute = orderby_absolute,
            entities = ['interaction', D.ligand_complex, D.receptor_complex]
        ).copy()

        top_lrs = top_lrs.sort_values('score', ascending = orderby_ascending) \
            .head(top_n).interaction

        # Filter liana_res to the interactions in top_lrs
        liana_res = liana_res[liana_res['interaction'].isin(top_lrs)].copy()
        # set categories to the order of top_lrs
        liana_res.loc[:, 'interaction'] = pd.Categorical(liana_res['interaction'], categories = top_lrs)

    return liana_res


def pivot_lr(
    liana_res: pd.DataFrame,
    source_key: str = 'source',
    target_key: str = 'target',
    score_key: str = 'lr.means',
    mode: Literal['counts', 'mean'] = 'counts'
) -> pd.DataFrame:
    
    if mode not in ['counts', 'mean']:
        error("`pivot_mode` must be 'counts' or 'mean'.")

    if mode == 'counts':
        pivot_table = liana_res.pivot_table(
            index = source_key, columns = target_key, 
            aggfunc = 'size', fill_value = 0
        )

    elif mode == 'mean':
        pivot_table = liana_res.pivot_table(
            index = source_key, columns = target_key, 
            values = score_key, aggfunc = 'mean', fill_value = 0
        )

    return pivot_table.loc[pivot_table.columns, :]


def scale(arr, min_val = 1, max_val = 5):
    arr = np.array(arr)
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    scaled_arr = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
    return scaled_arr


def set_color(adata, label, color_dict = None, hex = True):
    
    adata.obs[label] = adata.obs[label].astype("category")
    
    if color_dict:
        if not hex:
            from matplotlib.colors import to_hex
            color_dict = {x: to_hex(y) for x, y in color_dict.items()}

        _dt = get_color(adata, label)
        _dt.update(color_dict)
        color_dict = _dt
        adata.uns[f"{label}.colors"] = [
            color_dict[x] for x in adata.obs[label].cat.categories
        ]

    elif f"{label}.colors" not in adata.uns.keys():
        sc.pl._utils._set_default_colors_for_categorical_obs(adata, label)
        adata.uns[f"{label}.colors"] = adata.uns[f"{label}_colors"]
        del adata.uns[f"{label}_colors"]

    return adata


def get_color(adata, label):

    if f"{label}.colors" not in adata.uns.keys(): set_color(adata, label)
    return { 
        x: y for x, y in zip(
            adata.obs[label].cat.categories, 
            adata.uns[f"{label}.colors"]
    )}


def get_mask_df(
    pivot_table: pd.DataFrame,
    source_cell_type: Union[list, str] = None,
    target_cell_type: Union[list, str] = None,
    mode: Literal['and', 'or'] ='or'
) -> pd.DataFrame:

    if source_cell_type is None and target_cell_type is None:
        return pivot_table

    if isinstance(source_cell_type, str):
        source_cell_type = [source_cell_type]
    if isinstance(target_cell_type, str):
        target_cell_type = [target_cell_type]

    mask_df = pd.DataFrame(
        np.zeros_like(pivot_table), 
        index = pivot_table.index, 
        columns = pivot_table.columns, 
        dtype = bool
    )

    if mode == 'or':
        if source_cell_type is not None:
            mask_df.loc[source_cell_type] = True
        if target_cell_type is not None:
            mask_df.loc[:, target_cell_type] = True
    elif mode == 'and':
        if source_cell_type is not None and target_cell_type is not None:
            mask_df.loc[source_cell_type, target_cell_type] = True

    return pivot_table[mask_df].fillna(0)


def circleplot(
    adata: sc.AnnData,
    uns_key: Union[str, None] = K.uns_key,
    result: Union[pd.DataFrame, None] = None,
    groupby: str = None,
    source_key: str = 'source',
    target_key: str = 'target',
    score_key: str = None,
    inverse_score: bool = False,
    top_n: int = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fun: callable = None,
    source_labels: Union[list, str] = None,
    target_labels: Union[list, str] = None,
    ligand_complex: list | str | None = None,
    receptor_complex: list | str | None = None,
    pivot_mode: Literal['counts', 'mean'] = 'counts',
    mask_mode: Literal['and', 'or'] = 'or',
    figure_size: tuple = (5, 5),
    edge_alpha: float = .5,
    edge_arrow_size: int = 10,
    edge_width_scale: tuple = (1, 5),
    node_alpha: float = 1,
    node_size_scale: tuple = (100, 400),
    node_label_offset: tuple = (0.1, -0.2),
    node_label_size: int = 8,
    node_label_alpha: float = .7,
    ax = None
):
    
    if groupby is None:
        raise ValueError('`groupby` must be provided!')

    result = prepare_lr(
        adata = adata,
        source_labels = None,
        target_labels = None,
        ligand_complex = ligand_complex,
        receptor_complex = receptor_complex,
        uns_key=uns_key
    )

    result = filter_by(result, filter_fun)
    result = topn(result, top_n, orderby, orderby_ascending, orderby_absolute)

    if inverse_score:
        result[score_key] = invert_scores(result[score_key])

    pivot_table = pivot_lr(
        result,
        source_key = source_key,
        target_key = target_key,
        score_key = score_key,
        mode = pivot_mode
    )

    groupby_colors = get_color(adata, label = groupby)

    # mask pivot table
    pvt = get_mask_df(
        pivot_table,
        source_cell_type = source_labels,
        target_cell_type = target_labels,
        mode = mask_mode
    )

    G = nx.convert_matrix.from_pandas_adjacency(pvt, create_using = nx.DiGraph())
    pos = nx.circular_layout(G)

    edge_color = [groupby_colors[cell[0]] for cell in G.edges]
    edge_width = np.asarray([G.edges[e]['weight'] for e in G.edges()])
    edge_width = scale(edge_width, max_val = edge_width_scale[1], min_val = edge_width_scale[0])

    node_color = [groupby_colors[cell] for cell in G.nodes]
    node_size = pivot_table.sum(axis = 1).values
    node_size = scale(node_size, max_val = node_size_scale[1], min_val = node_size_scale[0])

    if ax is None: fig, ax = plt.subplots(figsize = figure_size)
    else: fig = ax.figure

    nx.draw_networkx_edges(
        G, pos, alpha = edge_alpha, arrowsize = edge_arrow_size,
        arrowstyle = '-|>',
        width = edge_width,
        edge_color = edge_color,
        connectionstyle = "arc3,rad=-0.3",
        ax = ax
    )

    nx.draw_networkx_nodes(
        G, pos, node_color = node_color,
        node_size = node_size,
        alpha = node_alpha,
        ax = ax
    )

    label_options = {"ec": "k", "fc": "white", "alpha": node_label_alpha}
    _ = nx.draw_networkx_labels(
        G, { k: v + np.array(node_label_offset) for k, v in pos.items() },
        font_size = node_label_size,
        bbox = label_options, ax = ax
    )

    ax.set_frame_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    coeff = 1.2
    ax.set_xlim((xlim[0] * coeff, xlim[1] * coeff))
    ax.set_ylim((ylim[0] * coeff, ylim[1]))
    ax.set_aspect('equal')

    return fig


def lr_dotplot(
    adata: sc.AnnData = None,
    uns_key = K.uns_key,
    liana_res: pd.DataFrame = None,
    colour: str = None,
    size: str = None,
    source_labels: list = None,
    target_labels: list = None,
    top_n: int = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fun: callable = None,
    ligand_complex: str | None = None,
    receptor_complex: str | None = None,
    inverse_colour: bool = False,
    inverse_size: bool = False,
    cmap: str = V.cmap,
    size_range: tuple = (2, 9),
    figure_size: tuple = (8, 6),
    ax = None
):
    
    liana_res = prepare_lr(
        adata = adata,
        liana_res = liana_res,
        source_labels = source_labels,
        target_labels = target_labels,
        ligand_complex = ligand_complex,
        receptor_complex = receptor_complex,
        uns_key = uns_key
    )

    check_variable(liana_res, var = colour, var_name = 'colour')
    check_variable(liana_res, var = size, var_name = 'size')

    liana_res = filter_by(liana_res, filter_fun)
    liana_res = topn(liana_res, top_n, orderby, orderby_ascending, orderby_absolute)

    # inverse sc if needed
    if inverse_colour: liana_res[colour] = invert_scores(liana_res[colour])
    if inverse_size: liana_res[size] = invert_scores(liana_res[size])
    
    if isinstance(target_labels, str): target_labels = [target_labels]
    if isinstance(source_labels, str): source_labels = [source_labels]
    x = liana_res['target'].unique().tolist() if target_labels is None else target_labels
    s = liana_res['source'].unique().tolist() if source_labels is None else source_labels
    y = liana_res['interaction'].unique().tolist()
    
    # g = sns.relplot(
    #     data = liana_res,
    #     x = 'target', y = 'interaction', hue = colour, size = size,
    #     palette = cmap, hue_norm = (-1, 1), edgecolor = ".7",
    #     height = figure_size[1], aspect = figure_size[0] / figure_size[1], 
    #     sizes = (10, 150), size_norm = (-.2, .8),
    # )

    # g.despine(left = True, bottom = True)
    # g.ax.margins(.02)
    # g.ax.set_xlim(-1, x)
    # g.ax.set_ylim(-1, y)

    if ax is None: fig, ax = plt.subplots(figsize = figure_size)
    else: fig = ax.figure

    # manually create scatter plot.
    
    xlabs = []
    ylabs = y[::-1]
    df = { 'x': [], 'y': [], 's': [], 'c': [] }

    for sx in s:
        for xx in x:
            xlabs += [f'{sx} > {xx}']
            for yx in y:
                mask = (
                    (liana_res['source'] == sx) & 
                    (liana_res['target'] == xx) & 
                    (liana_res['interaction'] == yx)
                )

                if mask.sum() == 1:
                    df['x'].append(len(xlabs) - 1)
                    df['y'].append(ylabs.index(yx))
                    df['s'].append(liana_res.loc[mask, size].iloc[0] * 32)
                    df['c'].append(liana_res.loc[mask, colour].iloc[0])



    ax.set_xlim(-0.9, len(xlabs) - 0.1)
    ax.set_ylim(-0.9, len(ylabs) - 0.1)
    ax.set_xticks(list(range(len(xlabs))), xlabs)
    ax.set_yticks(list(range(len(ylabs))), ylabs)
    
    # TODO: how to make the grids stay below ?
    # ax.set_frame_on(False)
    # ax.set_facecolor('lightgrey')
    # plt.grid(True, zorder = -1)

    p = ax.scatter(
        x = 'x', y = 'y', s = 's',
        edgecolors = "lightgrey", c = 'c', cmap = cmap,
        data = pd.DataFrame(df), marker = 'o'
    )

    cbar = plt.colorbar(
        p, shrink = 0.3, aspect = 10,
        anchor = (0.2, 0), location = "right"
    )

    cbar.ax.set_title(colour, loc = "left", fontsize = 10, pad = 15)

    for label in ax.get_xticklabels(): label.set_rotation(90)
    return fig


def heatmap(
    adata: sc.AnnData,
    uns_key: Union[str, None] = K.uns_key,
    result: Union[pd.DataFrame, None] = None,
    groupby: str = None,
    source_key: str = 'source',
    target_key: str = 'target',
    score_key: str = None,
    inverse_score: bool = False,
    top_n: int = None,
    orderby: str | None = None,
    orderby_ascending: bool | None = None,
    orderby_absolute: bool = False,
    filter_fun: callable = None,
    source_labels: Union[list, str] = None,
    target_labels: Union[list, str] = None,
    ligand_complex: list | str | None = None,
    receptor_complex: list | str | None = None,
    pivot_mode: Literal['counts', 'mean'] = 'counts',
    annotate = True,
    mask_mode: Literal['and', 'or'] = 'or',
    figure_size: tuple = (5, 5),
    cmap = 'turbo',
    ax = None
):
    
    if groupby is None:
        raise ValueError('`groupby` must be provided!')

    result = prepare_lr(
        adata = adata,
        source_labels = None,
        target_labels = None,
        ligand_complex = ligand_complex,
        receptor_complex = receptor_complex,
        uns_key=uns_key
    )

    result = filter_by(result, filter_fun)
    result = topn(result, top_n, orderby, orderby_ascending, orderby_absolute)

    if inverse_score:
        result[score_key] = invert_scores(result[score_key])

    pivot_table = pivot_lr(
        result,
        source_key = source_key,
        target_key = target_key,
        score_key = score_key,
        mode = pivot_mode
    )

    groupby_colors = get_color(adata, label = groupby)

    # mask pivot table
    pvt = get_mask_df(
        pivot_table,
        source_cell_type = source_labels,
        target_cell_type = target_labels,
        mode = mask_mode
    )

    if ax is None: fig, ax = plt.subplots(figsize = figure_size)
    else: fig = ax.figure

    sns.heatmap(
        pvt, ax = ax, annot = annotate, linewidths = .5, 
        fmt = 'd' if pivot_mode == 'counts' else '.1f',
        cmap = cmap
    )

    ax.set_xlabel('Receiver (receptors)')
    ax.set_ylabel('Sender (ligands)')
    for label in ax.get_xticklabels(): label.set_rotation(90)

    return ax.figure