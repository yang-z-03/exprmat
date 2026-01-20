
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from exprmat.descriptive.summary import to_long_form
from exprmat.ansi import error, info


def describe(
    adata, *,
    selected_rows = None, 
    selected_cols = None, 
    selected_layers = None,

    group_by = 'column',
    selected_groups = None,
    split_by = 'layer',
    selected_splits = None,

    discrete_plot = 'box',
    continuous_plot = 'scatter',
    split_palette = 'Reds',
    paired = False,
    ax = None, figsize = (5, 2), dpi = 100, 
    legend = 'auto'
):
    
    longf = to_long_form(
        adata, 
        selected_rows = selected_rows,
        selected_cols = selected_cols,
        selected_layers = selected_layers
    )

    if selected_groups is not None:
        longf = longf.loc[[x in selected_groups for x in longf[group_by].tolist()], :]
    if selected_splits is not None:
        longf = longf.loc[[x in selected_splits for x in longf[split_by].tolist()], :]

    touch_group = longf[group_by].tolist()[0]
    touch_split = longf[split_by].tolist()[0]
    if not isinstance(touch_split, str):
        error('split variable should not be numerical.')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize = figsize, dpi = dpi)
    else: fig, ax = ax.figure, ax
    
    if isinstance(touch_group, str):
        order = None
        if group_by is None: pass
        else: order = selected_groups
        if order is None: order = longf[group_by].value_counts().index.tolist()

        if discrete_plot == 'box':
            boxplot(
                longf, group_by, split_by, ax = ax, 
                palette = split_palette, order = order, legend = legend
            )
        
        elif discrete_plot == 'bar':
            barplot(
                longf, group_by, split_by, ax = ax, 
                palette = split_palette, order = order, legend = legend
            )
        
        elif discrete_plot == 'violin':
            violinplot(
                longf, group_by, split_by, ax = ax, 
                palette = split_palette, order = order, legend = legend
            )
        
        if paired:
            assert len(longf[split_by].value_counts().index.tolist()) == 2
            from matplotlib.lines import Line2D
            for x in range(len(order)):
                observations = longf['row'].value_counts().index.tolist()
                for obs in observations:
                    pairs = longf.loc[longf[group_by] == order[x], :].copy()
                    pairs = pairs.loc[longf['row'] == obs, :]
                    if len(pairs) != 2: continue
                    ax.add_line(Line2D(
                        [x - 0.15, x + 0.15], 
                        [pairs.iloc[0, 0], pairs.iloc[1, 0]],
                        linewidth = 0.5, color = '#00000080'
                    ))
    
    return fig


def boxplot(longf, x, split = None, ax = None, palette = 'Reds', order = None, legend = 'auto'):
    
    if ax is not None:
        sns.boxplot(
            data = longf,
            x = x, y = 'value', hue = split, ax = ax, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = order, gap = 0.3,
            flierprops = dict(markerfacecolor = '0.1', markersize = 1, linestyle = 'none'),
            showfliers = True, width = 0.6, legend = legend
        )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(order)
        ax.set_xticklabels(order, rotation = 45, ha = 'right')
        ax.legend(frameon = False)
        return ax.figure
    
    else: 
        fig = sns.boxplot(
            data = longf,
            x = x, y = 'value', hue = split, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = order, gap = 0.3,
            flierprops = dict(markerfacecolor = '0.1', markersize = 1, linestyle = 'none'),
            showfliers = True, width = 0.6, legend = legend
        )


def barplot(longf, x, split = None, ax = None, palette = 'Reds', order = None, legend = 'auto'):
    
    if ax is not None:
        sns.barplot(
            data = longf,
            x = x, y = 'value', hue = split, ax = ax, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', order = order, gap = 0.3, width = 0.6,
            err_kws = {'color': '0', 'linewidth': 0.6}, capsize = 0.2, legend = legend
        )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(order)
        ax.set_xticklabels(order, rotation = 45, ha = 'right')
        ax.legend(frameon = False)
        return ax.figure
    
    else: 
        fig = sns.barplot(
            data = longf,
            x = x, y = 'value', hue = split, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', order = order, gap = 0.3, width = 0.6,
            err_kws = {'color': '0', 'linewidth': 0.6}, capsize = 0.2, legend = legend
        )


def violinplot(longf, x, split = None, ax = None, palette = 'Reds', order = None, legend = 'auto'):
    
    if ax is not None:
        sns.violinplot(
            data = longf,
            x = x, y = 'value', hue = split, ax = ax, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = order, gap = 0.3,
            inner = 'points', width = 0.6, legend = legend
        )

        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks(order)
        ax.set_xticklabels(order, rotation = 45, ha = 'right')
        ax.legend(frameon = False)
        return ax.figure
    
    else: 
        fig = sns.violinplot(
            data = longf,
            x = x, y = 'value', hue = split, fill = True,
            palette = palette, saturation = 0.8, log_scale = False,
            color = '.8', linecolor = '0', order = order, gap = 0.3,
            inner = 'points', width = 0.6, legend = legend
        )