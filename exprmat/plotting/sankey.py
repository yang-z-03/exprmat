
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from exprmat.ansi import error
from exprmat.plotting.palettes import linear_palette, all_palettes


def check_data_matches_labels(labels, data, side):
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            if len(labels) <= 20: msg = "(labels: " + ",".join(labels) + ") "
            if len(data) < 20: msg += "(data: " + ",".join(data) + ')'
            error('{0} labels and data do not match. {1}'.format(side, msg))


def sankey(
    left, right, left_weights = None, right_weights = None, 
    colors = None, left_labels = None, right_labels = None, 
    aspect = 4, right_colors = False, cmap = 'Turbo',
    fontsize = 9, figsize = (4, 4), ax = None
):
    
    if left_weights is None: left_weights = []
    if right_weights is None: right_weights = []
    if left_labels is None: left_labels = []
    if right_labels is None: right_labels = []
    
    if len(left_weights) == 0: left_weights = np.ones(len(left))
    if len(right_weights) == 0: right_weights = left_weights

    if ax is None: fig, ax = plt.subplots(figsize = figsize)
    else: fig = ax.figure

    # create Dataframe
    if isinstance(left, pd.Series): left = left.reset_index(drop = True)
    if isinstance(right, pd.Series): right = right.reset_index(drop = True)
    df = pd.DataFrame({
        'left': left, 
        'right': right, 
        'left_weights': left_weights,
        'right_weights': right_weights
    }, index = range(len(left)))

    if len(df[(df.left.isnull()) | (df.right.isnull())]):
        error('sankey graph does not support null values.')

    # identify all labels that appear 'left' or 'right'
    all_labels = pd.Series(np.r_[
        df.left.unique(), 
        df.right.unique()
    ]).unique()

    if len(left_labels) == 0: left_labels = pd.Series(df.left.unique()).unique()
    else: check_data_matches_labels(left_labels, df['left'], 'left')

    if len(right_labels) == 0: right_labels = pd.Series(df.right.unique()).unique()
    else: check_data_matches_labels(left_labels, df['right'], 'right')

    # if no colorDict given, make one
    if colors is None:
        colors = {}
        pal = linear_palette(all_palettes[cmap][
            list(all_palettes[cmap].keys())[-1]
        ], len(all_labels))
        for i, label in enumerate(all_labels):
            colors[label] = pal[i]
    
    else:
        missing = [label for label in all_labels if label not in colors.keys()]
        if missing: error('you should give each label a color.')

    # determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for llabel in left_labels:
        leftDict = {}
        rightDict = {}
        for rlabel in right_labels:
            leftDict[rlabel] = df[(df.left == llabel) & (df.right == rlabel)].left_weights.sum()
            rightDict[rlabel] = df[(df.left == llabel) & (df.right == rlabel)].right_weights.sum()
        
        ns_l[llabel] = leftDict
        ns_r[llabel] = rightDict

    # Determine positions of left label patches and total widths
    lw = defaultdict()
    for i, llabel in enumerate(left_labels):
        myd = {}
        myd['left'] = df[df.left == llabel].left_weights.sum()
        if i == 0:
            myd['bottom'] = 0
            myd['top'] = myd['left']
        else:
            myd['bottom'] = lw[left_labels[i - 1]]['top'] + 0.02 * df.left_weights.sum()
            myd['top'] = myd['bottom'] + myd['left']
            topedge = myd['top']
        lw[llabel] = myd

    # Determine positions of right label patches and total widths
    rw = defaultdict()
    for i, rlabel in enumerate(right_labels):
        myd = {}
        myd['right'] = df[df.right == rlabel].right_weights.sum()
        if i == 0:
            myd['bottom'] = 0
            myd['top'] = myd['right']
        else:
            myd['bottom'] = rw[right_labels[i - 1]]['top'] + 0.02 * df.right_weights.sum()
            myd['top'] = myd['bottom'] + myd['right']
            topedge = myd['top']
        rw[rlabel] = myd

    # Total vertical extent of diagram
    xmax = topedge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for llabel in left_labels:
        
        plt.fill_between(
            [- 0.02 * xmax, 0],
            2 * [lw[llabel]['bottom']],
            2 * [lw[llabel]['bottom'] + lw[llabel]['left']],
            color = colors[llabel],
            alpha = 0.99
        )

        if lw[llabel]['top'] - lw[llabel]['bottom'] >= fontsize:
            plt.text(
                - 0.05 * xmax, lw[llabel]['bottom'] + 0.5 * lw[llabel]['left'], llabel,
                {'ha': 'right', 'va': 'center'},
                fontsize = fontsize
            )

    for rlabel in right_labels:

        plt.fill_between(
            [xmax, 1.02 * xmax], 2 * [rw[rlabel]['bottom']],
            2 * [rw[rlabel]['bottom'] + rw[rlabel]['right']],
            color = colors[rlabel],
            alpha = 0.99
        )

        if rw[rlabel]['top'] - rw[rlabel]['bottom'] >= fontsize:
            plt.text(
                1.05 * xmax,
                rw[rlabel]['bottom'] + 0.5 * rw[rlabel]['right'], rlabel,
                {'ha': 'left', 'va': 'center'},
                fontsize = fontsize
            )

    # strips
    for llabel in left_labels:
        for rlabel in right_labels:
            lab_color = llabel
            if right_colors:
                lab_color = rlabel
            if len(df[(df.left == llabel) & (df.right == rlabel)]) > 0:

                ys_d = np.array(50 * [lw[llabel]['bottom']] + 50 * [rw[rlabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [lw[llabel]['bottom'] + ns_l[llabel][rlabel]] + 50 * [rw[rlabel]['bottom'] + ns_r[llabel][rlabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                lw[llabel]['bottom'] += ns_l[llabel][rlabel]
                rw[rlabel]['bottom'] += ns_r[llabel][rlabel]
                plt.fill_between(
                    np.linspace(0, xmax, len(ys_d)), ys_d, ys_u, alpha = 0.65,
                    color = colors[lab_color]
                )

    ax.axis('off')
    return fig