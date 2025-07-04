
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def linregress(
    x, y, xscale = lambda z: z, yscale = lambda z: z, 
    xticks = None, yticks = None, 
    xlabel = None, ylabel = None,
    xticklabels = None, yticklabels = None, 
    xrange = None, yrange = None,
    title = None,
    show_stats = True, ax = None,
    figsize = (3, 3), dpi = 100
):

    if ax is not None: fig, axis = ax.figure, ax
    else: fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    sns.regplot(
        x = xscale(x), 
        y = yscale(y), 
        ax = axis,
        scatter_kws = { 's': 4, 'color': 'black' },
        line_kws = { 'linewidth': 1, 'color': 'red' }
    )

    if xlabel: axis.set_xlabel(xlabel, loc = 'left', linespacing = 1.5)
    if ylabel: axis.set_ylabel(ylabel, loc = 'bottom', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    from scipy import stats
    res = stats.linregress(xscale(x), yscale(y))
    
    r = (f"R: {res.rvalue:5.3f}")
    p = (f"p: {res.pvalue:.2e}")
    if title: axis.set_title(title)
    if xrange: axis.set_xlim(xrange)
    if yrange: axis.set_ylim(yrange)

    if xticks: 
        axis.set_xticks([xscale(z) for z in xticks])
        if xticklabels: axis.set_xticklabels(xticklabels)

    if yticks: 
        axis.set_yticks([yscale(z) for z in yticks])
        if yticklabels: axis.set_yticklabels(yticklabels)
    
    x1, x2 = axis.get_xlim()
    y1, y2 = axis.get_ylim()
    
    if show_stats:
        axis.text(
            s = r + '\n' + p, va = 'bottom', ha = 'right', 
            x = x1 + 0.95 * (x2 - x1), 
            y = y1 + 0.05 * (y2 - y1),
            linespacing = 1.5
        )

    return fig


def line(
    x, y, xscale = lambda z: z, yscale = lambda z: z, 
    xticks = None, yticks = None, 
    xlabel = None, ylabel = None,
    xticklabels = None, yticklabels = None, 
    xrange = None, yrange = None,
    title = None, ax = None,
    figsize = (3, 3), dpi = 100
):
    if ax is not None: fig, axis = ax.figure, ax
    else: fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    axis.plot(
        xscale(x), 
        yscale(y),
        linewidth = 1,
        color = 'black',
        linestyle = 'solid'
    )

    if xlabel: axis.set_xlabel(xlabel, loc = 'center', linespacing = 1.5)
    if ylabel: axis.set_ylabel(ylabel, loc = 'center', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    if title: axis.set_title(title)
    if xrange: axis.set_xlim(xrange)
    if yrange: axis.set_ylim(yrange)

    if xticks: 
        axis.set_xticks([xscale(z) for z in xticks])
        if xticklabels: axis.set_xticklabels(xticklabels)

    if yticks: 
        axis.set_yticks([yscale(z) for z in yticks])
        if yticklabels: axis.set_yticklabels(yticklabels)
    
    return fig


def stair(
    x, y, stair = 'pre', y2 = 0,
    xscale = lambda z: z, yscale = lambda z: z, 
    xticks = None, yticks = None, 
    xlabel = None, ylabel = None,
    xticklabels = None, yticklabels = None, 
    xrange = None, yrange = None,
    title = None, ax = None,
    figsize = (3, 3), dpi = 100
):

    if ax is not None: fig, axis = ax.figure, ax
    else: fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    axis.fill_between(
        x = xscale(x), 
        y1 = yscale(y),
        y2 = y2,
        step = stair,
        color = 'black',
        alpha = 0.6
    )

    if xlabel: axis.set_xlabel(xlabel, loc = 'left', linespacing = 1.5)
    if ylabel: axis.set_ylabel(ylabel, loc = 'bottom', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    if title: axis.set_title(title)
    if xrange: axis.set_xlim(xrange)
    if yrange: axis.set_ylim(yrange)

    if xticks: 
        axis.set_xticks([xscale(z) for z in xticks])
        if xticklabels: axis.set_xticklabels(xticklabels)

    if yticks: 
        axis.set_yticks([yscale(z) for z in yticks])
        if yticklabels: axis.set_yticklabels(yticklabels)
    
    return fig


def scatter(
    x, y, size = 2, color = 'k', alpha = 0.8,
    xscale = lambda z: z, yscale = lambda z: z, 
    xticks = None, yticks = None, 
    xlabel = None, ylabel = None,
    xticklabels = None, yticklabels = None, 
    xrange = None, yrange = None,
    title = None, ax = None,
    figsize = (3, 3), dpi = 100
):
    if ax is not None: fig, axis = ax.figure, ax
    else: fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    axis.scatter(
        x = xscale(x), 
        y = yscale(y),
        s = size,
        c = color,
        alpha = alpha
    )

    if xlabel: axis.set_xlabel(xlabel, loc = 'left', linespacing = 1.5)
    if ylabel: axis.set_ylabel(ylabel, loc = 'bottom', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    if title: axis.set_title(title)
    if xrange: axis.set_xlim(xrange)
    if yrange: axis.set_ylim(yrange)

    if xticks: 
        axis.set_xticks([xscale(z) for z in xticks])
        if xticklabels: axis.set_xticklabels(xticklabels)

    if yticks: 
        axis.set_yticks([yscale(z) for z in yticks])
        if yticklabels: axis.set_yticklabels(yticklabels)
    
    return fig


def density(
    x, y, size = 2, cmap = 'turbo', alpha = 0.8,
    xscale = lambda z: z, yscale = lambda z: z, 
    xticks = None, yticks = None, 
    xlabel = None, ylabel = None,
    xticklabels = None, yticklabels = None, 
    xrange = None, yrange = None,
    title = None, ax = None,
    figsize = (3, 3), dpi = 100
):
    if ax is not None: fig, axis = ax.figure, ax
    else: fig, axis = plt.subplots(1, 1, figsize = figsize, dpi = dpi)

    if isinstance(x, list):
        x = np.array(x)
    
    if isinstance(y, list):
        y = np.array(y)

    scx = xscale(x)
    scy = yscale(y)
    xmin, xmax = scx.min(), scx.max()
    ymin, ymax = scy.min(), scy.max()

    import scipy.stats as stats
    X, Y = np.mgrid[xmin : xmax : 500j, ymin : ymax : 500j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([scx, scy])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # retrieve levels from Z
    xfind = (499 * (scx - xmin) / (xmax - xmin)).astype(np.int32)
    yfind = (499 * (scy - ymin) / (ymax - ymin)).astype(np.int32)
    clist = [Z[xf, yf] for xf, yf in zip(xfind.tolist(), yfind.tolist())]

    axis.scatter(
        x = xscale(x), 
        y = yscale(y),
        s = size,
        c = clist,
        cmap = cmap,
        alpha = alpha
    )

    if xlabel: axis.set_xlabel(xlabel, loc = 'center', linespacing = 1.5)
    if ylabel: axis.set_ylabel(ylabel, loc = 'center', linespacing = 1.5)

    for pos in ['right', 'top']:
        axis.spines[pos].set_visible(False)

    if title: axis.set_title(title)
    if xrange: axis.set_xlim(xrange)
    if yrange: axis.set_ylim(yrange)

    if xticks: 
        axis.set_xticks([xscale(z) for z in xticks])
        if xticklabels: axis.set_xticklabels(xticklabels)

    if yticks: 
        axis.set_yticks([yscale(z) for z in yticks])
        if yticklabels: axis.set_yticklabels(yticklabels)
    
    return fig