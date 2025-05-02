
import numpy as np
import scipy
import scipy.sparse as sparse

from exprmat.ansi import warning, info


def running_quantile(x, y, p, n_bins = 20):
    '''
    Calculate the quantiles of y (on given cutoff probability p) running over sub-ranges
    of x. This gives the quantile ranges of y that is correlated with x.
    '''

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]

    dx = (x[-1] - x[0]) / n_bins
    # segment x axis into bins
    x_out = np.linspace(x[0] + dx / 2, x[-1] - dx / 2, n_bins)

    y_out = np.zeros(x_out.shape)
    for i in range(len(x_out)):
        ind = np.nonzero((x >= x_out[i] - dx / 2) & (x < x_out[i] + dx / 2))[0]
        if len(ind) > 0: # if this segment contains values
            y_out[i] = np.percentile(y[ind], p)
        else:
            if i > 0: y_out[i] = y_out[i - 1]
            else: y_out[i] = np.nan

    return x_out, y_out


def vscore(E, min_mean = 0, n_bins = 50, fit_percentile = 0.1, error_weight = 1):
    '''
    Calculate v-score (above poisson noise statistic) for genes in the input counts matrix
    
    Returns
    ----------

    A tuple containing the following components:

    *   V-scores for each gene.
    *   Coefficient of variation (CV) for effective counts.
    *   Coefficient of variation (CV) for input counts.
    *   Indices of genes considered in the analysis, namely genes that pass the minimal
        expression level filter. (given by ``min_mean``). Only expressing genes are
        candidates of variable gene selection.
    *   Mean expression values for selected genes.
    *   Fano factor (variance to mean ratio) for selected genes.
    *   Parameter 'a' obtained during optimization. (fitting y = log(c * exp(x)) + b)
    *   Parameter 'b' obtained during optimization.
    '''

    n_cell = E.shape[0]

    mu_gene = E.mean(axis = 0).A.squeeze() # mean expression of genes
    min_expr_filter = np.nonzero(mu_gene > min_mean)[0]
    mu_gene = mu_gene[min_expr_filter]

    tmp = E[:, min_expr_filter]
    tmp.data **= 2
    var_gene = tmp.mean(axis = 0).A.squeeze() - mu_gene ** 2
    ff_gene = var_gene / mu_gene # signal to noise ratio, fano factor
    del tmp

    data_x = np.log(mu_gene)
    data_y = np.log(ff_gene / mu_gene)

    # this plots the expression of genes as x, and the variance of genes as y
    # and both highly-variable genes and stable housekeeping genes selection can
    # be performed on this expression - variance plot using running quantile.
    x, y = running_quantile(data_x, data_y, fit_percentile, n_bins)
    x = x[~ np.isnan(y)]
    y = y[~ np.isnan(y)]

    generalize_log = lambda input: np.log(input[1] * np.exp(- input[0]) + input[2])
    h, b = np.histogram(np.log(ff_gene[mu_gene > 0]), bins = 200)
    b = b[:-1] + np.diff(b) / 2
    max_index = np.argmax(h)
    c = np.max((np.exp(b[max_index]), 1))
    # fit y = log(c * exp(x)) + b
    error_func = lambda b2: np.sum(abs(generalize_log([x, c, b2]) - y) ** error_weight)
    initial_b = 0.1
    b = scipy.optimize.fmin(func = error_func, x0 = [initial_b], disp = False)
    a = c / (1 + b) - 1

    v_scores = ff_gene / ((1 + a) * (1 + b) + b * mu_gene)
    cv_eff = np.sqrt((1 + a) * (1 + b) - 1)
    cv_input = np.sqrt(b)

    return v_scores, cv_eff, cv_input, min_expr_filter, mu_gene, ff_gene, a, b


def highly_variable_genes_ff(E, cells_id = [], min_vscore_pct = 85, min_counts = 3, min_cells = 3):
    ''' 
    Filter highly variable genes by fano factor threshold

    Parameters
    -----------

    E : np.array | scipy.sparse.csc_matrix | scipy.sparse.csr_matrix
        numpy array of feature counts matrix, either dense or sparse.

    cells_id : List
        Select a partition of cells to perform highly variable selection
    
    min_vscore_pct : int
        Filter the top (100 - min_vscore_pct) signal to noise ratio as highly variable

    min_counts : int
    min_cells : int
        Should only keep genes with at least ``min_counts`` transcripts detected in at
        least ``min_cells`` cells. Otherwise the gene is treated as lean.
    '''

    if len(cells_id) == 0: cells_id = np.arange(E.shape[0])

    vscores, _, _, gene_id, mu_gene, fano_factors, _, _ = vscore(E[cells_id, :])
    ix2 = vscores > 0
    vscores = vscores[ix2]
    gene_id = gene_id[ix2]
    mu_gene = mu_gene[ix2]
    fano_factors = fano_factors[ix2]
    min_vscore = np.percentile(vscores, min_vscore_pct)

    selector = (
        # only keep genes expressing at >= min_counts level in at least min_cells.
        ((E[:,gene_id] >= min_counts).sum(0).A.squeeze() >= min_cells) & \
        # select highly variable genes
        (vscores >= min_vscore)
    )
    
    return gene_id[selector]


def stable_genes(E, modality = 'rna', stable_number = None, use_signal_to_noise = True):
    '''
    Select stable features for modalities. Each modalities is characterized with an expected amount
    of stable features that are capable of fitting the multiplet distribution function.
    This function is only revelant in ``composite.py`` (the ``sccomposite`` routine).

    Parameters
    -----------

    E : np.array | scipy.sparse.csc_matrix | scipy.sparse.csr_matrix
        numpy array of feature counts matrix, either dense or sparse.
    
    modality : Literal['rna' | 'atac' | 'adt']
        Which modality do the matrix come from. This is used to specify expected stable feature
        counts (300 at least for scRNA dataset and scATAC datasets, and 15 for ADT.)
    
    stable_number : int | None
        Manually specify the number of expected stable features. This will override the modality
        parameter used for deciding them automatically.

    use_signal_to_noise : bool
        Use signal to noise ratio to determine selection.
    
    Returns
    ---------
    Returns the top ``stable_number`` signal-to-noise ratio genes. The returning matrix has a
    shape of (n_cells, n_genes).
    '''
    
    full = E.T
    if sparse.issparse(full) == True:
        full = full.todense()

    if stable_number == None:
        if modality == 'rna':
            stable_number = 300
        elif modality == 'atac':
            stable_number = 300
        elif modality == 'adt':
            stable_number = 15

    nonzero_proportion = np.array((full > 0).sum(axis = 1)) / full.shape[1]
    full = np.array(full)
    subset = full[np.squeeze(nonzero_proportion > 0.5),]
    high_nonzero = full[np.squeeze(nonzero_proportion > 0.5),]
    high_nonzero = np.log1p(high_nonzero)

    if high_nonzero.shape[0] < stable_number:
        warning('too few stable features to provide reliable inference.')
        result = subset

    elif use_signal_to_noise:
        mean = np.mean(high_nonzero, axis = 1)
        std = np.std(high_nonzero, axis = 1)
        signal = mean / std # signal to noise ratio

        # calculate the rank of the magnitude of signal for each cell
        order = (- signal).argsort()
        ranks = order.argsort()
        result = subset[ranks < stable_number]

    else:
        mean = np.mean(high_nonzero, axis = 1)
        order = (- mean).argsort()
        ranks = order.argsort()
        result = subset[ranks < stable_number]

    result = np.transpose(result)
    return result