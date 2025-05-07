
import itertools
import re
from collections.abc import Sequence
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.sparse
from anndata import AnnData
from scanpy import logging
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from exprmat.ansi import error, warning, info
from exprmat.utils import ensure_array


def infercnv(
    adata: AnnData,
    *,
    reference_key: str | None = None,
    reference_cat: None | str | Sequence[str] = None,
    reference: np.ndarray | None = None,
    lfc_clip: float = 3,
    window_size: int = 100,
    step: int = 10,
    dynamic_threshold: float | None = 1.5,
    exclude_chromosomes: Sequence[str] | None = ('X', 'Y'),
    chunksize: int = 5000,
    n_jobs: int | None = None,
    inplace: bool = True,
    layer: str = 'X',
    key_added: str = 'cnv',
    calculate_gene_values: bool = False,
) -> None | tuple[dict, scipy.sparse.csr_matrix, np.ndarray | None]:
    
    '''
    Infer Copy Number Variation (CNV) by averaging gene expression over genomic regions.
    InferCNV methods is suited most to SMARTseq protocol where the genomic coverage is higher
    throughout the chromosomes. On droplet based protocols, the prediction power is reduced.
    You may refer to <https://github.com/AntonioDeFalco/SCEVAN> for a solution on this.

    This method is heavily inspired by infercnv <https://github.com/broadinstitute/inferCNV>
    but more computationally efficient. The method is described in more detail there.
    There, you can also find instructions on how to prepare input data.

    Parameters
    ----------
    adata
        annotated data matrix

    reference_key
        Column name in adata.obs that contains tumor/normal annotations.
        If this is set to None, the average of all cells is used as reference.

    reference_cat
        One or multiple values in `adata.obs[reference_key]` that annotate
        normal cells.

    reference
        Directly supply an array of average normal gene expression. Overrides
        ``reference_key`` and ``reference_cat``.

    lfc_clip
        Clip log fold changes at this value

    window_size
        Size of the running window (number of genes in to include in the window)

    step
        Only compute every nth running window where n = `step`. Set to 1 to compute
        all windows. This saves computational efforts for large datasets.

    dynamic_threshold
        Values smaller than dynamic threshold * sd will be set to 0, where sd is the stadard 
        deviation of the smoothed gene expression. Set to `None` to disable this step.

    exclude_chromosomes
        List of chromosomes to exclude. The default is to exclude genosomes.
        For list of all available chromosomes in the dataset, you should manually check
        the adata.var['.seqid'] column, when you are not using the pre-built genome annotations.

    chunksize
        Process dataset in chunks of cells. This allows to run infercnv on
        datasets with many cells, where the dense matrix would not fit into memory.

    n_jobs
        Number of jobs for parallel processing. Default: use all cores.
        Data will be submitted to workers in chunks, see `chunksize`.

    inplace
        If True, save the results in adata.obsm, otherwise return the CNV matrix.

    layer
        Layer from adata to use. If `None`, use `X`.

    key_added
        Key under which the cnv matrix will be stored in adata if ``inplace = True``.
        Will store the matrix in ``adata.obsm['{key_added}']`` and additional information
        in `adata.uns[key_added]`.

    calculate_gene_values
        If True per gene CNVs will be calculated and stored in `adata.layers['gene_values_{key_added}']`.
        As many genes will be included in each segment the resultant per gene value will be 
        an average of the genes included in the segment. Additionally not all genes will be 
        included in the per gene CNV, due to the window size and step size not always being a 
        multiple of the number of genes. Any genes not included in the per gene CNV will be 
        filled with NaN. Note this will significantly increase the memory and computation 
        time, it is recommended to decrease the chunksize to ~ 100 if this is set to True.
    '''

    if not adata.var_names.is_unique:
        error('ensure your var_names are unique.')
    if {'.seqid', '.start', '.end'} - set(adata.var.columns) != set():
        warning('ensure your var table contains ".seqid", ".start", and ".end"')
        warning('this should be automatically queried for you if you specify a legal taxa name.')
        error('otherwise, you should parse the gff and configure these information yourself.')

    var_mask = adata.var['.seqid'].isnull()
    if np.sum(var_mask):
        warning(f'skipped {np.sum(var_mask)} genes because they don\'t have a genomic position annotated.')  # type: ignore
    if exclude_chromosomes is not None:
        var_mask = var_mask | adata.var['.seqid'].isin(exclude_chromosomes)

    valid_annot_adata = adata[:, ~ var_mask]
    reference = get_reference_cells(adata, reference_key, reference_cat, reference)[:, ~ var_mask]

    expr = valid_annot_adata.X if layer == 'X' else valid_annot_adata.layers[layer]
    if scipy.sparse.issparse(expr):
        expr = expr.tocsr()

    var = valid_annot_adata.var.loc[:, ['.seqid', '.start', '.end']]
    chr_pos, chunks, convolved_dfs = zip(
        *process_map(
            infercnv_proc,
            [expr[i : i + chunksize, :] for i in range(0, adata.shape[0], chunksize)],
            itertools.repeat(var),
            itertools.repeat(reference),
            itertools.repeat(lfc_clip),
            itertools.repeat(window_size),
            itertools.repeat(step),
            itertools.repeat(dynamic_threshold),
            itertools.repeat(calculate_gene_values),
            tqdm_class=tqdm,
            max_workers=cpu_count() if n_jobs is None else n_jobs,
        ),
        strict=False,
    )

    res = scipy.sparse.vstack(chunks)
    chr_pos = chr_pos[0]

    if calculate_gene_values:
        per_gene_df = pd.concat(convolved_dfs, axis=0)
        # ensure the dataframe has the correct row index
        per_gene_df.index = adata.obs.index
        # ensure the per gene cnv matches the adata var (genes) index, any genes
        # that are not included in the cnv will be filled with NaN
        per_gene_df = per_gene_df.reindex(columns = adata.var_names, fill_value = np.nan)
        # this needs to be a numpy array as colnames are too large to save in anndata
        per_gene_mtx = per_gene_df.values

    else: per_gene_mtx = None

    if inplace:
        adata.obsm[f'{key_added}'] = res
        adata.uns[key_added] = {'chr.pos': chr_pos}
        if calculate_gene_values:
            adata.layers[f'cnv.per.gene'] = per_gene_mtx

    else: return chr_pos, res, per_gene_mtx


def natural_sort(l: Sequence):

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)


def running_mean(
    x: np.ndarray | scipy.sparse.spmatrix,
    n: int = 50,
    step: int = 10,
    gene_list: list = None,
    calculate_gene_values: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | None]:
    '''
    Compute a pyramidially weighted running mean.
    This densifies the matrix. Use `step` and `chunksize` to save memory.

    Parameters
    ----------
    x
        Matrix to work on

    n
        Length of the running window

    step
        Only compute running windows every `step` columns, This saves memory and
        calculation time by reducing the parallel task step-fold.
    
    gene_list
        List of gene names to be used in the convolution

    calculate_gene_values
        If true, per gene CNVs will be calculated.
    '''

    if n < x.shape[1]: 
        # regular convolution: the filter is smaller than the #genes
        r = np.arange(1, n + 1)
        pyramid = np.minimum(r, r[::-1])
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode='valid'),
            axis = 1, arr = x,
        ) / np.sum(pyramid)

        # get the indices of the genes used in the convolution
        convolution_indices = get_convolution_indices(x, n)[np.arange(0, smoothed_x.shape[1], step)]
        # pull out the genes used in the convolution
        convolved_gene_names = gene_list[convolution_indices]
        smoothed_x = smoothed_x[:, np.arange(0, smoothed_x.shape[1], step)]

        if calculate_gene_values:
            convolved_gene_values = calculate_gene_averages(convolved_gene_names, smoothed_x)
        else: convolved_gene_values = None
        return smoothed_x, convolved_gene_values

    else:  
        # if there is less genes than the window size, set the window size to the 
        # number of genes and perform a single convolution
        n = x.shape[1]
        r = np.arange(1, n + 1)
        # as we are only doing one convolution the values should be equal
        pyramid = np.array([1] * n)
        smoothed_x = np.apply_along_axis(
            lambda row: np.convolve(row, pyramid, mode='valid'),
            axis = 1, arr = x,
        ) / np.sum(pyramid)

        if calculate_gene_values:
            # as all genes are used the convolution the values are identical for all genes
            convolved_gene_values = pd.DataFrame(
                np.repeat(smoothed_x, len(gene_list), axis = 1),
                columns = gene_list
            )

        else: convolved_gene_values = None
        return smoothed_x, convolved_gene_values


def calculate_gene_averages(
    convolved_gene_names: np.ndarray,
    smoothed_x: np.ndarray,
) -> pd.DataFrame:
    '''
    Calculate the average value of each gene in the convolution.

    Parameters
    ----------
    convolved_gene_names
        A numpy array with the gene names used in the convolution.

    smoothed_x
        A numpy array with the smoothed gene expression values.

    Returns
    -------
    convolved_gene_values
        A DataFrame with the average value of each gene in the convolution
    '''
    
    # create a dictionary to store the gene values per sample
    gene_to_values = {}
    # calculate the number of genes in each convolution,
    # will be same as the window size default = 100
    length = len(convolved_gene_names[0])
    # convert the flattened convolved gene names to a list
    flatten_list = list(convolved_gene_names.flatten())

    # for each sample in smoothed_x find the value for each gene and store it in a dictionary
    for sample, row in enumerate(smoothed_x):
        # create sample level in the dictionary
        if sample not in gene_to_values:
            gene_to_values[sample] = {}
        # for each gene in the flattened gene list find the value and store it in the dictionary
        for i, gene in enumerate(flatten_list):
            if gene not in gene_to_values[sample]:
                gene_to_values[sample][gene] = []
            # as the gene list has been flattend we can use the floor division of the index
            # to get the correct position of the gene to get the value and store it in the dictionary
            gene_to_values[sample][gene].append(row[i // length])

    for sample in gene_to_values:
        for gene in gene_to_values[sample]:
            gene_to_values[sample][gene] = np.mean(gene_to_values[sample][gene])

    convolved_gene_values = pd.DataFrame(gene_to_values).T
    return convolved_gene_values


def get_convolution_indices(x, n):
    
    indices = []
    for i in range(x.shape[1] - n + 1):
        indices.append(np.arange(i, i + n))
    return np.array(indices)


def running_mean_by_chr(
    expr, var, window_size, step, calculate_gene_values
) -> tuple[dict, np.ndarray, pd.DataFrame | None]:
    '''
    Compute the running mean for each chromosome independently. 
    Stack the resulting arrays ordered by chromosome.

    Parameters
    ----------
    expr
        A gene expression matrix, appropriately preprocessed

    var
        The var data frame of the associated AnnData object

    Returns
    -------
    chr_start_pos
        A dictionary mapping each chromosome to the index of ``running_mean`` where
        this chromosome begins, in the concatenated chromosome matrix.

    running_mean
        A numpy array with the smoothed gene expression, ordered by chromosome
        and genomic position. This is a row in the plotted chromosome CNV matrix.
    '''

    chromosomes = natural_sort([x for x in var['.seqid'].unique() if x != 'MT'])
    running_means = [
        running_mean_chr(
            chr, expr, var, window_size, step, calculate_gene_values
        ) for chr in chromosomes
    ]

    running_means, convolved_dfs = zip(*running_means, strict = False)
    chr_start_pos = {}
    for chr, i in zip(chromosomes, np.cumsum([0] + [x.shape[1] for x in running_means]), strict = False):
        chr_start_pos[chr] = i

    # concatenate the gene dfs
    if calculate_gene_values:
        convolved_dfs = pd.concat(convolved_dfs, axis = 1)
    return chr_start_pos, np.hstack(running_means), convolved_dfs


def running_mean_chr(chr, expr, var, window_size, step, calculate_gene_values):

    genes = var.loc[var['.seqid'] == chr].sort_values('.start').index.values
    tmp_x = expr[:, var.index.get_indexer(genes)]
    x_conv, convolved_gene_values = running_mean(
        tmp_x, n = window_size, step = step, 
        gene_list = genes, calculate_gene_values = calculate_gene_values
    )

    return x_conv, convolved_gene_values


def get_reference_cells(
    adata: AnnData,
    reference_key: str | None,
    reference_cat: None | str | Sequence[str],
    reference: np.ndarray | None,
) -> np.ndarray:
    '''Parameter validation extraction of reference gene expression.

    If multiple reference categories are given, compute the mean per
    category.

    Returns a 2D array with reference categories in rows, cells in columns.
    If there's just one category, it's still a 2D array.
    '''

    if reference is None:
        if reference_key is None or reference_cat is None:
            logging.warning(
                'Using mean of all cells as reference. For better results, '
                'provide either `reference`, or both `reference_key` and `reference_cat`. '
            )  # type: ignore
            reference = np.mean(adata.X, axis=0)

        else:
            obs_col = adata.obs[reference_key]
            if isinstance(reference_cat, str):
                reference_cat = [reference_cat]
            reference_cat = np.array(reference_cat)
            reference_cat_in_obs = np.isin(reference_cat, obs_col)
            if not np.all(reference_cat_in_obs):
                raise ValueError(
                    'The following reference categories were not found in '
                    'adata.obs[reference_key]: '
                    f'{reference_cat[~reference_cat_in_obs]}'
                )

            reference = np.vstack([np.mean(adata.X[obs_col.values == cat, :], axis=0) for cat in reference_cat])

    if reference.ndim == 1:
        reference = reference[np.newaxis, :]

    if reference.shape[1] != adata.shape[1]:
        raise ValueError('Reference must match the number of genes in AnnData. ')

    return reference


def infercnv_proc(tmp_x, var, reference, lfc_cap, window_size, step, dynamic_threshold, calculate_gene_values=False):
    
    # step 1: compute log fold change. This densifies the matrix.
    # as default, use 'bounded' difference calculation, if multiple references
    # are available. This mitigates cell-type specific biases (HLA, IGHG, ...)

    if reference.shape[0] == 1:
        x_centered = tmp_x - reference[0, :]

    else:
        ref_min = np.min(reference, axis=0)
        ref_max = np.max(reference, axis=0)
        # entries that are between the two 'bounds' are considered having a logFC of 0.
        x_centered = np.zeros(tmp_x.shape, dtype = tmp_x.dtype)
        above_max = tmp_x > ref_max
        below_min = tmp_x < ref_min
        x_centered[above_max] = ensure_array(tmp_x - ref_max)[above_max]
        x_centered[below_min] = ensure_array(tmp_x - ref_min)[below_min]

    x_centered = ensure_array(x_centered)
    # step 2: clip log fold changes
    x_clipped = np.clip(x_centered, -lfc_cap, lfc_cap)
    # step 3: smooth by genomic position
    chr_pos, x_smoothed, conv_df = running_mean_by_chr(
        x_clipped, var, window_size = window_size, step = step, 
        calculate_gene_values = calculate_gene_values
    )

    # step 4: center by cell
    x_res = x_smoothed - np.median(x_smoothed, axis=1)[:, np.newaxis]
    if calculate_gene_values:
        gene_res = conv_df - np.median(conv_df, axis=1)[:, np.newaxis]
    else: gene_res = None

    # step 5: standard deviation based noise filtering
    if dynamic_threshold is not None:
        noise_thres = dynamic_threshold * np.std(x_res)
        x_res[np.abs(x_res) < noise_thres] = 0
        if calculate_gene_values:
            gene_res[np.abs(gene_res) < noise_thres] = 0

    x_res = scipy.sparse.csr_matrix(x_res)
    return chr_pos, x_res, gene_res