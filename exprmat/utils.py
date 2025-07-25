
import numpy as np
import anndata as ad
import scipy.sparse as sp
from functools import singledispatch
from numba import njit
import numbers
import pandas as pd

import numba
from exprmat.ansi import error, warning, info
from exprmat.configuration import default as cfg


def setup_styles(
    font_name = cfg['plotting.font'], 
    backend = cfg['backend']
):

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # mpl.use(backend)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = font_name
    plt.rcParams["ytick.labelright"] = False
    plt.rcParams["ytick.labelleft"] = True

    # export text as fonts, not as paths.
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # do not show too much precision
    np.set_printoptions(precision = 3, floatmode = 'fixed', suppress = True)
    pass


def ensure_array(a):
    ''' If a is a matrix, turn it into an array. '''
    if isinstance(a, np.matrix): return a.A
    else: return a


def choose_layer(adata, use_raw = False, layer = None):
    is_layer = layer is not None
    if is_layer: 
        if layer == 'X': return adata.X
        elif layer in adata.layers.keys(): return adata.layers[layer]
        else: error(f'layer `{layer}` does not present in the annotated data.')
    elif use_raw: return adata.raw.X
    else: return adata.X


def choose_representation(
    adata: ad.AnnData, *, use_rep = 'pca', n_pcs = None, silent = False,
):
    if use_rep in adata.obsm and n_pcs is not None:
        if n_pcs > adata.obsm[use_rep].shape[1]:
            error(f'{use_rep} does not have enough dimensions.')
        X = adata.obsm[use_rep][:, :n_pcs]

    elif use_rep in adata.obsm and n_pcs is None:
        X = adata.obsm[use_rep]

    elif use_rep == 'X':
        X = adata.X

    else: error(f'{use_rep} does not exist is the obsm slots.')
    return X


def align(subset, parent):
    return [None if not x in parent else parent.index(x) for x in subset]


def find_variable(adata, gene_name, layer = 'X'):

    from scipy.sparse import issparse

    if gene_name in adata.obs.columns:
        vec = adata.obs[gene_name].tolist()

    elif gene_name in adata.var_names:
        X = adata[:, gene_name].X if layer == 'X' else adata[:, gene_name].layers[layer]
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

    # try some conventions
    elif 'gene' in adata.var.keys() and \
        gene_name in adata.var['gene'].tolist():
        genes = adata.var['gene'].tolist()
        X = adata.X[:, genes.index(gene_name)] if layer == 'X' \
            else adata[:, genes.index(gene_name)].layers[layer]
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

     # try some conventions
    elif 'ensembl' in adata.var.keys() and \
        gene_name in adata.var['ensembl'].tolist():
        genes = adata.var['ensembl'].tolist()
        X = adata.X[:, genes.index(gene_name)] if layer == 'X' \
            else adata[:, genes.index(gene_name)].layers[layer]
        
        if issparse(X): vec = X.toarray().reshape(-1)
        else: vec = X.reshape(-1)

    else: error(f'unable to find gene `{gene_name}` in metadata or variables.')
    return vec


def find_gene_index(X, genes):

    if genes is None: return None

    genes = np.array([genes]).flatten()
    if not issubclass(genes.dtype.type, numbers.Integral):
        # gene names
        if isinstance(X, pd.DataFrame): gene_names = X.columns
        elif is_anndata(X): 
            query = {}
            notfound = []
            for i, z in enumerate(X.var_names.tolist()): query[z] = i
            if 'gene' in X.var.keys():
                for i, z in enumerate(X.var['gene'].tolist()):
                    if str(z) != 'nan': query[str(z)] = i
            if 'ensembl' in X.var.keys():
                for i, z in enumerate(X.var['ensembl'].tolist()):
                    if str(z) != 'nan': query[str(z)] = i
            
            indices = []
            for x in genes:
                if x in query.keys(): indices.append(query[x])
                else: notfound.append(x)
            
            if len(notfound) > 0:
                warning("genes {} missing from input data".format(notfound))
            return indices
            
        else: error("you can only refer to gene names in a dataframe or anndata, with var names set.")
        if not np.all(np.isin(genes, gene_names)):
            warning("genes {} missing from input data".format(genes[~np.isin(genes, gene_names)]))
        genes = np.argwhere(np.isin(gene_names, genes)).reshape(-1)
    
    return genes


def select_columns(X, cols):
    
    integer_indices = find_gene_index(X, cols)
    if integer_indices is None: return X
    
    if isinstance(X, ad.AnnData):
        return X[:, integer_indices].copy()
    elif isinstance(X, pd.DataFrame):
        return X.iloc[:, integer_indices].copy()
    elif isinstance(X, np.ndarray) or isinstance(X, np.matrix) or isinstance(X, sp.spmatrix):
        return X[:, integer_indices].copy()
    else: return X[:, integer_indices]


def translate_variables(adata, gene_list, layer = 'X'):
    
    ensembls = adata.var['ensembl'].tolist()
    index = adata.var_names.tolist()
    names = adata.var['gene'].tolist()

    target = []
    for x in gene_list:
        if x in ensembls:
            target.append(index[ensembls.index(x)])
        elif x in names:
            target.append(index[names.index(x)])
        elif x in index:
            target.append(x)
        elif ('rna:' + x) in index:
            target.append('rna:' + x)
        else: warning(f'unable to find gene `{x}` in variables.')
    
    return target


def genes_from_names(gname, taxa):
    
    from exprmat.data.finders import get_mapper_name, get_mapper_ensembl

    gmask = []
    names = []
    not_in_list = []

    alt_finder_name = get_mapper_name(taxa)
    alt_finder_ens = get_mapper_ensembl(taxa)
    for x in gname:

        if x in alt_finder_ens.keys():
            gmask.append(True)
            names.append(alt_finder_ens[x])
            continue
        
        if x in alt_finder_name.keys():
            gmask.append(True)
            names.append(alt_finder_name[x])
            continue
        
        gmask.append(False)
        not_in_list.append(x)

    if len(not_in_list) > 0:
        warning(f'there are {len(not_in_list)} genes not present with the given taxa {taxa}.')

    names = ['rna:' + x for x in names]
    return gmask, names, not_in_list


@singledispatch
def is_constant(a, axis = None):
    """ Check whether values in array are constant. """
    raise NotImplementedError()


@is_constant.register(np.ndarray)
def _(a, axis = None):
    # should eventually support nd, not now.
    if axis is None: return bool((a == a.flat[0]).all())
    if axis == 0: return _is_constant_rows(a.T)
    elif axis == 1: return _is_constant_rows(a)
    else: error('not implemented.')


def _is_constant_rows(a):
    b = np.broadcast_to(a[:, 0][:, np.newaxis], a.shape)
    return (a == b).all(axis = 1)


@is_constant.register(sp.csr_matrix)
def _(a, axis = None):
    if axis is None:
        if len(a.data) == np.multiply(*a.shape):
            return is_constant(a.data)
        else: return (a.data == 0).all()
    if axis == 1: return _is_constant_csr_rows(a.data, a.indptr, a.shape)
    elif axis == 0:
        a = a.T.tocsr()
        return _is_constant_csr_rows(a.data, a.indptr, a.shape)
    else: error('not implemented.')


@njit
def _is_constant_csr_rows(data, indptr, shape,):
    n = len(indptr) - 1
    result = np.ones(n, dtype=np.bool_)
    for i in numba.prange(n):
        start = indptr[i]
        stop = indptr[i + 1]
        val = data[start] if stop - start == shape[1] else 0
        for j in range(start, stop):
            if data[j] != val:
                result[i] = False
                break
    return result


@is_constant.register(sp.csc_matrix)
def _(a, axis = None):
    if axis is None:
        if len(a.data) == np.multiply(*a.shape): return is_constant(a.data)
        else: return (a.data == 0).all()
    if axis == 0: return _is_constant_csr_rows(a.data, a.indptr, a.shape[::-1])
    elif axis == 1:
        a = a.T.tocsc()
        return _is_constant_csr_rows(a.data, a.indptr, a.shape[::-1])
    else: error('not implemented.')


def mdata_to_adata_2(
    mdata,
    x_mod, y_mod,
    x_layer = None, y_layer = None,
    x_use_raw = False, y_use_raw = False,
    x_transform = None,
    y_transform = None,
    verbose = True
):
    """
    Convert a MultiData object to an AnnData object. Returns an AnnData object with the 
    two modalities concatenated. Information related to observations (obs, obsp, obsm) 
    and `.uns` are copied from the original MuData object.
    """

    if x_mod is None or y_mod is None:
        error("both `x_mod` and `y_mod` must be provided.")

    xdata = handle_modality(mdata, x_mod, x_use_raw, x_layer, x_transform, verbose)
    ydata = handle_modality(mdata, y_mod, y_use_raw, y_layer, y_transform, verbose)

    adata = ad.concat([xdata, ydata], axis = 1, label = 'modality')

    adata.obs = mdata.obs.copy()
    adata.obsp = mdata.obsp.copy()
    adata.obsm = mdata.obsm.copy()
    adata.uns = mdata.uns.copy()

    return adata


def handle_modality(mdata, mod, use_raw, layer, transform, verbose):
    
    if mod not in mdata.mod.keys():
        error(f'`{mod}` is not in the mdata!')

    md = mdata.mod[mod].copy()
    if use_raw: md = md.raw.to_adata()
    else: md.X = choose_layer(md, use_raw = use_raw, layer = layer)

    if transform:
        if verbose: info(f'transforming {mod} using {transform.__name__}')
        md.X = transform(md.X)
    
    return md


def counts_to_10x(fpath, outdir, sep = '\t', comment = '#'):
    
    import pandas as pd
    countdf = pd.read_table(fpath, sep = sep, comment = comment, index_col = 0)
    genes = countdf.index.tolist()
    cells = countdf.columns.tolist()

    cells = pd.DataFrame(cells)
    genes = pd.DataFrame({
        'id': genes,
        'name': genes,
        'tag': ['Gene Expression'] * len(genes)
    })
    
    from scipy.sparse import csr_matrix
    from scipy.io import mmwrite
    import os

    mat = csr_matrix(countdf)
    info(f'count matrix of size {mat.shape} written.')
    mmwrite(os.path.join(outdir, 'matrix.mtx'), mat)
    genes.to_csv(os.path.join(outdir, 'features.tsv'), sep = '\t', header = False, index = False)
    cells.to_csv(os.path.join(outdir, 'barcodes.tsv'), sep = '\t', header = False, index = False)


'''
# r conversion from seurat to a three-row features table
seurat_to_10x <- function(srat, outdir) {
    
    features <- data.frame(
        ensembl = rownames(srat),
        gene = rownames(srat),
        tag = "Gene Expression"
    )

    write(
        x = colnames(srat),
        file = paste(outdir, 'barcodes.tsv', sep = '/')
    )
    
    write.table(
        x = features,
        file = paste(outdir, 'features.tsv', sep = '/'),
        sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE
    )

    require(Matrix)
    sparse <- Matrix(srat @ assays $ RNA @ counts, sparse = T)
    writeMM(sparse, file = paste(outdir, 'matrix.mtx', sep = '/'))
}

spmatrix_to_mtx <- function(sp, outdir) {
    features <- data.frame(
        ensembl = rownames(sp),
        gene = rownames(sp),
        tag = "Gene Expression"
    )

    write(
        x = colnames(sp),
        file = paste(outdir, 'barcodes.tsv', sep = '/')
    )
    
    write.table(
        x = features,
        file = paste(outdir, 'features.tsv', sep = '/'),
        sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE
    )

    require(Matrix)
    writeMM(sp, file = paste(outdir, 'matrix.mtx', sep = '/'))
}

seekspace_to_10x <- function(srat, outdir) {
    
    features <- data.frame(
        ensembl = rownames(srat),
        gene = rownames(srat),
        tag = "Gene Expression"
    )

    write(
        x = colnames(srat),
        file = paste(outdir, 'barcodes.tsv', sep = '/')
    )
    
    write.table(
        x = features,
        file = paste(outdir, 'features.tsv', sep = '/'),
        sep = "\t", quote = FALSE, col.names = FALSE, row.names = FALSE
    )

    spa <- srat @ reductions $ spatial @ cell.embeddings |> as.data.frame()
    spa $ Cell_Barcode <- rownames(spa)
    spa <- spa[c('Cell_Barcode', 'spatial_1', 'spatial_2')]
    colnames(spa) <- c('Cell_Barcode', 'X', 'Y')
    write.table(
        x = srat @ reductions $ spatial @ cell.embeddings,
        file = paste(outdir, 'cell_locations.tsv', sep = '/'),
        sep = "\t", quote = FALSE, col.names = TRUE, row.names = TRUE
    )

    require(Matrix)
    sparse <- Matrix(srat @ assays $ RNA @ counts, sparse = T)
    writeMM(sparse, file = paste(outdir, 'matrix.mtx', sep = '/'))
}
'''


def get_file_format(suffix):
    suffix = suffix.lower()
    _suffix = suffix

    if suffix.endswith(".gz"):
        compression = "gzip"
        _suffix = suffix[:-3]
    elif suffix.endswith(".zst"):
        compression = "zstandard"
        _suffix = suffix[:-4]
    else: compression = None
    
    if suffix.endswith(".bw") or suffix.endswith(".bigwig"):
        format = "bigwig"
    elif _suffix.endswith(".bedgraph") or _suffix.endswith(".bg") or _suffix.endswith(".bdg"):
        format = "bedgraph"
    else: format = None
    
    return format, compression


def anndata_rs_par(adatas, func, n_jobs = 4):
    return anndata_rs_ipar(list(enumerate(adatas)), lambda x: func(x[1]), n_jobs = n_jobs)

def anndata_rs_ipar(inputs, func, n_jobs = 4):

    from rich.progress import track
    import exprmat.snapatac as internal

    exist_in_memory_adata = False
    for _, adata in inputs:
        if isinstance(adata, ad.AnnData):
            exist_in_memory_adata = True
            break

    if exist_in_memory_adata:
        warning('in-memory anndata cannot be loaded with parallel.')
        warning('you can use backed anndata with multiprocessing')
        return [func((i, adata)) for i, adata in track(inputs, description = 'loading anndata')]
    
    else:
        from multiprocess import get_context
        def _func(x):
            adata = internal.read(x[1], backend=x[2])
            result = func((x[0], adata))
            adata.close() 
            return result

        # close the AnnData objects and return the filenames
        files = []
        for i, adata in inputs:
            files.append((i, adata.filename, adata.backend))
            adata.close()

        with get_context("spawn").Pool(n_jobs) as p:
            result = list(track(p.imap(_func, files), total = len(files), description = 'loading anndata'))
        
        # reopen the files if they were closed
        for _, adata in inputs:
            adata.open(mode = 'r+')
        
        return result
    

def get_igraph_from_adjacency(adj):
    
    import igraph as ig
    vcount = max(adj.shape)
    sources, targets = adj.nonzero()
    edgelist = list(zip(list(sources), list(targets)))
    weights = np.ravel(adj[(sources, targets)])
    gr = ig.Graph(n = vcount, edges = edgelist, directed = False, edge_attrs = {"weight": weights})
    return gr


def is_anndata(data) -> bool:
    from anndata import AnnData
    import exprmat.snapatac as internal
    return (
        isinstance(data, AnnData) or 
        isinstance(data, internal.AnnData) or 
        isinstance(data, internal.AnnDataSet)
    )


def check_positive(**params):
    for p in params:
        if params[p] <= 0: error("expected {} > 0, got {}".format(p, params[p]))


def check_int(**params):
    import numbers
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            error("expected {} integer, got {}".format(p, params[p]))


def check_if_not(x, *checks, **params):
    for p in params:
        if params[p] is not x and params[p] != x:
            [check(p = params[p]) for check in checks]


def check_in(choices, **params):
    for p in params:
        if params[p] not in choices:
            error("{} value {} not recognized. Choose from {}".format(p, params[p], choices))


def check_between(v_min, v_max, **params):
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            error("expected {} between {} and {}, got {}".format(p, v_min, v_max, params[p]))


def is_sparse_dataframe(x):
    if isinstance(x, pd.DataFrame):
        try:
            x.sparse
            return True
        except AttributeError: pass
    return False


def is_sparse_series(x):
    if isinstance(x, pd.Series):
        try:
            x.sparse
            return True
        except AttributeError: pass
    return False


def to_array_or_spmatrix(x):

    if is_sparse_dataframe(x) or is_sparse_series(x):
        x = x.sparse.to_coo()
    elif isinstance(
        x, (sp.spmatrix, np.ndarray, numbers.Number)
    ) and not isinstance(x, np.matrix): pass
    elif isinstance(x, list):
        x_out = []
        for xi in x:
            try: xi = to_array_or_spmatrix(xi)
            except TypeError: pass
            x_out.append(xi)
        # convert x_out from list to array
        x = np.array(x_out, dtype = check_numpy_dtype(x_out))
    else: x = toarray(x)
    return x


def check_numpy_dtype(x):
    try:
        if all([len(xi) == len(x[0]) for xi in x]):
            # all sequences of the same length; infer dtype
            return None
        else: return object

    except TypeError as e:
        if str(e).startswith("sparse matrix length is ambiguous"):
            # list contains sparse matrices; must be object
            return object
        
        elif str(e).endswith("has no len()"):
            if any([hasattr(xi, "__len__") for xi in x]):
                # some sequences and some not; must be object
                return object
            else:
                # no sequences; infer
                return None
        else: error('type error', e)


def toarray(x):
    
    if isinstance(x, (pd.DataFrame, pd.Series, pd.Index)):
        x = x.to_numpy()
    elif isinstance(x, sp.spmatrix):
        x = x.toarray()
    elif isinstance(x, np.matrix):
        x = x.A
    elif isinstance(x, list):
        x_out = []
        for xi in x:
            try: xi = toarray(xi)
            except TypeError: pass
            x_out.append(xi)
        # convert x_out from list to array
        x = np.array(x_out, dtype = check_numpy_dtype(x_out))
    elif isinstance(x, (np.ndarray, numbers.Number)): pass
    else: error("expected array-like. got {}".format(type(x)))
    return x


def matrix_is_equivalent(X, Y):
    ''' Check matrix equivalence with numpy, scipy and pandas. '''
    if X is Y:
        return True
    elif X.shape == Y.shape:
        if sp.issparse(X) or sp.issparse(Y):
            X = to_array_or_spmatrix(X)
            Y = to_array_or_spmatrix(Y)
        elif isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
            return np.all(X == Y)
        elif not (sp.issparse(X) and sp.issparse(Y)):
            X = toarray(X)
            Y = toarray(Y)
            return np.allclose(X, Y)
        else: return np.allclose((X - Y).data, 0)
    else: return False


def dataframe_to_sparse(x, fill_value = 0.0):

    x = pd.DataFrame.sparse.from_spmatrix(
        sp.coo_matrix(x.values), index = x.index, columns = x.columns
    )

    x.sparse.fill_value = fill_value
    return x


def as_sparse(X, columns = None, index = None, default_fill_value = 0.0):
    
    if sp.issparse(X):
        X = pd.DataFrame.sparse.from_spmatrix(X)
        X.sparse.fill_value = default_fill_value
    else:
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        X = dataframe_to_sparse(X, fill_value = default_fill_value)
    if columns is not None: X.columns = columns
    if index is not None: X.index = index
    return X


def convert_to_same_format(data, target_data, columns = None, prevent_sparse = False):
    ''' Convert data to same format as target data. '''
    # create new data object
    if is_sparse_dataframe(target_data):
        if prevent_sparse: data = pd.DataFrame(data)
        else: data = as_sparse(data)
        pandas = True
    elif isinstance(target_data, pd.DataFrame):
        data = pd.DataFrame(data)
        pandas = True
    elif is_anndata(target_data):
        data = ad.AnnData(data)
        pandas = False
    else:
        # nothing to do
        return data
    
    # retrieve column names
    target_columns = target_data.columns if pandas else target_data.var
    # subset column names
    try:
        if columns is not None:
            if pandas: target_columns = target_columns[columns]
            else: target_columns = target_columns.iloc[columns]
    except (KeyError, IndexError, ValueError):
        # keep the original column names
        if pandas: target_columns = columns
        else: target_columns = pd.DataFrame(index = columns)

    # set column names on new data object
    if pandas:
        data.columns = target_columns
        data.index = target_data.index
    else:
        data.var = target_columns
        data.obs = target_data.obs

    return data


def has_empty_columns(data):
    """ Check if an object has empty columns. """
    try: return np.any(np.array(data.sum(0)) == 0)
    except AttributeError as e:
        if is_anndata(data): return np.any(np.array(data.X.sum(0)) == 0)
        else: error('attribute error', e)


def in_ipynb():
    """
    Check if we are running in a Jupyter Notebook.
    Credit to https://stackoverflow.com/a/24937408/3996580
    """

    notebooks = [
        "<class 'google.colab._shell.Shell'>",
        "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>",
    ]

    # do not mind this get_ipython's exception. this variable is automatically
    # registered when running in an interactive notebook.
    try: return str(type(get_ipython())) in notebooks
    except NameError: return False
