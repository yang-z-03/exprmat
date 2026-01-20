
import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.random import choice

from exprmat import info, pprog
from exprmat.utils import choose_layer


def generate_simulated_data(
    reference, celltype_key = 'cell.type',
    counts_key = 'counts',
    d_prior = None,
    n = 500, samplenum = 5000,
    random_state = None, sparse = True, sparse_prob = 0.5,
    rare = False, rare_percentage = 0.4
):
    mat = choose_layer(reference, layer = counts_key)
    if isinstance(mat, np.ndarray): pass
    else: mat = mat.toarray()

    reference = pd.DataFrame(mat, index = reference.obs[celltype_key], columns = reference.var.index)
    reference.dropna(inplace = True)
    reference['celltype'] = reference.index
    reference.index = range(len(reference))
    
    num_celltype = len(reference['celltype'].value_counts())
    genename = reference.columns[:-1]

    celltype_groups = reference.groupby('celltype', observed = False).groups
    reference.drop(columns = 'celltype', inplace = True)
    reference = np.ascontiguousarray(reference.values, dtype = np.float32)

    if d_prior is None:
        info('generating cell fractions using dirichlet distribution without prior info')
        if isinstance(random_state, int): np.random.seed(random_state)
        prop = np.random.dirichlet(np.ones(num_celltype), samplenum)
        
    elif d_prior is not None:
        info('using prior info to generate cell fractions in dirichlet distribution')
        assert len(d_prior) == num_celltype, \
            'dirichlet prior is a vector, whose length should equals to the number of cell types'
        if isinstance(random_state, int): np.random.seed(random_state)
        prop = np.random.dirichlet(d_prior, samplenum)

    for key, value in celltype_groups.items():
        celltype_groups[key] = np.array(value)

    prop = prop / np.sum(prop, axis = 1).reshape(-1, 1)

    # sparse cell fractions
    if sparse:
        info("some cell's fraction will be zero")
        for i in range(int(prop.shape[0] * sparse_prob)):
            indices = np.random.choice(
                np.arange(prop.shape[1]), replace = False, 
                size = int(prop.shape[1] * sparse_prob)
            )
            prop[i, indices] = 0

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)

    if rare:
        np.random.seed(0)
        indices = np.random.choice(
            np.arange(prop.shape[1]), replace = False, 
            size = int(prop.shape[1] * rare_percentage)
        )

        prop = prop / np.sum(prop, axis=1).reshape(-1, 1)
        for i in range(int(0.5 * prop.shape[0]) + int(int(rare_percentage * 0.5 * prop.shape[0]))):
            prop[i, indices] = np.random.uniform(0, 0.03, len(indices))
            buf = prop[i, indices].copy()
            prop[i, indices] = 0
            prop[i] = (1 - np.sum(buf)) * prop[i] / np.sum(prop[i])
            prop[i, indices] = buf

    # precise number for each celltype
    cell_num = np.floor(n * prop)

    # precise proportion based on cell_num
    prop = cell_num / np.sum(cell_num, axis=1).reshape(-1, 1)

    # start sampling
    sample = np.zeros((prop.shape[0], reference.shape[1]))
    allcellname = celltype_groups.keys()

    info('sampling cells to compose pseudo-bulk data')
    for i, sample_prop in pprog(enumerate(cell_num), total = len(cell_num)):
        for j, cellname in enumerate(allcellname):
            select_index = choice(celltype_groups[cellname], size=int(sample_prop[j]), replace = True)
            sample[i] += reference[select_index].sum(axis = 0)

    prop = pd.DataFrame(prop, columns = celltype_groups.keys())
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        simudata = anndata.AnnData(X = sample, obs = prop, var = pd.DataFrame(index = genename))

    return simudata