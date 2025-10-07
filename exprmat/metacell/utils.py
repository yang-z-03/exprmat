
import torch
import numpy as np
import scanpy as sc
from scipy import sparse
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score


class MetaQDataset(Dataset):

    def __init__(self, x_list, sf_list, raw_list):
        super().__init__()
        self.x_list = x_list
        self.sf_list = sf_list
        self.raw_list = raw_list

        self.cell_num = self.x_list[0].shape[0]
        self.omics_num = len(self.x_list)

        for i in range(self.omics_num):
            self.x_list[i] = torch.from_numpy(self.x_list[i]).float()
            self.sf_list[i] = torch.from_numpy(self.sf_list[i]).float()
            self.raw_list[i] = torch.from_numpy(self.raw_list[i]).float()


    def __len__(self):
        return int(self.cell_num)


    def __getitem__(self, index):

        x_list = []
        sf_list = []
        raw_list = []
        for i in range(self.omics_num):
            x_list.append(self.x_list[i][index])
            sf_list.append(self.sf_list[i][index])
            raw_list.append(self.raw_list[i][index])
        data = { "x": x_list, "sf": sf_list, "raw": raw_list }
        return data


def preprocess(adata, data_type):

    # returns a tuple of four:
    # - scaled matrix in anndata format with only highly variable genes
    # - size factors by 1e4
    # - raw counts matrix with only highly variable genes
    # - log normalized (but not scaled) full size matrix

    if isinstance(adata.X, sparse.csr_matrix) or isinstance(adata.X, sparse.csc_matrix):
        adata.X = adata.X.toarray()
    raw = adata.X.copy()

    if data_type == "rna":
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sf = np.array((raw.sum(axis = 1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        if adata.shape[1] < 5000:
            sc.pp.highly_variable_genes(adata, n_top_genes = 3000)
        else:
            sc.pp.highly_variable_genes(adata)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]

    elif data_type == "adt":
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sf = np.array((raw.sum(axis = 1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()

    elif data_type == "atac":
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sf = np.array((raw.sum(axis=1) / 1e4).tolist()).reshape(-1, 1)
        sc.pp.log1p(adata)
        adata_ = adata.copy()
        sc.pp.highly_variable_genes(adata, n_top_genes = 30000)
        hvg_index = adata.var["highly_variable"].values
        raw = raw[:, hvg_index]
        adata = adata[:, hvg_index]

    sc.pp.scale(adata, max_value = 10)
    x = adata.X

    return x, sf, raw, adata_


def load_data(scaleds, size_factors, counts, n_metacells, batch_size = 512):

    x_list = []
    sf_list = []
    raw_list = []

    for x, sf, raw in zip(scaleds, size_factors, counts):
        x_list.append(x)
        sf_list.append(sf)
        raw_list.append(raw)

    dataset = MetaQDataset(x_list, sf_list, raw_list)
    if n_metacells > 1000 and batch_size <= 512:
        batch_size = 4096

    dataloader_train = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 4,
        pin_memory = True,
    )

    dataloader_eval = DataLoader(
        dataset = dataset,
        batch_size = batch_size * 4,
        shuffle = False,
        drop_last = False,
        num_workers = 4,
    )

    input_dims = [x.shape[1] for x in x_list]
    return dataloader_train, dataloader_eval, input_dims


def compute_metacell(adata, meta_ids):

    meta_ids = meta_ids.astype(int)
    non_empty_metacell = np.zeros(meta_ids.max() + 1).astype(bool)
    non_empty_metacell[np.unique(meta_ids)] = True

    data = adata.X
    data_meta = np.stack(
        [
            np.array(data[meta_ids == i].mean(axis = 0))[0] if (meta_ids == i).sum() > 0 \
                else np.zeros(shape = (data.shape[1],)) \
            for i in range(meta_ids.max() + 1)
        ]
    )

    data_meta = data_meta[non_empty_metacell]
    import scipy.sparse as sp
    metacell_adata = sc.AnnData(sp.csr_matrix(data_meta))

    return metacell_adata
