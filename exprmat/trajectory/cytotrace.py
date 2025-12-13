
import concurrent.futures
import math
import numpy as np
import os
import pandas as pd
import scanpy as sc
import anndata as ad
import subprocess
import torch
import warnings
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import scipy
from exprmat import warning, error, info, basepath


class ste_function(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class straight_estimator(nn.Module):
    def __init__(self):
        super(straight_estimator, self).__init__()

    def forward(self, x):
        x = ste_function.apply(x)
        return x


class binary_module(nn.Module):

    def __init__(
        self, input_size: int = 14271, hidden_size: int = 24, dropout: float = 0.5,
        *args, **kwargs
    ):
        super(binary_module, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop = dropout
        self.dropout = nn.Dropout(self.drop)
        self.maxrank = nn.Parameter(torch.FloatTensor(1, 1).uniform_(0, 1))
        self.maxrank_param = 1000
        self.weight = nn.Parameter(
            torch.FloatTensor(self.input_size, self.hidden_size).normal_(mean = -0.1, std = 0.055))
        self.ste = straight_estimator()
        num_enrich = 2
        num_labels = 1
        self.batchnorm = nn.BatchNorm1d(self.hidden_size*num_enrich, affine=False)
        self.out = nn.Linear(self.hidden_size*num_enrich, num_labels)

    def forward(self, x_rank, x_log2, B):
        # ucell calculation using x_rank
        W = self.ste(self.weight)
        n = W.sum(0).unsqueeze(0)
        maxrank = n.max()+10+torch.clamp(self.maxrank,min=0)*self.maxrank_param

        x_rank = torch.minimum(x_rank,maxrank)
        R = torch.matmul(x_rank,W)
        ucel = 1 - ((R - (n*(n+1))/2)/(n*maxrank))

        # ams calculation using x_log2
        gs_backgrounds = (B @ W) 
        bg_score = ((x_log2 @ gs_backgrounds) / gs_backgrounds.sum(0))
        raw_score = (x_log2 @ W) / W.sum(0)
        r_ams = raw_score - bg_score

        # concatenated gene set scores
        r_all = torch.cat((ucel, r_ams),1)

        # Batch normalization to transfer scores into shared space
        r_outnorm = self.batchnorm(r_all)

        # Apply gene set level dropout
        r_outnorm = self.dropout(r_outnorm)

        # Generate prediction
        pred = self.out(r_outnorm)

        return pred


class binary_encoder(nn.Module):

    def __init__(self, num_layers=6, **block_args):
        super().__init__()
        self.ste = straight_estimator()
        self.layers = nn.ModuleList([binary_module(**block_args) for i in range(num_layers)])

    def forward(self, x_rank, x_log2, B):
        pred_list = []
        for l in self.layers:
            pred = l(x_rank, x_log2, B)
            pred_list.append(pred.unsqueeze(1))
        pred_out = torch.cat(pred_list,1)
        return pred_out
    

def generic_data_loader(rank_expression, log2_expression, batch_size):
    rank_tensor = torch.Tensor(rank_expression)
    log2_tensor = torch.Tensor(log2_expression)
    train_tensor = torch.utils.data.TensorDataset(rank_tensor, log2_tensor)
    data_loader = torch.utils.data.DataLoader(
        train_tensor, batch_size = batch_size, 
        shuffle = False, drop_last = False
    )

    return data_loader


def validation(data_loader, B_in, model, device):
    prob_pred_list = []
    order_pred_list = []

    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    order_vector = torch.Tensor(np.arange(6).reshape(6, 1) / 5).to(device)
    for batch_idx, tensor in enumerate(data_loader):
        X_rank, X_log2 = tensor
        X_rank = X_rank.to(device)
        X_log2 = X_log2.to(device)

        model_output = model(X_rank, X_log2, B_in)
        prob_pred = model_output
        prob_pred = prob_pred.squeeze(2)
        prob_pred = softmax(prob_pred)
        prob_order = torch.matmul(prob_pred,order_vector)
        prob_pred_list.append(prob_pred.detach().cpu().numpy())
        order_pred_list.append(prob_order.squeeze(1).detach().cpu().numpy())
        
    return np.concatenate(prob_pred_list,0), np.concatenate(order_pred_list,0)


# dispersion function
def disp_fn(x):
    if len(np.unique(x)) == 1: return 0
    else: return np.var(x) / np.mean(x)


# choosing top variable genes
def top_var_genes(log2_data):

    dispersion_index = [disp_fn(log2_data[:, i]) for i in range(log2_data.shape[1])]
    top_col_inds = np.argsort(dispersion_index)[-1000:]   
    return top_col_inds


# cytotrace model is trained only on murine gene set.
# the official implementation of human relys on one-to-one homology mapping from
# human to mice etc. this may not make sense.

def preprocess(expression: ad.AnnData, counts_key = 'counts', gene_key = 'gene', species = 'mmu'):

    # expression is an adata
    expression = expression.copy()
    gene_names = expression.var[gene_key] if gene_key else expression.var_names
    
    # species homology mapping
    
    if species != 'mmu':
        from exprmat.data.orthologs import get_orthologs_symbol
        human = get_orthologs_symbol(species, 'mmu')
        lookup = {}
        for key, val in zip(human['source'].tolist(), human['dest'].tolist()):
            if (key not in lookup.keys()) and key != '-': lookup[key] = val
        gene_names = [lookup[x] if x in lookup.keys() else x for x in gene_names]

    features = os.path.join(basepath, 'mmu', 'cytotrace', 'model-features.tsv')
    features = pd.read_table(features, header = None)[0].tolist()

    # check the number of input genes mapped to model features
    intersection = set(gene_names).intersection(features)
    info(str(len(intersection)) + " input genes are present in the model features")
    if len(intersection) < 9000:
        warning('the mapped gene count is below 9000.')
        warning('make sure you supply the correct species.')

    # gene by cell, support dense matrix only.
    from scipy.sparse import issparse
    from exprmat.utils import choose_layer
    mat = choose_layer(expression, layer = counts_key)
    if issparse(mat):
        expression = pd.DataFrame(mat.todense(), index = expression.obs_names, columns = gene_names)
    else: expression = pd.DataFrame(mat, index = expression.obs_names, columns = gene_names)
    expression = pd.DataFrame(index = features).join(expression.T).T
    expression = expression.fillna(0)
    expression = expression.loc[:, ~expression.columns.duplicated()]
    
    cell_names = expression.index
    gene_names = expression.columns
    mat = expression.to_numpy()
    log2_data = np.log2(1000000 * mat.transpose() / mat.sum(1) + 1).transpose()
    rank_data = scipy.stats.rankdata(mat * -1, axis = 1, method = 'average')
    return cell_names, gene_names, rank_data, log2_data
    

def predict(
    rank_data, log2_data, B_in, cell_names, 
    model_dir, batch_size = 1000, device = "cpu"
):
    
    all_preds_test = []
    all_order_test = []
    all_models_path = pd.Series(
        np.array([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(model_dir)) for f in fn]))
    all_models_path = all_models_path[all_models_path.str.endswith('.pt')]

    data_loader = generic_data_loader(rank_data, log2_data, batch_size)
    
    for model_path in all_models_path:
        pytorch_model = torch.load(model_path, map_location = torch.device('cpu'))
        
        model = binary_encoder()
        model = model.to(device)
        model.load_state_dict(pytorch_model)
    
        prob_test, order_test = validation(data_loader, B_in, model, device)
    
        all_preds_test.append(prob_test.reshape(-1,6,1))
        all_order_test.append(order_test.reshape(-1,1))

    predicted_order = np.mean(np.concatenate(all_order_test,1),1)
    predicted_potency = np.argmax(np.concatenate(all_preds_test,2).mean(2),1)

    labels = ['Differentiated', 'Unipotent', 'Oligopotent', 'Multipotent', 'Pluripotent', 'Totipotent']
    labels_dict = dict(zip([0, 1, 2, 3, 4, 5], labels))
    predicted_df = pd.DataFrame({
        'score.cytotrace':predicted_order,
        'potency':predicted_potency
    })
    predicted_df['potency'] = predicted_df['potency'].map(labels_dict)
    predicted_df[labels] = np.concatenate(all_preds_test, 2).mean(2)
    predicted_df.index = cell_names

    return predicted_df


def get_markov_matrix(log2_data, top_col_inds):
    num_samples, num_genes = log2_data.shape
    
    sub_mat = log2_data[:, top_col_inds]
    with np.errstate(divide="ignore", invalid="ignore"): 
        D = np.corrcoef(sub_mat)  # Pairwise pearson-r corrs

    D[np.arange(num_samples), np.arange(num_samples)] = 0
    D[np.where(D != D)] = 0
    cutoff = max(np.mean(D), 0)
    D[np.where(D < cutoff)] = 0

    A = D / (D.sum(1, keepdims=True) + 1e-5)
    return A


def smooth_subset(chunk_log2_data, chunk_predicted_df, top_col_inds, maxiter):
    markov_mat = get_markov_matrix(chunk_log2_data, top_col_inds)
    score = chunk_predicted_df["score.cytotrace"]
    init_score = score.copy()
    prev_score = score.copy()
    traj = []

    for _ in range(int(maxiter)):
        cur_score = 0.9 * markov_mat.dot(prev_score) + 0.1 * init_score
        traj.append(np.mean(np.abs(cur_score - prev_score)) / (np.mean(init_score) + 1e-6))
        if np.mean(np.abs(cur_score - prev_score)) / (np.mean(init_score) + 1e-6) < 1e-6:
            break
        prev_score = cur_score

    return cur_score


def smoothing_by_diffusion(
    predicted_df, log2_data, top_col_inds, 
    smooth_batch_size = 1000, smooth_cores_to_use = 1, 
    seed = 42, maxiter = 1e4, rescale = True, rescale_deg = 1, rescale_ratio = None
):
    # set seed for reproducibility
    np.random.seed(seed)

    if smooth_batch_size > len(log2_data): chunk_number = 1
    else: chunk_number = math.ceil(len(log2_data) / smooth_batch_size)

    original_names = predicted_df.index
    subsamples_indices = np.arange(len(log2_data))
    np.random.shuffle(subsamples_indices)
    subsamples = np.array_split(subsamples_indices, chunk_number)

    smoothed_scores = []
    smooth_results = []
    # process each chunk separately
    with concurrent.futures.ProcessPoolExecutor(max_workers=smooth_cores_to_use) as executor:
        for subsample in subsamples:
            chunk_log2_data = log2_data[subsample, :]
            chunk_predicted_df = predicted_df.iloc[subsample, :]
            smooth_results.append(executor.submit(smooth_subset,chunk_log2_data,chunk_predicted_df,top_col_inds,maxiter))
        for f in concurrent.futures.as_completed(smooth_results):
            cur_score = f.result()
            smoothed_scores.append(cur_score)

    # concatenate the smoothed scores for all chunks
    smoothed_scores_concatenated = pd.concat(smoothed_scores)
   
    return smoothed_scores_concatenated[original_names]


def binning(predicted_df, scores): # scores is smoothed scores
    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent'
    ]
    
    pred_potencies = predicted_df["potency"]
    unique_potency = np.unique(pred_potencies)
    score = 'score.cytotrace'
    df_pred_potency = pd.DataFrame({'potency': pred_potencies, 'score.cytotrace': scores})
    limits = np.arange(7) / 6

    for potency_i, potency in enumerate(labels):
        lower = limits[potency_i]
        upper = limits[potency_i+1]
        if potency in unique_potency:
            data_order = df_pred_potency[df_pred_potency['potency'] == potency]['score.cytotrace'].sort_values()
            index = data_order.index
            n = len(index)
            scaler = MinMaxScaler(feature_range = (lower+1e-8, upper-1e-8))
            order = scaler.fit_transform(np.arange(n).reshape(-1,1))[:,0]
            df_pred_potency.loc[index,score] = order 

    predicted_df["score.cytotrace"] = df_pred_potency[score][predicted_df.index]
    return predicted_df


def map_score_to_potency(score):

    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent'
    ]

    ranges = np.linspace(0, 1, 7)  
    if score <= ranges[1]:
        return labels[0]
    elif score <= ranges[2]:
        return labels[1]
    elif score <= ranges[3]:
        return labels[2]
    elif score <= ranges[4]:
        return labels[3]   
    elif score <= ranges[5]:
        return labels[4]
    elif score <= ranges[6]:
        return labels[5]
    else: return np.nan
        

def shortest_consensus(neighbor_scores):
    idx_use = 2
    last_part = False
    for i in range(2,math.floor(len(neighbor_scores)/2+1)):
        if map_score_to_potency(np.mean(neighbor_scores[:i])) == \
            map_score_to_potency(np.mean(neighbor_scores[i:2*i])) and not last_part:
            idx_use = i
            last_part = True
    
    return 2 * idx_use


def neighborhood_smoothing_single_chunk(df_in, df_pca, chunk_cell_names):
    cell_names = df_pca.index        
    new_scores = []
    for cell in chunk_cell_names:
        cell_dist = pairwise_distances(df_pca.loc[cell,:].values.reshape(1, -1),df_pca)[0]
        cell_dist = cell_dist / np.max(cell_dist)
        neighbor_cells = np.argsort(cell_dist)[:30]
        neighbor_dists = cell_dist[neighbor_cells]
        neighbor_scores = df_in.loc[cell_names[neighbor_cells], 'score.cytotrace'].values
        num_neighbors_keep = shortest_consensus(neighbor_scores)
        if num_neighbors_keep > 1:
            new_neighbor_cells = neighbor_cells[:num_neighbors_keep]
            new_neighbor_dists = neighbor_dists[:num_neighbors_keep]
            new_neighbor_scores = neighbor_scores[:num_neighbors_keep]
            neighbor_score_weights = ((1 - new_neighbor_dists) ** 2) / (((1 - new_neighbor_dists) ** 2).sum())
            proposed_new_score = (new_neighbor_scores * (1 - new_neighbor_dists) ** 2).sum() / (
                        (1 - new_neighbor_dists) ** 2).sum()
            new_scores.append(proposed_new_score)
        else: new_scores.append(-1)
    return pd.DataFrame(new_scores, columns = ['Score'], index = chunk_cell_names)


def neighborhood_smoothing(df_in, log2_data, cores_to_use = 10):

    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent'
    ]

    ranges = np.linspace(0, 1, 7)  
    if len(df_in) < 100:
        info('dataset fewer than 100 cells, neighborhood smoothing disabled')
        df_in['.score.cytotrace'] = df_in['score.cytotrace']
        df_in['.potency'] = df_in['potency']
        
    data_scale = scale(log2_data, axis=1)
    df_out = df_in.copy()
    df_out['.score.cytotrace'] = 0.0
    df_out['.potency'] = ''

    df_out['.score.smooth'] = df_in['score.cytotrace'].copy()
    
    num_pcs = min(30, data_scale.shape[0] - 1)
    pca = PCA(n_components = num_pcs, svd_solver = 'arpack')
    cell_names = df_in.index
    df_pca = pd.DataFrame(
        pca.fit_transform(data_scale),
        index = cell_names,
        columns = ['PC' + str(i+1) for i in range(num_pcs)]
    )
    
    num_cells = df_in.shape[0]
    num_chunks = cores_to_use
    chunk_size = math.ceil(num_cells / num_chunks)
    results = []
    all_scores_out = []

    with concurrent.futures.ProcessPoolExecutor(max_workers = cores_to_use) as executor:
        for chunk in range(num_chunks):
            chunk_cell_names = cell_names[(chunk_size * chunk): np.min([chunk_size * (chunk + 1), num_cells])]
            results.append(executor.submit(
                neighborhood_smoothing_single_chunk, df_in, df_pca, chunk_cell_names
            ))

        for f in concurrent.futures.as_completed(results):
            chunk_scores = f.result()
            all_scores_out.append(chunk_scores)

    df_temp = pd.concat(all_scores_out,ignore_index = False)
    df_out.loc[df_temp.index, '.score.cytotrace'] = df_temp['Score'].values
    df_out.loc[df_out['.score.cytotrace'] < -0.1, '.score.cytotrace'] = \
        df_out.loc[df_out['.score.cytotrace'] < -0.1, '.score.cytotrace']
    
    df_out['.potency'] = ''
    for i in range(len(labels)):
        range_min = ranges[i]
        range_max = ranges[i + 1]
        df_out.loc[
            (range_min < df_out['.score.cytotrace']) * (
                    df_out['.score.cytotrace'] <= range_max), '.potency'
        ] = labels[i]

    return df_out



def process_subset(
    idx, chunked_expression, B_in, smooth_batch_size, smooth_cores_to_use, 
    species, use_model_dir, seed, counts_key, gene_key, device):

    # map and rank
    cell_names, gene_names, rank_data, log2_data = preprocess(
        chunked_expression, counts_key, gene_key, species
    )

    # Check gene counts as QC measure
    gene_counts = (log2_data>0).sum(1)
    low_gc_frac = (gene_counts < 500).mean()
    if low_gc_frac >= 0.2:
        warning(f'{(100 * low_gc_frac):.1f} % of input cells express fewer than 500 genes.')
        warning('for best performance and stability, a minimum gene count of 500 - 1000 is recommended.')

    # top variable genes
    top_col_inds = top_var_genes(log2_data)
    top_col_names = gene_names[top_col_inds]
    
    # predict by unrandomized chunked batches
    info('performing initial model prediction')
    predicted_df = predict(rank_data, log2_data, B_in, cell_names, use_model_dir, chunked_expression.shape[0], device)
    predicted_df['.score.raw'] = predicted_df['score.cytotrace'].copy()

    info('smoothing by diffusion')
    smooth_score = smoothing_by_diffusion(
        predicted_df, log2_data, top_col_inds, smooth_batch_size, smooth_cores_to_use, seed) 
    binned_score_pred_df = binning(predicted_df, smooth_score)

    if chunked_expression.shape[0] < 100:
        binned_score_pred_df['.score.cytotrace'] = binned_score_pred_df['score.cytotrace'].copy()
        return binned_score_pred_df
    else:
        info('smoothing by adaptive knn')
        smooth_by_knn_df = neighborhood_smoothing(binned_score_pred_df, log2_data, smooth_cores_to_use)
        return smooth_by_knn_df


def calculate_cores_to_use(chunk_number, smooth_chunk_number, max_cores, disable_parallelization):

    pred_cores_to_use = 1
    smooth_cores_to_use = 1
    if smooth_chunk_number == 1: pass

    if not disable_parallelization:
        # calculate number of available processors
        num_proc = os.cpu_count()
        if num_proc == 1: pass
        elif max_cores == None:
            pred_cores_to_use = max(1, num_proc // 2)
            smooth_cores_to_use = min(smooth_chunk_number,max(1, num_proc // 2))
        else:
            max_cores = min(max_cores,max(1, num_proc // 2))
            pred_cores_to_use = min(chunk_number,max_cores)
            smooth_cores_to_use = min(smooth_chunk_number,max_cores)
    
    return pred_cores_to_use, smooth_cores_to_use


def cytotrace2(
    adata, 
    key_counts = 'counts',
    key_gene = 'gene',
    taxa = 'mmu',
    batch_size = 20000,
    smooth_batch_size = 1000,
    disable_parallelization = False,
    max_cores = 4,
    seed = 42,
    device = 'cpu'
):
    if max_cores is None:
        cpus_detected = os.cpu_count()

    expression = adata
    
    np.random.seed(seed)
    if batch_size > expression.n_obs:
        batch_size <- expression.n_obs
    elif expression.n_obs > 50000 and batch_size > 50000:
        warning("consider reducing the batch_size to 50000 for runtime and memory efficiency.")
    
    info('preprocessing data')
    
    chunk_number = math.ceil(expression.n_obs / batch_size)
    smooth_chunk_number = math.ceil(batch_size / smooth_batch_size)
    if expression.n_obs < 1000:
        chunk_number = 1
        smooth_chunk_number = 1

    # determine multiprocessing parameters
    pred_cores_to_use, smooth_cores_to_use = calculate_cores_to_use(
        chunk_number, smooth_chunk_number, max_cores, disable_parallelization)
    torch.set_num_threads(pred_cores_to_use)

    use_model_dir = os.path.join(basepath, 'mmu', 'cytotrace', 'models')
    background_path = os.path.join(basepath, 'mmu', 'cytotrace', 'background.pt')
    B = torch.load(background_path)
    B = B.to_dense().T
    original_names = expression.obs_names
    subsamples_indices = np.arange(expression.n_obs) 
    if chunk_number > 1: np.random.shuffle(subsamples_indices)
    subsamples = np.array_split(subsamples_indices, chunk_number)
    
    predictions = []
   
    # process each chunk separately
    info(f'processing in {chunk_number} chunks ...')
    for idx in range(chunk_number):
        chunked_expression = expression[subsamples[idx], :]
        smooth_by_knn_df = process_subset(
            idx, chunked_expression, B, smooth_batch_size, smooth_cores_to_use, 
            taxa, use_model_dir, seed, key_counts, key_gene, device = device)
        predictions.append(smooth_by_knn_df)
    
    predicted_df_final = pd.concat(predictions, ignore_index=False)
    predicted_df_final = predicted_df_final.loc[original_names]
    ranges = np.linspace(0, 1, 7)  
    labels = [
        'Differentiated',
        'Unipotent',
        'Oligopotent',
        'Multipotent',
        'Pluripotent',
        'Totipotent'
    ]
    
    predicted_df_final['.potency'] = pd.cut(
        predicted_df_final['.score.cytotrace'], bins = ranges, 
        labels = labels, include_lowest = True
    )

    all_scores = predicted_df_final['.score.cytotrace'].values
    predicted_df_final['.relative'] = (all_scores - min(all_scores)) / (max(all_scores) - min(all_scores))
    predicted_df_final = predicted_df_final[[".score.cytotrace", ".potency" , ".relative", "score.cytotrace", "potency"]]
    predicted_df_final.columns = ['score', 'potency', 'relative', 'score.preknn', 'potency.preknn']
    
    return predicted_df_final
