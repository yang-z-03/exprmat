
import torch
import random
import numpy as np
import scanpy as sc
import seaborn as sns
from matplotlib import rcParams

from exprmat import warning, info, error, pprog
from exprmat.metacell.model import MetaQ
from exprmat.metacell.engine import train_one_epoch, warm_one_epoch, inference
from exprmat.metacell.utils import (
    load_data, compute_metacell
)


def assign_metacell(
    scaleds,
    size_factors,
    counts,
    n_metacell,
    datatypes = ['rna'],
    device = 'cuda',
    random_seed = 42,
    train_epoch = 300,
    adam_learning_rate = 1e-3,
    adam_weight_decay = 1e-2,
    codebook_init = 'random',
    converge_threshold = 10,
    batch_size = 32,
):

    # set random seeds
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device(device)

    dataloader_train, dataloader_eval, input_dims = load_data(
        scaleds, size_factors, counts,
        n_metacells = n_metacell, batch_size = batch_size
    )

    omics_num = len(scaleds)

    info(f"target metacell number: {n_metacell}")

    net = MetaQ(
        input_dims = input_dims,
        data_types = datatypes,
        entry_num = n_metacell,
    ).to(device)

    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr = adam_learning_rate, 
        weight_decay = adam_weight_decay
    )

    info('training metacell quantization model ...')

    loss_rec_his = loss_vq_his = 1e7
    stable_epochs = 0
    
    if codebook_init == "random":
        warm_epochs = 0
    else:
        # for Kmeans and Geometric initialization
        warm_epochs = min(50, int(train_epoch * 0.2))

    prog = pprog(range(train_epoch), desc = 'training network')
    for epoch in prog:
        
        if epoch < warm_epochs:
            warm_one_epoch(
                model = net,
                data_types = datatypes,
                dataloader = dataloader_train,
                optimizer = optimizer,
                epoch = epoch,
                device = device,
                prog = prog,
            )

        elif epoch == warm_epochs:
            embeds, ids, _, _, _ = inference(
                model = net,
                data_types = datatypes,
                data_loader = dataloader_eval,
                device = device,
            )
            net.quantizer.init_codebook(embeds, method = codebook_init)
            if omics_num == 1: net.copy_decoder_q()

        else:
            loss_rec, loss_vq = train_one_epoch(
                model = net,
                data_types = datatypes,
                dataloader = dataloader_train,
                optimizer = optimizer,
                epoch = epoch,
                device = device,
                prog = prog,
            )

            converge = (abs(loss_vq_his - loss_vq) <= 1e-5) and (
                abs(loss_rec_his - loss_rec) <= 1e-5)
            
            if converge:
                stable_epochs += 1
                if stable_epochs >= converge_threshold:
                    info(f'early stopping the training process at epoch {epoch}')
                    break

            else:
                stable_epochs = 0
                loss_rec_his = loss_rec
                loss_vq_his = loss_vq


    embeds, ids, delta_confs, rec_q_percent, loss_codebook = inference(
        model = net, 
        data_types = datatypes, 
        data_loader = dataloader_eval, 
        device = device
    )

    info(f"quantized reconstruction: {rec_q_percent:.2f}")
    info(f"delta assignment confidence: {np.mean(delta_confs):.2f}")
    info(f"codebook loss: {loss_codebook:.2f}")
    
    assignment = sc.AnnData(embeds.astype('float32'))
    assignment.obs["metacell"] = ids

    return assignment


def aggregate_metacell(adata_list, assignment):
    
    # supply a list of log normalized (X) full matrix.
    # note that the obs order must match that of the initial input.
    # and the previously computed assignment.

    omics_num = len(adata_list)
    metacells = []
    for i in range(omics_num):
        adata = adata_list[i]
        metacell_adata = compute_metacell(adata, assignment.obs["metacell"].to_numpy())
        metacells.append(metacell_adata)

    return metacells
