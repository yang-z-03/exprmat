
import anndata
import pandas as pd

import torch
import random
import warnings
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from exprmat.deconv.simulation import generate_simulated_data
from exprmat.utils import choose_layer
from exprmat import info, pprog


class simdatset(Dataset):

    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = torch.from_numpy(self.X[index]).float().to(self.device)
        y = torch.from_numpy(self.Y[index]).float().to(self.device)
        return x, y


class auto_encoder(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.name = 'ae'
        self.state = 'train' # or 'test'
        self.inputdim = input_dim
        self.outputdim = output_dim

        self.encoder = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.inputdim, 512),
            nn.CELU(),       

            nn.Dropout(),
            nn.Linear(512, 256),
            nn.CELU(),
            
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.CELU(),
            
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.CELU(),
            
            nn.Linear(64, output_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.outputdim, 64, bias = False),
            nn.Linear(64, 128, bias = False),
            nn.Linear(128, 256, bias = False),
            nn.Linear(256, 512, bias = False),
            nn.Linear(512, self.inputdim, bias = False)
        )


    def encode(self, x):
        return self.encoder(x)


    def decode(self, z):
        return self.decoder(z)


    def refraction(self,x):
        x_sum = torch.sum(x, dim = 1, keepdim = True)
        return x / x_sum
    
    def sigmatrix(self):

        w0 = (self.decoder[0].weight.T)
        w1 = (self.decoder[1].weight.T)
        w2 = (self.decoder[2].weight.T)
        w3 = (self.decoder[3].weight.T)
        w4 = (self.decoder[4].weight.T)
        w01 = (torch.mm(w0, w1))
        w02 = (torch.mm(w01, w2))
        w03 = (torch.mm(w02, w3))
        w04 = (torch.mm(w03, w4))
        return F.relu(w04)


    def forward(self, x):
        sigmatrix = self.sigmatrix()
        z = self.encode(x)
        if self.state == 'train': pass
        elif self.state == 'test':
            z = F.relu(z)
            z = self.refraction(z)
            
        x_recon = torch.mm(z, sigmatrix)
        return x_recon, z, sigmatrix


class mlp(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units, dropout_rates):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rates = dropout_rates
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.build()


    def forward(self, x):
        return self.model(x)


    def build(self):
        mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_units[0]),
            nn.Dropout(self.dropout_rates[0]),

            nn.ReLU(),
            nn.Linear(self.hidden_units[0], self.hidden_units[1]),
            nn.Dropout(self.dropout_rates[1]),

            nn.ReLU(),
            nn.Linear(self.hidden_units[1], self.hidden_units[2]),
            nn.Dropout(self.dropout_rates[2]),

            nn.ReLU(),
            nn.Linear(self.hidden_units[2], self.hidden_units[3]),
            nn.Dropout(self.dropout_rates[3]),

            nn.ReLU(),
            nn.Linear(self.hidden_units[3], self.output_dim),
            nn.Softmax(dim = 1)
        )

        return mlp


def initialize_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def reproducibility(seed = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def training_stage(model, train_loader, optimizer, epochs=128):
    
    model.train()
    model.state = 'train'
    loss = []
    recon_loss = []
    
    for i in pprog(range(epochs), desc = 'training'):
        for k, (data, label) in enumerate(train_loader):
            
            optimizer.zero_grad()
            x_recon, cell_prop, sigm = model(data)
            batch_loss = F.l1_loss(cell_prop, label) + F.l1_loss(x_recon, data) 
            batch_loss.backward()
            optimizer.step()
            loss.append(F.l1_loss(cell_prop, label).cpu().detach().numpy())
            recon_loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

    return model, loss, recon_loss


def adaptive_stage(
    model, data, optim_dec, optim_enc, device, step = 10, max_iter = 5
):
    
    data = torch.from_numpy(data).float().to(device)
    loss = []
    model.eval()
    model.state = 'test'
    _, ori_pred, ori_sigm = model(data)
    ori_sigm = ori_sigm.detach()
    ori_pred = ori_pred.detach()
    model.state = 'train'
    
    for k in range(max_iter):
        model.train()
        for i in range(step):
            reproducibility(seed = 0)
            optim_dec.zero_grad()
            x_recon, _, sigm = model(data)
            batch_loss = F.l1_loss(x_recon, data)+F.l1_loss(sigm,ori_sigm)
            batch_loss.backward()
            optim_dec.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())

        for i in range(step):
            reproducibility(seed = 0)
            optim_enc.zero_grad()
            x_recon, pred, _ = model(data)
            batch_loss = F.l1_loss(ori_pred, pred)+F.l1_loss(x_recon, data)
            batch_loss.backward()
            optim_enc.step()
            loss.append(F.l1_loss(x_recon, data).cpu().detach().numpy())


    model.eval()
    model.state = 'test'
    _, pred, sigm = model(data)
    return sigm.cpu().detach().numpy(), loss, pred.detach().cpu().numpy()


def train_model(
    train_x, train_y, device,
    model_name = None,
    batch_size = 128, epochs = 128,
):
    
    train_loader = DataLoader(simdatset(train_x, train_y, device), batch_size = batch_size, shuffle = True)
    model = auto_encoder(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = Adam(model.parameters(), lr = 1e-4)

    info('start training auto-encoder')
    model, loss, reconloss = training_stage(model, train_loader, optimizer, epochs = epochs)
    
    if model_name is not None:
        info(f'model saved to {model_name}.pth')
        torch.save(model, model_name + ".pth")

    return model


def predict(
    test_x, genename, celltypes, samplename,
    model_name = None, model = None,
    adaptive = True, mode = 'overall', device = None
):
    if adaptive is True:

        if mode == 'high-resolution':
            signature_list = np.zeros((test_x.shape[0], len(celltypes), len(genename)))
            tpred = np.zeros((test_x.shape[0], len(celltypes)))
            
            for i in pprog(range(len(test_x)), desc = 'predicting'):
                x = test_x[i,:].reshape(1,-1)
                if model_name is not None and model is None:
                    model = torch.load(model_name)
                elif model is not None and model_name is None: pass
                decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
                encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
                op_decode = torch.optim.Adam(decoder_parameters, lr = 1e-4)
                op_encode = torch.optim.Adam(encoder_parameters, lr = 1e-4)
                test_sigm, loss, test_pred = adaptive_stage(
                    model, x, op_decode, op_encode, step = 300, max_iter = 3,
                    device = device
                )
                signature_list[i, :, :] = test_sigm
                tpred[i,:] = test_pred

            tpred = pd.DataFrame(tpred, columns = celltypes, index = samplename)
            ct_signature = {}
            for i in range(len(celltypes)):
                cellname = celltypes[i]
                sigm = signature_list[:,i,:]
                sigm = pd.DataFrame(sigm, columns = genename, index = samplename)
                ct_signature[cellname] = sigm

            return ct_signature, tpred

        elif mode == 'overall':
            if model_name is not None and model is None:
                model = torch.load(model_name + ".pth")
            decoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'decoder' in n]}]
            encoder_parameters = [{'params': [p for n, p in model.named_parameters() if 'encoder' in n]}]
            op_decode = torch.optim.Adam(decoder_parameters, lr = 1e-4)
            op_encode = torch.optim.Adam(encoder_parameters, lr = 1e-4)
            test_sigm, loss, test_pred = adaptive_stage(
                model, test_x, op_decode, op_encode, step = 300, max_iter = 3,
                device = device
            )
            test_sigm = pd.DataFrame(test_sigm, columns = genename,index = celltypes)
            test_pred = pd.DataFrame(test_pred, columns = celltypes,index = samplename)

            return test_sigm, test_pred

    else:
        if model_name is not None and model is None:
            model = torch.load(model_name + ".pth")
        info('predict cell fractions without adaptive training')
        model.eval()
        model.state = 'test'
        data = torch.from_numpy(test_x).float().to(device)
        _, pred, _ = model(data)
        pred = pred.cpu().detach().numpy()
        pred = pd.DataFrame(pred, columns = celltypes, index = samplename)
        return pred


def deconvolute(
    reference, bulk, simulate = True, variance_threshold = 0.98,
    scaler = 'mms', reference_count_key = 'counts', reference_celltype_key = 'cell.type',
    d_prior = None, bulk_key = 'X',
    mode = 'overall', adaptive = True,
    save_model_name = None, sparse = True,
    batch_size = 128, epochs = 128, seed = 42,
    device = None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = torch.device(device)

    if simulate:
        simudata = generate_simulated_data(
            reference = reference, counts_key = reference_count_key, celltype_key = reference_celltype_key, 
            samplenum = 5000, d_prior = d_prior, sparse = sparse
        )

    else: simudata = reference

    train_x = pd.DataFrame(simudata.X, columns = simudata.var.index)
    train_y = simudata.obs
    test_x = pd.DataFrame(choose_layer(bulk, layer = bulk_key), columns = bulk.var.index)
    
    info('filtering low variance genes')
    var_cutoff = train_x.var(axis = 0).sort_values(ascending = False).iloc[
        int(train_x.shape[1] * variance_threshold)]
    train_x = train_x.loc[:, train_x.var(axis = 0) > var_cutoff]

    var_cutoff = test_x.var(axis = 0).sort_values(ascending = False).iloc[int(test_x.shape[1] * variance_threshold)]
    test_x = test_x.loc[:, test_x.var(axis = 0) > var_cutoff]

    inter = train_x.columns.intersection(test_x.columns)
    train_x = train_x[inter]
    test_x = test_x[inter]

    genename = list(inter)
    celltypes = train_y.columns
    samplename = test_x.index
    info(f'{len(inter)} genes taken into consideration')
    
    train_x = np.log(train_x + 1)
    test_x = np.log(test_x + 1)

    if scaler == 'ss':
        ss = StandardScaler()
        ss_train_x = ss.fit_transform(train_x.T).T
        ss_test_x = ss.fit_transform(test_x.T).T
        train_x, train_y, test_x, genename, celltypes, samplename = (
            ss_train_x, train_y.values, ss_test_x, genename, celltypes, samplename
        )

    elif scaler == 'mms':
        mms = MinMaxScaler()
        mms_train_x = mms.fit_transform(train_x.T).T
        mms_test_x = mms.fit_transform(test_x.T).T
        train_x, train_y, test_x, genename, celltypes, samplename = (
            mms_train_x, train_y.values, mms_test_x, genename, celltypes, samplename
        )
    
    info(f'training data shape: {train_x.shape}')
    info(f'prediction data shape: {test_x.shape}')

    reproducibility(seed)
    model = train_model(
        train_x, train_y, device, save_model_name, 
        batch_size = batch_size, epochs = epochs
    )
    
    if adaptive is True:
        signatures, predictions = \
            predict(
                test_x = test_x, 
                genename = genename, 
                celltypes = celltypes, 
                samplename = samplename,
                model = model, 
                model_name = save_model_name,
                adaptive = adaptive, 
                mode = mode,
                device = device
            )
        
        return signatures, predictions

    else:
        predictions = predict(
            test_x = test_x, 
            genename = genename, 
            celltypes = celltypes, 
            samplename = samplename,
            model = model, 
            model_name = save_model_name,
            adaptive = adaptive, 
            mode = mode,
            device = device
        )
        
        return None, predictions