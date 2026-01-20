
import numpy as np
import pandas as pd
import scipy
from scipy.io import mmread
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from scipy import special
import operator

from exprmat.preprocessing.filter import stable_genes
from exprmat.ansi import warning, info

# intialization: 
#   theta is based on estimated doublet proportion;
#   alpha and beta is estimated by assuming a simple gamma distribution and 
#   estimated by method of moment

def init_rna(theta, data):
  ''' Initialize torch parameters. Input data should be the expression matrix in tensor form. '''
  theta = torch.tensor(theta, device = dev)
  mu = torch.mean(data, dim=0)
  var = torch.var(data, dim=0)
  alpha = torch.square(mu)/var.to(dev)
  beta = mu/var.to(dev)

  theta = theta.requires_grad_()
  alpha = alpha.requires_grad_()
  beta = beta.requires_grad_()
  return theta, alpha, beta


def loglik_rna(N, theta, alpha, beta, decay, data):
  '''
  N is the user estimated maximum number of cells in a droplet.
  theta, alpha, and beta are outputs from initialization function.
  data should be the count matrix in tensor form.
  '''

  poisson = torch.distributions.Poisson(theta)
  for k in range(N):
    gamma = torch.distributions.Gamma(
       alpha * (1 + torch.tensor(k, device = dev) / (1 + torch.exp(-decay))), beta)
    if k == 0:
        sum_k = gamma.log_prob(data).exp() * \
            poisson.log_prob(torch.tensor([k], device = dev)).exp()
    else:
        sum_k = sum_k.clone() + gamma.log_prob(data).exp() * \
          poisson.log_prob(torch.tensor([k], device = dev)).exp()

  l = sum_k.log().sum()
  return l


def mle_regress_rna(data, N = 3, p = 0.7, lr = 0.001, tolerance = 15):
    '''
    Parameters
    -----------

    N : int
        Estimated number of cells in a droplet.

    p : float
        The initial guess of singlet rate; this value doesn't have a big impact on 
        parameter estimation, by default set to 0.70. Not all protocols have a multiplet
        rate as high as this.
    
    data : torch.Tensor
        Count matrix in torch tensor.

    tolerence : int
        Controls the exit point when the gradient descent loss function is considered to
        converge in iterations of training.
    '''

    x = data
    x.requires_grad_(False)

    decay = torch.tensor(0.0, device = dev)
    decay = decay.requires_grad_()

    theta = -math.log(p)
    theta, alpha, beta = init_rna(theta, data)
    alpha_0 = 0.5 * torch.mean(alpha).to('cpu').detach().item()
    parameters = [theta, alpha, beta, decay]
    
    # note that sgd optimizer often fails to work in this case.
    optimizer = optim.Adam(parameters, lr = lr)
    
    nll_0 = - loglik_rna(N, theta, alpha, beta, decay, x)
    l = []
    singlet_rate = []

    for i in range(5000):
        nll = - loglik_rna(N, theta, alpha, beta, decay, x)
        if i % 200 == 0:
            l.append(nll.to('cpu').detach())
            singlet_rate.append(torch.tensor([-theta]).exp())
            if len(l) > 2:
                if (l[-2] - l[-1]) < 0.01 * (l[0] - l[1]):
                    tolerance = tolerance - 1
        
        if tolerance == 0: break

        nll.backward()
        optimizer.step()
        optimizer.zero_grad()
        theta.data.clamp_(0.001, 1.0)
        decay.data.clamp_(-10, 10)
        alpha.data.clamp_(min = alpha_0)

    return theta, alpha, beta, decay, l, singlet_rate


def log_joint_k_rna(data, theta, alpha, beta, decay, k0):
    
    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    data = data.to('cpu').numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()

    alpha = alpha * (1 + k0 / (1 + np.exp(-decay)))

    log_conditional = np.log(gamma.pdf(data, alpha, loc = 0, scale = 1/beta))
    sum_gene = np.sum(log_conditional, axis = 1)
    log_joint = sum_gene + np.log(poisson.pmf(k0, theta))
    var_by_cell = np.var(np.exp(log_conditional), axis = 1)

    return log_joint


def prob_k0_rna(data, theta, alpha, beta, decay, k0, k=3):

    log_joint_k0 = log_joint_k_rna(data, theta, alpha, beta, decay, k0)
    one_ks = np.ones((data.shape[0],k))
    for i in np.arange(k):
      one_ks[:, i] = log_joint_k_rna(data, theta, alpha, beta, decay, i)

    logsumexp_ks = special.logsumexp(one_ks, axis = 1)
    log_prob = log_joint_k0 - logsumexp_ks
    log_prob = log_prob.astype('float128')
    prob = np.exp(log_prob, dtype = np.float128)

    return prob


def reliability_rna(data, theta, alpha, beta, decay, k=3):

    prob_singlet = prob_k0_rna(data, theta, alpha, beta, decay, 0, k)
    prob_doublet = 1-prob_singlet
    pred = np.where(prob_doublet > 0.5, True, False)

    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    data = data.to('cpu').numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()

    one_ks = np.ones((data.shape[0], data.shape[1], k))
    for i in np.arange(k):
        alpha_k = alpha*(1+i/(1+np.exp(-decay)))
        one_ks[:,:,i] = np.log(gamma.pdf(data,  alpha_k, loc=0, scale=1/beta))

    # probability of doublets predicted by individual feature
    reliability = 1 - (np.exp(one_ks[:, :, 0] - special.logsumexp(one_ks, axis = 2)))

    # if individual feature prediction result is the same as result by all features, 
    # then record as 1. otherwise record as 0.  then, calculate proportion of features 
    # that can individually provide correct prediction
    reliability[pred, :] = np.where(reliability[pred, :] > 0.5, 1, 0) # predicted doublets
    reliability[list(map(operator.not_, pred)), :] = \
        np.where(reliability[list(map(operator.not_, pred)), :] < 0.5, 1, 0)

    reliability = np.sum(reliability, axis = 1) / data.shape[1]
    result = np.zeros((2, data.shape[0]))
    result[0,:] = reliability
    # flags the cells whose prediction is subject to outliers
    result[1,:] = np.where(reliability <= 0.5, 1, 0)

    return result


def goodness_of_fit_rna(data, alpha, beta, theta, decay, k=3):

    data = torch.round(data)
    data = data.int()
    data = data.to('cpu').numpy()
    alpha = alpha.to('cpu').detach().numpy()
    beta = beta.to('cpu').detach().numpy()
    theta = theta.to('cpu').detach().numpy()
    decay = decay.to('cpu').detach().numpy()

    empirical = np.apply_along_axis(
       lambda x: np.bincount(x, minlength = np.max(data) + 1), axis = 0, arr = data)
    empirical_dist = empirical/data.shape[0]
    empirical_dist # each column is the empirical distribution of a gene

    for i in range(empirical_dist.shape[0]-1):
        empirical_dist[i+1,] += empirical_dist[i,] #empirical cdf

    # calculate theoretical cdf below
    grid = np.expand_dims(np.arange(0, empirical_dist.shape[0], 1, dtype = int) + 0.0001, axis = 1)
    grid = np.repeat(grid, empirical_dist.shape[1], axis = 1)

    one_ks = np.ones((grid.shape[0], grid.shape[1], k))

    for i in np.arange(k):
        alpha_k = alpha * (1 + i / (1 + np.exp(- decay)))
        one_ks[:, :, i] = np.log(gamma.cdf(grid, alpha_k, loc = 0, scale = 1 / beta)) + \
            np.log(poisson.pmf(i, theta))

    logsumexp_ks = special.logsumexp(one_ks, axis = 2)
    theoretical_dist = np.exp(logsumexp_ks)

    diff = np.abs(theoretical_dist-empirical_dist)
    mean_ks = np.mean(np.amax(diff, axis = 0))

    if mean_ks > 0.33:
        warning('the rna modality goodness-of-fit score is less than 3')
        warning('the composite model may not fit the data well')

    return mean_ks


def composite_rna(
        adata, layer = 'X', 
        N = 3, lr = 0.001, p = 0.7, tolerance = 10,
        use_signal_to_noise_ratio = True, stable_number = None
    ):

    '''
    Predict multiplets using composite poisson distribution. This is implemented in
    sccomposite package. It commonly yields more doublets prediction (be stricter)
    than any other (e.g. scrublet, DoubletFinder, ...) methods. It is based on torch,
    and will utilize GPU if available.

    Parameters
    -----------

    adata : anndata.AnnData
        Input anndata, you should run doublet prediction for each sequencing sample 
        (which is actually possible to form doublet between) on count matrix.
        Specify the layer in ``layer`` parameter, by default use the ``X`` slot.

    N : int
        Estimated number of cells in a droplet.

    p : float
        The initial guess of singlet rate; this value doesn't have a big impact on 
        parameter estimation, by default set to 0.70. Not all protocols have a multiplet
        rate as high as this.

    tolerence : int
        Controls the exit point when the gradient descent loss function is considered to
        converge in iterations of training.

    stable_number : int | None
        Manually specify the number of expected stable features. This will override the modality
        parameter used for deciding them automatically.

    use_signal_to_noise : bool
        Use signal to noise ratio to determine selection.

    Returns
    ----------
    A tuple of four:
    
    *   doublet classification vector, indicating whether the possibility of a cell's
        being multiplet is greater than 0.5.
    *   reliability, of the model's fit.
    *   predicted raw probability of being a doublet.
    *   overall weight on the RNA modality.
    '''
    
    global dev
    if torch.cuda.is_available(): dev = 'cuda:0'
    else: dev = 'cpu'
    device = torch.device(dev)

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        info('%d gpu\'s available. using gpu %d (%s) of cuda %d.%d with %.1f Gb total memory.' %
            (torch.cuda.device_count(), device_id, gpu_properties.name,
            gpu_properties.major, gpu_properties.minor, gpu_properties.total_memory / 1e9))
    else: info('no gpu available, will fit the model using cpu only.')

    rna_input = adata.X if layer == 'X' else adata.layers[layer]
    stable = stable_genes(
        rna_input, modality = 'rna', 
        stable_number = stable_number, 
        use_signal_to_noise = use_signal_to_noise_ratio
    )

    stable = torch.tensor(stable, device = dev)
    stable = stable.double()
    stable = stable + torch.tensor([0.0001], device = dev)
    theta, alpha, beta, decay, loss, p = mle_regress_rna(
        stable, N = N, p = p, lr = lr, tolerance = tolerance
    )

    rna_fit = goodness_of_fit_rna(stable, alpha, beta, theta, decay, k = N)
    prob_singlet = prob_k0_rna(stable, theta, alpha, beta, decay, 0, k = N)
    prob_doublet = 1 - prob_singlet
    doublet_classification = np.where(prob_doublet > 0.5, 1, 0)
    reliability_table = reliability_rna(stable, theta, alpha, beta, decay, k=N)
    rna_overall_weight = reliability_table[0,] / rna_fit

    info(f'rna modality goodness of fit: {1 / rna_fit}')
    info('(n < 3): poor fit.  (3 ~ 5): moderate fit.  (n > 5): good fit.')

    return (
        doublet_classification, 
        reliability_table[0, ],
        prob_doublet,
        rna_overall_weight
    )