
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import digamma, polygamma
from tqdm import tqdm


def estimate_parameters_all_genes(umi, model_matrix):
    
    results = []
    for gene, row in tqdm(umi.iterrows(), ncols = 80):

        gene_counts = row.values.reshape((-1, 1))
        coefs, theta = estimate_parameters(gene_counts, model_matrix)

        results.append(
            [gene, theta] + coefs.tolist()
        )

    model_pars = pd.DataFrame(
        results, columns = ['gene', 'theta'] + 
        model_matrix.design_info.column_names
    ).set_index("gene")

    return model_pars


def estimate_parameters(gene_counts, model_matrix):
    
    coefs, mu = estimate_parameters_poisson(gene_counts, model_matrix)
    theta = estimate_theta(gene_counts, mu)
    return coefs, theta


def estimate_parameters_poisson(gene_counts, model_matrix):

    model = sm.GLM(gene_counts, model_matrix, family = sm.families.Poisson())
    res = model.fit()
    return res.params, res.mu


def trigamma(x): return polygamma(n=1, x=x)


def estimate_theta(y, mu):
    
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]

    if mu.ndim == 2 and mu.shape[1] == 1:
        mu = mu[:, 0]

    assert y.ndim == 1
    assert mu.ndim == 1

    n = len(y)
    limit = 10
    eps = sys.float_info.epsilon ** 0.25

    def score(n, th, mu, y):
        r = digamma(th + y) - digamma(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th)
        return r.sum()

    def info(n, th, mu, y):
        r = -1 * trigamma(th + y) + trigamma(th) - 1 / th + 2 / (mu + th) - \
            (y + th) / (mu + th)**2
        return r.sum()

    t0 = n / np.sum((y / mu - 1)**2)
    it = 1
    delta = 1
    while it < limit:
        t0 = abs(t0)

        i = info(n, t0, mu, y)
        delta = score(n, t0, mu, y) / i
        t0 = t0 + delta

        if abs(delta) <= eps: break
        it = it + 1

    if t0 < 0: t0 = 0
    return t0