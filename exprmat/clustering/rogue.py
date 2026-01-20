
import numpy as np
import pandas as pd
import statsmodels.api as sm
from skmisc import loess
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from exprmat.ansi import error, warning, info


# the entropy of the genes across cell j is defined to be:
#
#     s1 = ln(mean(expr_j))
#
# this assumes that all cells in the cluster is homogenous. while the alternative
# assumption is that cells are totally heterogenous, and each cell represent its
# distinct state. the entropy for a totally heterogenous dataset is:
# 
#     s2 = mean(log(expr_ij))
#
# for calculation, this is simply a conversion of mean and log transform.
# r is a small positive value to avoid nan's been introduced due to invalid log.
# we require that the expression matrix have gene names as row names.

def rogue_entropy(C, varname, r = 1):
    '''
    Calculate entropy for genes. A completely homogenous gene will yield entropy
    as mean logrithmized counts (mean.expr). Expect raw count matrix.
    '''
  
    entropy = np.log(C + 1).mean(axis = 0) # per gene.
    mean_expr = np.log(C.mean(axis = 0) + r)
    return pd.DataFrame({
        'gene': varname,
        'entropy': entropy.tolist()[0],
        'mean.expr': mean_expr.tolist()[0]
    })


# fit the relationship between expression entropy and mean gene expression:
#
# in real-world dataset, there must be to some extent, heterogenous components
# in the dataset clusters. s1 > s2. and for different techniques for rna sequencing
# there is a systematic bias on the data distribution. we need a fitting step
# to make sure that the highly-variable genes in the clusters are picked.
#
#' @param span is a parameter for loess for controls of fit smoothness.
#' @param mt.method multiple testing method used in p.adjust.
#
# the returning dataframe consists of [1] gene names, [2] log mean expression (s1),
# [3] fit, the fitted curve of real world log mean expression trajectory (s1'),
# [4] ds, delta of entropy, the difference between s2 and s1',
# [5] p.value, assume that ds is normal, and the p value of normal distribution.
# [6] p.adj, the multi-test adjusted p value for normal approximation. and
# [7] entropy, the expression entropy per se.
#
# the returning values are sorted according to decreasing delta of entropy, so
# that the most highly-variable genes ranked on the top.

# available correction methods:
#  * bonferroni : one-step correction
#  * sidak : one-step correction
#  * holm-sidak : step down method using Sidak adjustments
#  * holm : step-down method using Bonferroni adjustments
#  * simes-hochberg : step-up method (independent)
#  * hommel : closed method based on Simes tests (non-negative)
#  * fdr_bh : Benjamini/Hochberg (non-negative)
#  * fdr_by : Benjamini/Yekutieli (negative)
#  * fdr_tsbh : two stage fdr correction (non-negative)
#  * fdr_tsbky : two stage fdr correction (non-negative)

def smoothen_entropy_lowess(df, span = 0.5, correction = "fdr_bh"):
    
    df = df.loc[
        np.isfinite(df['mean.expr']) & 
        df['entropy'] > 0, :
    ].copy()
    
    fit = sm.nonparametric.lowess(
        exog = df['mean.expr'], 
        endog = df['entropy'], frac = span,
        xvals = df['mean.expr']
    )

    df['fit'] = fit
    df['ds'] = df['fit'] - df['entropy']
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    training = df.loc[df['pval'] > 0.1, :].copy()
    
    fit2 = sm.nonparametric.lowess(
        exog = training['mean.expr'], 
        endog = training['entropy'], frac = span,
        xvals = df['mean.expr']
    )

    df['fit'] = fit2
    df['ds'] = df['fit'] - df['entropy']
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    training = df.loc[df['pval'] > 0.1, :].copy()

    fit3 = sm.nonparametric.lowess(
        exog = training['mean.expr'], 
        endog = training['entropy'], frac = span,
        xvals = df['mean.expr']
    )

    df['fit'] = fit3
    df['ds'] = df['fit'] - df['entropy']
    df = df.loc[np.isfinite(df['ds']), :]
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    df['qval'] = multipletests(df['pval'], method = correction)[1]
    
    return df.sort_values(['ds'], ascending = False)


def smoothen_entropy_loess(df, span = 0.5, correction = "fdr_bh"):
    
    df = df.loc[
        np.isfinite(df['mean.expr']) & 
        df['entropy'] > 0, :
    ].copy()
    
    model = loess.loess(df['mean.expr'], df['entropy'], span = span)
    model.fit()
    fit = model.predict(df['mean.expr'])

    df['fit'] = fit.values
    df['ds'] = df['fit'] - df['entropy']
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    training = df.loc[df['pval'] > 0.1, :].copy()
    
    df = df.loc[df['mean.expr'] <= np.max(training['mean.expr']), :].copy()
    model = loess.loess(training['mean.expr'], training['entropy'], span = span)
    model.fit()
    fit2 = model.predict(df['mean.expr'])

    df['fit'] = fit2.values
    df['ds'] = df['fit'] - df['entropy']
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    training = df.loc[df['pval'] > 0.1, :].copy()

    df = df.loc[df['mean.expr'] <= np.max(training['mean.expr']), :].copy()
    model = loess.loess(training['mean.expr'], training['entropy'], span = span)
    model.fit()
    fit3 = model.predict(df['mean.expr'])

    df['fit'] = fit3.values
    df['ds'] = df['fit'] - df['entropy']
    df = df.loc[np.isfinite(df['ds']), :]
    df['pval'] = 1 - norm.cdf(df['ds'], np.mean(df['ds']), np.std(df['ds']))
    df['qval'] = multipletests(df['pval'], method = correction)[1]
    
    return df.sort_values(['ds'], ascending = False)


smoothen_entropy = smoothen_entropy_loess

# identify highly informative genes using rogue's s-e model
# 
# returns a tibble object with seven columns:
# [1] gene, the gene name.
# [2] mean.expr, the mean expression levels of genes.
# [3] entropy, the expected expression entropy from a given mean gene expression.
# [4] fit, the mean expression levels of genes.
# [5] ds, the entropy reduction against the null expectation.
# [6] p.value, the significance of ds against normal distribution.
# [7] p.adj, adjusted p value, or a copy of p.value if specifying not to adjust it.

def rogue_hvg(C, varname, span = 0.5, r = 1, correction = "fdr_bh", adjust_p = True):
  
    entropy = rogue_entropy(C, varname, r = r)
    entropy = smoothen_entropy(entropy, span = span, correction = correction)
    if not adjust_p: entropy['qval'] = entropy["pval"]
    return entropy


# assess the purity of single cell population by rogue index.
#
# we may either specify the platform to access the default k value, or specify
# a k value manually. the rogue test is meaned to be run with highly variable
# genes. you can either specify them manually in `features`, or it is recommended
# to use the s-e model to pick the hvgs for you.

def rogue_index(df, platform = "umi", cutoff = 0.05, k = None, features = None):
    
    if k is None:
        if platform is None:
            warning("you should supply a constant k yourself, or specify the platform for ")
            warning("us to automatically pick the k constant for you. possible platforms ")
            warning("are `umi` or `full-length`, the default k value for each case is 45 ")
            warning("and 500 respectively.")
            error('routine exit.')

        elif platform == 'umi': k = 45
        elif platform == 'full-length': k = 500
        else: error('platform should be one of `umi` or `full-length`.')
    
    if features is not None:
        df = df.loc[[x in features for x in df['gene'].tolist()], :]
        sig = np.abs(df['ds'].to_numpy()).sum()
        rogue = 1 - sig / (sig + k)

    else:
        
        df = df.loc[
            (df['pval'] < cutoff) &
            (df['qval'] < cutoff), :
        ]

        sig = np.abs(df['ds'].to_numpy()).sum()
        rogue = 1 - sig / (sig + k)
    
    return rogue


# remove outlier cells when calculating ROGUE
#' @param n remove this many outlier cells.

def remove_outliers(df, C, varname, n = 2, span = 0.5, r = 1, correction = 'fdr_bh'):
    
    from exprmat.utils import choose_layer
    signif_genes = df.loc[df['qval'] < 0.05, :]['gene'].tolist()
    ngenes = len(signif_genes)
    bool_mask_gene = [x in signif_genes for x in varname]
    expr_cutoff = df['mean.expr'].to_numpy().min()
    
    expr = C[:, bool_mask_gene]
    signif_genes = []
    for imask in range(len(bool_mask_gene)):
        if bool_mask_gene[imask]: signif_genes.append(varname[imask])

    for i in range(ngenes):
        gene_expr = np.flip(np.array(np.sort(expr[:, i].T))[0])
        gene_expr = gene_expr[n:]
        mean = np.log(gene_expr.mean() + r)
        entropy = np.mean(np.log(gene_expr + 1))
        df.loc[df['gene'] == signif_genes[i], 'mean.expr'] = mean
        df.loc[df['gene'] == signif_genes[i], 'entropy'] = entropy

    df = df.loc[df['mean.expr'] > expr_cutoff, :]
    del df['qval']
    df = smoothen_entropy(df, span = span, correction = correction)
    return df


def rogue(
    adata, counts = 'X', platform = "umi", k = None, 
    min_cells = 10, min_genes = 10, outlier_n = 2,
    span = 0.5, r = 1, correction = 'fdr_bh'
):
    '''
    Calculate ROGUE heterogeneity index for the whole annotated data.
    Users are recommended to filter the minimally expressing genes and cells,
    and pick out the subset of interest before feeding into this method. Or
    more conveniently, run the analysis on given categorical properties via
    the wrapper function ``experiment.run_rna_rogue``.

    Parameters
    -----------

    adata : anndata.AnnData
        The set of data you would like to test for heterogeneity

    counts : str = 'X'
        ROGUE index is calculated on raw counts. Specify a layer of raw counts.
    
    platform : Literal['umi', 'full-length']
        Choose k value with preset technology

    k : int = None
        Manually picks a k value. It can be estimated from dataset using
        ``estimate_k`` function, or use the recommended value via ``platform``.

    outlier_n : int = 2
        Remove n top-expressing cells during analysis

    span : float = 0.5
        Controls the smoothness of LOESS regression
    
    correction : str = 'fdr_bh'
        Multitest correction of p-value. Available correction methods:
        *  bonferroni : one-step correction
        *  sidak : one-step correction
        *  holm-sidak : step down method using Sidak adjustments
        *  holm : step-down method using Bonferroni adjustments
        *  simes-hochberg : step-up method (independent)
        *  hommel : closed method based on Simes tests (non-negative)
        *  fdr_bh : Benjamini/Hochberg (non-negative)
        *  fdr_by : Benjamini/Yekutieli (negative)
        *  fdr_tsbh : two stage fdr correction (non-negative)
        *  fdr_tsbky : two stage fdr correction (non-negative)

    Returns
    -----------
        A floating point value, rogue index
    '''
    
    from exprmat.utils import choose_layer
    C = choose_layer(adata, layer = counts).todense()
    hvg = rogue_hvg(
        C, adata.var_names.tolist(), 
        span = span, r = r, correction = correction
    )

    hvg = remove_outliers(
        hvg, C, adata.var_names.tolist(),
        n = outlier_n, span = span, r = r, correction = correction
    )

    return rogue_index(hvg, platform = platform, k = k)


# calculate the value of the reference factor k

def estimate_k(C, varname, span = 0.5, r = 1, correction = "fdr_bh", adjust_p = True):
  
    entropy = rogue_entropy(C, varname, r = r)
    entropy = smoothen_entropy(entropy, span = span, correction = correction)
    if not adjust_p: entropy['qval'] = entropy["pval"]
    entropy = entropy.loc[entropy['qval'] < 0.05, :]
    k = entropy['ds'].to_numpy().sum()
    return k // 2


def se_plot(df):

    import seaborn as sns
    return sns.scatterplot(
        data = df, 
        x = 'mean.expr', 
        y = 'entropy', 
        hue = df['pval'].to_numpy() < 0.05
    )
