
# Peak-to-gene linkage
# --------------------
# 
# We estimated a linkage score for each peak-gene pair using linear regression models, 
# based on recent work described in the SHARE-seq method [Ma et al., 2020]. For 
# each gene, we computed the Pearson correlation coefficient r between the gene 
# expression and the accessibility of each peak within 500 kb of the gene TSS. 
# For each peak, we then computed a background set of expected correlation coefficients 
# given properties of the peak by randomly sampling 200 peaks located on a different 
# chromosome to the gene, matched for GC content, accessibility, and sequence length 
# (MatchRegionStats function in Signac). We then computed the Pearson correlation 
# between the expression of the gene and the set of background peaks. A z-score 
# was computed for each peak as z = (r − µ)/σ, where µ was the background mean 
# correlation coefficient and σ was the standard deviation of the background 
# correlation coefficients for the peak. We computed a p-value for each peak using 
# a one-sided z-test, and retained peak-gene links with a p-value < 0.05 and a 
# Pearson correlation coefficient > 0.05 or < -0.05. This was performed using the 
# LinkPeaks function in Signac.