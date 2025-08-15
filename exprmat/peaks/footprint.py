
# Motif footprinting
# ------------------
#
# We performed transcription factor motif footprinting following previously 
# described methods [Corces et al., 2018]. To account for Tn5 sequence insertion 
# bias, we first computed the observed Tn5 insertion frequency at each DNA 
# hexamer using all Tn5 insertions on chromosome 1. This was done by extracting 
# the base-resolution Tn5 insertion positions for each fragment mapped to 
# chromosome 1, and extending the insertion coordinate 3 bp upstream and 2 bp 
# downstream. We then extracted the DNA sequence corresponding to these 
# coordinates using the getSeq function from the Biostrings R package 
# [Pagès et al., 2020] and counted the frequency of each hexamer using the table 
# function in R. We next computed the expected Tn5 hexamer insertion frequencies 
# based on the frequency of each hexamer on chromosome 1. We counted the 
# frequency of each hexamer using the oligonucleotideFrequency function in the 
# Biostrings package with width=6 and names=“chr1”, using the hg38 genome via 
# the BSgenome R package [Pagès, 2020]. Finally, we computed the Tn5 insertion 
# bias as the observed Tn5 insertions divided by the expected insertions at each 
# hexamer. This was performed using the InsertionBias function in Signac.
# 
# To perform motif footprinting, we first identified the coordinates of each 
# instance of the motif to be footprinted using the matchMotifs function from 
# the motifmatchr package with out=“positions” to return the genomic coordinates 
# of each motif instance [Schep, 2020]. Motif coordinates were then resized to 
# include the +/-250 bp sequence. The Tn5 insertion frequency was counted at each 
# position in the region for each motif instance to produce a matrix containing 
# the total observed Tn5 insertion events at each position relative to the motif 
# center for each cell. We then found the expected Tn5 insertion frequency matrix 
# by computing the hexamer frequency matrix, M. The hexamer frequency matrix M 
# was defined as a matrix with i rows corresponding to i different DNA hexamers 
# and j columns corresponding to j positions centered on the motif, and each 
# entry Mi j corresponded to the hexamer count for hexamer i at position j. To 
# find the expected Tn5 insertion frequency at each position relative to the motif 
# given the Tn5 insertion bias (see above), we computed the matrix cross product 
# between the hexamer frequency matrix M and the Tn5 insertion bias vector. 
# Finally, the expected Tn5 insertion frequencies were normalized by dividing by 
# the mean expected frequency in the 50 bp flanking regions (the regions 200 to 
# 250 bp from the motif). To correct for Tn5 insertion bias we subtracted the 
# expected Tn5 insertion frequencies from the observed Tn5 insertion frequencies 
# at each position. This was performed using the Footprint function in Signac.

import numpy as np
from exprmat import pprog, warning, error


def footprint(
    segments, chromosome_size, peak_by_motif, peak_by_motif_coords, peak_table, 
    motif_names, given, grouping = None, motif_search_len = 200
):

    # get peaks annotated to the motif
    col = motif_names.index(given)
    mask = np.array(peak_by_motif[:, col].T.todense() > 0)[0]
    coords = np.array(peak_by_motif_coords[mask, col].T.todense())[0]
    peak_info = peak_table.loc[mask, :].copy()

    stat_regions = []
    for pid, signcoord, abscoord in zip(range(len(peak_info)), np.sign(coords), np.abs(coords)):
        
        chr, central = (
            peak_info.iloc[pid, :]['chr'], 
            (peak_info.iloc[pid, :]['start'] + peak_info.iloc[pid, :]['end']) // 2
        )

        stat_regions.append([
            chr, central + (abscoord - motif_search_len // 2) - 250 * signcoord,
            central + (abscoord - motif_search_len // 2) + 250 * signcoord
        ])
    
    # cumulate all cut sites within stat_regions.
    # encoded in (chr, start, end). if start > end, 
    # this means the mapping of motif occurs on the minus strand.

    # calculate chromosome starting index
    chr_start = {}
    cumulative = 0
    for chr, length in zip(chromosome_size['seqname'], chromosome_size['len']):
        chr_start[chr] = cumulative
        cumulative += length

    # grouping matches the obs list.
    if grouping == None:
        return summary_cutsites(segments, chr_start, stat_regions, [True] * segments.shape[0])
    
    else:
        summary = {}
        unique_grps = list(set(grouping))
        for group in unique_grps:
            mask = [x == group for x in grouping]
            summary[group] = summary_cutsites(segments, chr_start, stat_regions, mask)

    return summary


def summary_cutsites(segments, chr_start, stat_regions, mask):

    summary = np.zeros((501, ), dtype = np.int64)

    if len(stat_regions) == 0:
        warning('no occurance of this motif found in peaks.')
        return summary
    
    grouped_segment = segments[mask, :]
    for chr, start, end in pprog(stat_regions, desc = 'summarizing peaks'):
        xfrom = min(start, end) + chr_start[chr]
        xto = max(start, end) + chr_start[chr]
        # to consider the reads flanking
        frags = grouped_segment[:, xfrom - 180 : xto + 181]
        if start > end: frags = np.flip(frags, axis = 1)
        values = frags.data.copy().astype('int32')
        if start > end: values = -values

        for index in frags.indices:
            summary[max(0, index - 180 - 3) : max(0, index - 180 + 2)] += 1
        
        complement = frags.indices + values
        for index in complement:
            summary[max(0, index - 180 - 3) : max(0, index - 180 + 2)] += 1
    
    return summary


def run_footprint(
    adata_atac, adata_peaks, motif, motif_matches = 'motifs', groupby = None, 
    key_added = 'footprint.{motif}'
):

    if not 'paired' in adata_atac.obsm.keys():
        error('footprinting with single-ended reads is not yet implemented.')

    if not motif_matches in adata_peaks.varm.keys():
        warning(f'you should run match_motif(...) on the atac-p dataset before analysing footprinting.')
        warning(f'this will typically be done for you when running either motif_enrichment(...) or chromvar(...)')
        warning(f'however, your dataset is missing slots .varm[`{motif_matches}`] and properties.')
        error('you should check the spellings of the peak-by-motif matrix, or run `match_motif(...)`.')
    
    groups = None
    if groupby:
        if groupby in adata_peaks.obs.columns: groups = adata_peaks.obs[groupby].tolist()
        elif groupby in adata_atac.obs.columns: groups = adata_atac.obs[groupby].tolist()
        else: error(f'could not find groupings `{groupby}` in either .obs slots')

    adata_peaks.uns[key_added.format({'motif': motif})] = footprint(
        adata_atac.obsm['paired'],
        adata_atac.uns['assembly.size'],
        adata_peaks.varm[motif_matches],
        adata_peaks.varm[f'{motif_matches}.locs'],
        adata_peaks.var,
        adata_peaks.uns[f'{motif_matches}.names'],
        motif, groups, adata_peaks.uns[f'{motif_matches}.seqlen']
    )