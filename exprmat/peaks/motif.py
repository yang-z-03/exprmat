
# Motif enrichment
# ----------------
#
# A hypergeometric test was used to test for overrepresentation of each DNA 
# motif in the set of differentially accessible peaks compared to a background 
# set of peaks. We tested motifs present in the JASPAR database [Fornes et al., 2020] 
# for human (species code 9606) by first identifying which peaks contained each 
# motif using the motifmatchr R package [Schep, 2020]. We computed the GC 
# content (percentage of G and C nucleotides) for each differentially accessible 
# peak and sampled a background set of 40,000 peaks such that the background set 
# was matched for overall GC content, accessibility, and peak width. This was 
# performed using the FindMotifs function in Signac.
# 
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
