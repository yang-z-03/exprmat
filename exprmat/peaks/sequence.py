
import os
from exprmat import pprog
from exprmat.data.finders import basepath
from exprmat import config as default


def query_sequence(adata, seqlen = None):
    
    taxa = default['taxa.reference'][adata.uns['assembly']]
    assembly = adata.uns['assembly']

    genome = os.path.join(basepath, taxa, 'assemblies', assembly, 'genome.fa.gz')
    import pyfastx
    fa = pyfastx.Fasta(genome)

    adata.var['strand.pos'] = 'N'
    adata.var['strand.neg'] = 'N'
    segments = adata.var[['chr', 'start', 'end']].copy().sort_values(['chr', 'start'])
    chromosomes = segments['chr'].value_counts().index.tolist()

    for chr in pprog(chromosomes, desc = 'fetching chromosome'):
        positions = []
        subset = segments.loc[segments['chr'] == chr, :].sort_values('start').copy()
        for f, t in zip(subset['start'], subset['end']):
            # extract the central seqlen sequence
            if seqlen: positions.append((
                (f + t) // 2 - seqlen // 2,
                (f + t) // 2 - seqlen // 2 + seqlen - 1,
            ))
                
            # extract the full sequence
            else: positions.append((f, t))

        posstrand = fa.fetch(chr, positions, strand = '+')
        negstrand = fa.fetch(chr, positions, strand = '-')
        tot_length = len(negstrand)
        cum_start = 0
        for ind, leng in zip(
            subset.index, 
            ([seqlen] * len(subset)) if seqlen else (subset['end'] - subset['start'] + 1)
        ):
            adata.var.loc[ind, 'strand.pos'] = posstrand[cum_start : cum_start + leng]
            adata.var.loc[ind, 'strand.neg'] = negstrand[tot_length - cum_start - leng : tot_length - cum_start]
            cum_start += leng
        
        assert cum_start == len(posstrand)
        assert cum_start == len(negstrand)
    