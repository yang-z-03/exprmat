
import os
from rich.progress import track
from exprmat.data.finders import basepath
from exprmat.configuration import default


def query_sequence(adata):
    
    taxa = default['taxa.reference'][adata.uns['assembly']]
    assembly = adata.uns['assembly']

    genome = os.path.join(basepath, taxa, 'assemblies', assembly, 'genome.fa.gz')
    import pyfastx
    fa = pyfastx.Fasta(genome)

    adata.var['strand.pos'] = 'N'
    adata.var['strand.neg'] = 'N'
    segments = adata.var[['chr', 'start', 'end']].copy().sort_values(['chr', 'start'])
    chromosomes = segments['chr'].value_counts().index.tolist()
    for chr in track(chromosomes, description = 'fetching chromosome ...'):
        positions = []
        subset = segments.loc[segments['chr'] == chr, :].copy()
        for f, t in zip(subset['start'], subset['end']):
            positions.append((f, t))

        adata.var.loc[subset.index, 'strand.pos'] = fa.fetch(chr, positions, strand = '+')
        adata.var.loc[subset.index, 'strand.neg'] = fa.fetch(chr, positions, strand = '-')
    