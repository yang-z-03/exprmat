
import pyBigWig
import pandas
import pysam

from exprmat.ansi import error, warning, info


def bedgraph_to_bigwig(bedgraph, assembly, output):
    
    # read chromosome sizes
    chrom_sizes = []

    from exprmat.data.finders import get_genome_size
    sizes = get_genome_size(assembly)
    for chr in sizes.keys():
        chrom_sizes.append((chr, sizes[chr]))

    # create a new bigWig file for writing
    bw = pyBigWig.open(output, "w")

    # add the header with chromosome sizes
    bw.addHeader(chrom_sizes)

    # Read bedGraph entries and add them to the bigWig file
    with open(bedgraph, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            value = float(parts[3])
            bw.addEntries(chrom, start, ends=[end], values=[value])

    bw.close()


def bigwig_to_bedgraph(bigwig, bedgraph):
    
    bw = pyBigWig.open(bigwig)

    if not bw.isBigWig():
        error(f"{bigwig} is not a valid bigwig file.")
        return None

    with open(bedgraph, "w") as fout:
        for chrom in bw.chroms():
            # Iterate through intervals for each chromosome
            for start, end, value in bw.intervals(chrom):
                # Write each interval as a bedGraph entry
                fout.write(f"{chrom}\t{start}\t{end}\t{value}\n")


def bam_to_fragments(bam, sample_name, is_paired = True):
    
    import os
    if not os.path.exists(bam.replace('.bam', '.sample.bam')):
        info(f'appending sample tag to {bam} ...')
        pysam.addreplacerg(
            '-r', f'ID:{sample_name}\tSM:{sample_name}',
            '-o', bam.replace('.bam', '.sample.bam'),
            bam
        )

    if not os.path.exists(bam.replace('.bam', '.fragments.tsv.gz')):
        info(f'making fragments file ...')
        from exprmat.peaks.common import make_fragment_file
        make_fragment_file(
            bam.replace('.bam', '.sample.bam'), bam.replace('.bam', '.fragments.tsv.gz'),
            is_paired = is_paired, barcode_tag = 'RG',
            compression = 'gzip'
        )

        # read the fragment file and normalize its structure.
        # combine the duplicated cells.
        df = pandas.read_table(bam.replace('.bam', '.fragments.tsv.gz'), header = None, low_memory = False)
        from exprmat.plotting.track import ncbi_to_ucsc
        df[0] = [ncbi_to_ucsc(x) for x in df[0]]
        # gather duplicates
        del df[4]
        df.columns = ['chr', 'start', 'end', 'sample']
        pvt = df.pivot_table(index = ['chr', 'start', 'end', 'sample'], aggfunc = 'size').reset_index()
        pvt.to_csv(bam.replace('.bam', '.fragments.tsv.gz'), sep = '\t', header = False, index = False)
    
    return bam.replace('.bam', '.fragments.tsv.gz')
