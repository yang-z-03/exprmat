
import matplotlib.pyplot as plt
import math
import pandas as pd


def ncbi_to_ucsc(x): return 'chrM' if x == 'MT' else 'chr' + x
def ucsc_to_ncbi(x): return 'MT' if x == 'chrM' else x.replace('chr', '')


def initialize_tracks(
    ntracks = 1,
    heights = [1],
    xfrom = 0,
    xto = 1e9, # range of the total track. in absolute chromosomal positions
    sequence_name = 'chr1',
    show_x_axis = True,
    xticks = None, # automatic ticks (4 to 6 labels are inferred.)
    xticklabels = None,
    figsize = (4, 3),
    dpi = 100,
):
    
    fig, axes = plt.subplots(
        nrows = ntracks, ncols = 1, 
        gridspec_kw = { 'width_ratios': [1], 'height_ratios': heights },
        figsize = figsize, dpi = dpi
    )

    fig.subplots_adjust(
        left = 0.05, right = 0.95, top = 0.95, bottom = 0.05,
        wspace = 0.05, hspace = 0.05
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([xfrom, xto])
        for pos in ['right', 'top', 'bottom', 'left']:
            ax.spines[pos].set_visible(False)

    # get the bottom-most track.
    lastax = axes[-1]

    if show_x_axis:

        lastax.spines['bottom'].set_visible(True)
        lastax.set_xlabel(sequence_name)

        if xticks is None:

            interval = (xto - xfrom)
            scaler = 1
            while interval >= 10: 
                interval /= 10
                scaler *= 10

            if interval >= 7:
                interval /= 2
                scaler *= 2
            
            if interval <= 3:
                interval *= 2
                scaler /= 2

            xticksfrom = int(math.ceil(xfrom / scaler))
            xticksto =  int(math.floor(xto / scaler))
            xticks = [x * scaler for x in range(xticksfrom, xticksto + 1)]
            lastax.set_xticks([x * scaler for x in range(xticksfrom, xticksto + 1)])

            # format strings
            fmtstring = 'bp'

            if scaler >= 1000:
                scaler /= 1000
                fmtstring = 'kb'
            
            if scaler >= 1000:
                scaler /= 1000
                fmtstring = 'Mb'
            
            if scaler >= 1000:
                scaler /= 1000
                fmtstring = 'Gb'

            xticklabels = [
                str(x * scaler).replace('.0', '')
                for x in range(xticksfrom, xticksto + 1)]
            xticklabels[-1] = xticklabels[-1] + ' ' + fmtstring
            lastax.set_xticklabels(xticklabels)

        else:

            # manual x ticks
            lastax.set_xticks(xticks)
            if xticklabels is not None: lastax.set_xticklabels(xticklabels)

    return fig, axes, xticks, xticklabels, xfrom, xto


def genome_track(
    ax,
    show_x_axis = False,
    xticks = [],
    xticklabels = [],
    xlabel = None,
    yrange = (0, 1),
    ylabel = None,
):
    if show_x_axis:
        ax.spines['bottom'].set_visible(True)
        if xlabel is not None: ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    
    ax.set_ylim(yrange)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return ax


def chromosome_track(
    ax,
    xfrom, xto,
    chr = 'chr1',
    assembly = 'grcm39',
    color = 'assembly',
    draw_label = False
):
    from exprmat.data.finders import get_genome_architecture, get_genome_cytobands
    garch = get_genome_architecture(assembly)
    garch = garch.loc[garch['chr'] == chr, :]
    garch = garch.loc[(garch['end'] >= xfrom) & (garch['start'] <= xto), :].copy()
    
    ax.set_ylim((0, 1))
    def draw_box(ax, xf, xt, face = 'blue', stroke = 'blue', lw = 1, round = False, label = None, labelfg = None):
        from matplotlib.patches import Rectangle as rectpatch
        rect = rectpatch(
            xy = (max(xf, xfrom - 0.1 * (xto - xfrom)), 0.2), 
            width = min(xt, xto + 0.1 * (xto - xfrom)) - max(xf, xfrom - 0.1 * (xto - xfrom)), height = 0.6,
            edgecolor = stroke, facecolor = face, linewidth = lw
        )
        
        ax.add_patch(rect)
        
        if label is not None:
            ax.text(
                (min(xt, xto) - max(xf, xfrom)) // 2 + max(xf, xfrom), 0.5, 
                label, c = labelfg, verticalalignment = 'center', ha = 'center'
            )

    def draw_boxes(ax, df, face = 'blue', stroke = 'blue', lw = 1, round = False):
        for start, end in zip(df['start'], df['end']):
            draw_box(ax, start, end, face, stroke, lw, round)

    def draw_band_boxes(ax, df, lw = 1, round = False, draw_label = False):

        # for cytobands without color information
        if not 'stain' in df.columns:
            df['stain'] = 'gpos100'

        for start, end, txt, color in zip(df['start'], df['end'], df['name'], df['stain']):
            # acen  : regional centromere
            # gvar  : chromosomal structural element
            # stalk : chromosome arm for now; term request?
            # gneg  : interband
            if color == 'gneg':
                face, stroke, foreground = '#e0e0e0', '#999999', 'black'
            elif color == 'acen':
                # adjacency to centromere
                face, stroke, foreground = 'red', 'red', 'white'
            elif color == 'gvar':
                face, stroke, foreground = 'orange', 'orange', 'white'
            elif color == 'stalk':
                face, stroke, foreground = 'yellow', 'yellow', 'black'
            else:
                # percentile of giemsa staining intensity
                intens = int(color.replace('gpos', ''))
                from exprmat.plotting.palettes import rgb_color
                intens = int(0xe0 * (1 - (intens / 100)))
                face = rgb_color(intens, intens, intens).to_hex()
                stroke = face
                if intens < 100: foreground = 'white'
                else: foreground = 'black'

            draw_box(ax, start, end, face, stroke, lw, round, chr + txt if draw_label else None, foreground)

    # paint primary sequence
    primary = garch[garch['type'] == 'primary'].copy()
    draw_boxes(ax, primary, face = '#e0e0e0', stroke = '#999999', lw = 0.5, round = True)

    if color == 'assembly':
        fixes = garch[garch['type'] == 'fix-patch'].copy()
        draw_boxes(ax, fixes, face = '#ff0000', stroke = 'red', lw = 1)

        novels = garch[garch['type'] == 'novel-patch'].copy()
        draw_boxes(ax, novels, face = 'blue', stroke = 'blue', lw = 1)

        alts = garch[garch['type'] == 'alt-scaffold'].copy()
        draw_boxes(ax, alts, face = '#999999', stroke = '#999999', lw = 0.5)

        gaps = garch[garch['type'] == 'gap'].copy()
        draw_boxes(ax, gaps, face = 'black', stroke = 'black', lw = 1)

        telos = garch[garch['subtype'] == 'telomere'].copy()
        draw_boxes(ax, telos, face = 'purple', stroke = 'purple', lw = 1)

        cens = garch[garch['subtype'] == 'centromere'].copy()
        draw_boxes(ax, cens, face = 'green', stroke = 'green', lw = 2.5)

        pseudoautosomal = garch[garch['type'] == 'par'].copy()
        draw_boxes(ax, pseudoautosomal, face = 'orange', stroke = 'orange', lw = 1)
    
    elif color == 'cytobands':
        
        gbands = get_genome_cytobands(assembly)
        gbands['chr'] = [x for x in gbands['ucsc']]
        gbands = gbands.loc[gbands['chr'] == chr, :]
        gbands = gbands.loc[(gbands['end'] >= xfrom) & (gbands['start'] <= xto), :].copy()

        draw_band_boxes(ax, gbands, draw_label = draw_label)

    return garch, ax


def gene_track(
    ax,
    xfrom, xto,
    chr = '1',
    assembly = 'grcm39',
    show_gene_name = True
):
    from exprmat.data.finders import get_genome, get_genome_model
    from exprmat.configuration import default as cfg
    model = get_genome_model(assembly)
    gtable = get_genome(cfg['taxa.reference'][assembly.lower()])

    gmodel = model.loc[model['chr'] == chr, :]
    gmodel = gmodel.loc[(gmodel['end'] >= xfrom) & (gmodel['start'] <= xto), :].copy()
    gnames = gmodel.sort_values(['start'])['gid'].unique().tolist()
    model = model.loc[[x in gnames for x in model['gid']], :].copy()
    
    if len(gnames) <= 20:
        # render waterflow.
        waterflow = []

        for name in gnames:

            gene = model[model['gid'] == name].copy()
            requested_x = gene['start'].min()
            requested_y = -1
            for i, z in enumerate(waterflow):
                if z < requested_x:
                    requested_y = i
                    break

            if requested_y == -1:
                waterflow += [requested_x]
                requested_y = len(waterflow) - 1

            displayname = str(name)
            if show_gene_name:
                ens = gtable['id'].tolist()
                dname = gtable['gene'].tolist()
                if name in ens: displayname = dname[ens.index(name)] \
                    if str(ens.index(name)) != 'nan' else name
            
            if str(displayname) == 'nan':
                displayname = ''
                
            ntranscript = len(gene[gene['type'] == 'transcript'])
            if ntranscript > 1: displayname += f' ({ntranscript})'
            text = ax.text(max(requested_x, xfrom), requested_y + 0.25, displayname, verticalalignment = 'center')
            # plot the gene model here.

            def draw_gene(ax, df, requested_y):
                from matplotlib.patches import Rectangle as rectpatch
                for row in range(len(df)):
                    rowv = df.iloc[row, :]
                    if rowv['type'] == 'gene':

                        # the base line
                        rect = rectpatch(
                            xy = (rowv['start'], requested_y + 0.72), 
                            # to ensure very small genes visible
                            width = max((xto - xfrom) // 100, rowv['end'] - rowv['start']), height = 0.06,
                            facecolor = 'green' if rowv['strand'] == '+' else 'purple',
                            linewidth = 0, alpha = 0.5
                        )

                        ax.add_patch(rect)
                        # arrows alone the line

                        for x in range(xfrom, xto, (xto - xfrom) // 25):
                            if x > rowv['start'] and x < rowv['end']:
                                ax.arrow(
                                    x = x, y = requested_y + 0.75,
                                    dx = (xto - xfrom) / 100 * (1 if rowv['strand'] == '+' else -1), dy = 0,
                                    facecolor = 'green' if rowv['strand'] == '+' else 'purple',
                                    linewidth = 0,
                                    head_length = (xto - xfrom) / 50,
                                    head_width = 0.2, width = 0.001,
                                    overhang = 0.25,
                                    head_starts_at_zero = False,
                                    shape = 'full'
                                )
                        
                    # draw untranslated region boxes
                    elif rowv['type'] in ['utr', 'utr3', 'utr5']:
                        # the base line
                        rect = rectpatch(
                            xy = (rowv['start'], requested_y + 0.70), 
                            width = rowv['end'] - rowv['start'], height = 0.10,
                            facecolor = 'green' if rowv['strand'] == '+' else 'purple',
                            linewidth = 0, alpha = 0.5
                        )

                        ax.add_patch(rect)
                    
                    # draw exons
                    elif rowv['type'] in ['exon']:
                        # the base line
                        rect = rectpatch(
                            xy = (rowv['start'], requested_y + 0.60), 
                            width = rowv['end'] - rowv['start'], height = 0.30,
                            facecolor = 'green' if rowv['strand'] == '+' else 'purple',
                            linewidth = 0
                        )

                        ax.add_patch(rect)

            draw_gene(ax, gene, requested_y)
            bbox = text.get_window_extent().transformed(ax.transData.inverted())
            waterflow[requested_y] = max(gene['end'].max(), max(xfrom, requested_x) + bbox.width) + ((xto - xfrom) // 50)
            if requested_x + bbox.width > xto: text.remove()
        
        # waterflow plotting heights
        ax.set_ylim((-0.25, len(waterflow) + 0.25))

    else:
        
        # draw all the genome features in a unannotated boxplot
        # this may color the genes by its type.

        genes = model[model['type'] == 'gene'].copy()
        ax.set_ylim((0, 1))

        def draw_box(ax, xf, xt, face = 'blue', stroke = 'blue', lw = 1, round = False):
            from matplotlib.patches import Rectangle as rectpatch
            rect = rectpatch(
                xy = (max(xf, xfrom - 0.1 * (xto - xfrom)), 0.2), 
                width = min(xt, xto + 0.1 * (xto - xfrom)) - max(xf, xfrom - 0.1 * (xto - xfrom)), height = 0.6,
                edgecolor = stroke, facecolor = face, linewidth = lw
            )

            ax.add_patch(rect)

        def draw_boxes(ax, df, face = 'blue', stroke = 'blue', lw = 1, round = False):
            for start, end in zip(df['start'], df['end']):
                draw_box(ax, start, end, face, stroke, lw, round)
        
        draw_boxes(ax, genes, face = 'blue', stroke = 'blue', lw = 1)

    return ax, model


def bed_track(df, ax, chr, xfrom, xto):

    gmodel = df.loc[df['chr'] == chr, :]
    gmodel = gmodel.loc[(gmodel['end'] >= xfrom) & (gmodel['start'] <= xto), :].copy()

    # draw all the genome features in a unannotated boxplot
    # this may color the genes by its type.

    genes = gmodel
    ax.set_ylim((0, 1))

    def draw_box(ax, xf, xt, face = 'blue', stroke = 'blue', lw = 1, round = False):
        from matplotlib.patches import Rectangle as rectpatch
        rect = rectpatch(
            xy = (max(xf, xfrom - 0.1 * (xto - xfrom)), 0.2), 
            width = min(xt, xto + 0.1 * (xto - xfrom)) - max(xf, xfrom - 0.1 * (xto - xfrom)), height = 0.6,
            edgecolor = stroke, facecolor = face, linewidth = lw
        )

        ax.add_patch(rect)

    def draw_boxes(ax, df, face = 'blue', stroke = 'blue', lw = 1, round = False):
        for start, end in zip(df['start'], df['end']):
            draw_box(ax, start, end, face, stroke, lw, round)
    
    draw_boxes(ax, genes, face = 'orange', stroke = 'orange', lw = 1)

    return ax


def linking_track(
    ax,
    xfrom, xto,
    linkings, color = 'red',
    reversed = False, lw = 1
):
    
    if reversed: ax.set_ylim((-0.1, 1))
    else: ax.set_ylim((0, 1.1))

    def draw_link(ax, xf, xt, signif, cmap, lw = 1, reversed = False):
        import math
        x = [xf + (xt - xf) * (x / 100) for x in range(0, 101)]
        fx = [math.sin(_x * math.pi / 100) for _x in range(0, 101)]
        if reversed: fx = [1 - y for y in fx]
        ax.plot(x, fx, c = cmap, alpha = signif, lw = lw)

    def draw_links(ax, df, cmap, lw = 1, reversed = False):
        for start, end, signif in zip(df['start'], df['end'], df['intensity']):
            draw_link(ax, start, end, signif, cmap, lw, reversed)

    draw_links(ax, linkings, cmap = color, lw = lw, reversed = reversed)


def coverage_track(
    ax, xfrom, xto, xticks, coverage, 
    color = 'blue', xlabel = None, ylabel = None, show_x_axis = True
):
    
    if show_x_axis:
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#aaaaaa')
        ax.tick_params(axis = 'x', colors = '#aaaaaa')
        if xlabel is not None: ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels([])
    
    if ylabel is not None: ax.set_ylabel(ylabel)
    cov = coverage[(coverage['x'] >= xfrom) & (coverage['x'] <= xto)].copy()
    cov = cov.fillna(0) # fill nan's with zero
    covx = cov['x'].tolist()
    covy = cov['y'].tolist()
    
    import numpy as np
    ax.set_ylim((0, 1.1 * np.array(covy).max()))

    if covx[0] > xfrom + 1:
        covx = [xfrom, covx[0] - 1] + covx
        covy = [0, 0] + covy

    if covx[-1] < xto - 1:
        covx = covx + [covx[-1] + 1, 0]
        covy = covy + [0, 0]
    
    ax.fill_between(covx, covy, color = color, step = 'post', lw = 0.5)

    return ax, covx, covy


def coverage_track_from_bam(
    ax, bam, xfrom, xto, chrm, xticks, ncpus = 4,
    color = 'blue', xlabel = None, ylabel = None, show_x_axis = True
):
    '''
    Examples
    --------
    >>> coverage_track_from_bam(
    >>>     axes[3], './bam/align.bam', 
    >>>     xfrom, xto, chr_bam, xticks, ncpus = 4, xlabel = None, ylabel = 'WT', 
    >>>     show_x_axis = True
    >>> )
    '''
    import pysam
    import os
    import pandas

    pysam.depth(bam, '-r', f'{chrm}:{int(xfrom)}-{int(xto)}', '-o', '.temp.depth', '-@', str(ncpus))
    depth = pandas.read_table('.temp.depth', sep = '\t', header = None)
    depth.columns = ['chr', 'x', 'y']
    os.system('rm .temp.depth')
    return coverage_track(ax, xfrom, xto, xticks, depth[['x', 'y']], color, xlabel, ylabel, show_x_axis)


def coverage_track_from_bigwig(
    ax, bigwig, xfrom, xto, chrm, xticks, ncpus = 4, use_refseq_style = False,
    color = 'blue', xlabel = None, ylabel = None, show_x_axis = True
):
    '''
    Examples
    --------
    >>> coverage_track_from_bigwig(
    >>>     axes[3], './fc.bigwig', 
    >>>     xfrom, xto, chr_bam, xticks, ncpus = 4, xlabel = None, ylabel = 'WT', 
    >>>     show_x_axis = True
    >>> )
    '''
    import pyBigWig
    from exprmat.ansi import error, warning
    import os
    import pandas

    bw = pyBigWig.open(bigwig)
    if use_refseq_style: chrm = ucsc_to_ncbi(chrm)
    if not bw.chroms(chrm): error(f'specified chromosome {chrm} do not exist in the bigwig file.')
    depth = pandas.DataFrame({'x': [x for x in range(xfrom, xto)], 'y': bw.values(chrm, xfrom, xto)})
    return coverage_track(ax, xfrom, xto, xticks, depth[['x', 'y']], color, xlabel, ylabel, show_x_axis)


def architecture(assembly, figsize = (4, 8), dpi = 100, cby = 'assembly'):
    '''
    Examples
    --------
    >>> h38 = architecture('grch38', figsize = (4, 8), cby = 'cytobands')
    '''

    from exprmat.data.finders import get_genome_size
    import numpy as np

    sizes = get_genome_size(assembly)
    sizes = sorted([
        {'name': k, 'size': sizes[k]} for k in sizes.keys()], 
        key = lambda x: x['size'], reverse = True
    )
    
    maxsize = np.max([x['size'] for x in sizes])
    xright = int(maxsize * 1.05)
    nchrom = len(sizes)

    fig, axes, xticks, xticklabels, xfrom, xto = initialize_tracks(
        nchrom, xfrom = 0, xto = xright, heights = [1 / nchrom] * nchrom,
        figsize = figsize, dpi = dpi, sequence_name = assembly
    )

    for i, k in enumerate(sizes):
        seq = k['name']
        _ = genome_track(axes[i], ylabel = ucsc_to_ncbi(seq))
        _, _ = chromosome_track(axes[i], xfrom, xto, chr = seq, assembly = assembly, color = cby)
    
    return fig


def genes(
    assembly, chr = None, xfrom = None, xto = None, where = None, title = None, 
    flanking = 10000, show_gene_name = True, figsize = (4, 8), dpi = 100, 
    chr_track_height = 0.2
):
    '''
    Examples
    --------
    >>> fig = genes(
    >>>     assembly = 'grcm39', chr = '1', xfrom = 58000000, xto = 58400000, 
    >>>     figsize = (4, 2), title = 'Chromosome 1 (Mus musculus)'
    >>> )
    '''
    from exprmat.data.finders import get_genome

    def plot_there(asm, xf, xt, chr, figsize, dpi, title, showgn):

        fig, axes, xticks, xticklabels, xfrom, xto = initialize_tracks(
            2, xfrom = xf, xto = xt, heights = [chr_track_height, 1 - chr_track_height],
            figsize = figsize, dpi = dpi, sequence_name = title
        )

        _ = genome_track(axes[1], ylabel = 'Genes')
        _, _ = chromosome_track(
            axes[0], xfrom, xto, chr = chr, assembly = asm, color = 'cytobands', 
            draw_label = (xfrom - xto) <= 25e6
        )

        _, _ = gene_track(axes[1], xfrom, xto, chr = chr, assembly = asm, show_gene_name = showgn)
        return fig, axes

    if (xfrom is not None) and (xto is not None) and (chr is not None):
        
        fig, axes = plot_there(
            assembly, xfrom, xto, chr, figsize, dpi, 
            title = chr if title is None else title, showgn = show_gene_name
        )

        return fig
    
    elif where is not None:
        from exprmat.configuration import default as cfg
        gtable = get_genome(cfg['taxa.reference'][assembly.lower()])
        ens = gtable['id'].tolist()
        name = gtable['gene'].tolist()

        if where in ens:
            geneattr = gtable.iloc[ens.index(where), :].copy()
        elif where in name:
            geneattr = gtable.iloc[name.index(where), :].copy()
        else: geneattr = gtable[where, :].copy()

        chr = geneattr['chr']
        xfrom = geneattr['start'] - flanking
        xto = geneattr['end'] + flanking

        fig, axes = plot_there(
            assembly, xfrom, xto, chr, figsize, dpi, 
            title = chr if title is None else title + f' ({chr})', showgn = show_gene_name
        )

        return fig
    
    else:

        from exprmat.ansi import error
        error('must specify a range or a gene name flanking.')
        return None
    

def whereis(assembly, where, upstream = 10000, downstream = 10000):

    from exprmat.configuration import default as cfg
    from exprmat.data.finders import get_genome, get_genome_model
    from exprmat.ansi import error

    gtable = get_genome(cfg['taxa.reference'][assembly.lower()])
    gmodel = get_genome_model(assembly.lower())
    gmodel = gmodel.loc[gmodel['type'] == 'gene', :].copy()

    ens = gtable['id'].tolist()
    name = gtable['gene'].tolist()
    gene_ensembl_id = ''

    if where in ens: gene_ensembl_id = where
    elif where in name: gene_ensembl_id = ens[name.index(where)]
    else: gene_ensembl_id = ens[gtable.index.tolist().index(where)]

    gmodel = gmodel.loc[gmodel['id'] == gene_ensembl_id, :]
    if len(gmodel) != 1:
        error(f'found {len(gmodel)} items with identifier {where}.')
    else: geneattr = gmodel.iloc[0, :]
    
    chr = geneattr['chr']
    xfrom = geneattr['start'] - upstream
    xto = geneattr['end'] + downstream

    return chr, xfrom, xto
    

def plot_peaks(
    adata, group_key, peaks_key = None,
    gene = None, upstream = 5000, downstream = 5000, chr = None, xf = None, xt = None, 
    figsize = (4, 4), dpi = 100, 
    gene_track_height = 0.25, chrom_track_height = 0.05,  
    peak_annotation_height = 0.1,
    consensus_annotation_height = 0.05,
    linkage_height = 0.25,
    min_frag_length = None, max_frag_length = 180,
    title = None, showgn = True,
    plot_consensus_peak = None,
    plot_linkage = None,
    linkage_score = 'cor',
    sample_dir = '.', sample_name = 'temp'
):
    from exprmat.ansi import error
    x_groups = adata.obs[group_key].value_counts().index.tolist()
    n_groups = len(x_groups)

    height_per_channel = (1 - gene_track_height - chrom_track_height) / n_groups
    heights = [chrom_track_height] + [
        height_per_channel * (1 - peak_annotation_height), 
        height_per_channel * peak_annotation_height
    ] * n_groups

    cumulative = 0
    for x in heights: cumulative += x 
    if plot_consensus_peak and plot_linkage: 
        heights += [
            (1 - cumulative) * (1 - consensus_annotation_height - linkage_height), 
            (1 - cumulative) * consensus_annotation_height,
            (1 - cumulative) * linkage_height
        ]
    elif plot_consensus_peak and (not plot_linkage):
        heights += [
            (1 - cumulative) * (1 - consensus_annotation_height), 
            (1 - cumulative) * consensus_annotation_height,
        ]
    elif plot_linkage and (not plot_consensus_peak):
        heights += [
            (1 - cumulative) * (1 - linkage_height), 
            (1 - cumulative) * linkage_height,
        ]
    else: heights += [1 - cumulative]

    if gene: chr, xf, xt = whereis(adata.uns['assembly'], gene, upstream, downstream)
    elif (xf is None) or (xt is None) or (chr is None):
        error(f'you must specify chr, xf and xt manually if you don\'t tell us which gene to plot.')

    n_subplots = 2 + 2 * n_groups
    idx_consensus = -1
    idx_linkage = -1
    idx_gene = -1
    if plot_consensus_peak: n_subplots += 1
    if plot_linkage: n_subplots += 1

    if plot_consensus_peak and plot_linkage:
        idx_consensus = n_subplots - 2
        idx_linkage = n_subplots - 1
        idx_gene = n_subplots - 3
    elif plot_consensus_peak and (not plot_linkage):
        idx_consensus = n_subplots - 1
        idx_gene = n_subplots - 2
    elif plot_linkage and (not plot_consensus_peak):
        idx_linkage = n_subplots - 1
        idx_gene = n_subplots - 2
    else: idx_gene = n_subplots - 1

    fig, axes, xticks, xticklabels, xfrom, xto = initialize_tracks(
        n_subplots, xfrom = xf, xto = xt, heights = heights,
        figsize = figsize, dpi = dpi, sequence_name = title
    )

    _ = genome_track(axes[idx_gene], ylabel = 'Genes')
    _, _ = chromosome_track(
        axes[0], xfrom, xto, chr = chr, assembly = adata.uns['assembly'], color = 'cytobands', 
        draw_label = (xfrom - xto) <= 25e6
    )

    _, gdf = gene_track(
        axes[idx_gene], xfrom, xto, chr = chr, 
        assembly = adata.uns['assembly'], show_gene_name = showgn
    )

    from exprmat.peaks.coverage import export_coverage
    import os

    coverage_dir = os.path.join(sample_dir, 'coverage', sample_name, f'{group_key}-{max_frag_length}')
    built = True
    if os.path.exists(coverage_dir):
        for g in x_groups: built = built and (os.path.exists(os.path.join(coverage_dir, g + '.bigwig')))
    else: built = False

    if not os.path.exists(coverage_dir):
        os.makedirs(coverage_dir)

    if not built:
        export_coverage(
            adata, groupby = group_key, bin_size = 10, normalization = None,
            min_frag_length = min_frag_length,
            max_frag_length = max_frag_length,
            suffix = '.bigwig',
            out_dir = coverage_dir
        )

    for i, g in enumerate(x_groups):

        _ = coverage_track_from_bigwig(
            axes[i * 2 + 1], os.path.join(coverage_dir, g + '.bigwig'), 
            xfrom, xto, chr, 
            xticks, ylabel = g
        )

        _ = bed_track(
            adata.uns[f'peaks.{group_key}' if not peaks_key else peaks_key][g][
                ['chr', 'start', 'end']].copy(), 
            axes[i * 2 + 2], chr, xfrom, xto
        )
    
    # unify y axis
    max_y = 1
    for i, g in enumerate(x_groups):
        _, yrange = axes[i * 2 + 1].get_ylim()
        if yrange > max_y: max_y = yrange
    for i, g in enumerate(x_groups): axes[i * 2 + 1].set_ylim((0, max_y))  
    
    if plot_consensus_peak:
        _ = bed_track(
            adata.uns[plot_consensus_peak][
                ['chr', 'start', 'end']
            ].copy(), 
            axes[idx_consensus], chr, xfrom, xto
        )

    if plot_linkage:

        linkage = adata.uns[plot_linkage]
        linkage = linkage.loc[(linkage['chr'] == chr), :].copy()
        linkage['peak'] = (linkage['start'] + linkage['end']) / 2
        lnk_from = [min(x, y) for x, y in zip(linkage['peak'], linkage['tss'])]
        lnk_to = [max(x, y) for x, y in zip(linkage['peak'], linkage['tss'])]
        plotting = pd.DataFrame({
            'start': lnk_from,
            'end': lnk_to,
            'intensity': linkage[linkage_score],
            'gene': linkage['id']
        })

        plotting = plotting.loc[
            (plotting['start'] < xto) &
            (plotting['end'] > xfrom), :
        ].copy()

        genes_within = gdf.loc[gdf['type'] == 'gene', 'gid'].tolist()
        plotting = plotting.loc[[x in genes_within for x in plotting['gene']], :].copy()
        plotting['intensity'] = plotting['intensity'] / plotting['intensity'].max()

        _ = linking_track(
            axes[idx_linkage], xfrom, xto, plotting, 
            reversed = True, color = 'red'
        )

    return fig