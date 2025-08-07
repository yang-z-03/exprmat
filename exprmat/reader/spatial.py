
import anndata as ad
import pandas as pd
import numpy as np
import os

from exprmat.reader.metadata import metadata
from exprmat.ansi import error, warning


def read_table(prefix, **kwargs):

    if os.path.isfile(prefix + '.parquet'):
        return pd.read_parquet(prefix + '.parquet', **kwargs)
    elif os.path.isfile(prefix + '.parquet.gz'):
        return pd.read_parquet(prefix + '.parquet.gz', **kwargs)
    elif os.path.isfile(prefix + '.feather'):
        return pd.read_feather(prefix + '.feather', **kwargs)
    elif os.path.isfile(prefix + '.feather.gz'):
        return pd.read_feather(prefix + '.feather.gz', **kwargs)
    elif os.path.isfile(prefix + '.tsv'):
        return pd.read_table(prefix + '.tsv', **kwargs)
    elif os.path.isfile(prefix + '.tsv.gz'):
        return pd.read_table(prefix + '.tsv.gz', **kwargs)
    elif os.path.isfile(prefix + '.csv'):
        return pd.read_csv(prefix + '.csv', **kwargs)
    elif os.path.isfile(prefix + '.csv.gz'):
        return pd.read_csv(prefix + '.csv.gz', **kwargs)
    else: error(f'do not find {prefix} in any supported table format.')


def read_table_from_extension(fname, **kwargs):

    if fname.endswith('.parquet'):
        return pd.read_parquet(fname, **kwargs)
    elif fname.endswith('.parquet.gz'):
        return pd.read_parquet(fname, **kwargs)
    elif fname.endswith('.feather'):
        return pd.read_feather(fname, **kwargs)
    elif fname.endswith('.feather.gz'):
        return pd.read_feather(fname, **kwargs)
    elif fname.endswith('.tsv'):
        return pd.read_table(fname, **kwargs)
    elif fname.endswith('.tsv.gz'):
        return pd.read_table(fname, **kwargs)
    elif fname.endswith('.csv'):
        return pd.read_csv(fname, **kwargs)
    elif fname.endswith('.csv.gz'):
        return pd.read_csv(fname, **kwargs)
    else: error(f'do not find {fname} in any supported table format.')


def read_multiscale_image_from_spatial(folder):

    import json
    import sh
    from PIL import Image

    if os.path.exists(os.path.join(folder, 'scalefactors_json.json.gz')):
        sh.gunzip(os.path.join(folder, 'scalefactors_json.json.gz'))
    
    if os.path.exists(os.path.join(folder, 'tissue_hires_image.png.gz')):
        sh.gunzip(os.path.join(folder, 'tissue_hires_image.png.gz'))

    if os.path.exists(os.path.join(folder, 'tissue_lowres_image.png.gz')):
        sh.gunzip(os.path.join(folder, 'tissue_lowres_image.png.gz'))
    
    # read scale factors
    with open(os.path.join(folder, 'scalefactors_json.json'), 'r') as f:
        sf = json.load(f)
    
    highres = None
    lowres = None

    if os.path.exists(os.path.join(folder, 'tissue_lowres_image.png')):
        lowres = np.array(Image.open(os.path.join(folder, 'tissue_lowres_image.png')))
    
    if os.path.exists(os.path.join(folder, 'tissue_hires_image.png')):
        highres = np.array(Image.open(os.path.join(folder, 'tissue_hires_image.png')))
    
    template = {
        'images': {
            'hires': highres / 255,
            'lores': lowres / 255,
            'origin': None
        },
        'scalefactors': {
            'hires': sf['tissue_hires_scalef'],
            'lores': sf['tissue_lowres_scalef'],
            'origin': 1
        },
        'mask': None,
        'segmentation': None
    }

    return template


def get_lazyload_shape(paths):
    ''' ROI is supplied as (x_from, x_to, y_from, y_to). '''

    import tifffile
    import zarr
    
    if isinstance(paths, list):
        tzstore = tifffile.imread(paths[0], is_ome = False, aszarr = True)
        zstore = zarr.open(tzstore, 'r')
        c, y, x = len(paths), zstore['0'].shape[0], zstore['0'].shape[1]
        tzstore.close()
        return [y, x, c]
    
    else:

        tzstore = tifffile.imread(paths, is_ome = False, aszarr = True) # (c, y, x)
        zstore = zarr.open(tzstore, 'r')
        shape = list(zstore['0'].shape)
        tzstore.close()
        return shape[1:] + [shape[0]]
        

def read_fullres_from_lazyload(paths, roi = None):
    ''' ROI is supplied as (x_from, x_to, y_from, y_to). '''

    import tifffile

    # if roi is not given, read the whole file.
    # this will take up much memory.

    if roi is None:

        if isinstance(paths, list):
            channels = []
            for p in paths:
                channels += [tifffile.imread(p, is_ome = False, level = 0)]
                channels[len(channels) - 1] = channels[len(channels) - 1] / \
                    (1.0 * channels[len(channels) - 1].max())
                assert len(channels[len(channels) - 1].shape) == 2

            return np.stack(channels, axis = 2)

        else: 

            im = tifffile.imread(p, is_ome = False, level = 0) # (c, y, x)
            if len(im.shape) == 3: im = im.swapaxes(0, 2).swapaxes(0, 1)
            return im
    
    else:

        import zarr
        if isinstance(paths, list):
            channels = []
            for p in paths:
                tzstore = tifffile.imread(p, is_ome = False, aszarr = True)
                zstore = zarr.open(tzstore, 'r')
                channels += [zstore['0'][roi[2]:roi[3], roi[0]:roi[1]]]
                tzstore.close()
                channels[len(channels) - 1] = channels[len(channels) - 1] / \
                    (1.0 * channels[len(channels) - 1].max())
                assert len(channels[len(channels) - 1].shape) == 2

            return np.stack(channels, axis = 2)
        
        else:

            tzstore = tifffile.imread(p, is_ome = False, aszarr = True) # (c, y, x)
            zstore = zarr.open(tzstore, 'r')
            im = zstore['0'][..., roi[2]:roi[3], roi[0]:roi[1]]
            tzstore.close()
            if len(im.shape) == 3: im = im.swapaxes(0, 2).swapaxes(0, 1)
            return im


def read_multiscale_image_from_autofocus(
    folder, l_highres = 2, l_lowres = 4, base_zoom = 1, 
    invert_x = False, invert_y = False
):

    files = [
        'morphology_focus_0000.ome.tif', # b
        'morphology_focus_0001.ome.tif', # g
        'morphology_focus_0002.ome.tif', # r
        'morphology_focus_0003.ome.tif'  # yellow
    ]

    hi = []
    lo = []
    origins = []

    for fn in files:
        basefile = os.path.join(folder, fn)
        basefn = os.path.basename(basefile)
        basefull, baseext = os.path.splitext(basefile)
        basefn = basefn[:-len(baseext)]
        origins += [basefile]

        # tiff files
        if basefile.endswith('.gz'):
            import sh
            sh.gunzip(basefile)
            basefile = basefile[:-3]
        
        if basefile.endswith('.ome.tif') or basefile.endswith('.ome.tiff'):
            import tifffile

            highres = tifffile.imread(basefile, is_ome = False, level = l_highres)
            lowres = tifffile.imread(basefile, is_ome = False, level = l_lowres)

            if invert_y:
                highres = highres[::-1, :]
                lowres = lowres[::-1, :]

            if invert_x:
                highres = highres[:, ::-1]
                lowres = lowres[:, ::-1]

            hi += [highres / highres.max()]
            lo += [lowres / lowres.max()]


    if (len(hi) != 4) or (len(lo) != 4):
        warning('xenium generates 4 autofocus images with dapi, membrane, 18s and stromal.')
        warning('supplying less than 4 channels will make the color inconsistant.')

    if len(hi) != len(lo):
        error('inconsistant scaling image channels.')

    # support custom mixer. 
    # do not mix them here.

    # hi_r = np.zeros(hi[0].shape)
    # hi_g = np.zeros(hi[0].shape)
    # hi_b = np.zeros(hi[0].shape)
    # lo_r = np.zeros(lo[0].shape)
    # lo_g = np.zeros(lo[0].shape)
    # lo_b = np.zeros(lo[0].shape)

    # if (len(hi) <= 4) and (len(lo) <= 4):
    #     color = [(0,0,1), (0,1,0), (1,0,0), (1,1,0)]
    #     for h, l, c in zip(hi, lo, color[0:len(hi)]):
    #         hi_r += h * c[0]
    #         hi_g += h * c[1]
    #         hi_b += h * c[2]
    #         lo_r += l * c[0]
    #         lo_g += l * c[1]
    #         lo_b += l * c[2]
    # 
    # hi_r[hi_r > 1] = 1
    # hi_g[hi_g > 1] = 1
    # hi_b[hi_b > 1] = 1
    # lo_r[lo_r > 1] = 1
    # lo_g[lo_g > 1] = 1
    # lo_b[lo_b > 1] = 1

    return {
        'images': {
            'hires': np.stack(hi, axis = 2),
            'lores': np.stack(lo, axis = 2),
            'origin': origins
        },
        'scalefactors': {
            'hires': base_zoom / (2 ** l_highres),
            'lores': base_zoom / (2 ** l_lowres),
            'origin': base_zoom
        },
        'mask': None,
        'segmentation': None
    }


def read_multiscale_image(
    basefile, l_highres = 2, l_lowres = 4, base_zoom = 1, 
    invert_x = False, invert_y = False
):

    basefn = os.path.basename(basefile)
    basefull, baseext = os.path.splitext(basefile)
    basefn = basefn[:-len(baseext)]

    # tiff files
    if basefile.endswith('.gz'):
        import sh
        sh.gunzip(basefile)
        basefile = basefile[:-3]
    
    if basefile.endswith('.ome.tif') or basefile.endswith('.ome.tiff'):
        # must be in the format of c, y, x.
        import tifffile
        
        highres = tifffile.imread(basefile, is_ome = False, level = l_highres)
        lowres = tifffile.imread(basefile, is_ome = False, level = l_lowres)
        
        c, _, _ = highres.shape

        if c <= 3:

            if invert_y:
                highres = highres[:, ::-1, :]
                lowres = lowres[:, ::-1, :]

            if invert_x:
                highres = highres[:, ::-1, :]
                lowres = lowres[:, ::-1, :]

            # reform the array to (y, x, c).
            highres = highres.swapaxes(0, 2).swapaxes(0, 1)
            lowres = lowres.swapaxes(0, 2).swapaxes(0, 1)
        
        else:

            highres = highres[c // 2, :, :] / highres.max()
            lowres = lowres[c // 2, :, :] / lowres.max()

            if invert_y:
                highres = highres[::-1, :]
                lowres = lowres[::-1, :]

            if invert_x:
                highres = highres[:, ::-1]
                lowres = lowres[:, ::-1]
            
            # dapi is blue :)
            highres = np.stack([np.zeros(highres.shape), np.zeros(highres.shape), highres], axis = 2)
            lowres = np.stack([np.zeros(lowres.shape), np.zeros(lowres.shape), lowres], axis = 2)
        

        return {
            'images': {
                'hires': highres,
                'lores': lowres,
                'origin': basefile
            },
            'scalefactors': {
                'hires': base_zoom / (2 ** l_highres),
                'lores': base_zoom / (2 ** l_lowres),
                'origin': base_zoom
            },
            'mask': None,
            'segmentation': None
        }
    
    elif (
        os.path.exists(basefull + '.hires' + baseext) and
        os.path.exists(basefull + '.lores' + baseext)
    ):
        from PIL import Image

        png_highres = Image.open(basefull + '.hires' + baseext)
        png_lowres = Image.open(basefull + '.lores' + baseext)

        highres = np.array(png_highres)
        lowres = np.array(png_lowres)

        if invert_y:
            highres = highres[::-1, :, :]
            lowres = lowres[::-1, :, :]
        
        if invert_x:
            highres = highres[:, ::-1, :]
            lowres = lowres[:, ::-1, :]

        sp = {
            'images': {
                'hires': highres / 255,
                'lores': lowres / 255,
                'origin': basefile
            },
            'scalefactors': {
                'hires': base_zoom / (2 ** l_highres),
                'lores': base_zoom / (2 ** l_lowres),
                'origin': base_zoom
            },
            'mask': None,
            'segmentation': None
        }

        png_highres.close()
        png_lowres.close()
        return sp
    
    else: # handle and downsample them using pillow

        from PIL import Image

        fullres_multich_img = Image.open(basefile)

        w, h = fullres_multich_img.size
        png_highres = fullres_multich_img.resize((w // int(2 ** l_highres), h // int(2 ** l_highres)))
        png_lowres = fullres_multich_img.resize((w // int(2 ** l_lowres), h // int(2 ** l_lowres)))
        
        png_highres.save(basefull + '.hires' + baseext)
        png_lowres.save(basefull + '.lores' + baseext)

        highres = np.array(png_highres)
        lowres = np.array(png_lowres)

        if invert_y:
            highres = highres[::-1, :, :]
            lowres = lowres[::-1, :, :]
        
        if invert_x:
            highres = highres[:, ::-1, :]
            lowres = lowres[:, ::-1, :]

        sp = {
            'images': {
                'hires': highres / 255,
                'lores': lowres / 255,
                'origin': basefile
            },
            'scalefactors': {
                'hires': base_zoom / (2 ** l_highres),
                'lores': base_zoom / (2 ** l_lowres),
                'origin': base_zoom
            },
            'mask': None,
            'segmentation': None
        }

        png_highres.close()
        png_lowres.close()
        fullres_multich_img.close()
        return sp


def read_seekspace(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None
):

    from exprmat.reader.matcher import read_mtx_rna
    adata = read_mtx_rna(
        src, prefix, metadata = metadata, sample = sample, raw = raw,
        default_taxa = default_taxa, eccentric = eccentric
    )

    # attach morphology
    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[rows['modality'] == 'rnasp-c']
    assert len(rows) == 1
    props = rows.iloc[0]

    # tiff files are high quality files.
    if os.path.exists(os.path.join(src, 'morphology.tiff')):
        adata.uns['spatial'] = {
            props['sample']: read_multiscale_image(os.path.join(src, 'morphology.tiff'))
        }
    
    # png files are considered low-quality files without full resolution.
    elif os.path.exists(os.path.join(src, 'morphology.png')):
        adata.uns['spatial'] = {
            props['sample']: read_multiscale_image(
                # seekgene's magic number 55.
                # it will provide a slide capture where 1px -> 55 units in the x-y table.
                # it is also wierd that the given image have y inversed.
                os.path.join(src, 'morphology.png'), 0, 1, base_zoom = 1 / 55,
                invert_y = True
            )
        }

    # attach spatial coordinates
    spatial = read_table(os.path.join(src, 'cell_locations'), index_col = 0)
    barcodes = adata.obs['barcode'].tolist()
    spatial.index = (props['sample'] + ':') + spatial.index
    sorted_df = spatial.loc[barcodes, :].copy()
    sorted_df.columns = ['x', 'y']
    adata.obsm['spatial'] = sorted_df.values

    return adata


def read_visium(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None,
    accepts = 'rnasp-b'
):
    # read visium v1 data.

    if os.path.exists(os.path.join(src, 'filtered_feature_bc_matrix.h5')):
        from exprmat.reader.matcher import read_h5_rna
        adata = read_h5_rna(
            os.path.join(src, 'filtered_feature_bc_matrix.h5'), 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    elif os.path.exists(os.path.join(src, 'filtered_feature_bc_matrix')):
        from exprmat.reader.matcher import read_mtx_rna
        adata = read_mtx_rna(
            os.path.join(src, 'filtered_feature_bc_matrix'), prefix, 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    else: error('failed to find filtered feature barcode matrix.')

    # attach morphology
    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[rows['modality'] == accepts]
    assert len(rows) == 1
    props = rows.iloc[0]

    # tiff files are high quality files.
    if os.path.exists(os.path.join(src, 'spatial')):
        adata.uns['spatial'] = {
            props['sample']: read_multiscale_image_from_spatial(os.path.join(src, 'spatial'))
        }
    
    else: error('failed to find spatial folder.')

    # attach spatial coordinates
    tpos_file = None
    if os.path.exists(os.path.join(src, 'spatial', 'tissue_positions_list.parquet')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions_list')
    elif os.path.exists(os.path.join(src, 'spatial', 'tissue_positions.parquet')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions')
    elif os.path.exists(os.path.join(src, 'spatial', 'tissue_positions_list.csv')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions_list')
    elif os.path.exists(os.path.join(src, 'spatial', 'tissue_positions.csv')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions')
    elif os.path.exists(os.path.join(src, 'spatial', 'tissue_positions_list.csv.gz')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions_list')
    elif os.path.exists(os.path.join(src, 'spatial', 'tissue_positions.csv.gz')):
        tpos_file = os.path.join(src, 'spatial', 'tissue_positions')
    else: error('failed to find tissue position list.')

    spatial = read_table(tpos_file)
    spatial.columns = ['barcode', 'in.tissue', 'row', 'col', 'y', 'x']
    spatial = spatial.set_index('barcode')
    barcodes = adata.obs['barcode'].tolist()
    spatial.index = (props['sample'] + ':') + spatial.index
    sorted_df = spatial.loc[barcodes, :].copy()
    sorted_df.columns = ['in.tissue', 'row', 'col', 'y', 'x']
    adata.obsm['spatial'] = sorted_df[['x', 'y']].values
    adata.obsm['spatial.array'] = sorted_df[['row', 'col']].values
    adata.obs = adata.obs.join(sorted_df, on = 'barcode')

    return adata


def read_visium_hd(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None
):
    # if there is 2um bins, read that.
    # this contains the raw data of the visium hd slide. however, much of the 
    # later steps cannot be taken on that directly, you should merge them into
    # 8um bins or cell segmentation later, this operation will transform 
    # rnasp-s datasets to rnasp-c.

    if os.path.exists(os.path.join(src, 'binned_outputs', 'square_002um', 'filtered_feature_bc_matrix.h5')):
        from exprmat.reader.matcher import read_h5_rna
        adata = read_h5_rna(
            os.path.join(src, 'binned_outputs', 'square_002um', 'filtered_feature_bc_matrix.h5'), 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    elif os.path.exists(os.path.join(src, 'binned_outputs', 'square_002um', 'filtered_feature_bc_matrix')):
        from exprmat.reader.matcher import read_mtx_rna
        adata = read_mtx_rna(
            os.path.join(src, 'binned_outputs', 'square_002um', 'filtered_feature_bc_matrix'), prefix, 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    else: 
        warning('failed to find 2um bins for visium hd data.')
        error('you should otherwise specify modality rnasp-b like visium.')

    # attach morphology
    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[rows['modality'] == 'rnasp-s']
    assert len(rows) == 1
    props = rows.iloc[0]

    # tiff files are high quality files.
    if os.path.exists(os.path.join(src, 'spatial')):
        adata.uns['spatial'] = {
            props['sample']: read_multiscale_image_from_spatial(
                os.path.join(src, 'binned_outputs', 'square_002um', 'spatial')
            )
        }
    
    else: error('failed to find spatial folder (hd, 2um).')

    # attach spatial coordinates
    tpos_file = os.path.join(src, 'binned_outputs', 'square_002um', 'spatial', 'tissue_positions')
    spatial = read_table(tpos_file).set_index('barcode')
    
    spatial.columns = ['in.tissue', 'row', 'col', 'y', 'x']

    # load mapping for segmentation and other resolution bins.
    if os.path.exists(os.path.join(src, 'barcode_mappings.parquet')):
        mapping = pd.read_parquet(os.path.join(src, 'barcode_mappings.parquet')).set_index('square_002um')
        mapping.columns = ['barcode.8um', 'barcode.16um', 'cell', 'in.nucleus', 'in.cell']
        spatial = spatial.join(mapping, how = 'left')

    else: 
        warning('failed to find barcode mappings.')
        warning('you may need to re-segment the cells and bins with the original image.')

    barcodes = adata.obs['barcode'].tolist()
    spatial.index = (props['sample'] + ':') + spatial.index
    sorted_df = spatial.loc[barcodes, :].copy()
    adata.obsm['spatial'] = sorted_df[['x', 'y']].values
    adata.obsm['spatial.array'] = sorted_df[['row', 'col']].values
    adata.obs = adata.obs.join(sorted_df, on = 'barcode')

    # if there is a segmentation output, read it.
    # together with barcode mapping table we will be able to calculate the cell centers.

    cell_adata = None
    if os.path.exists(os.path.join(src, 'segmented_outputs')):

        cell_adata = read_visium_hd_segmentation(
            os.path.join(src, 'segmented_outputs'),
            prefix = '', metadata = metadata, sample = sample,
            raw = raw, default_taxa = default_taxa, eccentric = eccentric
        )

        # calculate cell centers
        # get a filtered 2um bins (assigned to cells)
        
        spatial = spatial.loc[
            spatial['in.tissue'] &
            (~spatial['cell'].isna()), :
        ].copy().groupby('cell')

        mean_x = spatial['x'].mean()
        mean_y = spatial['y'].mean()
        location = pd.DataFrame({'x': mean_x, 'y': mean_y})
        location.index = (props['sample'] + ':') + mean_x.index
        cell_adata.obs = cell_adata.obs.join(location, on = 'barcode', how = 'left')
        cell_adata.obsm['spatial'] = cell_adata.obs[['x', 'y']].values

        cell_row = props.copy()
        cell_row['modality'] = 'rnasp-c'
        metadata.insert_row(cell_row)

    else: 
        warning('cannot find segmented cells in spaceranger outputs.')
        warning('load the 2um bins for [rnasp-s] and 8um bins for [rnasp-c].')
        
        cell_adata = read_visium(
            os.path.join(src, 'binned_outputs', 'square_008um'),
            prefix = '', metadata = metadata, sample = sample,
            raw = raw, default_taxa = default_taxa, eccentric = eccentric,
            accepts = 'rnasp-s'
        )

        cell_row = props.copy()
        cell_row['modality'] = 'rnasp-c'
        metadata.dataframe.loc[len(metadata.dataframe)] = cell_row


    return adata, cell_adata


def read_visium_hd_segmentation(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None
):
    # if there is 2um bins, read that.
    # this contains the raw data of the visium hd slide. however, much of the 
    # later steps cannot be taken on that directly, you should merge them into
    # 8um bins or cell segmentation later, this operation will transform 
    # rnasp-s datasets to rnasp-c.

    if os.path.exists(os.path.join(src, 'filtered_feature_cell_matrix.h5')):
        from exprmat.reader.matcher import read_h5_rna
        adata = read_h5_rna(
            os.path.join(src, 'filtered_feature_cell_matrix.h5'), 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    elif os.path.exists(os.path.join(src, 'filtered_feature_cell_matrix')):
        from exprmat.reader.matcher import read_mtx_rna
        adata = read_mtx_rna(
            os.path.join(src, 'filtered_feature_cell_matrix'), prefix, 
            metadata = metadata, sample = sample, raw = raw,
            default_taxa = default_taxa, eccentric = eccentric,
            suppress_filter = True,
        )
    
    else: error('failed to find feature cell matrix.')

    # attach morphology
    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    # it is also visium hd
    rows = rows[rows['modality'] == 'rnasp-s']
    assert len(rows) == 1
    props = rows.iloc[0]

    # tiff files are high quality files.
    if os.path.exists(os.path.join(src, 'spatial')):
        adata.uns['spatial'] = {
            props['sample']: read_multiscale_image_from_spatial(
                os.path.join(src, 'spatial')
            )
        }
    
    else: error('failed to find spatial folder (hd, segmented).')

    return adata


def is_xenium_explorer(src):
    return (
        os.path.exists(os.path.join(src, 'cells.zarr.zip')),
        os.path.exists(os.path.join(src, 'cell_feature_matrix.zarr.zip'))
    )


def read_xenium_explorer(
    src: str, prefix: str, 
    metadata: metadata, sample: str, raw: bool = False,
    default_taxa = 'mmu', eccentric = None
):
    
    import zarr
    rows = metadata.dataframe[metadata.dataframe['sample'] == sample]
    assert len(rows) >= 1
    rows = rows[rows['modality'] == 'rnasp-c']
    assert len(rows) == 1
    props = rows.iloc[0]

    def open_zarr(path: str) -> zarr.Group:
        store = (
            zarr.storage.ZipStore(path, mode = "r") if path.endswith(".zip")
            else zarr.storage.LocalStore(path)
        )

        return zarr.open_group(store = store, mode = "r")

    if (not os.path.exists(os.path.join(src, 'cells.zarr.zip'))):
        error('failed to find cells.zarr.zip.')

    if (not os.path.exists(os.path.join(src, 'cell_feature_matrix.zarr.zip'))):
        error('failed to find cell_feature_matrix.zarr.zip.') 
    
    cells = open_zarr(os.path.join(src, 'cells.zarr.zip'))
    mat = open_zarr(os.path.join(src, 'cell_feature_matrix.zarr.zip'))

    cell_id = cells['cell_id'][...]
    cell_meta = pd.DataFrame(cells['cell_summary'][...], columns = [
        'x', 'y', 'area', 'x.nucleus', 'y.nucleus', 'area.nucleus', 'z', 'n.nucleus'
    ])

    cell_meta['cid'] = cell_id[:, 0]
    cell_meta['cid.tag'] = cell_id[:, 1]

    def int_to_hex(num):
        if num == 0:
            return "0"
        hex_chars = "abcdefghijklmnop"
        hex_str = ""
        while num > 0:
            hex_str = hex_chars[num & 0xf] + hex_str
            num >>= 4
        return hex_str
    
    string_bc = [
        int_to_hex(x).rjust(8, 'a') + '-' + str(y) 
        for x, y in zip(cell_meta['cid'], cell_meta['cid.tag'])
    ]

    cell_meta.index = [props['sample'] + ':' + x for x in string_bc]
    
    # create anndata
    n_cells = mat['cell_features'].attrs['number_cells']
    n_features = mat['cell_features'].attrs['number_features']

    from scipy.sparse import csr_matrix
    data = csr_matrix(
        (
            mat['cell_features/data'][:],
            mat['cell_features/indices'][:],
            mat['cell_features/indptr'][:]
        ), shape = (n_features, n_cells)
    ).T

    var_table = pd.DataFrame({
        'ensembl': mat['cell_features'].attrs['feature_ids'],
        'key': mat['cell_features'].attrs['feature_keys'],
        'type': mat['cell_features'].attrs['feature_types']
    }).set_index('ensembl')

    import anndata as ad
    adata = ad.AnnData(X = data, var = var_table)
    adata.obs_names = string_bc
    # keep only genes
    adata = adata[:, adata.var['type'] == 'gene'].copy()

    from exprmat.reader.matcher import match_matrix_rna
    final = match_matrix_rna(
        adata, metadata, sample, suppress_filter = True,
        force_filter = raw, default_taxa = default_taxa
    )

    # attach cell metadata
    final.obs = final.obs.join(cell_meta, on = 'barcode', how = 'left')
    final.obsm['spatial'] = final.obs[['x', 'y']].values

    # attach morphology
    
    # if there is he image, attach it.
    if os.path.exists(os.path.join(src, 'post', 'he.ome.tif')):
        final.uns['spatial'] = {
            props['sample']: read_multiscale_image(
                os.path.join(src, 'post', 'he.ome.tif'),
                l_highres = 3, l_lowres = 5,
                base_zoom = 1
            )
        }
    
    # otherwise, read autofocus 
    elif os.path.exists(os.path.join(src, 'morphology_focus')):
        scale = cells['masks/homogeneous_transform'][...][0, 0]
        final.uns['spatial'] = {
            props['sample']: read_multiscale_image_from_autofocus(
                os.path.join(src, 'morphology_focus'),
                l_highres = 3, l_lowres = 5,
                base_zoom = scale
            )
        }

    # otherwise, read the zstack dapi image with central z.
    elif os.path.exists(os.path.join(src, 'morphology.ome.tif')):
        scale = cells['masks/homogeneous_transform'][...][0, 0]
        final.uns['spatial'] = {
            props['sample']: read_multiscale_image(
                os.path.join(src, 'morphology.ome.tif'),
                l_highres = 3, l_lowres = 5,
                base_zoom = scale
            )
        }

    final.X = final.X.astype('float32')
    return final