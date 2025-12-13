
import numpy as np
from shapely import Point, Polygon


def polygon_gate(
    adata, gene_x, gene_y, 
    layer = 'X',
    
    remove_zero_expression = False,
    scale = 'asis', arcsinh_divider = None,

    polygon = [(0, 0), (1, 0), (1, 1), (0, 1)],
    key_added = 'gate'
):
    
    import pandas as pd
    import seaborn as sb
    from scipy.sparse import issparse
    from exprmat.plotting import palettes

    gx = None
    gy = None
    
    # assign gene X.
    from exprmat.utils import find_variable as find_var
    gx = find_var(adata, gene_name = gene_x, layer = layer)
    gy = find_var(adata, gene_name = gene_y, layer = layer)

    df = pd.DataFrame({
        'id': adata.obs_names.tolist(),
        'x': gx,
        'y': gy
    })

    if remove_zero_expression:
        df = df.loc[(df['x'] > 0) & (df['y'] > 0), :].copy()

    if scale == 'asis': pass
    elif scale == 'log':
        df['x'] = np.log1p(df['x'])
        df['y'] = np.log1p(df['y'])
    elif scale == 'expm1':
        df['x'] = np.expm1(df['x'])
        df['y'] = np.expm1(df['y'])
    elif scale == 'arcsinh':
        lx = np.expm1(df['x']).loc[df['x'] > 0]
        ly = np.expm1(df['y']).loc[df['x'] > 0]
        df['x'] = np.arcsinh(np.expm1(df['x']) / (
            arcsinh_divider if arcsinh_divider is not None 
            else (np.median(lx) / 20)))
        df['y'] = np.arcsinh(np.expm1(df['y']) / (
            arcsinh_divider if arcsinh_divider is not None 
            else (np.median(ly) / 20)))

    poly = Polygon(polygon)
    df['gate'] = [
        poly.contains(Point(x, y))
        for x, y in zip(df['x'], df['y'])
    ]

    df.index = df['id'].tolist()
    adata.obs[key_added] = False
    adata.obs.loc[df.index, key_added] = df['gate']
