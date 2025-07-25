
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import grid_sample

from exprmat.ansi import error


def normalize(arr, t_min = 0, t_max = 1):
    """
    Linearly normalizes an array between two specifed values.
    """
    
    diff = t_max - t_min
    diff_arr = np.max(arr) - np.min(arr)
    min_ = np.min(arr)
    norm_arr = ((arr - min_)/diff_arr * diff) + t_min
    return norm_arr


def rasterize(
    x, y, g = np.ones(1), dx = 30.0, blur = 1.0, expand = 1.1, draw = 10000, 
    wavelet_magnitude = False, use_windowing = True
):
    '''
    Rasterize a spatial transcriptomics dataset into a density image
    
    Parameters
    -------------
    x : numpy array of length N
        x location of cells

    y : numpy array of length N
        y location of cells

    g : numpy array of length N
        RNA count of cells. If not given, density image is created

    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)

    blur : float or list of floats
        Standard deviation of Gaussian interpolation kernel.  Units are in 
        number of pixels.  Can be aUse a list to do multi scale.

    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.

    draw : int
        If True, draw a figure every draw points return its handle. Defaults to False (0).

    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
    
        
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis

    Y  : numpy array
        Locations of pixels along the y axis

    M : numpy array
        A rasterized image with len(blur) channels along the first axis

    fig : matplotlib figure handle
        If draw=True, returns a figure handle to the drawn figure.
        
    Raises
    ------    
    Exception 
        If wavelet_magnitude is set to true but blur is not sorted from greatest to least.
        
    Examples
    --------
    Rasterize a dataset at 30 micron pixel size, with three kernels.
    >>> X, Y, M, fig = tools.rasterize(x, y, dx = 30.0, blur = [2.0,1.0,0.5], draw = 10000)
    
    Rasterize a dataset at 30 micron pixel size, with three kernels, using difference between scales.
    >>> X, Y, M, fig = tools.rasterize(
    >>>     x, y, dx = 30.0, blur = [2.0,1.0,0.5], draw = 10000, 
    >>>     wavelet_magnitude = True
    >>> )
    '''
    
    # set blur to a list
    if not isinstance(blur,list):
        blur = [blur]
    nb = len(blur)
    blur = np.array(blur)
    n = len(x)
    maxblur = np.max(blur) # for windowing
    
    if wavelet_magnitude and np.any(blur != np.sort(blur)[::-1]):
        raise Exception('When using wavelet magnitude, blurs must be sorted from greatest to least')
    
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)

    X = np.stack(np.meshgrid(X_,Y_)) # note this is xy order, not row col order
    W = np.zeros((X.shape[1],X.shape[2],nb))

    if draw: fig,ax = plt.subplots()
    count = 0
    
    g = np.resize(g,x.size)
    if(not (g==1.0).all()):
        g = normalize(g)
    
    for x_,y_,g_ in zip(x,y,g):
        
        if not use_windowing: # legacy version
            k = np.exp( - ( (X[0][...,None] - x_)**2 + (X[1][...,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            k /= np.sum(k,axis=(0,1),keepdims=True)
            k *= g_
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0:
                        continue
                    k[...,i] = k[...,i] - k[...,i-1]

            W += k

        else: # use a small window

            r = int(np.ceil(maxblur*4))
            col = np.round((x_ - X_[0])/dx).astype(int)
            row = np.round((y_ - Y_[0])/dx).astype(int)
            
            row0 = np.floor(row-r).astype(int)
            row1 = np.ceil(row+r).astype(int)                    
            col0 = np.floor(col-r).astype(int)
            col1 = np.ceil(col+r).astype(int)
            # we need boundary conditions
            row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
            row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
            col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
            col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)
            
            k =  np.exp( - ( (X[0][row0:row1+1,col0:col1+1,None] - x_)**2 + (X[1][row0:row1+1,col0:col1+1,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            k /= np.sum(k,axis=(0,1),keepdims=True)  
            k *= g_
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0:
                        continue
                    k[...,i] = k[...,i] - k[...,i-1]
            W[row0:row1+1,col0:col1+1,:] += k #range of voxels -oka
            
        if draw:
            if not count%draw or count==(x.shape[0]-1):
                print(f'{count} of {x.shape[0]}')

                ax.cla()
                toshow = W-np.min(W,axis=(0,1),keepdims=True)
                toshow = toshow / np.max(toshow,axis=(0,1),keepdims=True)
                
                if nb >= 3:
                    toshow = toshow[...,:3]
                elif nb == 2:
                    toshow = toshow[...,[0,1,0]]
                elif nb == 1:
                    toshow = toshow[...,[0,0,0]]
                
                ax.imshow(np.abs(toshow))
                fig.canvas.draw()

        count += 1
    W = np.abs(W)
    # we will permute so channels are on first axis
    W = W.transpose((-1,0,1))
    extent = (X_[0],X_[-1],Y_[0],Y_[-1])
    
    # rename
    X = X_
    Y = Y_
    if draw: output = X,Y,W,fig
    else: output = X,Y,W
    return output
    

def rasterize_with_signal(
    x, y, s = None, dx = 30.0, blur = 1.0, expand = 1.1, 
    draw = 0, wavelet_magnitude = False, use_windowing = True
):
    ''' 
    Rasterize a spatial transcriptomics dataset into a density image
    
    Parameters
    ------------

    x : numpy array of length N
        x location of cells

    y : numpy array of length N
        y location of cells

    s : numpy array of length N by M for M signals
        signal value should be length NxM

    dx : float
        Pixel size to rasterize data (default 30.0, in same units as x and y)

    blur : float or list of floats
        Standard deviation of Gaussian interpolation kernel.  Units are in 
        number of pixels.  Can be aUse a list to do multi scale.

    expand : float
        Factor to expand sampled area beyond cells. Defaults to 1.1.

    draw : int
        If True, draw a figure every draw points return its handle. Defaults to False (0).

    wavelet_magnitude : bool
        If True, take the absolute value of difference between scales for raster images.
        When using this option blur should be sorted from greatest to least.
    
    Returns
    -------
    X  : numpy array
        Locations of pixels along the x axis

    Y  : numpy array
        Locations of pixels along the y axis

    M : numpy array
        A rasterized image with len(blur) channels along the last axis

    fig : matplotlib figure handle
        If draw=True, returns a figure handle to the drawn figure.
    '''
    
    # set blur to a list
    if not isinstance(blur,list):
        blur = [blur]
    nb = len(blur)
    blur = np.array(blur)
    n = len(x)
    maxblur = np.max(blur) # for windowing
    
    if len(blur)>1 and s is not None:
        raise Exception('when using a signal, we can only have one blur')
    if s is not None:
        s = np.array(s)
        if s.ndim == 1:
            s = s[...,None] # add a column of size 1
        
    if wavelet_magnitude and np.any(blur != np.sort(blur)[::-1]):
        raise Exception('When using wavelet magnitude, blurs must be sorted from greatest to least')
    
    minx = np.min(x)
    maxx = np.max(x)
    miny = np.min(y)
    maxy = np.max(y)
    minx,maxx = (minx+maxx)/2.0 - (maxx-minx)/2.0*expand, (minx+maxx)/2.0 + (maxx-minx)/2.0*expand
    miny,maxy = (miny+maxy)/2.0 - (maxy-miny)/2.0*expand, (miny+maxy)/2.0 + (maxy-miny)/2.0*expand
    X_ = np.arange(minx,maxx,dx)
    Y_ = np.arange(miny,maxy,dx)
    
    X = np.stack(np.meshgrid(X_,Y_)) # note this is xy order, not row col order
    if s is None: W = np.zeros((X.shape[1],X.shape[2],nb))
    else: W = np.zeros((X.shape[1],X.shape[2],s.shape[1]))
    
    if draw: fig, ax = plt.subplots()
    count = 0
    
    for x_,y_ in zip(x,y):

        if not use_windowing: # legacy version
            k = np.exp( - ( (X[0][...,None] - x_)**2 + (X[1][...,None] - y_)**2 )/(2.0*(dx*blur*2)**2)  )
            k /= np.sum(k,axis=(0,1),keepdims=True)*dx**2
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0: continue
                    k[...,i] = k[...,i] - k[...,i-1]
            if s is None:
                factor = 1.0
            else:
                factor = s[count]
            W += k * factor

        else: # use a small window
            r = int(np.ceil(maxblur*4))
            col = np.round((x_ - X_[0])/dx).astype(int)
            row = np.round((y_ - Y_[0])/dx).astype(int)
            
            row0 = np.floor(row-r).astype(int)
            row1 = np.ceil(row+r).astype(int)                    
            col0 = np.floor(col-r).astype(int)
            col1 = np.ceil(col+r).astype(int)
            # we need boundary conditions
            row0 = np.minimum(np.maximum(row0,0),W.shape[0]-1)
            row1 = np.minimum(np.maximum(row1,0),W.shape[0]-1)
            col0 = np.minimum(np.maximum(col0,0),W.shape[1]-1)
            col1 = np.minimum(np.maximum(col1,0),W.shape[1]-1)
            
            k = np.exp( 
                - ( 
                    (X[0][row0:row1 + 1,col0:col1 + 1, None] - x_) ** 2 + 
                    (X[1][row0:row1 + 1,col0:col1 + 1, None] - y_) ** 2 
                ) / (2.0 * (dx * blur * 2) **2 )
            )

            k /= np.sum(k,axis=(0,1),keepdims=True)*dx**2
            if wavelet_magnitude:
                for i in reversed(range(nb)):
                    if i == 0: continue
                    k[...,i] = k[...,i] - k[...,i-1]
            if s is None: factor = 1.0
            else: factor = s[count]
            W[row0:row1+1,col0:col1+1,:] += k * factor
            
        if draw:
            if not count%draw or count==(x.shape[0]-1):
                print(f'{count} of {x.shape[0]}')

                ax.cla()
                toshow = W-np.min(W,axis=(0,1),keepdims=True)
                toshow = toshow / np.max(toshow,axis=(0,1),keepdims=True)
                
                if nb >= 3: toshow = toshow[...,:3]
                elif nb == 2: toshow = toshow[...,[0,1,0]]
                elif nb == 1: toshow = toshow[...,[0,0,0]]
                
                ax.imshow(np.abs(toshow))
                fig.canvas.draw()

        count += 1

    W = np.abs(W)
    # we will permute so channels are on first axis
    W = W.transpose((-1,0,1))
    extent = (X_[0],X_[-1],Y_[0],Y_[-1])
    
    # rename
    X = X_
    Y = Y_
    if draw: output = X,Y,W,fig
    else: output = X,Y,W
    return output

    
def interp(x, I, phii, **kwargs):
    '''
    Interpolate the 2D image I, with regular grid positions stored in x (1d arrays),
    at the positions stored in phii (3D arrays with first channel storing component)    
    
    Parameters
    ----------
    x : list of arrays
        List of arrays storing the pixel locations along each image axis. 
        convention is row column order not xy.

    I : array
        Image array. First axis should contain different channels.

    phii : array
        Sampling array. First axis should contain sample locations corresponding to each axis.

    **kwargs : dict
        Other arguments fed into the torch interpolation function torch.nn.grid_sample
        
    
    Returns
    -------
    out : torch tensor
            The image I resampled on the points defined in phii.
    
    Notes
    -----
    Convention is to use align_corners = True.
    This uses the torch library.
    '''
    
    # first we have to normalize phii to the range -1,1    
    I = torch.as_tensor(I)
    phii = torch.as_tensor(phii)
    phii = torch.clone(phii)
    for i in range(2):
        phii[i] -= x[i][0]
        phii[i] /= x[i][-1] - x[i][0]
    # note the above maps to 0,1
    phii *= 2.0
    # to 0 2
    phii -= 1.0
    # done

    # NOTE I should check that I can reproduce identity 
    # note that phii must now store x,y,z along last axis
    # is this the right order?
    # I need to put batch (none) along first axis
    # what order do the other 3 need to be in?    
    out = grid_sample(I[None], phii.flip(0).permute((1,2,0))[None], align_corners = True, **kwargs)
    # note align corners true means square voxels with points at their centers
    # post processing, get rid of batch dimension

    return out[0]


def clip(I):
    ''' Clip an arrays values between 0 and 1. '''
    Ic = torch.clone(I)
    Ic[Ic < 0] = 0
    Ic[Ic > 1] = 1
    return Ic


def v_to_phii(xv,v):
    ''' 
    Integrate a velocity field over time to return a position field (diffeomorphism).
    
    Parameters
    ----------
    xv : list of torch tensor 
        List of 1D tensors describing locations of sample points in v

    v : torch tensor
        5D (nt,2,v0,v1) velocity field
    
    Returns
    -------
    phii: torch tensor
        Inverse map (position field) computed by method of characteristics
    '''
    
    XV = torch.stack(torch.meshgrid(xv))
    phii = torch.clone(XV)
    dt = 1.0 / v.shape[0]
    for t in range(v.shape[0]):
        Xs = XV - v[t] * dt
        phii = interp(xv, phii - XV, Xs) + Xs
    return phii


def to_a(L,T):
    ''' 
    Convert a linear transform matrix and a translation vector into an affine matrix.
    
    Parameters
    ----------
    L : torch tensor
        2x2 linear transform matrix
        
    T : torch tensor
        2 element translation vector (note NOT 2x1)
        
    Returns
    -------
    
    A : torch tensor
        Affine transform matrix
    '''

    O = torch.tensor([0.,0.,1.], device = L.device, dtype = L.dtype)
    A = torch.cat((torch.cat((L, T[:, None]), 1), O[None]))
    return A


def extent_from_x(xJ):
    ''' 
    Given a set of pixel locations, returns an extent 4-tuple for use with np.imshow.
    Note inputs are locations of pixels along each axis, i.e. row column not xy.
    
    Parameters
    ----------
    xJ : list of torch tensors
        Location of pixels along each axis
    
    Returns
    -------
    extent : tuple
        (xmin, xmax, ymin, ymax) tuple
    
    Examples
    --------
    
    >>> extent_from_x(xJ)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(J, extent = extentJ)
    '''

    dJ = [x[1]-x[0] for x in xJ]
    extentJ = ( 
        (xJ[1][0] - dJ[1]/2.0).item(),
        (xJ[1][-1] + dJ[1]/2.0).item(),
        (xJ[0][-1] + dJ[0]/2.0).item(),
        (xJ[0][0] - dJ[0]/2.0).item()
    )

    return extentJ


def affine_transform_from_points(pointsI, pointsJ):
    '''
    Compute an affine transformation from points.
    Note for an affine transformation (6dof) we need 3 points.
    Outputs, L, T should be rconstructed blockwize like [L, T; 0, 0, 1]
    
    Returns
    -------
    L : array
        A 2-by-2 linear transform array.
    T : array
        A 2 element translation vector
    '''

    if pointsI is None or pointsJ is None:
        error('points are set to none')
        
    nI = pointsI.shape[0]
    nJ = pointsJ.shape[0]
    if nI != nJ: error(f'number of points_i ({nI}) is not equal to number of points_j ({nJ})')
    if pointsI.shape[1] != 2:
        error(f'number of components of points_i ({pointsI.shape[1]}) should be 2')
    if pointsJ.shape[1] != 2:
        error(f'number of components of points_j ({pointsJ.shape[1]}) should be 2')
    
    # transformation model
    if nI < 3:
        # translation only 
        L = np.eye(2)
        T = np.mean(pointsJ,0) - np.mean(pointsI,0)
    
    else:
        # we need an affine transform
        pointsI_ = np.concatenate((pointsI,np.ones((nI,1))),1)
        pointsJ_ = np.concatenate((pointsJ,np.ones((nI,1))),1)
        II = pointsI_.T @ pointsI_
        IJ = pointsI_.T @ pointsJ_
        A = (np.linalg.inv(II) @ IJ).T        
        L = A[:2,:2]
        T = A[:2,-1]

    return L, T


def lddmm(
    xI, I, xJ, J,
    points_i = None, points_j = None,
    L = None, T = None, A = None,
    velocity = None, x_velocity = None,
    a = 500.0, p = 2.0, expand = 2.0, nt = 3,
    niter = 5000, diffeo_start = 0, eps_L = 2e-8, eps_T = 2e-1, eps_V = 2e3,
    sigma_M = 1.0, sigma_B = 2.0, sigma_A = 5.0, sigma_R = 5e5, sigma_P = 2e1,
    device = 'cpu', dtype = torch.float64, mu_B = None, mu_A = None
):
    
    '''
    Run LDDMM between a pair of images.
    
    This jointly estimates an affine transform A, and a diffeomorphism phi.
    The map is off the form x -> A phi x
    
    Parameters
    ----------
    xI : list of torch tensor
        Location of voxels in source image I

    I : torch tensor
        Source image I, with channels along first axis  

    xJ : list of torch tensor
        Location of voxels in target image J

    J : torch tensor
        Target image J, with channels along first axis

    L : torch tensor
        Initial guess for linear transform (2x2 torch tensor). Defaults to None (identity).

    T : torch tensor
        Initial guess for translation (2 element torch tensor). Defaults to None (identity)

    A : torch tensor
        Initial guess for affine matrix.  Either L and T can be specified, or A, but not both.
        Defaults to None (identity).

    v : torch tensor
        Initial guess for velocity field

    xv : torch tensor
        Pixel locations for velocity field

    a : float
        Smoothness scale of velocity field (default 500.0)

    p : float
        Power of Laplacian in velocity regularization (default 2.0)

    expand : float
        Factor to expand size of velocity field around image boundaries (default 2.0)

    nt : int
        Number of timesteps for integrating velocity field (default 3). 
        Ignored if you input v.

    points_i : torch tensor
        N x 2 set of corresponding points for matching in source image. 
        Default None (no points).

    points_j : torch tensor
        N x 2 set of corresponding points for matching in target image. 
        Default None (no points).

    niter : int
        Number of iterations of gradient descent optimization

    diffeo_start : int
        Number of iterations of gradient descent optimization for affine only, 
        before nonlinear deformation.

    eps_L : float
        Gradient descent step size for linear part of affine.

    eps_T : float
        Gradient descent step size of translation part of affine.

    eps_V : float
        Gradient descent step size for velocity field.

    sigma_M : float
        Standard deviation of image matching term for Gaussian mixture modeling in cost function. 
        This term generally controls matching accuracy with smaller corresponding to more accurate.
        As an common example (rule of thumb), you could chose this parameter to be the variance 
        of the pixels in your target image.

    sigma_B : float
        Standard deviation of backtround term for Gaussian mixture modeling in cost function. 
        If there is missing tissue in target, we may label some pixels in target as background,
        and not enforce matching here.

    sigma_A : float
        Standard deviation of artifact term for Gaussian mixture modeling in cost function. 
        If there are artifacts in target or other lack of corresponding between template and target, 
        we may label some pixels in target as artifact, and not enforce matching here.

    sigma_R: float
        Standard deviation for regularization. Smaller sigmaR means a smoother resulting transformation. 
        Regularization is of the form: 0.5/sigmaR^2 int_0^1 int_X |Lv|^2 dx dt. 

    sigma_P: float
        Standard deviation for matching of points.  
        Cost is of the form 0.5/sigmaP^2 sum_i (source_point_i - target_point_i)^2

    device: str
        Torch device. defaults to 'cpu'. Can also be 'cuda:0' for example.

    dtype: torch dtype
        Torch data type. defaults to torch.float64

    mu_A: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, we will not estimate it.
        If the target is a RGB image, this should be a tensor of size 3.
        If the target is a grayscale image, this should be a tensor of size 1.

    mu_B: torch tensor whose dimension is the same as the target image
        Defaults to None, which means we estimate this. If you provide a value, 
        we will not estimate it.
        
    Returns
    -------
    'A': torch tensor
        Affine transform

    'v': torch tensor
        Velocity field

    'xv': list of torch tensor
        Pixel locations in v

    'WM': torch tensor
        Resulting weight 2D array (matching)

    'WB': torch tensor
        Resulting weight 2D array (background)

    'WA': torch tensor
        Resulting weight 2D array (artifact)
    '''
    
    # check initial inputs
    if A is not None:
        # if we specify an A
        if L is not None or T is not None:
            raise Exception('If specifying A, you must not specify L or T')
        L = torch.tensor(A[:2,:2],device=device,dtype=dtype,requires_grad=True)
        T = torch.tensor(A[:2,-1],device=device,dtype=dtype,requires_grad=True)   
    else:
        # if we do not specify A                
        if L is None: L = torch.eye(2,device=device,dtype=dtype,requires_grad=True)
        if T is None: T = torch.zeros(2,device=device,dtype=dtype,requires_grad=True)

    L = torch.tensor(L,device=device,dtype=dtype,requires_grad=True)
    T = torch.tensor(T,device=device,dtype=dtype,requires_grad=True)
    # change to torch
    I = torch.tensor(I,device=device,dtype=dtype)                         
    J = torch.tensor(J,device=device,dtype=dtype)
    
    # velocity
    
    if velocity is not None and x_velocity is not None:
        velocity = torch.tensor(velocity,device=device,dtype=dtype,requires_grad=True)
        x_velocity = [torch.tensor(x,device=device,dtype=dtype) for x in x_velocity]
        XV = torch.stack(torch.meshgrid(x_velocity),-1)
        nt = velocity.shape[0]        
    elif velocity is None and x_velocity is None:
        minv = torch.as_tensor([x[0] for x in xI],device=device,dtype=dtype)
        maxv = torch.as_tensor([x[-1] for x in xI],device=device,dtype=dtype)
        minv,maxv = (minv+maxv)*0.5 + 0.5*torch.tensor([-1.0,1.0],device=device,dtype=dtype)[...,None]*(maxv-minv)*expand
        x_velocity = [torch.arange(m,M,a*0.5,device=device,dtype=dtype) for m,M in zip(minv,maxv)]
        XV = torch.stack(torch.meshgrid(x_velocity),-1)
        velocity = torch.zeros((nt,XV.shape[0],XV.shape[1],XV.shape[2]),device=device,dtype=dtype,requires_grad=True)
    else:
        raise Exception(f'If inputting an initial v, must input both xv and v')
    extentV = extent_from_x(x_velocity)
    dv = torch.as_tensor([x[1]-x[0] for x in x_velocity],device=device,dtype=dtype)
    
    fv = [torch.arange(n,device=device,dtype=dtype)/n/d for n,d in zip(XV.shape,dv)]
    extentF = extent_from_x(fv)
    FV = torch.stack(torch.meshgrid(fv),-1)
    LL = (1.0 + 2.0*a**2* torch.sum( (1.0 - torch.cos(2.0*np.pi*FV*dv))/dv**2 ,-1))**(p*2.0)

    K = 1.0/LL
    
    DV = torch.prod(dv)
    Ki = torch.fft.ifftn(K).real

    # nt = 3

    WM = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.5
    WB = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.4
    WA = torch.ones(J[0].shape,dtype=J.dtype,device=J.device)*0.1
    if points_i is None and points_j is None:
        points_i = torch.zeros((0,2),device=J.device,dtype=J.dtype)
        points_j = torch.zeros((0,2),device=J.device,dtype=J.dtype) 
    elif (points_i is None and points_j is not None) or (points_j is None and points_i is not None):
        raise Exception('Must specify corresponding sets of points or none at all')
    else:
        points_i = torch.tensor(points_i,device=J.device,dtype=J.dtype)
        points_j = torch.tensor(points_j,device=J.device,dtype=J.dtype)
    
    
    xI = [torch.tensor(x,device=device,dtype=dtype) for x in xI]
    xJ = [torch.tensor(x,device=device,dtype=dtype) for x in xJ]
    XI = torch.stack(torch.meshgrid(*xI,indexing='ij'),-1)
    XJ = torch.stack(torch.meshgrid(*xJ,indexing='ij'),-1)
    dJ = [x[1]-x[0] for x in xJ]
    extentJ = (xJ[1][0].item()-dJ[1].item()/2.0,
          xJ[1][-1].item()+dJ[1].item()/2.0,
          xJ[0][-1].item()+dJ[0].item()/2.0,
          xJ[0][0].item()-dJ[0].item()/2.0)
    
    # sigma_M = 0.2
    # sigma_B = 0.19
    # sigma_A = 0.3
    # sigma_R = 5e5
    # sigma_P = 2e-1
    
    if mu_A is None: estimate_muA = True
    else: estimate_muA = False
    if mu_B is None: estimate_muB = True
    else: estimate_muB = False
    
    try: L.grad.zero_()
    except: pass

    try: T.grad.zero_()
    except: pass

    for it in range(niter):
        # make A
        A = to_a(L,T)
        # Ai
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:2,:2]@XJ[...,None])[...,0] + Ai[:2,-1]    
        # now diffeo, not semilagrange here
        for t in range(nt-1,-1,-1):
            Xs = Xs + interp(x_velocity,-velocity[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        # and points
        pointsIt = torch.clone(points_i)
        if pointsIt.shape[0] >0:
            for t in range(nt):            
                pointsIt += interp(x_velocity,velocity[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
            pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T

        # transform image
        AI = interp(xI,I,Xs.permute(2,0,1),padding_mode="border")

        # transform the contrast
        B = torch.ones(1+AI.shape[0],AI.shape[1]*AI.shape[2],device=AI.device,dtype=AI.dtype)
        B[1:AI.shape[0]+1] = AI.reshape(AI.shape[0],-1)
        with torch.no_grad():    
            BB = B@(B*WM.ravel()).T
            BJ = B@((J*WM).reshape(J.shape[0],J.shape[1]*J.shape[2])).T
            small = 0.1
            coeffs = torch.linalg.solve(BB + small*torch.eye(BB.shape[0],device=BB.device,dtype=BB.dtype),BJ)
        fAI = ((B.T@coeffs).T).reshape(J.shape)

        # objective function
        EM = torch.sum((fAI - J)**2*WM)/2.0/sigma_M**2
        ER = torch.sum(torch.sum(torch.abs(torch.fft.fftn(velocity,dim=(1,2)))**2,dim=(0,-1))*LL)*DV/2.0/velocity.shape[1]/velocity.shape[2]/sigma_R**2
        E = EM + ER
        tosave = [E.item(), EM.item(), ER.item()]
        if pointsIt.shape[0]>0:
            EP = torch.sum((pointsIt - points_j)**2)/2.0/sigma_P**2
            E += EP
            tosave.append(EP.item())
        
        # gradient update
        E.backward()
        with torch.no_grad():            
            L -= (eps_L/(1.0 + (it>=diffeo_start)*9))*L.grad
            T -= (eps_T/(1.0 + (it>=diffeo_start)*9))*T.grad

            L.grad.zero_()
            T.grad.zero_()
            
            # v grad
            vgrad = velocity.grad
            # smooth it
            vgrad = torch.fft.ifftn(torch.fft.fftn(vgrad,dim=(1,2))*K[...,None],dim=(1,2)).real
            if it >= diffeo_start:
                velocity -= vgrad*eps_V
            velocity.grad.zero_()

        # update weights
        if not it % 5:
            with torch.no_grad():
                # M step for these params
                if estimate_muA: mu_A = torch.sum(WA*J,dim=(-1,-2))/torch.sum(WA)
                if estimate_muB: mu_B = torch.sum(WB*J,dim=(-1,-2))/torch.sum(WB)
                
                if it >= 50:

                    W = torch.stack((WM,WA,WB))
                    pi = torch.sum(W,dim=(1,2))
                    pi += torch.max(pi)*1e-6
                    pi /= torch.sum(pi)

                    # now the E step, update the weights
                    WM = pi[0]* torch.exp( -torch.sum((fAI - J)**2,0)/2.0/sigma_M**2 )/np.sqrt(2.0*np.pi*sigma_M**2)**J.shape[0]
                    WA = pi[1]* torch.exp( -torch.sum((mu_A[...,None,None] - J)**2,0)/2.0/sigma_A**2 )/np.sqrt(2.0*np.pi*sigma_A**2)**J.shape[0]
                    WB = pi[2]* torch.exp( -torch.sum((mu_B[...,None,None] - J)**2,0)/2.0/sigma_B**2 )/np.sqrt(2.0*np.pi*sigma_B**2)**J.shape[0]
                    WS = WM+WB+WA
                    WS += torch.max(WS)*1e-6
                    WM /= WS
                    WB /= WS
                    WA /= WS

            
    return {
        'A': A.clone().detach(), 
        'v': velocity.clone().detach(), 
        'xv': x_velocity, 
        'WM': WM.clone().detach(),
        'WB': WB.clone().detach(),
        'WA': WA.clone().detach()
    }


def build_transform(xv, v, A, direction='b', XJ = None):
    ''' 
    Create sample points to transform source to target from affine and velocity.
    
    Parameters
    ----------
    xv : list of array
        Sample points for velocity

    v : array
        time dependent velocity field

    A : array
        Affine transformation matrix

    direction : char
        'f' for forward and 'b' for backward. 
        'b' is default and is used for transforming images.
        'f' is used for transforming points.

    XJ : array
        Sample points for target (meshgrid with ij index style).  
        Defaults to None to keep sampling on the xv.
    
    Returns
    -------
    Xs : array
        Sample points in mehsgrid format.
    '''
    
    A = torch.tensor(A)
    if v is not None: v = torch.tensor(v) 
    if XJ is not None:

        # check some types here
        if isinstance(XJ,list):
            if XJ[0].ndim == 1: # need meshgrid
                XJ = torch.stack(torch.meshgrid([torch.tensor(x) for x in XJ],indexing='ij'), -1)
            elif XJ[0].ndim == 2: # assume already meshgrid
                XJ = torch.stack([torch.tensor(x) for x in XJ],-1)
            else: error('could not understand variable XJ type')
            
        # if it is already in meshgrid form we just need to make sure it is a tensor
        XJ = torch.tensor(XJ)
    else: XJ = torch.stack(torch.meshgrid([torch.tensor(x) for x in xv],indexing='ij'),-1)
        
    if direction == 'b':
        Ai = torch.linalg.inv(A)
        # transform sample points
        Xs = (Ai[:-1,:-1]@XJ[...,None])[...,0] + Ai[:-1,-1]    
        # now diffeo, not semilagrange here
        if v is not None:
            nt = v.shape[0]
            for t in range(nt-1,-1,-1):
                Xs = Xs + interp(xv,-v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
    elif direction == 'f':
        Xs = torch.clone(XJ)
        if v is not None:
            nt = v.shape[0]
            for t in range(nt):
                Xs = Xs + interp(xv,v[t].permute(2,0,1),Xs.permute(2,0,1)).permute(1,2,0)/nt
        Xs = (A[:2,:2]@Xs[...,None])[...,0] + A[:2,-1]    
            
    else: error(f'direction must be one of: "f" or "b".')
    return Xs 


def transform_image_source_with_A(A, XI, I, XJ):
    '''
    Transform an image with an affine matrix
    
    Parameters
    ----------
    
    A  : torch tensor
         Affine transform matrix
        
    XI : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)  
    
    I  : numpy array
         A rasterized image with len(blur) channels along the first axis
        
    XJ : list of numpy arrays
         List of arrays storing the pixel location in image I along each image axis. 
         convention is row column order not xy. i.e, 
         locations of pixels along the y axis (rows) followed by
         locations of pixels along the x axis (columns)         
    
    Returns
    -------
    AI : torch tensor
        image I after affine transformation A, with channels along first axis
              
    '''
    xv = None
    v = None
    AI = transform_image_source_to_target(xv, v, A, XI, I, XJ = XJ)
    return AI


def transform_image_source_to_target(xv, v, A, xI, I, XJ = None):

    phii = build_transform(xv,v,A,direction='b',XJ=XJ)    
    phiI = interp(xI,I,phii.permute(2,0,1),padding_mode="border")
    return phiI
    
    
def transform_image_target_to_source(xv,v,A,xJ,J,XI = None):
    
    phi = build_transform(xv,v,A,direction='f',XJ=XI)    
    phiiJ = interp(xJ,J,phi.permute(2,0,1),padding_mode="border")
    return phiiJ
    

def transform_points_source_to_target(xv,v,A,pointsI):
    
    if isinstance(pointsI,torch.Tensor):
        pointsIt = torch.clone(pointsI)
    else: pointsIt = torch.tensor(pointsI)
    nt = v.shape[0]
    for t in range(nt):            
        pointsIt += interp(xv,v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
    pointsIt = (A[:2,:2]@pointsIt.T + A[:2,-1][...,None]).T
    return pointsIt


def transform_points_target_to_source(xv,v,A,pointsI):
    
    if isinstance(pointsI,torch.Tensor):
        pointsIt = torch.clone(pointsI)
    else:
        pointsIt = torch.tensor(pointsI)
    Ai = torch.linalg.inv(A)
    pointsIt = (Ai[:2,:2]@pointsIt.T + Ai[:2,-1][...,None]).T
    nt = v.shape[0]
    for t in range(nt):            
        pointsIt += interp(xv,-v[t].permute(2,0,1),pointsIt.T[...,None])[...,0].T/nt
    return pointsIt


def calculate_tre(pointsI, pointsJ):
    TRE_i = np.sqrt(np.sum((pointsI - pointsJ)**2,axis=1))
    meanTRE = np.mean(TRE_i)
    stdTRE = np.std(TRE_i)
    return meanTRE, stdTRE

