'''
Normalization in normal distribution space.
'''

import numpy as np


def log1p(E):
    '''
    Log transformation that keeps signs.
    '''
    return np.log1p(E)

