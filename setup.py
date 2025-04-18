
from setuptools import setup

setup(
    name                 = 'exprmat',
    version              = '0.1.0',
    description          = 'routines to process expression matrices',
    author               = 'Zheng Yang',
    author_email         = 'xornent@outlook.com',
    license              = 'GPLv3',
    packages             = ['exprmat'],
    install_requires     = [
        'umap-learn >= 0.5.0',
        'scvi-tools >= 1.3.0',
        'scanpy',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'torch >= 2.0'
    ],
    include_package_data = True,
    zip_safe             = False
)