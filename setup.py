
from setuptools import setup, find_packages

setup(
    name                 = 'exprmat',
    version              = '0.1.3',
    description          = 'routines to process expression matrices',
    url                  = 'https://github.com/yang-z-03/exprmat',
    author               = 'Zheng Yang',
    author_email         = 'xornent@outlook.com',
    license              = 'GPLv3',
    packages             = find_packages(),
    install_requires     = [
        'umap-learn >= 0.5.0',
        'scvi-tools >= 1.3.0',
        'scanpy',
        'anndata',
        'mudata',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'torch >= 2.0',
        'annoy',
        'scikit-image',
        'scikit-misc',
        'igraph',
        'bbknn',
        'scanorama',
        'harmonypy',
        'pymde',
        'datashader',
        'fa2_modified',
        'pynndescent',
        'gseapy'
    ],
    include_package_data = True,
    package_data         = {
        'exprmat.data': ['mmu/*.gz', 'hsa/*.gz']
    },
    zip_safe             = False
)
