
from setuptools import setup, find_packages

setup(
    name                 = 'exprmat',
    version              = '0.1.15',
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
        'gseapy',
        'metacells',
        'sh',
        'networkx',
        'palantir',
        'scfates',
        'rich'
    ],
    include_package_data = False,
    package_data         = {
        'exprmat.data': [
            'mmu/*', 
            'mmu/genesets/*',
            'mmu/orthologs/*',
            'mmu/lr/*',
            'hsa/*',
            'hsa/genesets/*',
            'hsa/orthologs/*',
            'hsa/lr/*',
        ]
    },
    zip_safe             = False
)
