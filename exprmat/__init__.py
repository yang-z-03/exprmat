
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import importlib.metadata
import mudata as mu
import anndata as ad
import scanpy as sc

# core method exports
from exprmat.utils import setup_styles, plotting_styles
from exprmat.reader.experiment import experiment, load_experiment
from exprmat.reader.metadata import metadata, load_metadata
import exprmat.configuration as config

mu.set_options(pull_on_update = False)
setup_styles(**plotting_styles)


def version(): 
    import os
    import sys
    import platform
    from exprmat.ansi import info, error, format_file_size

    ver_string = importlib.metadata.version("exprmat")
    MAJOR, MINOR, REVISION = [int(x) for x in ver_string.split('.')]
    info(f'exprmat {MAJOR}.{MINOR}.{REVISION}')
    info(f'os: {os.name} ({sys.platform})  platform version: {platform.release()}')
    info(f'current working directory: {os.getcwd()}')
    info(f'current database directory: {config.default['data']}')
    memory()
    return (MAJOR, MINOR, REVISION)


def memory():
    import psutil
    from exprmat.ansi import info, error, format_file_size
    meminfo = psutil.Process().memory_info()
    resident = meminfo.rss
    virtual = meminfo.vms
    info(f'resident memory: {format_file_size(resident)}')
    info(f'virtual memory: {format_file_size(virtual)}')


setwd = os.chdir
getwd = os.getcwd
def locate_data(dir): config.default.update_config('data', dir)


__all__ = [
    'setup_styles',
    'plotting_styles',
    'experiment',
    'metadata',
    'load_experiment',
    'load_metadata',
    'version',
    'memory',
    'setwd',
    'getwd'
]