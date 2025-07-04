
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import importlib.metadata
import mudata as mu
import anndata as ad
import scanpy as sc

# core method exports
from exprmat.utils import setup_styles
from exprmat.reader.experiment import experiment, load_experiment
from exprmat.reader.metadata import metadata, load_metadata
import exprmat.configuration as config
from exprmat.ansi import warning

mu.set_options(pull_on_update = False)
setup_styles()

DATABASE_SETUP_MESSAGE = """
The exprmat package do not come with a included database after version 0.1.25, 
due to the extensive size of the reference data. You need to setup the database
and configure it properly with configuration file `exprmat.config` under your
current working directory.

For more guide on how to setup the database folder, and more information about
what the database includes, see <https://github.com/yang-z-03/exprmat>
"""

# load configuration
PERFORM_DATABASE_CHECK = True
if os.path.exists('exprmat.config'):
    with open('exprmat.config', 'r') as f:
        import json
        workspace_config = json.load(f)
        config.default.update(workspace_config)


# perform database integrity check.
if PERFORM_DATABASE_CHECK:
    dbpath = config.default['data']
    version_file = os.path.join(dbpath, '.bsync', 'current')
    db_setup = True
    
    if not os.path.exists(version_file): db_setup = False
    else:
        with open(version_file, 'r') as vf:
            db_ver = vf.read().strip()
        if db_ver != importlib.metadata.version("exprmat"):
            pkg_ver = importlib.metadata.version("exprmat")
            warning('the exprmat database version ({db_ver}) is not compatible with package version ({pkg_ver})')
            db_setup = False
    
    if not db_setup: 
        warning('the database is not set.')
        print(DATABASE_SETUP_MESSAGE)


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
    info(f'current database directory: {config.default["data"]}')
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
    'experiment',
    'metadata',
    'load_experiment',
    'load_metadata',
    'version',
    'memory',
    'setwd',
    'getwd'
]