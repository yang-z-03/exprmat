
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['GOTO_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import importlib.metadata
import mudata as mu
import anndata as ad
import scanpy as sc
import pandas as pd
import pathlib

from exprmat.configuration import default as config
from exprmat.ansi import warning, info

mu.set_options(pull_on_update = False)


DATABASE_SETUP_MESSAGE = """
The exprmat package do not come with a included database after version 0.1.25, 
due to the extensive size of the reference data. You need to setup the database
and configure it properly with configuration file `.exprmatrc` under your
user root or current working directory

For more guide on how to setup the database folder, and more information about
what the database includes, see <https://github.com/yang-z-03/exprmat>
"""

# load configuration
PERFORM_DATABASE_CHECK = True
default_finders = [
    'exprmat.config',
    '.exprmat.config',
    '.exprmatrc',
    os.path.join(str(pathlib.Path.home()), 'exprmat.config'),
    os.path.join(str(pathlib.Path.home()), '.exprmat.config'),
    os.path.join(str(pathlib.Path.home()), '.exprmatrc')
]

for finder in default_finders:
    if os.path.exists(finder):
        info(f'load configuration from {finder}')
        with open(finder, 'r') as f:
            import json
            workspace_config = json.load(f)
            config.update(workspace_config)
            break

basepath = config['data']

# core method exports
from exprmat.utils import setup_styles
from exprmat.reader.experiment import experiment, load_experiment
from exprmat.reader.metadata import metadata, load_metadata

setup_styles()


def version_db():
    dbpath = config['data']
    version_file = os.path.join(dbpath, '.bsync', 'current')

    if not os.path.exists(version_file): 
        db_ver = None
    else:
        with open(version_file, 'r') as vf:
            db_ver = vf.read().strip()
    
    return db_ver


# perform database integrity check.
if PERFORM_DATABASE_CHECK:
    db_ver = version_db()
    if db_ver is None: 
        warning('the database is not installed.')
        print(DATABASE_SETUP_MESSAGE)
    elif db_ver != importlib.metadata.version("exprmat"):
        warning('the database version do not match the package version')
        warning(f'db version: {db_ver}  package version: {importlib.metadata.version("exprmat")}')


def version(): 
    import os
    import sys
    import platform
    from exprmat.ansi import info, error, format_file_size

    ver_string = importlib.metadata.version("exprmat")
    db_ver = version_db()

    MAJOR, MINOR, REVISION = [int(x) for x in ver_string.split('.')]
    info(f'exprmat {MAJOR}.{MINOR}.{REVISION} / exprmat-db {db_ver if db_ver is not None else "(Not installed)"}')
    info(f'os: {os.name} ({sys.platform})  platform version: {platform.release()}')
    info(f'current working directory: {os.getcwd()}')
    info(f'current database directory: {config["data"]} ({db_ver if db_ver is not None else "(Not found)"})')
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
def locate_data(dir): config.update_config('data', dir)


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