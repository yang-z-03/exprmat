
import mudata as mu
import anndata as ad

from exprmat.ansi import info, error, format_file_size
mu.set_options(pull_on_update = False)


MAJOR = 0
MINOR = 1
REVISION = 19
PATCH = 1


def version(): 
    import os
    import sys
    import platform
    info(f'exprmat {MAJOR}.{MINOR}.{REVISION} (patch {PATCH})')
    info(f'os: {os.name} ({sys.platform})  platform version: {platform.release()}')
    info(f'current working directory: {os.getcwd()}')
    memory()
    return (MAJOR, MINOR, REVISION, PATCH)


def memory():
    import psutil
    meminfo = psutil.Process().memory_info()
    resident = meminfo.rss
    virtual = meminfo.vms
    info(f'resident memory: {format_file_size(resident)}')
    info(f'virtual memory: {format_file_size(virtual)}')

