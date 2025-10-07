
import re
import os
import sys

from functools import partial
from tqdm import tqdm
from tqdm.auto import tqdm as tqdma

SILENT = False


def fore_black() -> None:
    print('\033[30m', end = '')

def fore_red() -> None:
    print('\033[31m', end = '')

def fore_green() -> None:
    print('\033[32m', end = '')

def fore_yellow() -> None:
    print('\033[33m', end = '')

def fore_blue() -> None:
    print('\033[34m', end = '')

def fore_purple() -> None:
    print('\033[35m', end = '')

def fore_cyan() -> None:
    print('\033[36m', end = '')

def fore_white() -> None:
    print('\033[37m', end = '')

def fore_gray() -> None:
    print('\033[8m', end = '')


def ansi_reset() -> None:
    print('\033[0m', end = '')


def back_black() -> None:
    print('\033[40m', end = '')

def back_red() -> None:
    print('\033[41m', end = '')

def back_green() -> None:
    print('\033[42m', end = '')

def back_yellow() -> None:
    print('\033[43m', end = '')

def back_blue() -> None:
    print('\033[44m', end = '')

def back_purple() -> None:
    print('\033[45m', end = '')

def back_cyan() -> None:
    print('\033[46m', end = '')

def back_white() -> None:
    print('\033[47m', end = '')


def ansi_move_cursor(line: int, col: int) -> None:
    if line == 0: pass
    elif line > 0: print('\033[{0}B'.format(str(line)), end = '', flush = True) # moves down
    elif line < 0: print('\033[{0}A'.format(str(-line)), end = '', flush = True) # moves up

    if col == 0: pass
    elif col > 0: print('\033[{0}C'.format(str(col)), end = '', flush = True) # moves right
    elif col < 0: print('\033[{0}D'.format(str(-col)), end = '', flush = True) # moves left


def common_length(string: str, limit: int) -> str:

    cn_re = '[\u201c-\u201d\u3001-\u3011\uff08-\uff1f\u4e00-\u9fa5]'
    cn = re.findall(cn_re, string)
    cn_length = len(cn)
    
    if limit < 5:
        x_str = string[-limit:]
        cn = len(re.findall(cn_re, x_str))
        while cn + len(x_str) > limit:
            x_str = x_str[1:]
            cn = len(re.findall(cn_re, x_str))
        return x_str
    
    elif len(string) + cn_length > limit:
        x_str = string[-(limit - 3):]
        cn = len(re.findall(cn_re, x_str))
        while cn + len(x_str) > limit - 3:
            x_str = x_str[1:]
            cn = len(re.findall(cn_re, x_str))
        return '.. ' + x_str
        
    else: return ('{:<' + str(limit - cn_length) + '}').format(string)


def fill_blank(blanks: int, string: str) -> None:
    print(common_length(string, blanks), end = '', flush = True)


def format_file_size(size):
    for suffix in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size < 1024.0 or suffix == 'TiB': break
        size /= 1024.0
    return f"{size:.2f} {suffix}"


def error(text: str, error = None) -> None:
    if SILENT: raise Exception(text) from error
    fore_red()
    print('[error]', end = ' ')
    ansi_reset()
    print(text)

    if error is None: raise Exception(text)
    else: raise Exception(text) from error


def warning(text: str) -> None:
    if SILENT: return
    fore_yellow()
    print('[!]', end = ' ')
    ansi_reset()
    print(text)
    ansi_reset()


def info(text: str) -> None:
    if SILENT: return
    fore_cyan()
    print('[i]', end = ' ')
    ansi_reset()
    print(text)
    ansi_reset()


def clear():
    # for windows
    if os.name == 'nt':
        os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else: os.system('clear')


def red(x): return '\033[31m' + x + '\033[0m'
def green(x): return '\033[32m' + x + '\033[0m'
def yellow(x): return '\033[33m' + x + '\033[0m'
def blue(x): return '\033[34m' + x + '\033[0m'
def purple(x): return '\033[35m' + x + '\033[0m'
def cyan(x): return '\033[36m' + x + '\033[0m'
def annot(x): return '\033[90;3m' + x + '\033[0m'


def dtypestr(dty):
    string = str(dty)
    if string == 'category': return 'cat'
    elif 'float' in string: return string.replace('float', 'f')
    elif 'int' in string: return string.replace('int', 'i')
    elif string == 'object': return 'o'
    return string


def dtypemat(dty):
    from numpy import matrix, ndarray
    from scipy.sparse import csr_array, csr_matrix, csc_array, csc_matrix, coo_array, coo_matrix
    from pandas import DataFrame

    classdef = ''
    if isinstance(dty, DataFrame): return 'df'
    elif isinstance(dty, matrix): classdef = 'dense'
    elif isinstance(dty, ndarray): classdef = 'arr'
    elif isinstance(dty, csr_array): classdef = 'csra'
    elif isinstance(dty, csr_matrix): classdef = 'csr'
    elif isinstance(dty, csc_array): classdef = 'csca'
    elif isinstance(dty, csc_matrix): classdef = 'csc'
    elif isinstance(dty, coo_array): classdef = 'cooa'
    elif isinstance(dty, coo_matrix): classdef = 'coo'
    else: classdef = str(type(dty))
    
    size = ''
    try:
        if dty.shape[1] != dty.shape[0]: size = dty.shape[1]
    except: pass
    
    ret = ':'.join([classdef, dtypestr(dty.dtype)])
    if size != '': ret += f'({size})'
    return ret


def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def lenp(string: str):
    return len(remove_ansi_escape_sequences(string))


def wrap(array: list[str], sep = ' ', n = 100) -> list[str]:
    results = []
    cumulative = ''
    lencum = 0

    for a in array:

        lenprint = lenp(a)

        if lencum + lenprint + len(sep) <= n:
            cumulative += (sep + a)
            lencum += (len(sep) + lenprint)
        else:
            results += [cumulative[1:] if cumulative[0] == ' ' else cumulative]
            cumulative = a
            lencum = lenprint

    if cumulative != '':
        results += [cumulative[1:] if cumulative[0] == ' ' else cumulative]

    return results


progress_styles = {
    'ncols': 80,
    'ascii': '-━',
    'bar_format': '   {bar} {desc:20} {n:5d} / {total:<5d} ({elapsed} < {remaining})',
    'file': sys.stderr
}


class pprog(tqdm):
    def __init__(self, iterable = None, **kwargs):
        super().__init__(iterable, **progress_styles, **kwargs)


class pproga(tqdma):
    def __init__(self, iterable = None, **kwargs):
        super().__init__(iterable, **progress_styles, **kwargs)


def tqinfo(x):
    if SILENT: return
    tqdm.write(cyan('[i] ') + x)