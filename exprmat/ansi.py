
# ./shared/ansi.py:
#   provides basic ansi console coloring and cursor manipulation support.
#   considering adding later another alternative implementation of the same 
#   interface for terminal/emulators that do not support ansi.
# 
# license: gplv3. <https://www.gnu.org/licenses>
# contact: yang-z <xornent at outlook dot com>

import re
import os
import sys

# foregrounds -----------------------------------------------------------------

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

# reset ansi controls ---------------------------------------------------------
    
def ansi_reset() -> None:
    print('\033[0m', end = '')

# backgrounds -----------------------------------------------------------------
    
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

# cursor operations

def ansi_move_cursor(line: int, col: int) -> None:
    if line == 0: pass
    elif line > 0: print('\033[{0}B'.format(str(line)), end = '', flush = True) # moves down
    elif line < 0: print('\033[{0}A'.format(str(-line)), end = '', flush = True) # moves up

    if col == 0: pass
    elif col > 0: print('\033[{0}C'.format(str(col)), end = '', flush = True) # moves right
    elif col < 0: print('\033[{0}D'.format(str(-col)), end = '', flush = True) # moves left

def fill_blank(blanks: int) -> None:
    print(' ' * blanks, end = '', flush = True)

def common_length(string: str, limit: int) -> str:

    cn_re = '[\u201c-\u201d\u3001-\u3011\uff08-\uff1f\u4e00-\u9fa5]'
    cn = re.findall(cn_re, string)
    cn_length = len(cn)
    
    if limit <= 5:
        x_str = string[-limit:]
        cn = len(re.findall(cn_re, x_str))
        while cn + len(x_str) > limit:
            x_str = x_str[1:]
            cn = len(re.findall(cn_re, x_str))
        return x_str
    
    elif len(string) + cn_length > limit:
        x_str = string[-(limit - 4):]
        cn = len(re.findall(cn_re, x_str))
        while cn + len(x_str) > limit - 4:
            x_str = x_str[1:]
            cn = len(re.findall(cn_re, x_str))
        return '... ' + x_str
        
    else: return ('{:<' + str(limit - cn_length) + '}').format(string)

def fill_blank(blanks: int, string: str) -> None:
    print(common_length(string, blanks), end = '', flush = True)

def line_start() -> None:
    print('\r', end = '')

def print_message(color, title, path, overwrite = True):
    if overwrite:
        print('\r{0}[{1}]\033[0m'.format(color, title), common_length(path, 70), end = '')
        return '{0}[{1}]\033[0m'.format(color, title) + ' ' + common_length(path, 70)
    else:
        print('{0}[{1}]\033[0m'.format(color, title), common_length(path, 70))
        return '{0}[{1}]\033[0m'.format(color, title) + ' ' + common_length(path, 70)
    
def format_file_size(size):
    for suffix in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if size < 1024.0 or suffix == 'TiB': break
        size /= 1024.0
    return f"{size:.2f} {suffix}"

# prompt messages -------------------------------------------------------------
    
def error(text: str) -> None:
    fore_red()
    print('[error]', end = ' ')
    ansi_reset()
    print(text)
    raise Exception()

def warning(text: str) -> None:
    fore_yellow()
    print('[!]', end = ' ')
    ansi_reset()
    print(text)
    ansi_reset()

def info(text: str) -> None:
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
