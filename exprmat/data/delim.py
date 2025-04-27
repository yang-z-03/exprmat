
# utilities of reading delimited files from raw or gzipped archives.

from exprmat.ansi import error


def readdelim(fname, encoding = 'utf-8', 
              delim = '\t', ignore_first_column = True):
    
    with open(fname, 'r', encoding = encoding) as f:
        return readdelim_fp(f, delim, ignore_first_column)
    
    pass


def readdelim_gzipped(fname, encoding = 'utf-8', 
                      delim = '\t', ignore_first_column = True):

    import gzip
    import io

    with gzip.open(fname, 'rb') as f:
        with io.TextIOWrapper(f, encoding = encoding) as enc:
            return readdelim_fp(enc, delim, ignore_first_column)
    
    pass


def readdelim_fp(fp, delim = '\t', ignore_first_column = True):
    
    line = fp.readline().replace('\n', '')
    print('', end = '')
    
    header = False
    header_cols = []

    lines = { }
    lineno = 0
    rowno = 0

    while line:

        # the file has been scanned by scangff. so there is no format problems.
        # we do not perform any checks.

        if len(line) == 0:
            line = fp.readline().replace('\n', '')
            lineno += 1; continue
        
        elif line[0] == '#':
            line = fp.readline().replace('\n', '')
            lineno += 1; continue
        
        else: rowno += 1

        if not header:

            cols = line.split(delim)
            if len(cols) > 1:
                if ignore_first_column:
                    for x in cols[1:]: lines[x] = []
                    header_cols = cols[1:]
                else:
                    for x in cols: lines[x] = []
                    header_cols = cols
            else:
                if ignore_first_column: return {}
                else: lines[cols[0]] = []; header_cols = cols

            header = True
            line = fp.readline().replace('\n', '')
            lineno += 1; continue
        
        cols = line.split(delim)
        if ignore_first_column: cols = cols[1:]

        i = 0
        for c in header_cols:
            lines[c] += [cols[i]]
            i += 1

        line = fp.readline().replace('\n', '')
        lineno += 1

        if rowno % 1000 == 0:
            print('\rreaddelim_fp(fp) progress:', rowno, end = '')
    
    print('')
    return lines