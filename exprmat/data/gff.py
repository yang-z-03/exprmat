
# utilities of reading gff files from raw or gzipped archives.

from exprmat.ansi import error


def readgff(fname, encoding = 'utf-8'):
    
    with open(fname, 'r', encoding = encoding) as f:
        col, _, _ = scangff(f)
        return readgff_fp(f, col)
    
    pass


def readgff_gzipped(fname, encoding = 'utf-8'):

    import gzip
    import io

    with gzip.open(fname, 'rb') as f:
        with io.TextIOWrapper(f, encoding = encoding) as enc:
            col, _, _ = scangff(enc)
            return readgff_fp(enc, col)
    
    pass


def gffcols(fname, encoding = 'utf-8'):
    
    with open(fname, 'r', encoding = encoding) as f:
        col, _, type = scangff(f)
    
    return col, type


def gffcols_gzipped(fname, encoding = 'utf-8'):

    import gzip
    import io

    with gzip.open(fname, 'rb') as f:
        with io.TextIOWrapper(f, encoding = encoding) as enc:
            col, _, type = scangff(enc)
    
    return col, type


# scan the gff table, and get the full union of available columns (though some
# may not present in certain kind of rows.) this help later specify the classes
# and columns of a full union table, which contains all of the gff metadata
# column information. also returns the available data row numbers.

def scangff(fp):

    line = fp.readline().replace('\n', '')
    lineno = 0
    rowno = 0
    cols = ['.seqid', '.source', '.type', '.start', '.end', '.score', '.strand', '.phase']
    attrcols = []
    typecols = {}
    print('', end = '')

    while line:
        
        if len(line) == 0:
            line = fp.readline().replace('\n', '')
            lineno += 1; continue
        
        elif line[0] == '#':
            line = fp.readline().replace('\n', '')
            lineno += 1; continue
        
        else: rowno += 1

        tabdelim = line.split('\t')
        if len(tabdelim) != 9: 
            print('')
            error('line {0} does not have exactly 9 columns!'.format(lineno))
        
        if len(tabdelim[8]) > 0:
            attribs = tabdelim[8].split(';')
            rnames = [x.split('=')[0] for x in attribs]
            
            typename = tabdelim[2]
            if not typename in typecols.keys(): typecols[typename] = { '.': 1 }
            else: typecols[typename]['.'] += 1

            for rname in rnames:
                if not rname in attrcols: attrcols += [rname]

                if not rname in typecols[typename].keys(): typecols[typename][rname] = 1
                else: typecols[typename][rname] += 1

        line = fp.readline().replace('\n', '')
        lineno += 1

        if rowno % 1000 == 0:
            print('\rscangff(fp) progress:', rowno, end = '')
    
    fp.seek(0) # seek back to the origin.
    print('')
    
    return cols + attrcols, rowno, typecols


def readgff_fp(fp, columns):
    
    line = fp.readline().replace('\n', '')
    print('', end = '')

    lines = { }
    for i in columns: lines[i] = []
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

        # ['.seqid', '.source', '.type', '.start', '.end', '.score', '.strand', '.phase']
        
        tabdelim = line.split('\t')
        lines['.seqid'] += [tabdelim[0]]
        lines['.source'] += [tabdelim[1]]
        lines['.type'] += [tabdelim[2]]
        lines['.start'] += [tabdelim[3]]
        lines['.end'] += [tabdelim[4]]
        lines['.score'] += [tabdelim[5]]
        lines['.strand'] += [tabdelim[6]]
        lines['.phase'] += [tabdelim[7]]

        attl = columns[8:]
        if len(tabdelim[8]) > 0:
            attribs = tabdelim[8].split(';')
            for attrib in attribs:
                namevalue = attrib.split('=')
                lines[namevalue[0]] += [namevalue[1]]
                attl.remove(namevalue[0])
        
        for a in attl:
            lines[a] += [None]

        line = fp.readline().replace('\n', '')
        lineno += 1

        if rowno % 1000 == 0:
            print('\rreadgff_fp(fp, columns) progress:', rowno, end = '')
    
    print('')
    return lines