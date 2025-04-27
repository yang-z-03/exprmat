
from exprmat.ansi import error

class range(list):

    # a range is an ordered set of tuples.
    # for example [(1, 4), (5, 7)]

    def __init__(self):
        pass

    def __init__(self, content, strand, is_dedup = False):
        super(range, self).__init__(content)
        self.strand = strand
        self.is_dedup = is_dedup
        if len(content) <= 1: self.is_dedup = True

    pass


def merge(r1:range, r2:range) -> range:

    if r1.strand != r2.strand: 
        error('attempt to merge ranges on different strand.')

    if not r1.is_dedup: r1 = dedup(r1)
    if not r2.is_dedup: r2 = dedup(r2)

    if len(r2) == 0: return r1
    if len(r2) > 1: return merge(merge(r1, 
                                       range([r2[0]], r2.strand, True)), 
                                       range(r2[1:], r2.strand, True))

    x, y = r2[0]

    # we would by default create a new range item for this
    result_ranges = []
    for r in r1:
        rx, ry = r # where ry must >= rx.
        if x > ry: result_ranges += [r]; continue # falls entirely to the right.
        if y < rx: result_ranges += [r]; continue # falls entirely to the left
        
        # by now, x <= ry and y >= rx.
        # this may be the case of several subranges in r1.
        if rx < x: x = rx
        if ry > y: y = ry
    
    result_ranges += [(x, y)]
    rr = range(result_ranges, r2.strand, True)
    sort(rr); return rr


def dedup(r:range) -> range:

    if len(r) <= 1: return r
    if len(r) >= 3: return merge(range([r[0]], r.strand, True), 
                                 dedup(range(r[1:], r.strand)))
    
    return merge(range([r[0]], r.strand, True), 
                 range([r[1]], r.strand, True))


def exclude(r1:range, r2:range):

    if r1.strand != r2.strand: 
        error('attempt to merge ranges on different strand.')

    if not r1.is_dedup: r1 = dedup(r1)
    if not r2.is_dedup: r2 = dedup(r2)

    if len(r2) == 0: return r1
    if len(r2) > 1: return exclude(exclude(r1, range([r2[0]], r2.strand)), 
                                   range(r2[1:], r2.strand, True))

    x, y = r2[0]

    # we would by default create a new range item for this
    result_ranges = []
    for r in r1:
        rx, ry = r # where ry must >= rx.
        if x > ry: result_ranges += [r]; continue # falls entirely to the right.
        if y < rx: result_ranges += [r]; continue # falls entirely to the left
        
        if x <= rx:
            if y < ry: result_ranges += [(y + 1, ry)]
            else: continue # completely diminished
        else:
            if y < ry: result_ranges += [(rx, x - 1), (y + 1, ry)]
            else: result_ranges += [(rx, x - 1)]

    rr = range(result_ranges, r2.strand, True)
    sort(rr); return rr


def flank(r1:range, l):

    return exclude(extend(r1, l), r1)


def extend(r1:range, l):

    result_ranges = []
    for r in r1:
        rx, ry = r
        result_ranges += [(rx - l, ry + l)]

    return dedup(range(result_ranges, r1.strand))


def shrink(r1:range, l):

    if not r1.is_dedup: r1 = dedup(r1)
    result_ranges = []
    for r in r1:
        rx, ry = r
        leng = ry - rx + 1
        if leng <= 2 * l: continue
        else: result_ranges += [(rx + l, ry - l)]

    return range(result_ranges, r1.strand, True)


def locate(r1:range, i) -> tuple | None:

    if len(r1) == 0: return None
    if i < r1[0][0]: return None
    if i > r1[len(r1) - 1][1]: return None

    for t in r1:
        if i >= t[0] and i <= t[1]: return t

    return None


def inside(r1:range, i) -> bool:

    if len(r1) == 0: return False
    if i < r1[0][0]: return False
    if i > r1[len(r1) - 1][1]: return False

    for t in r1:
        if i >= t[0] and i <= t[1]: return True

    return False


def outside(r1:range, i) -> bool:

    return not inside(r1, i)


def sort(r:range) -> range:

    r.sort(key = lambda x: x[0])


def length(r:range) -> int:

    if not r.is_dedup: r = dedup(r)
    sum = 0
    for rg in r:
        sum += (rg[1] - rg[0] + 1)

    return sum