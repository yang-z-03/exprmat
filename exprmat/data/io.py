
import gzip


def zopen(filename: str, mode = "r"):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode, encoding = "utf-8")
    else: return open(filename, mode, encoding = "utf-8")
