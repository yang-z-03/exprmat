
from exprmat.lr.methods.cellchat import cellchat
from exprmat.lr.methods.cellphonedb import cellphonedb
from exprmat.lr.methods.connectome import connectome
from exprmat.lr.methods.geomean import geomean
from exprmat.lr.methods.logfc import logfc
from exprmat.lr.methods.natmi import natmi
from exprmat.lr.methods.scseqcomm import scseqcomm
from exprmat.lr.methods.singlecellsignalr import singlecellsignalr

from exprmat.lr.methods.rank_aggregate import ra_config, aggregate_method

rank_aggregate = aggregate_method(
    scoring_methods = ra_config,
    methods = [cellchat, cellphonedb, connectome, logfc, natmi, scseqcomm, singlecellsignalr]
)

flavors = {
    'cellchat': cellchat,
    'cellphonedb': cellphonedb,
    'connectome': connectome,
    'geomean': geomean,
    'logfc': logfc,
    'natmi': natmi,
    'scseqcomm': scseqcomm,
    'singlecellsignalr': singlecellsignalr,
    'ra': rank_aggregate
}