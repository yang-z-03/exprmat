
s_gene_sets = {
    'mmu': [ # the original list
        'Atad2', 'Blm', 'Brip1', 'Casp8ap2', 'Ccne2', 'Cdc45', 'Cdc6', 'Cdca7', 
        'Chaf1b', 'Clspn', 'Dscc1', 'Dtl', 'E2f8', 'Exo1', 'Fen1', 'Gins2', 'Gmnn', 
        'Hells', 'Mcm2', 'Mcm4', 'Mcm5', 'Mcm6', 'Msh2', 'Nasp', 'Pcna', 'Pola1', 
        'Pold3', 'Prim1', 'Rad51', 'Rad51ap1', 'Rfc2', 'Rpa2', 'Rrm1', 'Rrm2', 
        'Slbp', 'Tipin', 'Tyms', 'Ubr7', 'Uhrf1', 'Ung', 'Usp1', 'Wdr76'
    ],
    'hsa': [ # translated from mice
        'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2', 
        'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'MLF1IP', 'HELLS', 'RFC2', 
        'RPA2', 'NASP', 'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 
        'POLD3', 'MSH2', 'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 
        'TIPIN', 'DSCC1', 'BLM', 'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 
        'BRIP1', 'E2F8'
    ]
}

g2m_genesets = {
    'mmu': [
        'Anln', 'Anp32e', 'Aurka', 'Aurkb', 'Birc5', 'Bub1', 'Cbx5', 'Ccnb2', 
        'Cdc20', 'Cdc25c', 'Cdca2', 'Cdca3', 'Cdca8', 'Cdk1', 'Cenpa', 'Cenpe', 
        'Cenpf', 'Ckap2', 'Ckap2l', 'Ckap5', 'Cks1b', 'Cks2', 'Ctcf', 'Dlgap5', 
        'Ect2', 'G2e3', 'Gas2l3', 'Gtse1', 'Hjurp', 'Hmgb2', 'Hmmr', 'Kif11', 
        'Kif20b', 'Kif23', 'Kif2c', 'Lbr', 'Mki67', 'Ncapd2', 'Ndc80', 'Nek2', 
        'Nuf2', 'Nusap1', 'Psrc1', 'Rangap1', 'Smc4', 'Tacc3', 'Tmpo', 'Top2a', 
        'Tpx2', 'Ttk', 'Tubb4b', 'Ube2c'
    ],
    'hsa': [
        'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80', 
        'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A', 
        'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E', 
        'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK', 
        'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 
        'ECT2', 'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE', 
        'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
    ]
}

preset_genesets = {
    's': s_gene_sets,
    'g2m': g2m_genesets
}
