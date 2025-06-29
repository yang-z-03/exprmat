
# Exprmat

![wheels](https://img.shields.io/pypi/wheel/exprmat)
![version](https://img.shields.io/pypi/v/exprmat)

Exprmat (short for **expr**ession **mat**rix) is a routine package for manipulation 
of single cell expression matrices. It is built based on commonly accepted python
infrastructures for single cell data management (Scanpy, SnapAtac2, and MuData) and
provides integration and wrappers of common routines for preprocessing, annotating,
clustering, visualizations, and downstream analyses with a common interface.

You may refer to these places:

* [Source code repository](https://github.com/yang-z-03/exprmat)
* [Releases in PyPI](https://pypi.org/project/exprmat)
* [Documentation](https://exprmat.readthedocs.io/en/latest)
* Due to the extensive size (and the size limit imposed by PyPI repositories), we
  do not contain the genomic references and databases for transcriptional factors
  and motifs, thus functions related to accessible genomic regions are not available
  by default. (We still include the gene tables, feature tables, prebuilt genesets, 
  homology tables, and ligand-receptor databases for *Homo sapiens* and *Mus musculus* 
  with the package content). You may download and locate the full reference databases 
  for your taxa and instruct the package to load it correctly. 
  Pre-built databases are distributed elsewhere, and for now we are still finding 
  sites to host them. 