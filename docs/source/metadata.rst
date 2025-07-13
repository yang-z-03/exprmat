
Metadata class
==============

Creating your metadata table
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You need a metadata table to instruct `exprmat` package on where to find and how to load
the given data directories. It is also convenient to append sample-level metadata
information attached to the datasets.


References
^^^^^^^^^^

Reference manual of export functions

.. autoclass:: exprmat.reader.metadata.metadata
    :members: save, define_column, set_paste, set_fraction, set_if_starts, set_if_ends, set_if_contains

.. autofunction:: exprmat.reader.metadata.load_metadata