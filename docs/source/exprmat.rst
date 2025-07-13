
Export module contents
======================

Utility functions
^^^^^^^^^^^^^^^^^

These are the top level utility functions to examine and configure the package
behaviors and versioning information.

.. py:function:: version_db()

    Returns the version string of the database registered and found by the package.
    Example: ``"0.1.29"``.


.. py:function:: version()

    Returns the version of the main source package. You should ensure the version
    of the package be identical to the database. Otherwise bugs may happen anywhere.
    In the format of major, minor and revision in integral tuples of 3.
    Example: ``(0, 1, 29)``


.. py:function:: memory()

    Print the currently dedicated and virtual memory.
    Do not return anything.


.. py:function:: setwd(path: str = '.')

    Set the working directory. An alias of :py:func:`os.chdir`.


.. py:function:: getwd()

    Get the working directory. An alias of :py:func:`os.getcwd`.


.. py:function:: locate_data(path: str = '.')

    Set the database folder directly.


Main export classes
^^^^^^^^^^^^^^^^^^^

Metadata and experiments are the main export entries of the package. Nearly all
of the package's functions is performed on these objects. You may also dump the
objects to on-disk formats for dataset sharing.

.. py:class:: experiment

    See :py:class:`exprmat.reader.experiment`


.. py:class:: metadata

    See :py:class:`exprmat.reader.metadata`