
元数据
==============

创建元数据
-------------------

手动或使用某些脚本创建一个元数据表是加载数据集的第一步。这个表格是一个 TSV 表，你可以使用代码
:py:class:`exprmat.metadata` 创建这个表并保存成 TSV 文件，也可以采用其他方式或直接手动输入。
这个表具有至少六列，这六列的列名是固定的，分别是：

- `location`：指定原始数据集的位置


参考手册
-------------------

导出函数参考

.. autoclass:: exprmat.metadata
    :members: save, define_column, set_paste, set_fraction, set_if_starts, set_if_ends, set_if_contains

.. autofunction:: exprmat.load_metadata