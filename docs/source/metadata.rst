
元数据
==============

创建元数据
-------------------

手动或使用某些脚本创建一个元数据表是加载数据集的第一步。这个表格是一个 TSV 表，你可以使用代码
:py:class:`exprmat.metadata` 创建这个表并保存成 TSV 文件，也可以采用其他方式或直接手动输入。
这个表具有至少六列，这六列的列名是固定的，分别是：

- `location`：指定原始数据集的位置

最后，你应该调用 :py:func:`exprmat.metadata.save` 保存到硬盘上的 TSV 文件。

.. dropdown:: ``metadata`` 参考
   :open:

   你可以在构造函数中提供六个必需列的信息，并使用 :py:func:`define_column` 定义新的自定义
   数据列。你可以在 :py:attr:`metadata.dataframe` 中直接操作这个数据表对象。
   这个类只是对 Pandas 数据表的一个简单封装。尽管如此，我们还是建议你符合一定的命名规则，
   以便使用更加容易读懂的内建方法。

   .. autoclass:: exprmat.metadata
      :members: save, define_column, set_paste, set_fraction, set_if_starts,
                set_if_ends, set_if_contains

从硬盘上读取元数据
---------------------------

你可以使用 :py:func:`exprmat.load_metadata` 读入 TSV 元数据表。

.. dropdown:: ``load_metadata`` 参考
   :open:
   
   从一个保存的 TSV 表读入元数据。这个函数会检查表中含有约定的必需列。否则会提示错误。

   .. autofunction:: exprmat.load_metadata
   
从元数据新建数据集
----------------------------

元数据准备好后，你已经拥有了一个指挥软件包如何、何处读取各个样本数据的指南，你需要根据这个指南
创建一个数据集。程序包将会按照你的指定类型自动处理并加载各个样本，创建一个内存数据集。
参见 :py:class:`exprmat.experiment`.