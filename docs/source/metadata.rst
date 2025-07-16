
.. title::
   元数据

元数据
==============

.. _create-metadata:

创建元数据
-------------------

手动或使用某些脚本创建一个元数据表是加载数据集的第一步。这个表格是一个 TSV 表，你可以使用代码
:py:class:`exprmat.metadata` 创建这个表并保存成 TSV 文件，也可以采用其他方式或直接手动输入。
这个表具有至少六列，这六列的列名是固定的，分别是：

- `location`：指定原始数据集的位置
   这个位置可以是一个文件夹，也可以是一个文件。指定不同格式的规范是由 `modality` 列约定的。
   一个特定的 `modality` 可以接受几种受支持格式的文件。具体受支持的范围参见下表：

   .. table:: 各个模态可接受的文件类型
      :width: 100%
      :widths: 20 80

      +--------------+------------------------------------------------------------+
      | Modality     | Acceptable formats as locations                            |
      +==============+============================================================+
      | rna          | - **A folder** containing three files: `matrix.mtx(.gz)`,  |
      |              |   `barcodes.tsv(.gz)` and `features.tsv(.gz)` (the legacy  |
      |              |   10X format of `genes.tsv` is also supported.)            |
      |              | - **A `.h5ad` or `.h5` file**. If the file have variable   |
      |              |   names starting with `rna:` prefix, it is considered as   |
      |              |   reuseable processed files by the `exprmat` pipeline. It  |
      |              |   is thus expected that the `obs` table should contain the |
      |              |   columns `sample`, `modality`, `batch`, `gene`, `taxa`.   |
      |              |   Any missing columns will lead to an error. If not        |
      |              |   starting with `rna:`, it is considered as files from     |
      |              |   other pipelines e.g. 10X standard H5 files, and the      |
      |              |   `obs` columns will be overwritten by the ones given in   |
      |              |   the metadata table (even if there is already in the H5   |
      |              |   file). Variable names will be normalized to `rna:`       |
      |              |   automatically.                                           |
      |              | - **A `.csv` or `.tsv` file**. Rows are genes and columns  |
      |              |   are cells. This will be transposed automatically, so do  |
      |              |   not edit yourself.                                       |
      +--------------+------------------------------------------------------------+
      | rna.splicing | - **A `.loom` file** from the standard velocyto pipeline.  |
      |              | - **A folder**, containing `barcodes.tsv.gz`,              |
      |              |   `features.tsv.gz`, `spanning.mtx.gz`, `spliced.mtx.gz`   |
      |              |   and `unspliced.mtx.gz`. These typically from separate    |
      |              |   feature counting pipelines or from BGI's DNBC4 tools     |
      +--------------+------------------------------------------------------------+
      | rna.tcr      | - **A `.csv` file** from 10X's standard CellRanger         |
      |              |   pipelines: `filtered_contig_annotations.csv`.            |
      +--------------+------------------------------------------------------------+
      | atac         | - **A `.tsv.gz` file** containing mapped fragments. The    |
      |              |   genomic reference is automatically determined by `taxa`. |
      |              |   This results in the use of latest genome reference by    |
      |              |   default, only can be configured manually to use          |
      |              |   user-defined versions.                                   |
      +--------------+------------------------------------------------------------+

- `modality`：指定数据样本的模态类型，参见上表
- `sample`：指定样本的名称
   样本名称在每个独立模态中必须唯一，而在不同独立模态中可以重复。非独立模态的样本必须能在独立模态
   中找到同名的母样本。
   
   .. important::
      样本的名称是描述样本的唯一标识符，你需要在各个独立模态中保持唯一。 **独立模态** 包括
      ``rna`` 和 ``atac``，是样本字典的根模态，他们读取数据后会为每个样本单独生成一个
      AnnData 对象储存在样本字典中。因此他们的名字必须唯一。这种唯一性是准确指示样本所必要的。
      在不同的独立模态之间样本名称可以重复，例如，你可以有一个 ``wt1`` 的 ``rna`` 样本，同时
      有一个 ``wt1`` 的 ``atac`` 样本，这种情况下，程序将认为这两个模态来源于一个样本，并
      进行多组学整合。 **非独立模态** 是一个独立模态样本上的附加数据，他们一般在模态名称中
      含有 “.”（例如 ``rna.tcr`` ， ``rna.splicing`` ） 这些模态并不会新建一个样本，而是
      根据指定的样本名寻找独立模态样本，并将附加数据插入他们的 AnnData 对象中。因此，他们必须
      在其所属的独立模态（非独立模态名的根名称，即 ``rna``）中有一个同名的独立模态样本。 

- `batch`：指定样本的批次，用于批次整合和校正
- `group`：内容格式是自由的，一般用来储存实验分组条件
- `taxa`：样本来源物种名
   对于 RNA 数据，我们会根据物种名在自带的数据库中查找物种的基因列表。对于 ATAC 数据，我们会
   根据物种名找到最新的参考基因组。默认情况下，安装的数据库只包含 ``hsa`` 和 ``mmu`` 两个物种。
   你可以自行构建或下载其他感兴趣的物种数据库。

.. code-block:: python
   :linenos:
   :caption: 使用构造函数直接创建元数据表

   import exprmat as em

   meta = em.metadata(
       locations    = [
           # one sample from over-expression group.
           './oe1/filtered',
           './oe1/velocyto/splicing.loom',
           # one sample from wild type group.
           './wt1/filtered',
           './wt1/velocyto/splicing.loom',
       ],
       modality     = ['rna', 'rna.splicing', 'rna', 'rna.splicing'],
       default_taxa = ['mmu'] * 4,
       batches      = ['b1', 'b1', 'b2', 'b2'],
       names        = ['oe', 'oe', 'wt', 'wt'],
       groups       = [
           'somatic(cond(cd8, oe(A)))',
           'somatic(cond(cd8, oe(A)))',
           'somatic(wt)',
           'somatic(wt)'
       ]
   )

.. note::
   由于向后兼容问题，:py:class:`exprmat.metadata` 的构造函数中的参数名并不完全是 TSV 表的
   列名，例如 ``default_taxa`` 参数对应 ``taxa`` 列， ``names`` 参数对应 ``sample`` 列。如果
   你决定手写元数据表，请使用下面示例表中规定的列名称。

最后，你应该调用 :py:func:`exprmat.metadata.save` 保存到硬盘上的 TSV 文件。

.. code-block:: python
   :linenos:
   :caption: 保存元数据表

   meta.save('metadata.tsv')

这将会被保存为一个 TSV 文件：

.. csv-table::
   :header-rows: 1
   :file: _static/files/example-metadata.csv

.. dropdown:: ``metadata`` 参考

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

.. caution::

   如果你选择手动创建元数据表，请一定确认你的表格是 Tab 分隔的。读取的 TSV 表必须含有列名，
   且不允许行名。列的顺序是可以颠倒的，但是一般推荐使用默认的顺序。注意一些编辑器在没有设置的
   情况下会自动将输入的 Tab 转换成空格，你需要确认你输入的 Tab 确实是制表符。例如在
   Visual Studio Code 中，你需要如下设置：

   .. figure:: _static/images/change-tab-1.png

.. dropdown:: ``load_metadata`` 参考
   :open:
   
   从一个保存的 TSV 表读入元数据。这个函数会检查表中含有约定的必需列。否则会提示错误。

   .. autofunction:: exprmat.load_metadata
   
从元数据新建数据集
----------------------------

元数据准备好后，你已经拥有了一个指挥软件包如何、何处读取各个样本数据的指南，你需要根据这个指南
创建一个数据集。程序包将会按照你的指定类型自动处理并加载各个样本，创建一个内存数据集。
参见 :py:class:`exprmat.experiment`.