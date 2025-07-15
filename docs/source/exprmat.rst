
exprmat 主模块
======================

核心导出类 
^^^^^^^^^^^^^^^^^^^

程序包主要导出两个核心类：`metadata` 和 `experiment`. 这两个类分别是实验元数据表的封装，
以及内存中的数据集表示。一般情况下，你需要先创建元数据表，根据其中的信息指定在何处，以何种
类型方式读取硬盘上的数据，软件包根据指定的数据类型自动转换成统一的内存数据格式，并储存在一个
`experiment` 对象中。你可以操作这个对象，并保存到硬盘上。一个硬盘数据集格式是一个具有特定
结构的目录。你可以用 tar 打包并传输共享。

.. py:class:: experiment

    See :py:class:`exprmat.experiment` and :py:func:`exprmat.load_experiment`


.. py:class:: metadata

    See :py:class:`exprmat.metadata` and :py:func:`exprmat.load_metadata`


实用函数
^^^^^^^^^^^^^^^^^

程序包的主模块导出一些标识程序版本、数据库版本、内存占用、和操作工作目录的函数，你可以使用这些
函数为接下来的程序操作做准备。

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

开始使用 exprmat 程序包
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

你接下来可以阅读以下文档，这些教程简要地介绍了程序包的基本工作流程