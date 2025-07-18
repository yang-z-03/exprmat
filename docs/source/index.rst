
.. title::
   Exprmat 文档

exprmat 文档
=====================

.. figure:: https://img.shields.io/pypi/v/exprmat

安装和配置
------------------

为了避免版本冲突，我们推荐你在安装包之前创建一个 Python 或 Anaconda 虚拟环境，哪怕你不使用
Anaconda 来管理包。请选择使用一个固定的包管理器，例如，如果你喜欢使用 ``pip``，那就请尽可能
使用 ``pip`` 安装所有 Python 包。本软件包发布在 PyPI 储存库上，是一个 Python & Rust 
混合包，所以你需要安装 Rust 编译器才可以从源代码编译该包。Cargo 要求你在编译过程中保持网络连接，
否则将无法自动下载 Rust 依赖项而导致编译失败。对于常见的 Linux 平台，我们提供了预编译的 Wheel
二进制分发。本包支持几乎所有 Linux 系统，不支持 macOS 和 Windows。

除了软件包，你还需要安装对应版本的物种数据库。否则大多数功能将无法使用。物种数据库是我们手动
整理的，包含物种的基因表、参考基因组注释、染色体结构、配体受体数据库、基因集数据库、同源基因表、
和转录因子库。目前我们只提供人 ``hsa`` 和小鼠 ``mmu`` 的物种数据库。目前处于早期开发阶段，
并没有生成其他任意物种数据库的自动化方法。

1. 首先根据你的 GPU 和 CPU 情况安装对应版本的 PyTorch （要求 2.0 版本以上）

   .. code-block:: bash 

      # for a cpu-only version:
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

      # you may install the corresponding version if you have a supported gpu.
      # this is only an example.

2. 安装 Rust 编译器

   .. code-block:: bash

      conda install rust

3. 安装软件包

   .. code-block:: bash

      pip install -U exprmat

4. 安装数据库
   软件包安装后，我们自动安装了一个配套辅助工具，可以增量下载和安装 exprmat 数据库。我们的数据库
   发布在阿里云对象储存服务上，由请求者付费下载。因此，你需要一个阿里云注册账号和 Token 标识符及密码
   你可以通过下面的命令自动使用你的账号安装某一个版本的数据库。注意，数据库的版本 **务必** 与软件
   包的版本吻合，否则将出现错误。

   .. code-block:: bash
      
      # first, navigate to your intended location to install.
      # this can be any location that are accessible as you like.
      cd /your/path/to/database

      # please fill in the token and secret. and select the same version as your
      # python package (e.g. "0.1.30")
      bsync-fetch --id "your-aliyun-token" \
                  --secret "your-token-secret" \
                  --endpoint oss-cn-wuhan-lr.aliyuncs.com \
                  --bucket exprmat-data \
                  --version "exprmat-version"

5. 配置软件包数据库位置
   你可以在你想要的任何路径下安装数据库，安装完成后，你需要告诉软件包数据库位置。你可以在用户根目录
   下创建配置文件如下：

   .. code-block:: json
      :linenos:
      :caption: ~/.exprmatrc

      {
          "data": "/your/path/to/database"
      }

6. 配置完成，你可以在 Python REPL 中输入 ``import exprmat``，若无异常说明安装成功。


开始使用
--------------

1. :ref:`create-metadata`

.. toctree::
    :titlesonly:
    :hidden:

    exprmat
    metadata
    experiment/index