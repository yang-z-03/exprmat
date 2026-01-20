
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

### Database installation

The package `exprmat` do not come with its reference database. You need to setup
the database and configure the package to find it properly, or else most of the
features from the package will fail.

This package ships with a tool to fetch database distribution on Alibaba Cloud
object storage service. You will first need to register an account and request
a key-secret pair to authenticate your identity, before starting out to download
specified version of the database distribution using the tool `bsync-fetch` 
installed alongside the `exprmat` package.

```
usage: bsync-fetch [-h] --id ID --secret SECRET 
                        --bucket BUCKET [--endpoint ENDPOINT] --version VERSION

fetch from remote bucket.

options:
  -h, --help           show this help message and exit
  --id ID              The requester access id.
  --secret SECRET      The requester access secret.
  --bucket BUCKET      The name of the bucket.
  --endpoint ENDPOINT  The domain names that other services can use to access OSS.
  --version VERSION    The version to fetch from remote.
```

Suppose you are installing the database version `0.1.50` to `~/database`.

```bash
# enter the intended path of installation
cd ~/database

# download the version 0.1.50 of database distribution
# you must download the same version of the database with the package. the database
# and package content are published strictly synchronically.
bsync-fetch --id <your-id> --secret <your-api-secret> \
            --bucket exprmat-data --endpoint oss-cn-wuhan-lr.aliyuncs.com \
            --version 0.1.50
```

The authentication tokens passed to `--id` and `--secret` is provided by the
Alibaba Cloud service (See the [Documentation](https://help.aliyun.com/zh/ram/user-guide/create-an-accesskey-pair)
for details)

### Licensing

The original part of the source code is licensed under GNU GPLv3.
The database download tool bsync-fetch (developed as a dependency package for
exprmat, in python package `bincsync`) is only allowed for internal use and not
for re-distribution.

```
exprmat - Routines to manipulate expression matrices
Copyright (C) 2025 - 2026 Zheng Yang (杨政) <xornent@outlook.com>

exprmat is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions. 
You should have received a copy of the GNU General Public License
version 3 along with this program. If not, see <http://www.gnu.org/licenses/>.

The GNU General Public License does not permit incorporating 
your program into proprietary programs.
```

I acknowledge the following author(s) for modifying and integrating their work
in the form of source code to this package.

```
parts of the source code under /preprocessing comes from the scanpy project.

  BSD 3-Clause License. For full text of the license, see 
  <https://opensource.org/license/BSD-3-clause>
  Copyright (c) 2025 scverse®
  Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab

function embedding_atlas (file /reduction/plot.py) comes from the omicverse project.

  GNU GPLv3. (same as this project)
  Copyright (c) 2024 112 Lab

most of the code from /snapatac2 and /snapatac2-core comes from the SnapATAC
project, including the rust build system and the compiled binaries.

  The MIT license. For full text of the license, see
  <https://opensource.org/license/MIT>
  Copyright (c) 2022-2024 Kai Zhang

file /clustering/sc3.py comes from the sc3s project.

  GNU GPLv3. (same as this project)
  Copyright (c) Quah, F.X. and Hemberg, M. and contributors.

file /clustering/nmf.py comes from cNMF project.

  The MIT license
  Copyright (c) 2019 Dylan Kotliar

directory /cnv comes from infercnvpy project.
  
  BSD 3-Clause License
  Copyright (c) 2022, Gregor Sturm

directory /deconv comes from the TAPE project.

  GNU GPLv3. (same as this project)
  Copyright (c) The contributors.

directory /metacell comes from the MetaQ project.
  
  MIT License
  Copyright (c) 2024 XLearning Group

directory /grn comes from the scenicplus project.

  Academic Non-commercial Software License Agreement.
  For full text of the license, see <https://github.com/aertslab/scenicplus>
  Copyright (c) The contributors.

directory /lr (but not /lr/icnet) comes from the lianapy project

  BSD 3-Clause License
  Copyright (c) 2025, Daniel Dimitrov

directory /peaks/idr.py comes from the idr project

  GNU GPLv2. For copyright details, see <https://github.com/nboley/idr>
  Copyright (c) The contributors

function diffmap comes from the pydiffmap project

  MIT License
  Copyright (c) 2018 Ralf Banisch, Erik Henning Thiede, Zofia Trstanova

principle tree fitting and trajectory analysis comes from the scFates project

  BSD 3-Clause License
  Copyright (c) 2020, Louis Faure

file /trajectory/cytotrace.py comes from the CytoTRACE2 project

  Stanford Non-commercial Software License Agreement
  Copyright (c) The contributors.

rna velocity fitting comes from the scvelo project

  BSD 3-Clause License
  Copyright (c) 2018, Theis Lab

directory /deseq comes from the pyDEseq2 project

  MIT License
  Copyright (c) 2022 Owkin

```
