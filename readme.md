
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

Suppose you are installing the database version `0.1.25` to `~/database`.

```bash
# enter the intended path of installation
cd ~/database

# download the version 0.1.25 of database distribution
# you must download the same version of the database with the package. the database
# and package content are published strictly synchronically.
bsync-fetch --id <your-id> --secret <your-api-secret> \
            --bucket exprmat-data --endpoint oss-cn-wuhan-lr.aliyuncs.com \
            --version 0.1.25
```

The authentication tokens passed to `--id` and `--secret` is provided by the
Alibaba Cloud service (See the [Documentation](https://help.aliyun.com/zh/ram/user-guide/create-an-accesskey-pair)
for details)