python setup.py sdist build
twine upload dist/*.tar.gz
rm -r build
rm -r dist
pip install -U -e .
