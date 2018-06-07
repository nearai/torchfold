from setuptools import setup, find_packages
from os import path

VERSION = "0.1.0"

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='torchfold',
    version=VERSION,
    description='Dynamic Batching with PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=["*_test.py"]),
    license='Apache License, Version 2.0',
    author='Illia Polosukhin, NEAR Inc',
    author_email="illia@near.ai",
    project_urls={
        'Blog Post': "http://near.ai/articles/2017-09-06-PyTorch-Dynamic-Batching/",
        'Source': "https://github.com/nearai/torchfold",
    },
)

